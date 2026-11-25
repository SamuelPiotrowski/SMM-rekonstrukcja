"""
Porównanie cech powierzchni: oznaczone powierzchnie styku (przełomy) vs reszta kości (kora).

Po co: żeby wiedzieć, po czym algorytm wykrywania (src/contacts.py, @detector) może
odróżnić przełom od kory - i jak bardzo każda cecha się do tego nadaje.

Dla każdego wierzchołka siatki kości liczone są cechy (FEATURES):
    - szorstkość i rozrzut normalnych w kilku promieniach (kształt w małej skali),
    - wypukłość (znak: > 0 wypukła, < 0 wklęsła),
    - długość krawędzi siatki (skaner zagęszcza siatkę tam, gdzie widzi szczegóły),
    - kolor ze skanera: jasność, nasycenie, plamistość (zmienność jasności w 1 mm),
    - ułożenie względem osi kości: kąt normalnej do osi i położenie wzdłuż osi.
Potem dla każdej ręcznie oznaczonej powierzchni styku porównywane są rozkłady cechy
na niej i na korze tej samej kości. Pomijane są:
    - pas boundary_margin_mm przy otwartym brzegu skanu (postrzępiony, to nie przełom),
    - pas label_band_mm po obu stronach granicy oznaczenia (ręczne oznaczenie nie jest
      idealnie dokładne, a cechy liczone w promieniu "widzą" obie strony).

Miara rozdzielności: AUC = prawdopodobieństwo, że losowy punkt przełomu ma WIĘKSZĄ
wartość cechy niż losowy punkt kory (ważone polem). 0.5 = cecha nic nie mówi,
1.0 = przełom zawsze większy, 0.0 = przełom zawsze mniejszy (też dobra cecha, odwrotnie).
"Siła" = |AUC - 0.5| x 2 (0 = bezużyteczna, 1 = idealna).

Na koniec regresja logistyczna na wszystkich cechach naraz, sprawdzana "zostaw jedną kość":
uczona na pozostałych kościach, oceniana na tej jednej - tyle mniej więcej dałby prosty
algorytm łączący cechy na nowej kości.

Użycie: scripts/compare_surfaces.py (strona out/3d/<zestaw>/surface_features.html + CSV).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

import contacts as ct
from data_parsing import sample_surface
from visualization_3d import FONT, LIGHTING, View3DParams, decimate, lying_rotation

MARKED_COLOR = "#e8508f"   # przełom (kolor 1. powierzchni styku)
CORTEX_COLOR = "#8a9aab"   # kora


@dataclass(frozen=True)
class SurfaceFeatureParams:
    """Ustawienia porównania cech (settings.json, sekcja "surface_features")."""

    radii_mm: tuple = (0.5, 1.0, 2.0)  # promienie sąsiedztwa szorstkości / rozrzutu normalnych [mm]
    boundary_margin_mm: float = 1.0    # pomiń pas przy otwartym brzegu skanu [mm]
    label_band_mm: float = 0.5         # pomiń pas po obu stronach granicy oznaczenia [mm]


# -----------------
# CECHY
# -----------------

def feature_info(params: SurfaceFeatureParams = SurfaceFeatureParams()) -> dict[str, dict]:
    """Opis cech liczonych przez vertex_features, w kolejności wyświetlania.

    Args:
        params: promienie (od nich zależą nazwy cech szorstkości i rozrzutu).

    Returns:
        {klucz: {"name": krótka nazwa, "desc": opis, "log": czy rozkład jest skośny
        (do regresji brany log), "sign": kierunek na przełomie wg raporty/przelom_vs_kora.md
        (+1 większa, -1 mniejsza, 0 niespójny)}}.
    """
    info = {}
    for r in params.radii_mm:
        info[f"rough_{r:g}"] = dict(name=f"szorstkość {r:g} mm", log=True, sign=1,
                                    desc=f"odchylenie powierzchni od płaszczyzny w promieniu {r:g} mm / promień")
    for r in params.radii_mm:
        info[f"spread_{r:g}"] = dict(name=f"rozrzut normalnych {r:g} mm", log=True, sign=1,
                                     desc=f"1 - |średnia normalnych| w promieniu {r:g} mm (0 = płasko, rośnie na grudkach i krawędziach)")
    rmax = max(params.radii_mm)
    info["convexity"] = dict(name=f"wypukłość {rmax:g} mm", log=False, sign=1,
                             desc=f"jak bardzo punkt wystaje ponad średnią okolicy {rmax:g} mm wzdłuż normalnej (> 0 wypukła, < 0 wklęsła)")
    info["edge_mm"] = dict(name="krawędź siatki [mm]", log=True, sign=-1,
                           desc="średnia długość krawędzi trójkątów przy wierzchołku - skaner zagęszcza siatkę na szczegółach")
    info["brightness"] = dict(name="jasność koloru", log=False, sign=0, desc="średnia R, G, B ze skanera (0-1)")
    info["saturation"] = dict(name="nasycenie koloru", log=False, sign=0, desc="(max - min) / max z R, G, B (0 = szary)")
    info["color_texture"] = dict(name="plamistość koloru", log=True, sign=1,
                                 desc="odchylenie standardowe jasności w promieniu 1 mm")
    info["normal_axis"] = dict(name="normalna ∥ oś kości", log=False, sign=1,
                               desc="|cos| kąta normalnej do długiej osi kości: 0 = ściana boczna (kora trzonu), 1 = ściana poprzeczna")
    info["axial_pos"] = dict(name="położenie na osi", log=False, sign=0,
                             desc="odległość od środka kości wzdłuż długiej osi: 0 = środek, 1 = koniec")
    return info


def _neighbours(points: np.ndarray, targets: np.ndarray, radius: float):
    """Macierz sąsiedztwa (len(points), len(targets)): 1, gdy target jest w promieniu od punktu.

    Args:
        points:  (V, 3) punkty, dla których szukamy sąsiadów.
        targets: (P, 3) kandydaci na sąsiadów.
        radius:  promień [mm].

    Returns:
        (W - CSR float32, counts - (V,) liczba sąsiadów).
    """
    nb = cKDTree(targets).query_ball_point(points, radius, workers=-1)
    counts = np.fromiter((len(x) for x in nb), dtype=np.int64, count=len(points))
    rows = np.repeat(np.arange(len(points)), counts)
    cols = np.fromiter((j for x in nb for j in x), dtype=np.int64, count=counts.sum())
    W = coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(points), len(targets))).tocsr()
    return W, counts


def bone_axis(mesh: o3d.geometry.TriangleMesh) -> tuple[np.ndarray, np.ndarray]:
    """Środek i długa oś kości (pierwsza oś główna PCA wierzchołków ważonych polem).

    Args:
        mesh: siatka kości.

    Returns:
        (środek (3,), jednostkowy kierunek osi (3,)).
    """
    v = np.asarray(mesh.vertices)
    va = ct.vertex_areas(mesh)
    c = (v * va[:, None]).sum(axis=0) / max(va.sum(), 1e-12)
    _, _, vt = np.linalg.svd((v - c) * np.sqrt(va)[:, None], full_matrices=False)
    return c, vt[0]


def vertex_features(mesh: o3d.geometry.TriangleMesh,
                    params: SurfaceFeatureParams = SurfaceFeatureParams()) -> dict[str, np.ndarray]:
    """Liczy wszystkie cechy z feature_info dla każdego wierzchołka siatki.

    Sąsiedzi do cech kształtu to punkty równomiernej próbki powierzchni (co r / 3), więc
    wynik nie zależy od gęstości siatki (tak jak contacts.local_shape). Wierzchołki z mniej
    niż 6 sąsiadami dostają 0 w cechach kształtu.

    Args:
        mesh:   siatka kości z normalnymi wierzchołków (liczone, gdy ich brak).
        params: promienie.

    Returns:
        {klucz cechy: (V,) wartości}.
    """
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    v, vn = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
    t = np.asarray(mesh.triangles)
    out: dict[str, np.ndarray] = {}
    convexity = None
    for r in sorted(params.radii_mm):
        s = sample_surface(mesh, r / 3)
        p, n = s.points, s.normals
        W, counts = _neighbours(v, p, r)
        cnt = np.maximum(counts, 1)[:, None].astype(float)
        mean = (W @ p) / cnt
        outer = np.einsum("ni,nj->nij", p, p).reshape(len(p), 9)
        cov = ((W @ outer) / cnt).reshape(-1, 3, 3) - np.einsum("ni,nj->nij", mean, mean)
        smallest = np.clip(np.linalg.eigvalsh(cov)[:, 0], 0, None)
        few = counts < 6
        out[f"rough_{r:g}"] = np.where(few, 0, np.sqrt(smallest) / r)
        out[f"spread_{r:g}"] = np.where(few, 0, 1 - np.linalg.norm((W @ n) / cnt, axis=1))
        # wypukłość: wysokość wierzchołka nad średnią okolicy wzdłuż normalnej, względem r^2
        # (dla sfery o promieniu R wychodzi ~ r^2 / (4R) / r^2 * 4 = 1/R, czyli krzywizna [1/mm])
        convexity = np.where(few, 0, 4 * np.einsum("ij,ij->i", v - mean, vn) / r**2)
    # rozrzut i szorstkość w kolejności promieni z ustawień
    out = {k: out[k] for k in feature_info(params) if k in out}
    out["convexity"] = convexity

    e = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    length = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    ends = np.concatenate([e[:, 0], e[:, 1]])        # każda krawędź liczy się dla obu końców
    sums = np.bincount(ends, weights=np.concatenate([length, length]), minlength=len(v))
    out["edge_mm"] = sums / np.maximum(np.bincount(ends, minlength=len(v)), 1)

    if mesh.has_vertex_colors():
        c = np.asarray(mesh.vertex_colors)
    else:
        c = np.full((len(v), 3), 0.8)
    mx, mn = c.max(axis=1), c.min(axis=1)
    out["brightness"] = c.mean(axis=1)
    out["saturation"] = (mx - mn) / np.maximum(mx, 1e-6)
    W, counts = _neighbours(v, v, 1.0)
    cnt = np.maximum(counts, 1).astype(float)
    b = out["brightness"]
    m1, m2 = (W @ b) / cnt, (W @ (b * b)) / cnt
    out["color_texture"] = np.sqrt(np.clip(m2 - m1**2, 0, None))

    centre, axis = bone_axis(mesh)
    out["normal_axis"] = np.abs(vn @ axis)
    proj = np.abs((v - centre) @ axis)
    out["axial_pos"] = proj / max(proj.max(), 1e-12)
    return out


# -----------------
# STATYSTYKI
# -----------------

def weighted_auc(pos: np.ndarray, neg: np.ndarray, wpos: np.ndarray, wneg: np.ndarray) -> float:
    """AUC ważone: P(losowy pos > losowy neg), remisy liczone po połowie.

    Args:
        pos, neg:   wartości cechy w obu grupach.
        wpos, wneg: wagi (pola wierzchołków).

    Returns:
        AUC w [0, 1]; nan, gdy któraś grupa jest pusta.
    """
    if len(pos) == 0 or len(neg) == 0 or wpos.sum() <= 0 or wneg.sum() <= 0:
        return float("nan")
    vals, inv = np.unique(np.concatenate([pos, neg]), return_inverse=True)
    wp = np.bincount(inv[:len(pos)], weights=wpos, minlength=len(vals))
    wn = np.bincount(inv[len(pos):], weights=wneg, minlength=len(vals))
    below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wpos.sum() * wneg.sum()))


def weighted_quantiles(x: np.ndarray, w: np.ndarray, qs) -> np.ndarray:
    """Kwantyle ważone (np. pole wierzchołków).

    Args:
        x:  wartości.
        w:  wagi >= 0.
        qs: kwantyle w [0, 1].

    Returns:
        Tablica kwantyli (nan dla pustego x).
    """
    if len(x) == 0:
        return np.full(len(qs), np.nan)
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    cw = (cw - 0.5 * w[o]) / cw[-1]
    return np.interp(qs, cw, x[o])


QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass
class BoneComparison:
    """Cechy jednej kości i podział jej wierzchołków na przełom / korę / pominięte."""

    bone: object                       # bones.Bone
    contacts: object                   # contacts.Contacts (ręczne)
    features: dict[str, np.ndarray]    # {klucz: (V,)}
    areas: np.ndarray                  # (V,) pola wierzchołków
    marked: np.ndarray                 # (V,) bool - przełom użyty w porównaniu
    cortex: np.ndarray                 # (V,) bool - kora użyta w porównaniu
    stats: list[dict]                  # wiersze: jedna powierzchnia x jedna cecha


def _distance_to(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Odległość [mm] każdego wierzchołka do najbliższego wierzchołka z maski (inf, gdy maska pusta).

    Liczona w przestrzeni, nie po siatce - pierścienie siatki dałyby różne mm tam, gdzie
    skaner zagęścił siatkę (na przełomie trójkąty są ~2x mniejsze).
    """
    if not mask.any():
        return np.full(len(v), np.inf)
    return cKDTree(v[mask]).query(v, workers=-1)[0]


def split_marked(mesh: o3d.geometry.TriangleMesh, labels: np.ndarray,
                 params: SurfaceFeatureParams = SurfaceFeatureParams()) -> tuple[np.ndarray, np.ndarray]:
    """Wierzchołki przełomu i kory do porównania / uczenia - bez pasów, którym nie można ufać.

    Pomijane: pas boundary_margin_mm przy otwartym brzegu skanu i pas label_band_mm po obu
    stronach granicy oznaczenia (przełom blisko kory i kora blisko przełomu).

    Args:
        mesh:   siatka kości.
        labels: (V,) ręczne etykiety (0 = kora, > 0 = przełom).
        params: szerokości pasów.

    Returns:
        (przełom, kora) - dwie rozłączne maski (V,) bool.
    """
    va = ct.vertex_areas(mesh)
    lab = labels > 0
    v = np.asarray(mesh.vertices)
    usable = (va > 0) & (_distance_to(v, ct.boundary_vertices(mesh)) > params.boundary_margin_mm)
    usable &= np.where(lab, _distance_to(v, ~lab), _distance_to(v, lab)) > params.label_band_mm
    return lab & usable, ~lab & usable


def compare_bone(bone, contacts, params: SurfaceFeatureParams = SurfaceFeatureParams()) -> BoneComparison:
    """Liczy cechy kości i porównuje każdą jej oznaczoną powierzchnię z korą.

    Args:
        bone:     kość (bones.Bone).
        contacts: ręczne oznaczenia tej kości (contacts.Contacts z co najmniej jedną powierzchnią).
        params:   ustawienia.

    Returns:
        BoneComparison; stats ma wiersze {"bone", "surface" (nazwa; "wszystkie" = suma powierzchni),
        "surface_id" (0 = wszystkie), "feature", "auc", "area_marked", "area_cortex",
        "q_marked", "q_cortex" (kwantyle QUANTILES)}.
    """
    mesh = bone.mesh
    feats = cached_features(bone, params)
    va = ct.vertex_areas(mesh)
    marked, cortex = split_marked(mesh, contacts.labels, params)
    groups = [(0, "wszystkie", marked)]
    if len(contacts.surface_ids) > 1:
        groups += [(k, contacts.name(k), marked & (contacts.labels == k)) for k in contacts.surface_ids]
    stats = []
    for sid, name, m in groups:
        for key, x in feats.items():
            stats.append({"bone": bone.label, "surface": name, "surface_id": sid, "feature": key,
                          "auc": weighted_auc(x[m], x[cortex], va[m], va[cortex]),
                          "area_marked": float(va[m].sum()), "area_cortex": float(va[cortex].sum()),
                          "q_marked": weighted_quantiles(x[m], va[m], QUANTILES),
                          "q_cortex": weighted_quantiles(x[cortex], va[cortex], QUANTILES)})
    return BoneComparison(bone, contacts, feats, va, marked, cortex, stats)


# -----------------
# MODEL: WSZYSTKIE CECHY NARAZ (regresja logistyczna - tylko do analizy w compare_surfaces.py)
# -----------------

def design_matrix(features: dict[str, np.ndarray], keys,
                  params: SurfaceFeatureParams = SurfaceFeatureParams()) -> np.ndarray:
    """Macierz cech (V, F) do regresji; cechy o skośnym rozkładzie (feature_info "log") są logarytmowane.

    Args:
        features: wynik vertex_features.
        keys:     które cechy, w jakiej kolejności.
        params:   ustawienia (opisy cech).

    Returns:
        (V, len(keys)) float.

    Raises:
        ValueError: nieznana cecha.
    """
    info = feature_info(params)
    unknown = [k for k in keys if k not in info]
    if unknown:
        raise ValueError(f"Nieznane cechy {unknown}, dostępne: {', '.join(info)}")
    cols = [np.log(features[k] + 1e-3) if info[k]["log"] else features[k] for k in keys]
    return np.column_stack(cols).astype(float)


@dataclass
class FeatureModel:
    """Regresja logistyczna: cechy wierzchołka -> prawdopodobieństwo, że to przełom."""

    keys: tuple                # cechy (klucze feature_info), kolejność kolumn
    mu: np.ndarray             # średnie cech (standaryzacja)
    sd: np.ndarray             # odchylenia cech (standaryzacja)
    beta: np.ndarray           # [wyraz wolny, współczynniki standaryzowanych cech...]
    bones: tuple = ()          # id kości, na których uczono

    def predict(self, features: dict[str, np.ndarray],
                params: SurfaceFeatureParams = SurfaceFeatureParams()) -> np.ndarray:
        """Prawdopodobieństwo przełomu dla każdego wierzchołka.

        Args:
            features: wynik vertex_features (te same promienie co przy uczeniu).
            params:   ustawienia (opisy cech).

        Returns:
            (V,) w [0, 1].
        """
        X = design_matrix(features, self.keys, params)
        z = self.beta[0] + ((X - self.mu) / self.sd) @ self.beta[1:]
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    @property
    def weights(self) -> dict[str, float]:
        """{cecha: współczynnik standaryzowanej cechy} (> 0: więcej = bardziej przełom)."""
        return dict(zip(self.keys, self.beta[1:].tolist()))


def fit_model(samples: list[tuple], keys, params: SurfaceFeatureParams = SurfaceFeatureParams(),
              l2: float = 1.0, iters: int = 30) -> FeatureModel:
    """Uczy regresję logistyczną (metoda Newtona, cechy standaryzowane, kara L2).

    Każda kość ma w uczeniu tę samą łączną wagę przełomu i tę samą łączną wagę kory - inaczej
    duże kości i duża kora by dominowały, a próg 0.5 zależałby od proporcji przełom / kora.

    Args:
        samples: [(features, marked, cortex, areas, id kości), ...] - features z vertex_features,
                 marked / cortex z split_marked, areas z contacts.vertex_areas.
        keys:    cechy modelu.
        params:  ustawienia (opisy cech).
        l2:      kara L2 na współczynniki (bez wyrazu wolnego).
        iters:   kroki Newtona.

    Returns:
        FeatureModel.

    Raises:
        ValueError: żadna kość nie ma jednocześnie przełomu i kory do uczenia.
    """
    keys = tuple(keys)
    Xs, ys, ws, ids = [], [], [], []
    for feats, marked, cortex, areas, bone_id in samples:
        if not marked.any() or not cortex.any():
            continue
        X = design_matrix(feats, keys, params)
        use = marked | cortex
        w = np.where(marked, areas / areas[marked].sum(), areas / areas[cortex].sum())
        Xs.append(X[use])
        ys.append(marked[use].astype(float))
        ws.append(w[use] * 1000)
        ids.append(bone_id)
    if not Xs:
        raise ValueError("Brak kości z ręcznie oznaczonym przełomem i korą do uczenia modelu")
    X, y, w = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    beta = np.zeros(Z.shape[1])
    reg = l2 * np.eye(len(beta))
    reg[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ beta, -30, 30)))
        g = Z.T @ (w * (p - y)) + reg @ beta
        H = (Z * (w * p * (1 - p))[:, None]).T @ Z + reg
        beta -= np.linalg.solve(H, g)
    return FeatureModel(keys, mu, sd, beta, tuple(ids))


def cached_features(bone, params: SurfaceFeatureParams = SurfaceFeatureParams(),
                    cache_dir: Path | None = None) -> dict[str, np.ndarray]:
    """vertex_features z cache na dysku (liczone raz na kość i ustawienia).

    Plik: <cache_dir>/<id kości>.npz; liczony od nowa, gdy zmieniła się siatka kości
    (contacts.mesh_fingerprint) albo ustawienia cech.

    Args:
        bone:      kość.
        params:    ustawienia cech.
        cache_dir: folder cache; domyślnie <folder wyników>/cache/features (z bone.folder),
                   a dla kości bez folderu - bez cache.

    Returns:
        {klucz cechy: (V,)}.
    """
    import json
    from dataclasses import asdict

    if cache_dir is None and bone.folder is not None:
        cache_dir = Path(bone.folder).parent.parent / "cache" / "features"
    tag = json.dumps({"mesh": ct.mesh_fingerprint(bone.mesh), "params": asdict(params)}, sort_keys=True)
    path = Path(cache_dir) / f"{bone.id}.npz" if cache_dir is not None else None
    if path is not None and path.exists():
        try:
            with np.load(path) as d:
                if str(d["_tag"]) == tag:
                    return {k: d[k].astype(float) for k in d.files if k != "_tag"}
        except Exception:       # uszkodzony albo stary plik - policz od nowa
            pass
    feats = vertex_features(bone.mesh, params)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, _tag=np.array(tag), **{k: v.astype(np.float32) for k, v in feats.items()})
    return feats


def combined_model(comps: list[BoneComparison], params: SurfaceFeatureParams = SurfaceFeatureParams()) -> dict:
    """Wszystkie cechy naraz (regresja logistyczna), sprawdzane "zostaw jedną kość".

    Każda kość ma w uczeniu tę samą łączną wagę przełomu i kory (inaczej duże kości i duża
    kora by dominowały). Potrzebne co najmniej 2 kości.

    Args:
        comps:  wyniki compare_bone.
        params: ustawienia (lista cech).

    Returns:
        {"auc": {id kości: AUC na tej kości przy uczeniu na pozostałych},
         "iou": {id kości: IoU przełomu dla progu 0.5 (bez wygładzania obszarów)},
         "proba": {id kości: (V,) prawdopodobieństwo przełomu z modelu bez tej kości},
         "weights": {klucz cechy: współczynnik standaryzowanej cechy w modelu na wszystkich kościach}}
        albo {} przy mniej niż 2 kościach.
    """
    if len(comps) < 2:
        return {}
    keys = tuple(feature_info(params))
    samples = [(c.features, c.marked, c.cortex, c.areas, c.bone.id) for c in comps]
    out = {"auc": {}, "iou": {}, "proba": {}}
    for i, c in enumerate(comps):
        model = fit_model([sm for j, sm in enumerate(samples) if j != i], keys, params)
        p = model.predict(c.features, params)
        out["proba"][c.bone.id] = p
        out["auc"][c.bone.id] = weighted_auc(p[c.marked], p[c.cortex], c.areas[c.marked], c.areas[c.cortex])
        hit = p > 0.5
        use = c.marked | c.cortex
        out["iou"][c.bone.id] = float(c.areas[use & hit & c.marked].sum() / max(c.areas[use & (hit | c.marked)].sum(), 1e-12))
    out["weights"] = fit_model(samples, keys, params).weights
    return out


def summary(comps: list[BoneComparison], params: SurfaceFeatureParams = SurfaceFeatureParams()) -> list[dict]:
    """Ranking cech: średnie AUC po kościach (wszystkie powierzchnie kości razem), od najsilniejszej.

    Args:
        comps:  wyniki compare_bone.
        params: ustawienia (opisy cech).

    Returns:
        [{"feature", "name", "desc", "auc_mean", "auc_min", "auc_max", "strength", "direction",
          "consistent" (czy na każdej kości po tej samej stronie 0.5), "median_marked",
          "median_cortex" (mediany median po kościach)}, ...] posortowane wg strength malejąco.
    """
    info = feature_info(params)
    rows = []
    for key, fi in info.items():
        st = [s for c in comps for s in c.stats if s["feature"] == key and s["surface_id"] == 0]
        a = np.array([s["auc"] for s in st])
        mean = float(np.nanmean(a))
        rows.append({"feature": key, "name": fi["name"], "desc": fi["desc"], "auc_mean": mean,
                     "auc_min": float(np.nanmin(a)), "auc_max": float(np.nanmax(a)),
                     "strength": abs(mean - 0.5) * 2,
                     "direction": "większa na przełomie" if mean > 0.5 else "mniejsza na przełomie",
                     "consistent": bool((a > 0.5).all() or (a < 0.5).all()),
                     "median_marked": float(np.median([s["q_marked"][2] for s in st])),
                     "median_cortex": float(np.median([s["q_cortex"][2] for s in st]))})
    return sorted(rows, key=lambda r: -r["strength"])


def write_csv(path: Path, comps: list[BoneComparison], params: SurfaceFeatureParams = SurfaceFeatureParams()) -> Path:
    """Zapisuje wszystkie porównania (powierzchnia x cecha) do CSV (separator ";", otwiera się w Excelu).

    Args:
        path:   plik wynikowy.
        comps:  wyniki compare_bone.
        params: ustawienia (nazwy cech).

    Returns:
        Ścieżka pliku.
    """
    info = feature_info(params)
    q = [f"p{int(x * 100)}" for x in QUANTILES]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["kosc", "powierzchnia", "cecha", "klucz", "AUC", "pole_przelomu_mm2", "pole_kory_mm2"]
                   + [f"przelom_{x}" for x in q] + [f"kora_{x}" for x in q])
        for c in comps:
            for s in c.stats:
                w.writerow([s["bone"], s["surface"], info[s["feature"]]["name"], s["feature"], f"{s['auc']:.3f}",
                            f"{s['area_marked']:.1f}", f"{s['area_cortex']:.1f}"]
                           + [f"{x:.5g}" for x in s["q_marked"]] + [f"{x:.5g}" for x in s["q_cortex"]])
    return path


# -----------------
# WYKRESY
# -----------------

def _layout(fig: go.Figure, params: View3DParams, **kw) -> go.Figure:
    """Wspólny wygląd wykresów strony."""
    fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=FONT, **kw)
    return fig


def ranking_figure(rows: list[dict], model: dict, comps: list[BoneComparison],
                   params: View3DParams = View3DParams()) -> go.Figure:
    """Poziome słupki: średnie AUC każdej cechy (wąsy = najgorsza i najlepsza kość), oś od 0.5.

    Args:
        rows:   summary().
        model:  combined_model() - dopisuje słupek "wszystkie cechy razem".
        comps:  wyniki compare_bone (do liczby kości w opisie).
        params: wygląd.

    Returns:
        Figura.
    """
    rows = list(rows)
    names = [r["name"] for r in rows]
    mean = [r["auc_mean"] for r in rows]
    lo = [r["auc_mean"] - r["auc_min"] for r in rows]
    hi = [r["auc_max"] - r["auc_mean"] for r in rows]
    colors = [MARKED_COLOR if r["auc_mean"] > 0.5 else "#2b9be0" for r in rows]
    hover = [f"<b>{r['name']}</b><br>{r['desc']}<br>AUC średnio {r['auc_mean']:.2f} "
             f"(kości: {r['auc_min']:.2f}-{r['auc_max']:.2f})<br>{r['direction']}"
             f"{'' if r['consistent'] else '<br><i>na różnych kościach w różne strony</i>'}<br>"
             f"mediana: przełom {r['median_marked']:.3g}, kora {r['median_cortex']:.3g}" for r in rows]
    if model:
        a = np.array(list(model["auc"].values()))
        names.insert(0, "<b>wszystkie cechy razem</b>")
        mean.insert(0, float(a.mean()))
        lo.insert(0, float(a.mean() - a.min()))
        hi.insert(0, float(a.max() - a.mean()))
        colors.insert(0, "#1f2a36")
        hover.insert(0, f"<b>regresja logistyczna na wszystkich cechach</b><br>uczona na pozostałych kościach, "
                        f"oceniana na każdej z {len(a)} kości osobno<br>AUC średnio {a.mean():.2f} "
                        f"({a.min():.2f}-{a.max():.2f})")
    fig = go.Figure(go.Bar(
        y=names, x=[m - 0.5 for m in mean], base=0.5, orientation="h", marker_color=colors,
        error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo, color="#6b7b8c", thickness=1.2),
        hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
        text=[f"{m:.2f}" for m in mean], textposition="outside", cliponaxis=False))
    fig.add_vline(x=0.5, line_color="#1f2a36", line_width=1)
    _layout(fig, params, height=60 + 30 * len(names), margin=dict(l=10, r=40, t=30, b=40), showlegend=False,
            xaxis=dict(range=[0, 1.05], title="AUC  (←  mniejsza na przełomie   |   większa na przełomie  →)",
                       gridcolor="#e7ebef", tickvals=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            yaxis=dict(autorange="reversed"))
    return fig


def auc_heatmap_figure(comps: list[BoneComparison], model: dict, rows: list[dict],
                       params: SurfaceFeatureParams = SurfaceFeatureParams(),
                       view: View3DParams = View3DParams()) -> go.Figure:
    """Mapa AUC: wiersze = kości (i osobno ich powierzchnie), kolumny = cechy (kolejność z rankingu).

    Args:
        comps:  wyniki compare_bone.
        model:  combined_model() - dodatkowa kolumna "razem".
        rows:   summary() - kolejność kolumn.
        params: ustawienia (nazwy cech).
        view:   wygląd.

    Returns:
        Figura.
    """
    keys = [r["feature"] for r in rows]
    info = feature_info(params)
    ylab, z = [], []
    for c in comps:
        for sid in sorted({s["surface_id"] for s in c.stats}):
            st = {s["feature"]: s for s in c.stats if s["surface_id"] == sid}
            first = next(iter(st.values()))
            label = c.bone.label if sid == 0 else f"   └ {first['surface']}"
            ylab.append(f"{label} ({first['area_marked']:.0f} mm²)")
            extra = [model["auc"][c.bone.id]] if model and sid == 0 else ([np.nan] if model else [])
            z.append(extra + [st[k]["auc"] for k in keys])
    xlab = (["razem"] if model else []) + [info[k]["name"] for k in keys]
    z = np.array(z)
    fig = go.Figure(go.Heatmap(
        z=z, x=xlab, y=ylab, zmin=0, zmax=1, xgap=1, ygap=1,
        colorscale=[[0, "#2b9be0"], [0.5, "#f4f6f8"], [1, MARKED_COLOR]],
        text=[["" if np.isnan(v) else f"{v:.2f}" for v in row] for row in z], texttemplate="%{text}",
        textfont=dict(size=10), colorbar=dict(title="AUC", tickvals=[0, 0.25, 0.5, 0.75, 1]),
        hovertemplate="%{y}<br>%{x}<br>AUC %{z:.2f}<extra></extra>"))
    _layout(fig, view, height=max(320, 28 * len(ylab) + 200), margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(tickangle=-35, side="top"), yaxis=dict(autorange="reversed"))
    return fig


def distributions_figure(comps: list[BoneComparison], rows: list[dict],
                         params: SurfaceFeatureParams = SurfaceFeatureParams(),
                         view: View3DParams = View3DParams(), columns: int = 3) -> go.Figure:
    """Rozkłady każdej cechy: przełom vs kora na każdej kości (pudełka: 5-25-50-75-95 percentyl).

    Args:
        comps:   wyniki compare_bone.
        rows:    summary() - kolejność cech.
        params:  ustawienia (nazwy cech).
        view:    wygląd.
        columns: liczba kolumn wykresów.

    Returns:
        Figura z jednym wykresem na cechę.
    """
    info = feature_info(params)
    keys = [r["feature"] for r in rows]
    nrows = -(-len(keys) // columns)
    fig = make_subplots(rows=nrows, cols=columns, subplot_titles=[info[k]["name"] for k in keys],
                        vertical_spacing=0.25 / nrows, horizontal_spacing=0.06)
    xs = np.arange(1, len(comps) + 1, dtype=float)
    for n, key in enumerate(keys):
        r, col = n // columns + 1, n % columns + 1
        for group, color, qk, dx in (("przełom", MARKED_COLOR, "q_marked", -0.19), ("kora", CORTEX_COLOR, "q_cortex", 0.19)):
            q = np.array([next(s for s in c.stats if s["feature"] == key and s["surface_id"] == 0)[qk] for c in comps])
            fig.add_trace(go.Box(
                x=xs + dx, q1=q[:, 1], median=q[:, 2], q3=q[:, 3], lowerfence=q[:, 0], upperfence=q[:, 4],
                name=group, legendgroup=group, width=0.34, showlegend=n == 0,
                marker_color=color, line_width=1.3,
                fillcolor=color + "55", hoverinfo="y+name"), row=r, col=col)
    fig.update_layout(boxmode="overlay")
    fig.update_xaxes(tickvals=xs, range=[0.4, len(comps) + 0.6], title_text="kość", title_font_size=10, tickfont_size=10)
    fig.update_yaxes(gridcolor="#e7ebef", tickfont_size=10)
    _layout(fig, view, height=270 * nrows + 80, margin=dict(l=10, r=10, t=70, b=10),
            legend=dict(orientation="h", y=0.995, yanchor="top", x=0.01, xanchor="left",
                        yref="container", font_size=13))
    return fig


def weights_figure(model: dict, params: SurfaceFeatureParams = SurfaceFeatureParams(),
                   view: View3DParams = View3DParams()) -> go.Figure:
    """Współczynniki regresji (cechy standaryzowane): co model łączący cechy naprawdę wykorzystuje.

    Args:
        model:  combined_model().
        params: ustawienia (nazwy cech).
        view:   wygląd.

    Returns:
        Figura.
    """
    info = feature_info(params)
    items = sorted(model["weights"].items(), key=lambda kv: -abs(kv[1]))
    fig = go.Figure(go.Bar(
        y=[info[k]["name"] + (" (log)" if info[k]["log"] else "") for k, _ in items], x=[w for _, w in items],
        orientation="h", marker_color=[MARKED_COLOR if w > 0 else "#2b9be0" for _, w in items],
        hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.add_vline(x=0, line_color="#1f2a36", line_width=1)
    _layout(fig, view, height=60 + 26 * len(items), margin=dict(l=10, r=20, t=10, b=40),
            xaxis=dict(title="współczynnik (> 0: więcej = bardziej przełom)", gridcolor="#e7ebef"),
            yaxis=dict(autorange="reversed"))
    return fig


def _outline(mesh: o3d.geometry.TriangleMesh, tri_mask: np.ndarray, v: np.ndarray) -> tuple[list, list, list]:
    """Linie brzegu zbioru trójkątów (krawędzie należące do dokładnie jednego zaznaczonego trójkąta).

    Args:
        mesh:     siatka.
        tri_mask: (T,) bool - zaznaczone trójkąty.
        v:        (V, 3) wierzchołki do rysowania.

    Returns:
        (x, y, z) z None między odcinkami (format Scatter3d).
    """
    t = np.asarray(mesh.triangles)[tri_mask]
    if len(t) == 0:
        return [], [], []
    e = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    uniq, count = np.unique(e, axis=0, return_counts=True)
    b = uniq[count == 1]
    seg = np.full((len(b) * 3, 3), np.nan)
    seg[0::3], seg[1::3] = v[b[:, 0]], v[b[:, 1]]
    out = [[None if np.isnan(x) else round(float(x), 2) for x in seg[:, i]] for i in range(3)]
    return out[0], out[1], out[2]


FEATURES_3D = ("proba", "rough_1", "spread_2", "convexity", "edge_mm", "brightness", "color_texture", "normal_axis")


def features_3d_figure(comps: list[BoneComparison], model: dict,
                       params: SurfaceFeatureParams = SurfaceFeatureParams(),
                       view: View3DParams = View3DParams()) -> go.Figure:
    """Kość w 3D pokolorowana wybraną cechą, z obrysem ręcznie oznaczonych powierzchni.

    Lista po lewej wybiera kość, po prawej - cechę. Kolory: 2-98 percentyl cechy na wszystkich
    kościach razem (te same kolory = te same wartości na każdej kości).

    Args:
        comps:  wyniki compare_bone.
        model:  combined_model() - cecha "połączenie cech" (prawdopodobieństwo przełomu z modelu
                uczonego BEZ tej kości).
        params: ustawienia (nazwy cech).
        view:   wygląd (triangles_total dzielone między kości).

    Returns:
        Figura 3D.
    """
    info = feature_info(params)
    keys = [k for k in FEATURES_3D if k in info or (k == "proba" and model)]
    names = {k: ("połączenie cech (model)" if k == "proba" else info[k]["name"]) for k in keys}
    per_bone = max(5000, view.triangles_total // 2 // max(len(comps), 1))   # połowa: każda cecha to osobne kolory
    traces, values = [], {k: [] for k in keys}
    for c in comps:
        d = decimate(c.bone.mesh, per_bone)
        _, near = cKDTree(np.asarray(c.bone.mesh.vertices)).query(np.asarray(d.vertices))
        R = lying_rotation(c.bone.mesh)
        v = (np.asarray(d.vertices) - np.asarray(c.bone.mesh.vertices).mean(axis=0)) @ R.T
        for k in keys:
            x = model["proba"][c.bone.id] if k == "proba" else c.features[k]
            values[k].append(x[near])
        tri = np.asarray(d.triangles)
        lab = c.contacts.labels[near]
        traces.append((c, d, v, tri, (lab[tri] > 0).all(axis=1)))
    ranges = {}
    for k in keys:
        allv = np.concatenate(values[k])
        ranges[k] = (0.0, 1.0) if k == "proba" else tuple(np.percentile(allv, [2, 98]).tolist())

    def scaled(k: str, i: int) -> list[int]:
        """Wartości cechy k kości i przeskalowane do 0..100 (krótszy HTML)."""
        lo, hi = ranges[k]
        return np.clip(np.round((values[k][i] - lo) / max(hi - lo, 1e-12) * 100), 0, 100).astype(int).tolist()

    def colorbar(k: str) -> dict:
        """Opis paska kolorów cechy k (wartości rzeczywiste na 0..100)."""
        lo, hi = ranges[k]
        ticks = np.linspace(0, 100, 5)
        return {"colorbar.tickvals": [ticks.tolist()] * len(traces),
                "colorbar.ticktext": [[f"{lo + (hi - lo) * t / 100:.3g}" for t in ticks]] * len(traces),
                "colorbar.title.text": [names[k]] * len(traces)}

    fig = go.Figure()
    first = keys[0]
    for i, (c, d, v, tri, inside) in enumerate(traces):
        fig.add_trace(go.Mesh3d(
            x=v[:, 0].astype(np.float32), y=v[:, 1].astype(np.float32), z=v[:, 2].astype(np.float32),
            i=tri[:, 0].astype(np.int32), j=tri[:, 1].astype(np.int32), k=tri[:, 2].astype(np.int32),
            intensity=scaled(first, i), cmin=0, cmax=100, colorscale="Viridis", flatshading=False,
            lighting=LIGHTING, hoverinfo="skip", visible=i == 0, name=c.bone.label,
            colorbar=dict(title=dict(text=names[first], side="right"), len=0.7, thickness=14,
                          tickvals=np.linspace(0, 100, 5).tolist(),
                          ticktext=[f"{ranges[first][0] + (ranges[first][1] - ranges[first][0]) * t / 100:.3g}"
                                    for t in np.linspace(0, 100, 5)])))
    for i, (c, d, v, tri, inside) in enumerate(traces):
        x, y, z = _outline(d, inside, v)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="#ff2d6f", width=5),
                                   hoverinfo="skip", visible=i == 0, name="oznaczony przełom", showlegend=False))
    n = len(traces)
    bone_buttons = []
    for i, (c, *_rest) in enumerate(traces):
        vis = [j == i for j in range(n)] * 2
        bone_buttons.append(dict(label=f"{i + 1}. {c.bone.label}", method="update", args=[{"visible": vis}]))
    feat_buttons = []
    for k in keys:
        args = {"intensity": [scaled(k, i) for i in range(n)], **colorbar(k)}
        feat_buttons.append(dict(label=names[k], method="restyle", args=[args, list(range(n))]))
    scene = dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                 camera=dict(eye=dict(x=0.0, y=-1.2, z=1.4)), bgcolor="#ffffff")
    _layout(fig, view, height=680, margin=dict(l=0, r=0, t=50, b=0), scene=scene, showlegend=False,
            updatemenus=[dict(buttons=bone_buttons, x=0.01, y=1.06, xanchor="left", yanchor="top",
                              bgcolor="#ffffff", bordercolor="#c9d2db"),
                         dict(buttons=feat_buttons, x=0.99, y=1.06, xanchor="right", yanchor="top",
                              bgcolor="#ffffff", bordercolor="#c9d2db")])
    return fig

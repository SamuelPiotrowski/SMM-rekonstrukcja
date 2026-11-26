"""
Dopasowywanie fragmentów kości po POWIERZCHNIACH STYKU (podproblem 6 z README.md).

Bierze tylko punkty powierzchni styku (przełomów), każdą powierzchnię osobno (kość może
mieć kilka), i dopasowuje każdą do każdej powierzchni INNEJ kości.

SKĄD POWIERZCHNIE (SurfaceMatchParams.source, settings.json "matching" -> "source"):
    "manual"      - tylko ręczne oznaczenia contacts/<zestaw>/<id>.json; kości bez nich są pomijane,
    "auto"        - tylko wynik algorytmu (settings.json "contacts" -> "method" i jego ustawienia):
                    z contacts/<zestaw>/auto/<id>.json, a gdy pliku brak albo jest nieaktualny (inny
                    algorytm, inne ustawienia, zmieniona siatka) - liczony i zapisywany
                    (contacts.auto_contacts),
    "manual-auto" - ręczne, a dla kości bez ręcznych - z algorytmu jak w "auto".
    Powierzchnie mniejsze niż `min_area_au2` są pomijane.

DLA KAŻDEJ PARY POWIERZCHNI (A z jednej kości, B z innej):
1. PRÓBKOWANIE: równomierne punkty co `sample_au` z normalnymi (na zewnątrz kości).
2. ODWRÓCENIE NORMALNYCH A. Dwa przełomy stykają się "twarzą w twarz": wypukłość na A
   to wklęsłość na B, a ich normalne są przeciwne. Po odwróceniu normalnych A wygląda
   dla deskryptorów jak B, więc działa ta sama rejestracja co przy łączeniu skanów.
3. ZGRUBNE USTAWIENIE - RANSAC na deskryptorach FPFH (punkty co `coarse_voxel_au`,
   promień `fpfh_radius_au`): losuje trójki podobnych punktów, odrzuca ustawienia
   z normalnymi różnymi o więcej niż `ransac_normal_angle_deg`, zostawia to, z którym
   zgadza się najwięcej punktów. Zawsze wszystkie `ransac_iterations` (open3d liczy
   wielowątkowo, więc wynik jest losowy - mało iteracji = często złe ustawienie).
4. DOKŁADNE USTAWIENIE - ICP point-to-plane: pary punktów szukane do `icp_start_au`,
   potem do połowy tego, na koniec do 2 x `sample_au`.
   Kroki 3-4 są powtarzane `ransac_seeds` razy, wygrywa ustawienie z najlepszym wynikiem.
5. OCENA ustawienia T (A -> B):
   - styk [a.u.²]:    punkty A bliżej niż `tolerance_au` od B, z normalnymi przeciwnymi
                    z dokładnością do `normal_angle_deg` (i odwrotnie, B na A);
                    pole styku = mniejsze z pól obu stron (część pokryta x pole powierzchni),
   - RMSE [a.u.]:     średnia kwadratowa odległość punktów A w styku od B,
   - kolizja [a.u.²]: pole CAŁEJ kości A, które leży pod powierzchnią kości B głębiej niż
                    `collision_au` (sprawdzane do `collision_search_au` od B), i odwrotnie -
                    prawdziwe dopasowanie przełomów nie może wbijać jednej kości w drugą,
   - WYNIK = styk - kolizja [a.u.²]; pary są sortowane od najlepszego wyniku.

ATLAS ANATOMICZNY (use_atlas, settings.json "matching"; src/atlas.py, cache out/3d/<zestaw>/cache/atlas/):
    Dla każdej kości z powierzchniami atlas mówi, z której kompletnej kości z bazy pochodzi fragment
    (prawdopodobieństwa, też "żadna z bazy") i jak na niej leży (T: model -> fragment, ze skalą).
    "Ta sama kość" = ten sam model i ta sama strona (odbicie lustrzane albo nie); modele w jednym
    podfolderze bazy (typ kości) liczą się razem, bez strony - mogą być z różnych stron ciała.
    Z tego (atlas_relation): P(ta sama kość) = suma p_A x p_B po kościach, P(różne) = reszta znanych;
    "żadna z bazy" nie liczy się do żadnej, więc niepewny fragment nie wyklucza pary.
    a) POMIJANIE PAR: P(różne kości) >= atlas_skip_different -> para nie jest dopasowywana
       (przełomy różnych kości nie mogą do siebie pasować).
    b) START Z ATLASU (atlas_starts): gdy A i B leżą na tym samym modelu, ich położenie względem
       siebie wynika z ustawień na modelu: x_B = T_B T_A⁻¹ x_A (bez różnicy skal - sam obrót i
       przesunięcie zgodne w środku powierzchni A). Z tego startu ICP (pierwszy promień
       atlas_icp_start_au, dalej jak w kroku 4). Gdy najlepszy start pokrył obie powierzchnie co
       najmniej w atlas_enough_cover, RANSAC (krok 3) jest pomijany - inaczej liczony jak zwykle.
    c) PREMIA: WYNIK = styk x (1 + atlas_boost x P(ta sama kość)) - kolizja; para bez styku nic nie zyskuje.

Użycie: scripts/match_surfaces.py liczy i zapisuje out/3d/<zestaw>/matching.json,
scripts/tune_matching.py (src/match_tool.py) go pokazuje: suwaki ustawień z przeliczaniem
jednej pary albo (--html) statyczna strona out/3d/<zestaw>/matching.html.
W kodzie:
    surfaces, info = extract_surfaces(load_bones(features=True), params, contact_params)
    results = match_all(surfaces, params)     # [{"a", "b", "score", "contact_au2", "T", ...}, ...]
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from bones import Bone
from contacts import (ContactParams, Contacts, auto_contacts, is_excluded, load_contacts, scan_frames,
                      triangle_labels)
from data_parsing import compute_fpfh, sample_surface, submesh, to_cloud, voxel_keep
from merging import covered, transform_points

log = logging.getLogger(__name__)
reg = o3d.pipelines.registration

SOURCES = ("manual", "auto", "manual-auto")
# Opisy trybów "source" do komunikatów i stron (match_surfaces.py, match_tool).
SOURCE_NAMES = {"manual": "tylko ręczne", "auto": "tylko z algorytmu",
                "manual-auto": "ręczne, a gdy brak - z algorytmu"}


@dataclass(frozen=True)
class SurfaceMatchParams:
    """Ustawienia dopasowania powierzchni styku (settings.json, sekcja "matching" - tam pełne opisy).

    Jak każde ustawienie wpływa na algorytm: kroki 1-5 w docstringu modułu.
    """

    source: str = "manual"                 # "manual" / "auto" / "manual-auto"
    min_area_au2: float = 20.0             # mniejsze powierzchnie styku są pomijane [a.u.²]
    sample_au: float = 0.3                 # odstęp punktów powierzchni [a.u.]
    coarse_voxel_au: float = 1.0           # odstęp punktów zgrubnego ustawienia (RANSAC) [a.u.]
    fpfh_radius_au: float = 4.0            # promień deskryptora FPFH [a.u.]
    ransac_iterations: int = 1000000       # zawsze wszystkie (bez wczesnego końca)
    ransac_seeds: int = 2                  # niezależnych prób RANSAC + ICP na parę
    ransac_normal_angle_deg: float = 35.0  # RANSAC: maks. różnica normalnych par punktów [stopnie]
    icp_start_au: float = 2.0              # ICP: pierwszy promień szukania par punktów [a.u.]
    tolerance_au: float = 0.5              # styk: punkty bliżej niż to [a.u.] ...
    normal_angle_deg: float = 45.0         # ... z normalnymi przeciwnymi z tą dokładnością [stopnie]
    collision_au: float = 1.0              # kolizja: głębiej niż to pod powierzchnią drugiej kości [a.u.]
    collision_search_au: float = 4.0       # ... sprawdzane do tej odległości od drugiej kości [a.u.]
    max_shown: int = 15                    # ile najlepszych par pokazać w 3D
    # atlas anatomiczny (docstring modułu, "ATLAS ANATOMICZNY")
    use_atlas: bool = False                # kontekst z atlasu: pomijanie par, start z atlasu, premia
    atlas_skip_different: float = 0.95     # pomiń parę, gdy P(różne kości) >= tyle; > 1 = nie pomijaj
    atlas_starts: int = 2                  # najwięcej startów z atlasu na parę (wspólne modele); 0 = bez
    atlas_icp_start_au: float = 6.0        # ICP ze startu z atlasu: pierwszy promień szukania par [a.u.]
    atlas_enough_cover: float = 0.35       # start z atlasu pokrył obie powierzchnie w tylu -> bez RANSAC; > 1 = zawsze
    atlas_boost: float = 0.25              # styk liczy się do tylu razy mocniej przy P(ta sama kość) = 1; 0 = bez


# -----------------
# POWIERZCHNIE
# -----------------

@dataclass
class Surface:
    """Jedna powierzchnia styku jednej kości, w układzie tej kości (jak out/3d/<zestaw>/bones/<id>/mesh.ply).

    Tworzona przez extract_surfaces. Punkty są równomierną próbką samej powierzchni
    styku (co sample_au), a bone_tree - punktami całej kości (do liczenia kolizji).
    """

    bone: Bone
    contacts: Contacts
    sid: int                       # numer powierzchni w etykietach kości
    points: np.ndarray             # (N, 3)
    normals: np.ndarray            # (N, 3), na zewnątrz kości
    area_au2: float                # pole powierzchni (z siatki, nie z punktów)
    bone_tree: cKDTree | None = None   # punkty CAŁEJ kości (do kolizji), wspólne dla powierzchni kości
    atlas: dict | None = None          # z której kości z bazy pochodzi fragment (atlas_context); None = bez atlasu
    _cache: dict = field(default_factory=dict, repr=False)   # chmury i KD-drzewo (clouds, tree)

    @property
    def key(self) -> str:
        """Unikalny klucz "<id kości>#<numer powierzchni>" - tak zapisany w matching.json."""
        return f"{self.bone.id}#{self.sid}"

    @property
    def name(self) -> str:
        """Czytelna nazwa "<nazwa kości> · <nazwa powierzchni>"."""
        return f"{self.bone.label} · {self.contacts.name(self.sid)}"

    @property
    def is_manual(self) -> bool:
        """True = powierzchnia oznaczona ręcznie, False = z algorytmu."""
        return self.contacts.source == "manual"

    def clouds(self, p: SurfaceMatchParams, flip: bool):
        """Chmury punktów powierzchni do rejestracji - liczone raz dla każdej wartości `flip`.

        Pamięć podręczna pamięta ustawienia, z którymi chmury policzono (sample_au,
        coarse_voxel_au, fpfh_radius_au) - po ich zmianie chmury są liczone od nowa.

        Args:
            p:    ustawienia (sample_au, coarse_voxel_au, fpfh_radius_au).
            flip: True = normalne odwrócone - tak przygotowuje się stronę "A" pary.

        Returns:
            (gęsta - wszystkie punkty, średnia - co 2 x sample_au, zgrubna - co coarse_voxel_au,
             FPFH zgrubnej w promieniu fpfh_radius_au); chmury z normalnymi.
        """
        key = ("clouds", flip, p.sample_au, p.coarse_voxel_au, p.fpfh_radius_au)
        if key not in self._cache:
            nrm = -self.normals if flip else self.normals
            mid = voxel_keep(self.points, 2 * p.sample_au)
            coarse_idx = voxel_keep(self.points, p.coarse_voxel_au)
            coarse = to_cloud(self.points[coarse_idx], nrm[coarse_idx])
            self._cache[key] = (to_cloud(self.points, nrm), to_cloud(self.points[mid], nrm[mid]), coarse,
                                compute_fpfh(coarse, p.fpfh_radius_au))
        return self._cache[key]

    def tree(self) -> cKDTree:
        """KD-drzewo punktów powierzchni (szukanie najbliższych sąsiadów) - liczone raz.

        Returns:
            cKDTree z `points` (te same indeksy co points i normals).
        """
        if "tree" not in self._cache:
            self._cache["tree"] = cKDTree(self.points)
        return self._cache["tree"]


def bone_contacts(bone: Bone, source: str, contact_params: ContactParams = ContactParams(),
                  root: Path | None = None) -> tuple[Contacts | None, str]:
    """Powierzchnie styku kości wg trybu `source` (opis trybów: docstring modułu).

    W trybach "auto" i "manual-auto" może uruchomić algorytm wykrywania i zapisać jego
    wynik w contacts/<zestaw>/auto/ (contacts.auto_contacts) - za pierwszym razem trwa to dłużej.

    Args:
        bone:           kość.
        source:         "manual" / "auto" / "manual-auto".
        contact_params: algorytm wykrywania i jego ustawienia (dla "auto").
        root:           folder oznaczeń zamiast contacts/ z settings.json (np. w testach).

    Returns:
        (etykiety albo None, skąd: "ręczne" / "algorytm (cache)" / "algorytm (policzone)" / "brak ...").

    Raises:
        ValueError: nieznany tryb `source`.
    """
    if source not in SOURCES:
        raise ValueError(f"matching.source = {source!r}, dostępne: {', '.join(SOURCES)}")
    if source in ("manual", "manual-auto"):
        c = load_contacts(bone.id, bone.mesh, "manual", root=root, frames=scan_frames(bone))
        if c is not None and c.labels.any():
            return c, "ręczne"
        if source == "manual":
            return None, "brak ręcznych oznaczeń"
    if is_excluded(bone, contact_params):
        return None, "pominięta (contacts -> exclude)"
    c, computed = auto_contacts(bone, contact_params, root)
    if not c.labels.any():
        return None, "brak (algorytm nic nie znalazł)"
    return c, "algorytm (policzone)" if computed else "algorytm (cache)"


def make_surface(bone: Bone, contacts: Contacts, sid: int, sample_au: float,
                 tri_labels: np.ndarray | None = None, bone_tree: cKDTree | None = None) -> Surface | None:
    """Jedna powierzchnia styku kości: punkty co `sample_au` z normalnymi i pole.

    Args:
        bone:       kość wczytana z chmurą punktów (load_bones(features=True)).
        contacts:   etykiety powierzchni tej kości.
        sid:        numer powierzchni.
        sample_au:  odstęp punktów [a.u.].
        tri_labels: etykiety trójkątów (triangle_labels) - podane, żeby nie liczyć ich dla każdej powierzchni.
        bone_tree:  KD-drzewo punktów całej kości (do kolizji); None = zbudowane tutaj.

    Returns:
        Powierzchnia albo None, gdy etykieta `sid` nie ma żadnego trójkąta.
    """
    tl = triangle_labels(bone.mesh, contacts.labels) if tri_labels is None else tri_labels
    m = submesh(bone.mesh, tl == sid)
    if not len(m.triangles):
        return None
    if bone_tree is None:
        bone_tree = cKDTree(np.asarray(bone.cloud.points))
    s = sample_surface(m, sample_au)
    return Surface(bone=bone, contacts=contacts, sid=sid, points=s.points, normals=s.normals,
                   area_au2=float(m.get_surface_area()), bone_tree=bone_tree)


def extract_surfaces(bones: list[Bone], p: SurfaceMatchParams, contact_params: ContactParams = ContactParams(),
                     root: Path | None = None, log=None) -> tuple[list[Surface], dict[str, str]]:
    """Powierzchnie styku wszystkich kości (każda osobno), bez mniejszych niż p.min_area_au2.

    Kości muszą być wczytane z chmurą punktów (load_bones(features=True)) - do kolizji.
    Przy p.use_atlas każda kość z powierzchniami dostaje kontekst z atlasu (Surface.atlas,
    atlas_context) - z cache, a gdy go brak, liczony (jak algorytm "atlas", za pierwszym razem dłużej).

    Args:
        bones:          kości do dopasowania.
        p:              ustawienia (source, min_area_au2, sample_au, use_atlas).
        contact_params: algorytm wykrywania (tryby "auto" i "manual-auto"); też analiza lokalna atlasu.
        root:           folder oznaczeń zamiast contacts/ z settings.json.
        log:            komunikaty, gdy atlas liczy dopasowanie fragmentu od nowa (None = cisza).

    Returns:
        (powierzchnie, {id kości: skąd wzięte powierzchnie - jak w bone_contacts; gdy
         wszystkie odpadły jako za małe, z dopiskiem ", powierzchnie < ... a.u.²"; przy
         use_atlas z dopiskiem ", atlas: <najbardziej prawdopodobna kość> <p>"}).

    Raises:
        ValueError: kość bez chmury punktów albo nieznany p.source.
    """
    out, info = [], {}
    atlas_off = ""                 # powód, gdy atlasu nie da się użyć (pusta baza) - wtedy bez atlasu
    for b in bones:
        if b.cloud is None:
            raise ValueError(f"{b.id}: brak chmury punktów - wczytaj kości przez load_bones(features=True)")
        c, info[b.id] = bone_contacts(b, p.source, contact_params, root)
        if c is None:
            continue
        tl = triangle_labels(b.mesh, c.labels)
        tree = None
        for k in c.surface_ids:
            if tree is None:
                tree = cKDTree(np.asarray(b.cloud.points))
            s = make_surface(b, c, k, p.sample_au, tl, tree)
            if s is not None and s.area_au2 >= p.min_area_au2:
                out.append(s)
        mine = [s for s in out if s.bone.id == b.id]
        if not mine:
            info[b.id] += f", powierzchnie < {p.min_area_au2:g} a.u.²"
        elif p.use_atlas and not atlas_off:
            try:
                ctx = atlas_context(b, contact_params, log)
            except ValueError:     # pusta baza modeli (atlas.load_atlas)
                atlas_off = "baza modeli pusta - python scripts/build_atlas.py"
                ctx = None
            for s in mine:
                s.atlas = ctx
            info[b.id] += f", atlas: {atlas_off or atlas_top(ctx)}"
    return out, info


# -----------------
# ATLAS: z której kości pochodzi fragment (src/atlas.py)
# -----------------

ATLAS_MIN_JOINT = 0.02     # start z atlasu tylko z modelu, na którym p_A x p_B >= tyle


def atlas_context(bone: Bone, contact_params: ContactParams, log=None) -> dict:
    """Ranking kości z bazy dla fragmentu, z ustawieniami na modelach (atlas.cached_match).

    Ten sam cache co algorytm "atlas" (out/3d/<zestaw>/cache/atlas/<id>.npz) - gdy wykrywanie
    liczono algorytmem "atlas" z tymi samymi ustawieniami, nic nie jest liczone od nowa.

    Args:
        bone:           kość (fragment).
        contact_params: analiza lokalna (feature_keys, feature_threshold) - jak przy wykrywaniu.
        log:            komunikaty przy liczeniu od nowa (brak cache); None = cisza.

    Returns:
        {"p_unknown": p "żadna z bazy", "candidates": [{"model", "group", "mirror", "label", "p",
         "rel_scale", "T": 4x4 model -> fragment (lista)}] od najbardziej prawdopodobnego}.

    Raises:
        ValueError: pusta baza modeli.
    """
    import atlas
    import config

    ap, sfp = config.atlas_params(), config.surface_feature_params()
    started: list = []

    def note(text: str) -> None:
        """Pierwszy komunikat atlasu dla tej kości poprzedzony jej nazwą (tylko przy liczeniu od nowa)."""
        if not started:
            started.append(True)
            log(f"  atlas: {bone.label} - liczę dopasowanie do bazy (raz, potem z cache)...")
        log(text)

    m = atlas.cached_match(bone, atlas.load_atlas(ap), ap, contact_params, sfp, log=note if log else None)
    keys = ("model", "group", "mirror", "label", "p", "rel_scale", "T")
    return {"p_unknown": round(float(m.p_unknown), 4),
            "candidates": [{k: c[k] for k in keys} for c in m.candidates if c["p"] >= 0.001]}


def atlas_top(ctx: dict | None) -> str:
    """Najbardziej prawdopodobna kość z kontekstu atlasu, np. "FIBULA_LEFT 82%" (do konsoli)."""
    if ctx is None:
        return "brak"
    import atlas

    best = ctx["candidates"][0] if ctx["candidates"] else None
    if best is None or best["p"] < ctx["p_unknown"]:
        return f"{atlas.UNKNOWN} {ctx['p_unknown']:.0%}"
    return f"{best['label']} {best['p']:.0%}"


def atlas_kinds(ctx: dict) -> dict[str, float]:
    """P(kość) fragmentu: model + strona (odbicie albo nie); modele z podfolderu bazy = jeden typ, bez strony.

    Args:
        ctx: kontekst z atlas_context.

    Returns:
        {klucz kości: prawdopodobieństwo}; suma = 1 - p "żadna z bazy".
    """
    acc: dict[str, float] = {}
    for c in ctx["candidates"]:
        key = c["group"] or f"{c['model']}|{int(bool(c['mirror']))}"
        acc[key] = acc.get(key, 0.0) + c["p"]
    return acc


def atlas_relation(a: Surface, b: Surface) -> tuple[float, float]:
    """Czy fragmenty a i b pochodzą z tej samej kości - z rankingów atlasu obu fragmentów.

    Args:
        a, b: powierzchnie (Surface.atlas z atlas_context; None = bez wiedzy).

    Returns:
        (P(ta sama kość), P(różne kości)); reszta do 1 = niewiadomo (któryś może być "żadna
        z bazy"). Bez kontekstu któregoś fragmentu: (0, 0).
    """
    if a.atlas is None or b.atlas is None:
        return 0.0, 0.0
    ka, kb = atlas_kinds(a.atlas), atlas_kinds(b.atlas)
    same = sum(q * kb.get(k, 0.0) for k, q in ka.items())
    return same, max(sum(ka.values()) * sum(kb.values()) - same, 0.0)


def atlas_starts(a: Surface, b: Surface, p: SurfaceMatchParams) -> list[tuple[str, float, np.ndarray]]:
    """Ustawienia startowe A -> B z atlasu: oba fragmenty leżą na tym samym modelu (i stronie).

    x_A = T_A x_modelu, x_B = T_B x_modelu  =>  x_B = T_B T_A⁻¹ x_A. Skale obu fragmentów na
    modelu mogą się różnić (osobno dopasowane), a skan jest w jednych jednostkach - zostaje
    sam obrót (najbliższy macierzy T_B T_A⁻¹) i przesunięcie zgodne w środku powierzchni A.

    Args:
        a, b: powierzchnie z kontekstem atlasu (Surface.atlas).
        p:    ustawienia (use_atlas, atlas_starts).

    Returns:
        [(nazwa modelu, p_A x p_B, macierz 4x4 A -> B)] od najbardziej prawdopodobnego, najwyżej
        p.atlas_starts, tylko modele z p_A x p_B >= ATLAS_MIN_JOINT; [] = bez startów z atlasu.
    """
    if not p.use_atlas or p.atlas_starts <= 0 or a.atlas is None or b.atlas is None:
        return []
    on_b = {(c["model"], bool(c["mirror"])): c for c in b.atlas["candidates"]}
    center = a.points.mean(axis=0)
    out = []
    for ca in a.atlas["candidates"]:
        cb = on_b.get((ca["model"], bool(ca["mirror"])))
        joint = ca["p"] * cb["p"] if cb is not None else 0.0
        if joint < ATLAS_MIN_JOINT:
            continue
        M = np.asarray(cb["T"], dtype=float) @ np.linalg.inv(np.asarray(ca["T"], dtype=float))
        U, _, Vt = np.linalg.svd(M[:3, :3])
        R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, transform_points(M, center[None])[0] - R @ center
        out.append((ca["label"], joint, T))
    out.sort(key=lambda s: -s[1])
    return out[:p.atlas_starts]


# -----------------
# DOPASOWANIE PARY
# -----------------

def contact_score(a: Surface, b: Surface, T: np.ndarray, p: SurfaceMatchParams) -> dict:
    """Jak dobrze powierzchnia a, przeniesiona przekształceniem T, przylega do b (krok 5 modułu).

    Punkt jest "w styku", gdy druga powierzchnia jest bliżej niż p.tolerance_au, a normalne
    są przeciwne z dokładnością do p.normal_angle_deg. Sprawdzane w obie strony.

    Args:
        a, b: powierzchnie (każda w układzie swojej kości).
        T:    przekształcenie 4x4 z układu kości a do układu kości b.
        p:    ustawienia (tolerance_au, normal_angle_deg).

    Returns:
        {"cover_a", "cover_b": ułamek punktów a / b w styku,
         "contact_au2": min(cover_a x pole a, cover_b x pole b),
         "rmse_au": RMSE odległości punktów a w styku od b (nan, gdy styku brak)}.
    """
    min_dot = float(np.cos(np.deg2rad(p.normal_angle_deg)))
    pa = transform_points(T, a.points)
    na = -(a.normals @ T[:3, :3].T)                  # odwrócone: w styku mają być zgodne z normalnymi b
    ok_a = covered(pa, na, b.tree(), b.normals, p.tolerance_au, min_dot)
    Ti = np.linalg.inv(T)
    pb = transform_points(Ti, b.points)
    nb = -(b.normals @ Ti[:3, :3].T)
    ok_b = covered(pb, nb, a.tree(), a.normals, p.tolerance_au, min_dot)
    d = b.tree().query(pa[ok_a])[0] if ok_a.any() else np.zeros(0)
    fa, fb = float(ok_a.mean()), float(ok_b.mean())
    return {"cover_a": fa, "cover_b": fb, "contact_au2": min(fa * a.area_au2, fb * b.area_au2),
            "rmse_au": float(np.sqrt(np.mean(d**2))) if len(d) else float("nan")}


def collision_area(a: Surface, b: Surface, T: np.ndarray, p: SurfaceMatchParams) -> float:
    """Pole kości a (po przekształceniu T), które wchodzi w kość b głębiej niż collision_au, i odwrotnie [a.u.²].

    Punkt jest "w środku" kości, jeśli najbliższy jej punkt jest bliżej niż collision_search_au,
    a punkt leży za jej powierzchnią (po przeciwnej stronie niż normalna) o więcej niż
    collision_au. Skany są otwarte, więc to przybliżenie - wystarcza do odrzucenia
    ustawień "kość w kości". Pole = liczba takich punktów chmury x (pole kości / liczba
    punktów chmury).

    Args:
        a, b: powierzchnie - liczone są ich CAŁE kości (bone.cloud, bone_tree).
        T:    przekształcenie 4x4 z układu kości a do układu kości b.
        p:    ustawienia (collision_au, collision_search_au).

    Returns:
        Suma pól z obu kierunków [a.u.²].
    """
    def one_way(src: Surface, dst: Surface, M: np.ndarray) -> float:
        """Pole kości src (przeniesionej M) wbite w kość dst [a.u.²].

        Args:
            src: powierzchnia, której kość jest przenoszona.
            dst: powierzchnia kości, w którą src może się wbijać.
            M:   przekształcenie 4x4 z układu kości src do układu kości dst.

        Returns:
            Pole punktów chmury src leżących głębiej niż p.collision_au pod powierzchnią dst.
        """
        ps = transform_points(M, np.asarray(src.bone.cloud.points))
        pd, nd = np.asarray(dst.bone.cloud.points), np.asarray(dst.bone.cloud.normals)
        d, idx = dst.bone_tree.query(ps, distance_upper_bound=p.collision_search_au)
        ok = np.isfinite(d)
        depth = np.einsum("ij,ij->i", pd[idx[ok]] - ps[ok], nd[idx[ok]])
        per_point = src.bone.meta.get("stats", {}).get("area_au2", 0) / max(len(ps), 1)
        return float((depth > p.collision_au).sum() * per_point)

    return one_way(a, b, T) + one_way(b, a, np.linalg.inv(T))


def evaluate(a: Surface, b: Surface, T: np.ndarray, p: SurfaceMatchParams) -> dict:
    """Pełna ocena ustawienia T (krok 5 modułu): contact_score + collision_area + wynik.

    Bez szukania ustawienia - szybkie, więc nadaje się do sprawdzenia, jak wynik
    zmienia się z ustawieniami oceny (tolerance_au, normal_angle_deg, collision_*).

    Args:
        a, b: powierzchnie (każda w układzie swojej kości).
        T:    przekształcenie 4x4 z układu kości a do układu kości b.
        p:    ustawienia oceny (tolerance_au, normal_angle_deg, collision_au, collision_search_au,
              przy use_atlas też atlas_boost).

    Returns:
        {"cover_a", "cover_b", "contact_au2", "rmse_au", "collision_au2",
         "score" = contact_au2 - collision_au2 (+ atlas_bonus_au2)}; przy use_atlas dodatkowo
        "atlas_same", "atlas_diff" (atlas_relation) i "atlas_bonus_au2" = styk x atlas_boost x atlas_same.
    """
    sc = contact_score(a, b, T, p)
    sc["collision_au2"] = collision_area(a, b, T, p)
    sc["score"] = sc["contact_au2"] - sc["collision_au2"]
    if p.use_atlas:
        same, diff = atlas_relation(a, b)
        sc.update(atlas_same=round(same, 4), atlas_diff=round(diff, 4),
                  atlas_bonus_au2=sc["contact_au2"] * p.atlas_boost * same)
        sc["score"] += sc["atlas_bonus_au2"]
    return sc


def refine_icp(src_clouds, tgt_clouds, T: np.ndarray, p: SurfaceMatchParams,
               first_au: float | None = None) -> np.ndarray:
    """ICP point-to-plane: icp_start_au i połowa tego na rzadszych chmurach, potem 2 x sample_au na gęstych.

    Args:
        src_clouds: chmury strony A z Surface.clouds(flip=True) (używane [0] i [1]).
        tgt_clouds: chmury strony B z Surface.clouds(flip=False).
        T:          przekształcenie startowe A -> B (z RANSAC albo z atlasu).
        p:          ustawienia (icp_start_au, sample_au).
        first_au:   dodatkowy pierwszy krok z tym promieniem (start z atlasu jest mniej dokładny
                    niż z RANSAC); None = bez.

    Returns:
        Poprawione przekształcenie 4x4 A -> B.
    """
    crit = reg.ICPConvergenceCriteria(max_iteration=50)
    steps = ((p.icp_start_au, 1), (p.icp_start_au / 2, 1), (2 * p.sample_au, 0))
    if first_au and first_au > p.icp_start_au:
        steps = ((first_au, 1),) + steps
    for dist, k in steps:
        T = reg.registration_icp(src_clouds[k], tgt_clouds[k], dist, T,
                                 reg.TransformationEstimationPointToPlane(), crit).transformation
    return T


def match_pair(a: Surface, b: Surface, p: SurfaceMatchParams) -> dict:
    """Szuka T kładącego powierzchnię a na b (twarzą w twarz). Zwraca najlepszą próbę.

    Próby: najpierw starty z atlasu (atlas_starts -> refine_icp z promieniem atlas_icp_start_au),
    potem p.ransac_seeds prób RANSAC na FPFH (ziarno = numer próby RANSAC) -> refine_icp. RANSAC
    jest pomijany, gdy najlepszy start z atlasu pokrył obie powierzchnie co najmniej w
    p.atlas_enough_cover. Każda próba oceniana przez evaluate; wygrywa największy wynik.

    Args:
        a, b: powierzchnie dwóch różnych kości.
        p:    ustawienia dopasowania.

    Returns:
        {"score", "contact_au2", "collision_au2", "cover_a", "cover_b", "rmse_au",
         "T": macierz 4x4 (lista) z układu kości a do układu kości b, "seed": numer wygranej
         próby (od 0, starty z atlasu pierwsze), "start": "RANSAC" albo "atlas: <model>",
         "ransac": czy liczono RANSAC} (+ pola atlasu z evaluate przy use_atlas).
    """
    ca, cb = a.clouds(p, flip=True), b.clouds(p, flip=False)
    best, n = None, 0

    def keep(T: np.ndarray, start: str) -> None:
        """Ocenia próbę n i zapamiętuje ją, jeśli jest najlepsza."""
        nonlocal best, n
        sc = evaluate(a, b, T, p)
        if best is None or sc["score"] > best["score"]:
            best = {**sc, "T": T.tolist(), "seed": n, "start": start}
        n += 1

    for label, _, T0 in atlas_starts(a, b, p):
        keep(refine_icp(ca, cb, T0, p, first_au=p.atlas_icp_start_au), f"atlas: {label}")
    ransac = best is None or min(best["cover_a"], best["cover_b"]) < p.atlas_enough_cover
    if ransac:
        dist = 1.5 * p.coarse_voxel_au             # maks. odległość punktów uznanych za pasujące w RANSAC
        for seed in range(p.ransac_seeds):
            o3d.utility.random.seed(seed)
            coarse = reg.registration_ransac_based_on_feature_matching(
                ca[2], cb[2], ca[3], cb[3], True, dist,
                reg.TransformationEstimationPointToPoint(False), 3,
                [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                 reg.CorrespondenceCheckerBasedOnDistance(dist),
                 reg.CorrespondenceCheckerBasedOnNormal(np.deg2rad(p.ransac_normal_angle_deg))],
                reg.RANSACConvergenceCriteria(p.ransac_iterations, 1.0))    # 1.0 = bez wczesnego końca
            keep(refine_icp(ca, cb, coarse.transformation, p), "RANSAC")
    if best is not None:
        best["ransac"] = ransac
    return best


def split_pairs(surfaces: list[Surface], p: SurfaceMatchParams) -> tuple[list[tuple[int, int]], list[dict]]:
    """Pary powierzchni z różnych kości: do dopasowania i pominięte (atlas: na pewno różne kości).

    Args:
        surfaces: powierzchnie z extract_surfaces.
        p:        ustawienia (use_atlas, atlas_skip_different).

    Returns:
        ([(i, j) indeksy w `surfaces`] do dopasowania,
         [{"a", "b", "a_name", "b_name", "atlas_same", "atlas_diff"}] pominięte).
    """
    pairs, skipped = [], []
    for i in range(len(surfaces)):
        for j in range(i + 1, len(surfaces)):
            a, b = surfaces[i], surfaces[j]
            if a.bone.id == b.bone.id:
                continue
            same, diff = atlas_relation(a, b) if p.use_atlas else (0.0, 0.0)
            if diff > 0 and diff >= p.atlas_skip_different:
                skipped.append({"a": a.key, "b": b.key, "a_name": a.name, "b_name": b.name,
                                "atlas_same": round(same, 4), "atlas_diff": round(diff, 4)})
            else:
                pairs.append((i, j))
    return pairs, skipped


def match_all(surfaces: list[Surface], p: SurfaceMatchParams, progress=print) -> list[dict]:
    """Dopasowuje każdą parę powierzchni z różnych kości. Wynik posortowany od najlepszej pary.

    Liczba par rośnie kwadratowo z liczbą powierzchni (~N²/2), każda para trwa ~1-3 s.
    Przy use_atlas pary na pewno różnych kości są pomijane (split_pairs).

    Args:
        surfaces: powierzchnie z extract_surfaces.
        p:        ustawienia dopasowania.
        progress: funkcja do wypisywania postępu (co 10 par); lambda _: None = cisza.

    Returns:
        Wyniki match_pair uzupełnione o opis pary: "a", "b" (Surface.key), "a_name",
        "b_name", "a_area_au2", "b_area_au2", "a_manual", "b_manual"; od największego "score".
    """
    pairs, skipped = split_pairs(surfaces, p)
    note = f" (pominięte {len(skipped)} - atlas: różne kości)" if skipped else ""
    progress(f"Powierzchnie: {len(surfaces)}, pary do sprawdzenia: {len(pairs)}{note}")
    results = []
    start = time.perf_counter()
    for n, (i, j) in enumerate(pairs, 1):
        a, b = surfaces[i], surfaces[j]
        r = match_pair(a, b, p)
        r.update(a=a.key, b=b.key, a_name=a.name, b_name=b.name, a_area_au2=a.area_au2, b_area_au2=b.area_au2,
                 a_manual=a.is_manual, b_manual=b.is_manual)
        results.append(r)
        if n % 10 == 0 or n == len(pairs):
            progress(f"  {n}/{len(pairs)} par ({time.perf_counter() - start:.0f} s)")
    results.sort(key=lambda r: -r["score"])
    return results


def json_safe(obj):
    """Kopia `obj` z NaN / nieskończonościami zamienionymi na None (w JSON: null).

    Python zapisuje float("nan") jako NaN, a to nie jest poprawny JSON - przeglądarka
    (JSON.parse) odrzuca cały plik. NaN ma np. rmse_au pary bez styku.

    Args:
        obj: słownik / lista / liczba / tekst (zagnieżdżone dowolnie).

    Returns:
        Ten sam obiekt z niekończonymi liczbami zamienionymi na None.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def report(surfaces: list[Surface], results: list[dict], p: SurfaceMatchParams,
           contact_params: ContactParams = ContactParams()) -> dict:
    """Słownik do zapisu w matching.json (i do visualization_3d.match_matrix_figure).

    Args:
        surfaces:       powierzchnie z extract_surfaces.
        results:        wyniki match_all dla tych powierzchni.
        p:              ustawienia dopasowania, z którymi liczono wyniki.
        contact_params: ustawienia wykrywania (ważne w trybach "auto" i "manual-auto").

    Returns:
        {"params": ustawienia dopasowania, "contact_params": ustawienia wykrywania,
         "surfaces": [{"key", "bone", "bone_label", "surface", "name", "source",
                       "area_au2", "points"}],
         "pairs": wyniki match_all (liczby zaokrąglone do 4 miejsc),
         przy use_atlas także "atlas": {id kości: kontekst z atlas_context} (do match_tool) i
         "skipped": pary pominięte (split_pairs)}.
    """
    rep = {"params": asdict(p), "contact_params": asdict(contact_params),
           "surfaces": [{"key": s.key, "bone": s.bone.id, "bone_label": s.bone.label, "surface": s.sid,
                         "name": s.contacts.name(s.sid), "source": s.contacts.source,
                         "area_au2": round(s.area_au2, 1), "points": len(s.points)} for s in surfaces],
           "pairs": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in results]}
    if p.use_atlas:
        rep["atlas"] = {s.bone.id: s.atlas for s in surfaces if s.atlas is not None}
        rep["skipped"] = split_pairs(surfaces, p)[1]
    return rep

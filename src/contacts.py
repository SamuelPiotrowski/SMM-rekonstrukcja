"""
Powierzchnie styku (przełomy) na kościach: format zapisu, statystyki, algorytmy wykrywania.

Etykiety są przypisane do WIERZCHOŁKÓW siatki kości (out/3d/<zestaw>/bones/<id>/mesh.ply):
    labels[i] = 0  -> wierzchołek nie należy do żadnej powierzchni styku,
    labels[i] = k  -> wierzchołek należy do powierzchni styku nr k (1, 2, ...).
Jedna kość może mieć kilka powierzchni styku (np. przełom na każdym końcu).

Skąd się biorą etykiety (oba rodzaje w jednym folderze contacts/, ten sam format):
    - RĘCZNIE: scripts/mark_contacts.py (przeglądarka) -> contacts/<zestaw>/<id_kości>.json
      To ręczna praca - jest w repozytorium, warto ją commitować.
    - ALGORYTM: scripts/detect_contacts.py -> contacts/<zestaw>/auto/<id_kości>.json
      Pole "source" = "auto:<nazwa algorytmu>", pole "detector" = algorytm i jego ustawienia.
      Generowane (nie ma ich w repozytorium) - auto_contacts() liczy je na nowo, gdy brakuje
      pliku albo zmienił się algorytm, jego ustawienia lub siatka kości.

Jak dodać własny algorytm:
    @detector("moja_metoda")
    def moja_metoda(bone: Bone, params: ContactParams) -> np.ndarray:
        ...  # zwróć labels (V,) jak wyżej
        return labels
    potem: python scripts/detect_contacts.py --method moja_metoda
    Jeśli wynik zależy od czegoś poza sekcją "contacts" (np. od ustawień innej sekcji),
    podaj @detector("nazwa", data_key=funkcja) - zmiana jej wyniku = auto_contacts liczy od nowa.
    albo na stałe (też dla dopasowania i narzędzia): settings.json, "contacts" -> "method".
    Skrypt porówna wynik z ręcznymi oznaczeniami (IoU, precyzja, czułość), jeśli są.

Format pliku JSON:
    {
      "format": "bones-smm-contacts/1",
      "bone": "<id kości>", "source": "manual" | "auto:<metoda>", "updated": "<data>",
      "mesh": {"vertices": V, "fingerprint": "..."},    # do wykrycia zmiany siatki
      "detector": {"method": "...", "params": {...}},   # tylko wynik algorytmu
      "labels_zlib_b64": "...",          # labels (V,) uint8 skompresowane zlib + base64
      "surfaces": [
        {"id": 1, "name": "...", "area_mm2": ..., "centroid": [x, y, z], "normal": [...],
         "max_diameter_mm": ..., "extent_mm": [dł., szer., grub.], "vertices": liczba,
         "points": [[x, y, z], ...]}     # punkty co ~0.5 mm - do przeniesienia na nową siatkę
      ]
    }
Najprościej czytać przez load_contacts(). Jeśli siatka kości się zmieni (np. inne ustawienia
prepare_data), etykiety są przenoszone na nową siatkę po położeniu punktów
(najbliższy punkt bliżej niż REMAP_MM).
"""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

from data_parsing import voxel_keep

FORMAT = "bones-smm-contacts/1"
POINTS_MM = 0.5       # odstęp zapisanych punktów powierzchni
REMAP_MM = 0.5        # promień przenoszenia etykiet na zmienioną siatkę
# Kolory powierzchni (te same w narzędziu do oznaczania i w wizualizacjach).
PALETTE = ("#e8508f", "#18b0a6", "#f2b134", "#7b61ff", "#2b9be0", "#9bc53d", "#ff8a3d", "#00c2d1")


@dataclass(frozen=True)
class ContactParams:
    """Ustawienia narzędzia do oznaczania i algorytmu wykrywania (settings.json, sekcja "contacts")."""

    method: str = "roughness"         # algorytm wykrywania (detect_contacts.py, narzędzie, dopasowanie "auto")
    brush_mm: float = 2.0             # startowy promień pędzla w narzędziu [mm]
    wand_angle_deg: float = 25.0      # startowa tolerancja kąta różdżki [stopnie]
    roughness_radius_mm: float = 1.0  # algorytm bazowy: promień liczenia szorstkości
    rough_threshold: float = 0.09     # algorytm bazowy: szorstkość, od której powierzchnia = przełom
    min_surface_mm2: float = 15.0     # oba algorytmy: mniejsze obszary są pomijane
    feature_keys: tuple = ("spread_2", "rough_1", "edge_mm")  # "features": łączone cechy (surface_features)
    feature_threshold: float = 0.15   # "features": o ile wynik ponad typowy dla tej kości (w odchyleniach) = przełom
    feature_smooth_mm: float = 1.0    # "features": uśrednianie wyniku w tym promieniu przed progiem
    feature_open_mm: float = 1.0      # "features": otwarcie morfologiczne (rozcina wąskie mostki między obszarami)


@dataclass
class Contacts:
    """Etykiety powierzchni styku jednej kości (zawartość jednego pliku contacts/...json).

    Etykiety dotyczą konkretnej siatki kości - labels[i] odpowiada wierzchołkowi i
    siatki mesh.ply (znaczenie wartości: docstring modułu).
    """

    bone_id: str
    labels: np.ndarray                          # (V,) uint8: 0 = brak, k = powierzchnia k
    names: dict[int, str] = field(default_factory=dict)   # {k: nazwa nadana w narzędziu}
    source: str = "manual"                      # "manual" albo "auto:<metoda>"
    updated: str = ""                           # data i czas zapisu (ISO), "" = jeszcze niezapisane
    remapped: bool = False                      # True = przeniesione ze starej wersji siatki
    detector: dict = field(default_factory=dict)  # wynik algorytmu: {"method", "params"}

    @property
    def surface_ids(self) -> list[int]:
        """Numery powierzchni obecnych w etykietach (bez 0), rosnąco."""
        return [int(k) for k in np.unique(self.labels) if k > 0]

    def name(self, k: int) -> str:
        """Nazwa powierzchni k do wyświetlania.

        Args:
            k: numer powierzchni (1, 2, ...).

        Returns:
            Nazwa nadana w narzędziu; gdy nie nadano (albo pusta) - "Powierzchnia k".
        """
        return self.names.get(k) or f"Powierzchnia {k}"


def surface_color(k: int) -> str:
    """Kolor powierzchni nr k - ten sam w narzędziu do oznaczania i w wizualizacjach.

    Args:
        k: numer powierzchni (1, 2, ...); po wyczerpaniu PALETTE kolory się powtarzają.

    Returns:
        Kolor "#rrggbb" z PALETTE.
    """
    return PALETTE[(k - 1) % len(PALETTE)]


# -----------------
# GEOMETRIA
# -----------------

def mesh_fingerprint(mesh: o3d.geometry.TriangleMesh) -> str:
    """Odcisk siatki: SHA-1 (16 znaków) współrzędnych wierzchołków zaokrąglonych do 0.001 mm.

    Zapisywany razem z etykietami. Inny odcisk przy odczycie = siatka kości się zmieniła
    (np. inne ustawienia prepare_data), więc etykiet nie da się wziąć wprost - trzeba je
    przenieść po położeniu punktów.

    Args:
        mesh: siatka kości.

    Returns:
        16 znaków hex.
    """
    v = np.round(np.asarray(mesh.vertices), 3).astype(np.float32)
    return hashlib.sha1(v.tobytes()).hexdigest()[:16]


def vertex_areas(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Pole przypadające na wierzchołek = 1/3 pól sąsiednich trójkątów [mm^2].

    Suma po wierzchołkach = pole całej siatki, więc suma po wierzchołkach powierzchni
    styku = jej pole.

    Args:
        mesh: siatka kości.

    Returns:
        (V,) pola; 0 dla wierzchołków bez trójkątów.
    """
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    a = 0.5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
    return np.bincount(t.ravel(), weights=np.repeat(a / 3, 3), minlength=len(v))


def vertex_adjacency(mesh: o3d.geometry.TriangleMesh):
    """Macierz sąsiedztwa wierzchołków (krawędzie siatki), scipy.sparse CSR (V, V).

    Przykład: (adj @ maska_0_1) > 0 = wierzchołki, które mają zaznaczonego sąsiada.

    Args:
        mesh: siatka kości.

    Returns:
        Macierz int8 (V, V): element [i, j] > 0, gdy i oraz j łączy krawędź (symetryczna;
        wartość = liczba trójkątów z tą krawędzią).
    """
    t = np.asarray(mesh.triangles)
    n = len(mesh.vertices)
    i = np.concatenate([t[:, 0], t[:, 1], t[:, 2], t[:, 1], t[:, 2], t[:, 0]])
    j = np.concatenate([t[:, 1], t[:, 2], t[:, 0], t[:, 0], t[:, 1], t[:, 2]])
    return coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n, n)).tocsr()


def triangle_labels(mesh: o3d.geometry.TriangleMesh, labels: np.ndarray) -> np.ndarray:
    """Etykiety wierzchołków -> etykiety trójkątów.

    Trójkąt dostaje etykietę, którą mają co najmniej 2 z 3 jego wierzchołków (inaczej 0).

    Args:
        mesh:   siatka kości.
        labels: (V,) etykiety wierzchołków.

    Returns:
        (T,) etykiety trójkątów.
    """
    lt = labels[np.asarray(mesh.triangles)]
    out = np.where((lt[:, 0] == lt[:, 1]) | (lt[:, 0] == lt[:, 2]), lt[:, 0], lt[:, 1])
    out[(out != lt[:, 0]) & (out != lt[:, 1]) & (out != lt[:, 2])] = 0
    agree = (lt == out[:, None]).sum(axis=1) >= 2
    return np.where(agree, out, 0)


def max_diameter(points: np.ndarray) -> float:
    """Największa odległość między dwoma punktami ("średnica" powierzchni) [mm].

    Liczona tylko na wierzchołkach otoczki wypukłej (najdalsza para zawsze na niej leży).
    Gdy wierzchołków otoczki jest ponad 3000, brana jest ich losowa próbka (stałe ziarno) -
    wynik może być wtedy minimalnie zaniżony.

    Args:
        points: (N, 3) punkty.

    Returns:
        Średnica; 0 dla mniej niż 2 punktów.
    """
    if len(points) < 2:
        return 0.0
    try:
        pts = points[ConvexHull(points).vertices] if len(points) > 4 else points
    except Exception:           # punkty współliniowe / współpłaszczyznowe
        pts = points
    if len(pts) > 3000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 3000, replace=False)]
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    return float(d.max())


def surface_stats(mesh: o3d.geometry.TriangleMesh, labels: np.ndarray) -> list[dict]:
    """Miary każdej powierzchni styku - do wstępnej selekcji par kości (podproblem 5).

    Środek i normalna są ważone polem wierzchołków, wymiary liczone wzdłuż osi
    głównych (PCA) punktów powierzchni. Jeśli siatka nie ma normalnych wierzchołków,
    są liczone (siatka jest zmieniana).

    Args:
        mesh:   siatka kości.
        labels: (V,) etykiety wierzchołków.

    Returns:
        Lista słowników {"id", "area_mm2", "centroid", "normal" (średnia ważona polem),
        "max_diameter_mm" (największa odległość dwóch punktów), "extent_mm" (długość,
        szerokość, grubość wzdłuż osi głównych), "vertices" (liczba)}.
    """
    v = np.asarray(mesh.vertices)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    n = np.asarray(mesh.vertex_normals)
    va = vertex_areas(mesh)
    out = []
    for k in [int(x) for x in np.unique(labels) if x > 0]:
        m = labels == k
        w = va[m]
        pts = v[m]
        c = (pts * w[:, None]).sum(axis=0) / max(w.sum(), 1e-12)
        nn = (n[m] * w[:, None]).sum(axis=0)
        nn = nn / max(np.linalg.norm(nn), 1e-12)
        if len(pts) >= 3:
            _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
            proj = (pts - c) @ vt.T
            extent = np.ptp(proj, axis=0)
        else:
            extent = np.zeros(3)
        out.append({"id": k, "area_mm2": round(float(w.sum()), 2), "centroid": np.round(c, 3).tolist(),
                     "normal": np.round(nn, 4).tolist(), "max_diameter_mm": round(max_diameter(pts), 2),
                     "extent_mm": np.round(extent, 2).tolist(), "vertices": int(m.sum())})
    return out


# -----------------
# ZAPIS / ODCZYT
# -----------------

def contacts_path(bone_id: str, source: str = "manual", root: Path | None = None) -> Path:
    """Ścieżka pliku etykiet kości.

    Args:
        bone_id: id kości.
        source:  "manual" -> <root>/<id>.json, cokolwiek innego (np. "auto")
                 -> <root>/auto/<id>.json.
        root:    folder oznaczeń (domyślnie CONTACTS_DIR z settings.json, np. inny w testach).

    Returns:
        Ścieżka pliku (plik nie musi istnieć).
    """
    if root is None:
        from config import CONTACTS_DIR as root
    root = Path(root)
    return root / f"{bone_id}.json" if source == "manual" else root / "auto" / f"{bone_id}.json"


def save_contacts(mesh: o3d.geometry.TriangleMesh, contacts: Contacts, path: Path | None = None) -> Path | None:
    """Zapisuje etykiety do JSON (format: docstring modułu) i zwraca ścieżkę.

    Ręczne bez żadnej powierzchni = plik jest usuwany (zwraca None). Wynik algorytmu jest
    zapisywany zawsze, także pusty - żeby nie liczyć go na nowo przy każdym uruchomieniu.
    Oprócz etykiet zapisuje statystyki każdej powierzchni (surface_stats) i jej punkty
    co POINTS_MM - po nich etykiety da się przenieść na zmienioną siatkę.

    Args:
        mesh:     siatka kości, do której należą etykiety.
        contacts: etykiety; contacts.source decyduje o domyślnym folderze.
        path:     plik docelowy (domyślnie contacts_path(...) wg contacts.source).

    Returns:
        Ścieżka zapisanego pliku albo None (usunięte puste ręczne oznaczenia).

    Raises:
        ValueError: liczba etykiet różni się od liczby wierzchołków siatki.
    """
    manual = contacts.source == "manual"
    path = Path(path or contacts_path(contacts.bone_id, "manual" if manual else "auto"))
    labels = np.asarray(contacts.labels, dtype=np.uint8)
    if len(labels) != len(mesh.vertices):
        raise ValueError(f"{contacts.bone_id}: {len(labels)} etykiet, a siatka ma {len(mesh.vertices)} wierzchołków")
    if manual and not labels.any():
        if path.exists():
            path.unlink()
        return None
    v = np.asarray(mesh.vertices)
    stats = {s["id"]: s for s in surface_stats(mesh, labels)}
    surfaces = []
    for k in sorted(stats):
        idx = np.flatnonzero(labels == k)
        pts = v[idx][voxel_keep(v[idx], POINTS_MM)]
        surfaces.append({"id": k, "name": contacts.name(k), **{a: b for a, b in stats[k].items() if a != "id"},
                         "points": np.round(pts, 2).tolist()})
    data = {"format": FORMAT, "bone": contacts.bone_id, "source": contacts.source,
            "updated": datetime.now().isoformat(timespec="seconds"),
            "mesh": {"vertices": len(v), "fingerprint": mesh_fingerprint(mesh)},
            **({"detector": contacts.detector} if contacts.detector else {}),
            "labels_zlib_b64": base64.b64encode(zlib.compress(labels.tobytes(), 9)).decode("ascii"),
            "surfaces": surfaces}
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=1)
    # listy liczb w jednej linii - plik jest dużo krótszy i czytelniejszy w gicie
    text = _compact_lists(text)
    path.write_text(text, encoding="utf-8")
    return path


def _compact_lists(text: str) -> str:
    """Skraca tekst JSON: listy liczb i listy punktów [[x, y, z], ...] zapisuje w jednej linii.

    Args:
        text: JSON z json.dumps(..., indent=...).

    Returns:
        Ten sam JSON (te same dane), krótszy.
    """
    import re

    text = re.sub(r"\[\s*([-0-9.,\s]+?)\s*\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", text)
    # lista punktów [[x, y, z], ...] też w jednej linii
    return re.sub(r"\[\s*(\[[^\[\]]*\](?:\s*,\s*\[[^\[\]]*\])*)\s*\]",
                  lambda m: "[" + re.sub(r"\]\s*,\s*\[", "], [", m.group(1)) + "]", text)


def load_contacts(bone_id: str, mesh: o3d.geometry.TriangleMesh, source: str = "manual",
                  path: Path | None = None, root: Path | None = None) -> Contacts | None:
    """Wczytuje zapisane etykiety kości: "manual" (ręczne) albo "auto" (wynik algorytmu).

    Jeśli siatka zmieniła się od zapisu (inny fingerprint), etykiety są przenoszone
    po położeniu punktów, a `remapped` = True: wierzchołek dostaje etykietę najbliższego
    zapisanego punktu, jeśli ten jest bliżej niż REMAP_MM (inaczej 0). Plik nie jest
    przy tym zmieniany - przeniesione etykiety trzeba zapisać samemu.

    Args:
        bone_id: id kości.
        mesh:    aktualna siatka kości (out/3d/<zestaw>/bones/<id>/mesh.ply).
        source:  "manual" albo "auto" - który plik czytać (gdy nie podano `path`).
        path:    konkretny plik zamiast contacts_path(bone_id, source, root).
        root:    folder oznaczeń zamiast CONTACTS_DIR.

    Returns:
        Contacts z etykietami (V,) dla `mesh` albo None, gdy pliku nie ma.
    """
    path = Path(path or contacts_path(bone_id, source, root))
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    n = len(mesh.vertices)
    labels = np.zeros(n, dtype=np.uint8)
    names = {int(s["id"]): s.get("name", "") for s in data.get("surfaces", [])}
    same = data.get("mesh", {}).get("fingerprint") == mesh_fingerprint(mesh)
    if same:
        labels = np.frombuffer(zlib.decompress(base64.b64decode(data["labels_zlib_b64"])), dtype=np.uint8).copy()
    else:
        pts, ids = [], []
        for s in data["surfaces"]:
            p = np.asarray(s.get("points", []), dtype=float).reshape(-1, 3)
            pts.append(p)
            ids.append(np.full(len(p), int(s["id"])))
        if pts and sum(len(p) for p in pts):
            d, idx = cKDTree(np.vstack(pts)).query(np.asarray(mesh.vertices), distance_upper_bound=REMAP_MM)
            ok = np.isfinite(d)
            labels[ok] = np.concatenate(ids)[idx[ok]]
    return Contacts(bone_id=bone_id, labels=labels, names=names, source=data.get("source", source),
                    updated=data.get("updated", ""), remapped=not same, detector=data.get("detector", {}))


# -----------------
# ALGORYTMY WYKRYWANIA
# -----------------

DETECTORS: dict[str, Callable] = {}
DATA_KEYS: dict[str, Callable] = {}     # algorytm -> funkcja (bone, params) -> tekst (patrz detector)


def detector(name: str, data_key: Callable | None = None):
    """Dekorator rejestrujący algorytm wykrywania w DETECTORS pod nazwą `name`.

    Funkcja dostaje (bone: Bone, params: ContactParams) i zwraca labels (V,) dla
    wierzchołków bone.mesh. Przykład: docstring modułu, "Jak dodać własny algorytm".

    Args:
        name:     nazwa algorytmu (argument --method, klucz "method" w settings.json).
        data_key: dla algorytmów, których wynik zależy od czegoś poza sekcją "contacts" (np. od
                  ustawień innej sekcji): (bone, params) -> krótki tekst, który się zmienia,
                  gdy te dane się zmienią. Zapisywany w wyniku ("detector" -> "data").

    Returns:
        Dekorator, który rejestruje funkcję i zwraca ją bez zmian.
    """
    def wrap(fn):
        """Rejestruje `fn` w DETECTORS pod nazwą `name` i zwraca ją bez zmian."""
        DETECTORS[name] = fn
        if data_key is not None:
            DATA_KEYS[name] = data_key
        return fn
    return wrap


def split_regions(mesh: o3d.geometry.TriangleMesh, mask: np.ndarray, min_area: float) -> np.ndarray:
    """Maska wierzchołków -> etykiety 1..K spójnych obszarów (od największego), małe pominięte.

    Obszar = wierzchołki z maski połączone krawędziami siatki. Przydatne na końcu
    każdego algorytmu: zamienia "te wierzchołki wyglądają na przełom" na osobne powierzchnie.

    Args:
        mesh:     siatka kości.
        mask:     (V,) bool.
        min_area: obszary o mniejszym polu [mm²] dostają 0.

    Returns:
        (V,) uint8 etykiety; najwyżej 255 obszarów (największych).
    """
    adj = vertex_adjacency(mesh)
    idx = np.flatnonzero(mask)
    labels = np.zeros(len(mask), dtype=np.uint8)
    if len(idx) == 0:
        return labels
    sub = adj[idx][:, idx]
    _, comp = connected_components(sub, directed=False)
    va = vertex_areas(mesh)[idx]
    areas = np.bincount(comp, weights=va)
    order = [c for c in np.argsort(-areas) if areas[c] >= min_area][:255]
    for k, c in enumerate(order, start=1):
        labels[idx[comp == c]] = k
    return labels


def local_shape(mesh: o3d.geometry.TriangleMesh, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Lokalne miary kształtu dla każdego wierzchołka (sąsiedzi w promieniu `radius`).

    Sąsiedzi to punkty równomiernej próbki powierzchni (co radius / 3), więc wynik nie
    zależy od gęstości siatki. Wierzchołki z mniej niż 6 sąsiadami dostają 0.

    Args:
        mesh:   siatka kości.
        radius: promień sąsiedztwa [mm].

    Returns:
        (szorstkość: odchylenie sąsiadów od dopasowanej płaszczyzny / promień,
         rozrzut normalnych: 1 - |średnia normalnych sąsiadów|, 0 = płasko, rośnie na krawędziach)
    """
    from data_parsing import sample_surface

    v = np.asarray(mesh.vertices)
    s = sample_surface(mesh, radius / 3)            # sąsiedzi z równomiernej próbki (~30 na wierzchołek)
    p, n = s.points, s.normals
    nb = cKDTree(p).query_ball_point(v, radius, workers=-1)
    counts = np.fromiter((len(x) for x in nb), dtype=np.int64, count=len(v))
    rows = np.repeat(np.arange(len(v)), counts)
    cols = np.fromiter((j for x in nb for j in x), dtype=np.int64, count=counts.sum())
    W = coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(v), len(p))).tocsr()
    cnt = np.maximum(counts, 1)[:, None].astype(float)
    mean = (W @ p) / cnt
    outer = np.einsum("ni,nj->nij", p, p).reshape(len(p), 9)
    cov = ((W @ outer) / cnt).reshape(-1, 3, 3) - np.einsum("ni,nj->nij", mean, mean)
    smallest = np.linalg.eigvalsh(cov)[:, 0]
    rough = np.sqrt(np.clip(smallest, 0, None)) / radius
    spread = 1 - np.linalg.norm((W @ n) / cnt, axis=1)
    rough[counts < 6] = 0
    spread[counts < 6] = 0
    return rough, spread


def _rings(mesh: o3d.geometry.TriangleMesh, mm: float) -> int:
    """Ile pierścieni sąsiadów siatki odpowiada odległości `mm` (wg średniej długości krawędzi).

    Średnia jest liczona z jednej krawędzi każdego trójkąta (v0-v1) - to wystarcza.

    Args:
        mesh: siatka kości.
        mm:   odległość po powierzchni [mm].

    Returns:
        Liczba pierścieni, co najmniej 1.
    """
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    edge = np.linalg.norm(v[t[:, 0]] - v[t[:, 1]], axis=1).mean()
    return max(1, int(round(mm / edge)))


def morph(mesh: o3d.geometry.TriangleMesh, mask: np.ndarray, mm: float, grow: bool) -> np.ndarray:
    """Dylatacja (grow=True) albo erozja (grow=False) maski wierzchołków o ~`mm` po siatce.

    Jeden krok = jeden pierścień sąsiadów (krawędzie siatki); liczba kroków z _rings.

    Args:
        mesh: siatka kości.
        mask: (V,) bool - nie jest zmieniana.
        mm:   o ile poszerzyć / zwęzić [mm].
        grow: True = poszerz, False = zwęź.

    Returns:
        Nowa maska (V,) bool.
    """
    adj = vertex_adjacency(mesh)
    m = mask.copy()
    for _ in range(_rings(mesh, mm)):
        hits = adj @ (m if grow else ~m).astype(np.int8)
        m = (m | (hits > 0)) if grow else (m & (hits == 0))
    return m


@detector("roughness")
def detect_roughness(bone, params: ContactParams) -> np.ndarray:
    """ALGORYTM BAZOWY (punkt wyjścia, nie rozwiązanie): przełom jest szorstki, kora kości gładka.

    1. Szorstkość każdego wierzchołka: odchylenie powierzchni od płaszczyzny w promieniu
       roughness_radius_mm, względem promienia (kora ~0.04, przełom gąbczasty ~0.15).
    2. Szorstkie = powyżej rough_threshold, z pominięciem pasa ~roughness_radius_mm
       wzdłuż otwartego brzegu skanu (brzeg jest postrzępiony, ale to nie przełom).
    3. Otwarcie morfologiczne (erozja, potem dylatacja o ~0.6 x promień): znikają cienkie
       linie ostrych krawędzi, zostają szerokie ściany przełomu.
    4. Spójne obszary >= min_surface_mm2 = wykryte powierzchnie (od największej).
    Słabo działa na gładkich, świeżych przełomach kości zbitej - tu potrzebny lepszy algorytm.

    Args:
        bone:   kość (wystarczy bone.mesh).
        params: roughness_radius_mm, rough_threshold, min_surface_mm2.

    Returns:
        (V,) uint8 etykiety wierzchołków bone.mesh.
    """
    mesh = bone.mesh
    rough, _ = local_shape(mesh, params.roughness_radius_mm)
    mask = rough > params.rough_threshold
    mask &= ~morph(mesh, boundary_vertices(mesh), params.roughness_radius_mm, grow=True)   # brzeg skanu
    r = 0.6 * params.roughness_radius_mm
    mask = morph(mesh, morph(mesh, mask, r, grow=False), r, grow=True)
    return split_regions(mesh, mask, params.min_surface_mm2)


def _features_data_key(bone, params: ContactParams) -> str:
    """Ustawienia cech (settings.json, sekcja "surface_features") - od nich zależy wynik "features"."""
    import config

    return json.dumps(asdict(config.surface_feature_params()), sort_keys=True)


def bone_scores(features: dict[str, np.ndarray], areas: np.ndarray, keys) -> np.ndarray:
    """Wynik "jak bardzo to przełom" każdego wierzchołka - bez żadnych oznaczeń.

    Każda cecha jest porównywana z typową wartością na TEJ SAMEJ kości: większość kości to
    kora, więc mediana cechy (ważona polem) to "poziom kory". Cecha (log) jest zamieniana na
    liczbę odchyleń od mediany: (log x - mediana) / (1.4826 * MAD) - odporne na przełom
    i niezależne od skanera (inna gęstość siatki, inna skala szorstkości = te same liczby).
    Cechy mniejsze na przełomie (np. długość krawędzi siatki) dostają znak minus
    (feature_info "sign"). Wynik = średnia po cechach: ~0 na typowej korze, > 0 na przełomie.

    Args:
        features: wynik surface_features.vertex_features.
        areas:    (V,) pola wierzchołków (wagi mediany).
        keys:     które cechy połączyć.

    Returns:
        (V,) wynik.

    Raises:
        ValueError: nieznana cecha albo cecha bez stałego kierunku (np. jasność koloru).
    """
    import surface_features as sf

    info = sf.feature_info()
    out = np.zeros(len(areas))
    for k in keys:
        if k not in features:
            raise ValueError(f"Nieznana cecha {k!r}, dostępne: {', '.join(features)}")
        sign = info.get(k, {}).get("sign", 0)
        if sign == 0:
            raise ValueError(f"Cecha {k!r} nie ma stałego kierunku (na różnych kościach raz większa, raz mniejsza "
                             f"na przełomie) - nie nadaje się do \"features\"")
        x = np.log(features[k] + 1e-3)
        med = sf.weighted_quantiles(x, areas, [0.5])[0]
        mad = 1.4826 * sf.weighted_quantiles(np.abs(x - med), areas, [0.5])[0]
        out += sign * (x - med) / max(mad, 1e-9)
    return out / max(len(keys), 1)


@detector("features", data_key=_features_data_key)
def detect_features(bone, params: ContactParams) -> np.ndarray:
    """KILKA CECH NARAZ, BEZ OZNACZEŃ: szorstkość, rozrzut normalnych i gęstość siatki.

    1. Cechy każdego wierzchołka (src/surface_features.py; domyślnie rozrzut normalnych 2 mm,
       szorstkość 1 mm, długość krawędzi siatki - lista: feature_keys, promienie:
       settings.json -> "surface_features"; opis: raporty/przelom_vs_kora.md).
    2. Każda cecha względem typowej wartości na tej samej kości (bone_scores): przełom jest
       bardziej szorstki, ma większy rozrzut normalnych i gęstszą siatkę niż reszta kości.
       Wynik = średnia tych odchyleń.
    3. Uśrednienie wyniku w promieniu feature_smooth_mm (pojedyncze szorstkie punkty kory
       i gładkie punkty przełomu znikają) i próg feature_threshold.
    4. Bez pasu przy otwartym brzegu skanu, otwarcie morfologiczne o feature_open_mm (znikają
       cienkie linie krawędzi i wąskie mostki - bez tego np. oba końce długiej kości potrafią
       połączyć się w jeden obszar), spójne obszary >= min_surface_mm2 (od największego).
    Założenie: większość zeskanowanej powierzchni to kora (mediana = kora). Na kości, która
    jest prawie cała przełomem, algorytm zaznaczy za mało.

    Args:
        bone:   kość.
        params: feature_keys, feature_threshold, feature_smooth_mm, feature_open_mm, min_surface_mm2.

    Returns:
        (V,) uint8 etykiety wierzchołków bone.mesh.

    Raises:
        ValueError: zła nazwa cechy w feature_keys.
    """
    import config
    import surface_features as sf

    sfp = config.surface_feature_params()
    mesh = bone.mesh
    v, va = np.asarray(mesh.vertices), vertex_areas(mesh)
    score = bone_scores(sf.cached_features(bone, sfp), va, params.feature_keys)
    if params.feature_smooth_mm > 0:
        W, _ = sf._neighbours(v, v, params.feature_smooth_mm)
        score = (W @ (score * va)) / np.maximum(W @ va, 1e-12)
    mask = score > params.feature_threshold
    mask &= sf._distance_to(v, boundary_vertices(mesh)) > sfp.boundary_margin_mm
    if params.feature_open_mm > 0:
        r = params.feature_open_mm
        mask = morph(mesh, morph(mesh, mask, r, grow=False), r, grow=True)
    return split_regions(mesh, mask, params.min_surface_mm2)


def boundary_vertices(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Maska wierzchołków na otwartym brzegu siatki (krawędź tylko jednego trójkąta).

    Skan ma brzeg tam, gdzie skaner przestał widzieć powierzchnię - to NIE jest przełom,
    a brzeg bywa postrzępiony, więc wygląda na szorstki.

    Args:
        mesh: siatka kości.

    Returns:
        (V,) bool; same False dla siatki zamkniętej.
    """
    t = np.asarray(mesh.triangles)
    e = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    uniq, count = np.unique(e, axis=0, return_counts=True)
    mask = np.zeros(len(mesh.vertices), dtype=bool)
    mask[uniq[count == 1].ravel()] = True
    return mask


def detection_params(params: ContactParams) -> dict:
    """Ustawienia wpływające na wynik algorytmu (bez ustawień narzędzia i nazwy algorytmu).

    Zapisywane w pliku wyniku ("detector" -> "params"); auto_contacts porównuje je
    z bieżącymi, żeby wiedzieć, czy zapisany wynik jest aktualny.

    Args:
        params: wszystkie ustawienia sekcji "contacts".

    Returns:
        {nazwa: wartość} bez method, brush_mm i wand_angle_deg (krotki jako listy - jak po odczycie JSON).
    """
    return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(params).items()
            if k not in ("method", "brush_mm", "wand_angle_deg")}


def detector_info(bone, method: str, params: ContactParams) -> dict:
    """Opis algorytmu zapisywany z wynikiem ("detector"): nazwa, ustawienia i - jeśli algorytm
    ma data_key - dane spoza sekcji "contacts", od których zależy.

    Args:
        bone:   kość.
        method: nazwa algorytmu.
        params: ustawienia.

    Returns:
        {"method", "params"[, "data"]}.
    """
    info = {"method": method, "params": detection_params(params)}
    if method in DATA_KEYS:
        info["data"] = DATA_KEYS[method](bone, params)
    return info


def detect_contacts(bone, method: str | None = None, params: ContactParams = ContactParams()) -> Contacts:
    """Uruchamia algorytm `method` na kości. Nic nie zapisuje (zapis: save_contacts).

    Args:
        bone:   kość.
        method: nazwa z DETECTORS; None = params.method.
        params: ustawienia algorytmu.

    Returns:
        Contacts z source = "auto:<method>", nazwami "Wykryta k" i opisem algorytmu
        w `detector` (żeby auto_contacts mógł sprawdzić aktualność).

    Raises:
        ValueError: nieznany algorytm.
    """
    method = method or params.method
    if method not in DETECTORS:
        raise ValueError(f"Nieznany algorytm {method!r}, dostępne: {', '.join(DETECTORS)}")
    labels = np.asarray(DETECTORS[method](bone, params), dtype=np.uint8)
    names = {int(k): f"Wykryta {k}" for k in np.unique(labels) if k > 0}
    return Contacts(bone_id=bone.id, labels=labels, names=names, source=f"auto:{method}",
                    detector=detector_info(bone, method, params))


def auto_contacts(bone, params: ContactParams = ContactParams(),
                  root: Path | None = None) -> tuple[Contacts, bool]:
    """Wynik algorytmu params.method dla kości: z cache (contacts/<zestaw>/auto/) albo policzony i zapisany.

    Liczy na nowo, gdy pliku nie ma albo powstał innym algorytmem, z innymi ustawieniami,
    na innej siatce kości lub (algorytmy z data_key, np. "features") z innymi danymi spoza sekcji.

    Args:
        bone:   kość.
        params: algorytm (params.method) i jego ustawienia.
        root:   folder oznaczeń zamiast CONTACTS_DIR (wynik w <root>/auto/).

    Returns:
        (etykiety - puste, gdy algorytm nic nie znalazł; True = właśnie policzone, False = z cache).
    """
    want = detector_info(bone, params.method, params)
    cached = load_contacts(bone.id, bone.mesh, "auto", root=root)
    if cached is not None and not cached.remapped and cached.detector == want:
        return cached, False
    found = detect_contacts(bone, params.method, params)
    save_contacts(bone.mesh, found, contacts_path(bone.id, "auto", root))
    return found, True


# -----------------
# OCENA ALGORYTMU WZGLĘDEM RĘCZNYCH OZNACZEŃ
# -----------------

def compare(mesh: o3d.geometry.TriangleMesh, predicted: np.ndarray, truth: np.ndarray) -> dict:
    """Porównuje wynik algorytmu z ręcznymi oznaczeniami (ważone polem wierzchołków).

    IoU = część wspólna / suma, precyzja = jaka część wykrytego to naprawdę styk,
    czułość = jaka część prawdziwego styku została wykryta.

    Args:
        mesh:      siatka kości.
        predicted: (V,) etykiety z algorytmu.
        truth:     (V,) etykiety ręczne.

    Returns:
        {"iou", "precision", "recall"} dla podziału styk / nie-styk oraz "surfaces":
        dla każdej ręcznej powierzchni najlepsze IoU z którąkolwiek wykrytą.
    """
    va = vertex_areas(mesh)
    p, t = predicted > 0, truth > 0
    inter, union = va[p & t].sum(), va[p | t].sum()
    out = {"iou": inter / union if union else 1.0,
           "precision": inter / va[p].sum() if p.any() else 0.0,
           "recall": inter / va[t].sum() if t.any() else 1.0, "surfaces": []}
    for k in [int(x) for x in np.unique(truth) if x > 0]:
        tk = truth == k
        best = 0.0
        for j in [int(x) for x in np.unique(predicted[tk]) if x > 0]:
            pj = predicted == j
            best = max(best, va[pj & tk].sum() / va[pj | tk].sum())
        out["surfaces"].append({"id": k, "best_iou": best, "area_mm2": float(va[tk].sum())})
    return out

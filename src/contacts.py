"""
Powierzchnie styku (przełomy) na kościach: format zapisu, statystyki, algorytmy wykrywania.

Etykiety są przypisane do WIERZCHOŁKÓW siatki kości (out/3d/bones/<id>/mesh.ply):
    labels[i] = 0  -> wierzchołek nie należy do żadnej powierzchni styku,
    labels[i] = k  -> wierzchołek należy do powierzchni styku nr k (1, 2, ...).
Jedna kość może mieć kilka powierzchni styku (np. przełom na każdym końcu).

Skąd się biorą etykiety:
    - RĘCZNIE: scripts/mark_contacts.py (przeglądarka) -> contacts/<id_kości>.json
      Folder contacts/ jest w repozytorium - to ręczna praca, warto ją commitować.
    - ALGORYTM: scripts/detect_contacts.py -> out/3d/bones/<id>/contacts_auto.json
      Ten sam format, pole "source" = "auto:<nazwa algorytmu>".

Jak dodać własny algorytm:
    @detector("moja_metoda")
    def moja_metoda(bone: Bone, params: ContactParams) -> np.ndarray:
        ...  # zwróć labels (V,) jak wyżej
        return labels
    potem: python scripts/detect_contacts.py --method moja_metoda
    Skrypt porówna wynik z ręcznymi oznaczeniami (IoU, precyzja, czułość), jeśli są.

Format pliku JSON:
    {
      "format": "bones-smm-contacts/1",
      "bone": "<id kości>", "source": "manual" | "auto:<metoda>", "updated": "<data>",
      "mesh": {"vertices": V, "fingerprint": "..."},    # do wykrycia zmiany siatki
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
from dataclasses import dataclass, field
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
    brush_mm: float = 2.0             # startowy promień pędzla w narzędziu [mm]
    wand_angle_deg: float = 25.0      # startowa tolerancja kąta różdżki [stopnie]
    roughness_radius_mm: float = 1.0  # algorytm bazowy: promień liczenia szorstkości
    rough_threshold: float = 0.09     # algorytm bazowy: szorstkość, od której powierzchnia = przełom
    min_surface_mm2: float = 15.0     # algorytm bazowy: mniejsze obszary są pomijane


@dataclass
class Contacts:
    """Etykiety powierzchni styku jednej kości."""

    bone_id: str
    labels: np.ndarray                          # (V,) uint8
    names: dict[int, str] = field(default_factory=dict)
    source: str = "manual"
    updated: str = ""
    remapped: bool = False                      # True = przeniesione ze starej wersji siatki

    @property
    def surface_ids(self) -> list[int]:
        return [int(k) for k in np.unique(self.labels) if k > 0]

    def name(self, k: int) -> str:
        return self.names.get(k) or f"Powierzchnia {k}"


def surface_color(k: int) -> str:
    return PALETTE[(k - 1) % len(PALETTE)]


# -----------------
# GEOMETRIA
# -----------------

def mesh_fingerprint(mesh: o3d.geometry.TriangleMesh) -> str:
    v = np.round(np.asarray(mesh.vertices), 3).astype(np.float32)
    return hashlib.sha1(v.tobytes()).hexdigest()[:16]


def vertex_areas(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Pole przypadające na wierzchołek = 1/3 pól sąsiednich trójkątów [mm^2]."""
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    a = 0.5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
    return np.bincount(t.ravel(), weights=np.repeat(a / 3, 3), minlength=len(v))


def vertex_adjacency(mesh: o3d.geometry.TriangleMesh):
    """Macierz sąsiedztwa wierzchołków (krawędzie siatki), scipy.sparse CSR."""
    t = np.asarray(mesh.triangles)
    n = len(mesh.vertices)
    i = np.concatenate([t[:, 0], t[:, 1], t[:, 2], t[:, 1], t[:, 2], t[:, 0]])
    j = np.concatenate([t[:, 1], t[:, 2], t[:, 0], t[:, 0], t[:, 1], t[:, 2]])
    return coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n, n)).tocsr()


def triangle_labels(mesh: o3d.geometry.TriangleMesh, labels: np.ndarray) -> np.ndarray:
    """Etykieta trójkąta: ta, którą mają co najmniej 2 z 3 wierzchołków (inaczej 0)."""
    lt = labels[np.asarray(mesh.triangles)]
    out = np.where((lt[:, 0] == lt[:, 1]) | (lt[:, 0] == lt[:, 2]), lt[:, 0], lt[:, 1])
    out[(out != lt[:, 0]) & (out != lt[:, 1]) & (out != lt[:, 2])] = 0
    agree = (lt == out[:, None]).sum(axis=1) >= 2
    return np.where(agree, out, 0)


def max_diameter(points: np.ndarray) -> float:
    """Największa odległość między dwoma punktami ("średnica" powierzchni) [mm]."""
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

def contacts_path(bone_id: str, source: str = "manual") -> Path:
    """manual -> contacts/<id>.json, auto -> out/3d/bones/<id>/contacts_auto.json."""
    from config import CONTACTS_DIR, DATA_OUT

    if source == "manual":
        return CONTACTS_DIR / f"{bone_id}.json"
    return DATA_OUT / "bones" / bone_id / "contacts_auto.json"


def save_contacts(mesh: o3d.geometry.TriangleMesh, contacts: Contacts, path: Path | None = None) -> Path | None:
    """Zapisuje etykiety do JSON. Brak oznaczeń = plik jest usuwany. Zwraca ścieżkę (albo None)."""
    path = Path(path or contacts_path(contacts.bone_id, "manual" if contacts.source == "manual" else "auto"))
    labels = np.asarray(contacts.labels, dtype=np.uint8)
    if len(labels) != len(mesh.vertices):
        raise ValueError(f"{contacts.bone_id}: {len(labels)} etykiet, a siatka ma {len(mesh.vertices)} wierzchołków")
    if not labels.any():
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
            "labels_zlib_b64": base64.b64encode(zlib.compress(labels.tobytes(), 9)).decode("ascii"),
            "surfaces": surfaces}
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=1)
    # listy liczb w jednej linii - plik jest dużo krótszy i czytelniejszy w gicie
    text = _compact_lists(text)
    path.write_text(text, encoding="utf-8")
    return path


def _compact_lists(text: str) -> str:
    import re

    text = re.sub(r"\[\s*([-0-9.,\s]+?)\s*\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", text)
    # lista punktów [[x, y, z], ...] też w jednej linii
    return re.sub(r"\[\s*(\[[^\[\]]*\](?:\s*,\s*\[[^\[\]]*\])*)\s*\]",
                  lambda m: "[" + re.sub(r"\]\s*,\s*\[", "], [", m.group(1)) + "]", text)


def load_contacts(bone_id: str, mesh: o3d.geometry.TriangleMesh, source: str = "manual",
                  path: Path | None = None) -> Contacts | None:
    """Wczytuje etykiety kości (None = brak pliku).

    Jeśli siatka zmieniła się od zapisu (inny fingerprint), etykiety są przenoszone
    po położeniu punktów, a `remapped` = True.
    """
    path = Path(path or contacts_path(bone_id, source))
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
                    updated=data.get("updated", ""), remapped=not same)


# -----------------
# ALGORYTMY WYKRYWANIA
# -----------------

DETECTORS: dict[str, Callable] = {}


def detector(name: str):
    """Dekorator rejestrujący algorytm: funkcja(bone, params) -> labels (V,)."""
    def wrap(fn):
        DETECTORS[name] = fn
        return fn
    return wrap


def split_regions(mesh: o3d.geometry.TriangleMesh, mask: np.ndarray, min_area: float) -> np.ndarray:
    """Maska wierzchołków -> etykiety 1..K spójnych obszarów (od największego), małe pominięte."""
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
    """Ile pierścieni sąsiadów siatki odpowiada odległości `mm` (wg średniej długości krawędzi)."""
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    edge = np.linalg.norm(v[t[:, 0]] - v[t[:, 1]], axis=1).mean()
    return max(1, int(round(mm / edge)))


def morph(mesh: o3d.geometry.TriangleMesh, mask: np.ndarray, mm: float, grow: bool) -> np.ndarray:
    """Dylatacja (grow=True) albo erozja maski wierzchołków o ~`mm` po siatce."""
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
    2. Szorstkie = powyżej rough_threshold.
    3. Otwarcie morfologiczne (erozja, potem dylatacja o ~0.6 x promień): znikają cienkie
       linie ostrych krawędzi, zostają szerokie ściany przełomu.
    4. Spójne obszary >= min_surface_mm2 = wykryte powierzchnie (od największej).
    Słabo działa na gładkich, świeżych przełomach kości zbitej - tu potrzebny lepszy algorytm.
    """
    mesh = bone.mesh
    rough, _ = local_shape(mesh, params.roughness_radius_mm)
    mask = rough > params.rough_threshold
    mask &= ~morph(mesh, boundary_vertices(mesh), params.roughness_radius_mm, grow=True)   # brzeg skanu
    r = 0.6 * params.roughness_radius_mm
    mask = morph(mesh, morph(mesh, mask, r, grow=False), r, grow=True)
    return split_regions(mesh, mask, params.min_surface_mm2)


def boundary_vertices(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Maska wierzchołków na otwartym brzegu siatki (krawędź tylko jednego trójkąta).

    Skan ma brzeg tam, gdzie skaner przestał widzieć powierzchnię - to NIE jest przełom,
    a brzeg bywa postrzępiony, więc wygląda na szorstki.
    """
    t = np.asarray(mesh.triangles)
    e = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    uniq, count = np.unique(e, axis=0, return_counts=True)
    mask = np.zeros(len(mesh.vertices), dtype=bool)
    mask[uniq[count == 1].ravel()] = True
    return mask


def detect_contacts(bone, method: str | None = None, params: ContactParams = ContactParams()) -> Contacts:
    """Uruchamia algorytm `method` (z DETECTORS; None = pierwszy, czyli "edges") na kości."""
    method = method or next(iter(DETECTORS))
    if method not in DETECTORS:
        raise ValueError(f"Nieznany algorytm {method!r}, dostępne: {', '.join(DETECTORS)}")
    labels = np.asarray(DETECTORS[method](bone, params), dtype=np.uint8)
    names = {k: f"Wykryta {k}" for k in np.unique(labels) if k > 0}
    return Contacts(bone_id=bone.id, labels=labels, names=names, source=f"auto:{method}")


# -----------------
# OCENA ALGORYTMU WZGLĘDEM RĘCZNYCH OZNACZEŃ
# -----------------

def compare(mesh: o3d.geometry.TriangleMesh, predicted: np.ndarray, truth: np.ndarray) -> dict:
    """Porównuje wynik algorytmu z ręcznymi oznaczeniami (ważone polem wierzchołków).

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

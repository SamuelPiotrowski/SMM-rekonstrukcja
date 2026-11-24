"""
Dopasowywanie fragmentów kości po POWIERZCHNIACH STYKU (podproblem 6 z README.md).

Bierze tylko punkty powierzchni styku (przełomów), każdą powierzchnię osobno (kość może
mieć kilka), i dopasowuje każdą do każdej powierzchni INNEJ kości.

SKĄD POWIERZCHNIE (SurfaceMatchParams.source, settings.json "matching" -> "source"):
    "manual"      - tylko ręczne oznaczenia contacts/<id>.json; kości bez nich są pomijane,
    "auto"        - tylko wynik algorytmu (settings.json "contacts" -> "method" i jego ustawienia):
                    z contacts/auto/<id>.json, a gdy pliku brak albo jest nieaktualny (inny
                    algorytm, inne ustawienia, zmieniona siatka) - liczony i zapisywany
                    (contacts.auto_contacts),
    "manual-auto" - ręczne, a dla kości bez ręcznych - z algorytmu jak w "auto".
    Powierzchnie mniejsze niż `min_area_mm2` są pomijane.

DLA KAŻDEJ PARY POWIERZCHNI (A z jednej kości, B z innej):
1. PRÓBKOWANIE: równomierne punkty co `sample_mm` z normalnymi (na zewnątrz kości).
2. ODWRÓCENIE NORMALNYCH A. Dwa przełomy stykają się "twarzą w twarz": wypukłość na A
   to wklęsłość na B, a ich normalne są przeciwne. Po odwróceniu normalnych A wygląda
   dla deskryptorów jak B, więc działa ta sama rejestracja co przy łączeniu skanów.
3. ZGRUBNE USTAWIENIE - RANSAC na deskryptorach FPFH (punkty co `coarse_voxel_mm`,
   promień `fpfh_radius_mm`): losuje trójki podobnych punktów, odrzuca ustawienia
   z normalnymi różnymi o więcej niż `ransac_normal_angle_deg`, zostawia to, z którym
   zgadza się najwięcej punktów. Zawsze wszystkie `ransac_iterations` (open3d liczy
   wielowątkowo, więc wynik jest losowy - mało iteracji = często złe ustawienie).
4. DOKŁADNE USTAWIENIE - ICP point-to-plane: pary punktów szukane do `icp_start_mm`,
   potem do połowy tego, na koniec do 2 x `sample_mm`.
   Kroki 3-4 są powtarzane `ransac_seeds` razy, wygrywa ustawienie z najlepszym wynikiem.
5. OCENA ustawienia T (A -> B):
   - styk [mm²]:    punkty A bliżej niż `tolerance_mm` od B, z normalnymi przeciwnymi
                    z dokładnością do `normal_angle_deg` (i odwrotnie, B na A);
                    pole styku = mniejsze z pól obu stron (część pokryta x pole powierzchni),
   - RMSE [mm]:     średnia odległość punktów A w styku od B,
   - kolizja [mm²]: pole CAŁEJ kości A, które leży pod powierzchnią kości B głębiej niż
                    `collision_mm` (sprawdzane do `collision_search_mm` od B), i odwrotnie -
                    prawdziwe dopasowanie przełomów nie może wbijać jednej kości w drugą,
   - WYNIK = styk - kolizja [mm²]; pary są sortowane od najlepszego wyniku.

Użycie: scripts/match_surfaces.py (wyniki: out/3d/matching.json, out/3d/matching.html).
W kodzie:
    surfaces, info = extract_surfaces(load_bones(features=True), params, contact_params)
    results = match_all(surfaces, params)     # [{"a", "b", "score", "contact_mm2", "T", ...}, ...]
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from bones import Bone
from contacts import ContactParams, Contacts, auto_contacts, load_contacts, triangle_labels
from data_parsing import compute_fpfh, sample_surface, submesh, to_cloud, voxel_keep
from merging import covered, transform_points

log = logging.getLogger(__name__)
reg = o3d.pipelines.registration

SOURCES = ("manual", "auto", "manual-auto")


@dataclass(frozen=True)
class SurfaceMatchParams:
    """Ustawienia dopasowania - opisy w settings.json, sekcja "matching"."""

    source: str = "manual"                 # "manual" / "auto" / "manual-auto"
    min_area_mm2: float = 20.0             # mniejsze powierzchnie styku są pomijane [mm²]
    sample_mm: float = 0.3                 # odstęp punktów powierzchni [mm]
    coarse_voxel_mm: float = 1.0           # odstęp punktów zgrubnego ustawienia (RANSAC) [mm]
    fpfh_radius_mm: float = 4.0            # promień deskryptora FPFH [mm]
    ransac_iterations: int = 1000000       # zawsze wszystkie (bez wczesnego końca)
    ransac_seeds: int = 2                  # niezależnych prób RANSAC + ICP na parę
    ransac_normal_angle_deg: float = 35.0  # RANSAC: maks. różnica normalnych par punktów [stopnie]
    icp_start_mm: float = 2.0              # ICP: pierwszy promień szukania par punktów [mm]
    tolerance_mm: float = 0.5              # styk: punkty bliżej niż to [mm] ...
    normal_angle_deg: float = 45.0         # ... z normalnymi przeciwnymi z tą dokładnością [stopnie]
    collision_mm: float = 1.0              # kolizja: głębiej niż to pod powierzchnią drugiej kości [mm]
    collision_search_mm: float = 4.0       # ... sprawdzane do tej odległości od drugiej kości [mm]
    max_shown: int = 15                    # ile najlepszych par pokazać w 3D


# -----------------
# POWIERZCHNIE
# -----------------

@dataclass
class Surface:
    """Jedna powierzchnia styku jednej kości, w układzie tej kości (jak out/3d/bones/<id>/mesh.ply)."""

    bone: Bone
    contacts: Contacts
    sid: int                       # numer powierzchni w etykietach kości
    points: np.ndarray             # (N, 3)
    normals: np.ndarray            # (N, 3), na zewnątrz kości
    area_mm2: float
    bone_tree: cKDTree | None = None   # punkty CAŁEJ kości (do kolizji), wspólne dla powierzchni kości
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def key(self) -> str:
        return f"{self.bone.id}#{self.sid}"

    @property
    def name(self) -> str:
        return f"{self.bone.label} · {self.contacts.name(self.sid)}"

    @property
    def is_manual(self) -> bool:
        return self.contacts.source == "manual"

    def clouds(self, p: SurfaceMatchParams, flip: bool):
        """(gęsta, 2x rzadsza, zgrubna, FPFH zgrubnej); flip = normalne odwrócone (strona "A")."""
        key = ("clouds", flip)
        if key not in self._cache:
            nrm = -self.normals if flip else self.normals
            mid = voxel_keep(self.points, 2 * p.sample_mm)
            coarse_idx = voxel_keep(self.points, p.coarse_voxel_mm)
            coarse = to_cloud(self.points[coarse_idx], nrm[coarse_idx])
            self._cache[key] = (to_cloud(self.points, nrm), to_cloud(self.points[mid], nrm[mid]), coarse,
                                compute_fpfh(coarse, p.fpfh_radius_mm))
        return self._cache[key]

    def tree(self) -> cKDTree:
        if "tree" not in self._cache:
            self._cache["tree"] = cKDTree(self.points)
        return self._cache["tree"]


def bone_contacts(bone: Bone, source: str, contact_params: ContactParams = ContactParams(),
                  root: Path | None = None) -> tuple[Contacts | None, str]:
    """Powierzchnie styku kości wg trybu `source` (opis trybów: docstring modułu).

    Args:
        root: folder oznaczeń zamiast contacts/ z settings.json (np. w testach).

    Returns:
        (etykiety albo None, skąd: "ręczne" / "algorytm (cache)" / "algorytm (policzone)" / "brak ...").
    """
    if source not in SOURCES:
        raise ValueError(f"matching.source = {source!r}, dostępne: {', '.join(SOURCES)}")
    if source in ("manual", "manual-auto"):
        c = load_contacts(bone.id, bone.mesh, "manual", root=root)
        if c is not None and c.labels.any():
            return c, "ręczne"
        if source == "manual":
            return None, "brak ręcznych oznaczeń"
    c, computed = auto_contacts(bone, contact_params, root)
    if not c.labels.any():
        return None, "brak (algorytm nic nie znalazł)"
    return c, "algorytm (policzone)" if computed else "algorytm (cache)"


def extract_surfaces(bones: list[Bone], p: SurfaceMatchParams, contact_params: ContactParams = ContactParams(),
                     root: Path | None = None) -> tuple[list[Surface], dict[str, str]]:
    """Powierzchnie styku wszystkich kości (każda osobno), bez mniejszych niż p.min_area_mm2.

    Kości muszą być wczytane z chmurą punktów (load_bones(features=True)) - do kolizji.

    Returns:
        (powierzchnie, {id kości: skąd wzięte powierzchnie - jak w bone_contacts}).
    """
    out, info = [], {}
    for b in bones:
        if b.cloud is None:
            raise ValueError(f"{b.id}: brak chmury punktów - wczytaj kości przez load_bones(features=True)")
        c, info[b.id] = bone_contacts(b, p.source, contact_params, root)
        if c is None:
            continue
        tl = triangle_labels(b.mesh, c.labels)
        tree = None
        for k in c.surface_ids:
            m = submesh(b.mesh, tl == k)
            area = float(m.get_surface_area()) if len(m.triangles) else 0.0
            if area < p.min_area_mm2:
                continue
            if tree is None:
                tree = cKDTree(np.asarray(b.cloud.points))
            s = sample_surface(m, p.sample_mm)
            out.append(Surface(bone=b, contacts=c, sid=k, points=s.points, normals=s.normals, area_mm2=area,
                               bone_tree=tree))
        if not any(s.bone.id == b.id for s in out):
            info[b.id] += f", powierzchnie < {p.min_area_mm2:g} mm²"
    return out, info


# -----------------
# DOPASOWANIE PARY
# -----------------

def contact_score(a: Surface, b: Surface, T: np.ndarray, p: SurfaceMatchParams) -> dict:
    """Styk powierzchni a (przekształconej T) z b: części pokryte, pole styku, RMSE."""
    min_dot = float(np.cos(np.deg2rad(p.normal_angle_deg)))
    pa = transform_points(T, a.points)
    na = -(a.normals @ T[:3, :3].T)                  # odwrócone: w styku mają być zgodne z normalnymi b
    ok_a = covered(pa, na, b.tree(), b.normals, p.tolerance_mm, min_dot)
    Ti = np.linalg.inv(T)
    pb = transform_points(Ti, b.points)
    nb = -(b.normals @ Ti[:3, :3].T)
    ok_b = covered(pb, nb, a.tree(), a.normals, p.tolerance_mm, min_dot)
    d = b.tree().query(pa[ok_a])[0] if ok_a.any() else np.zeros(0)
    fa, fb = float(ok_a.mean()), float(ok_b.mean())
    return {"cover_a": fa, "cover_b": fb, "contact_mm2": min(fa * a.area_mm2, fb * b.area_mm2),
            "rmse_mm": float(np.sqrt(np.mean(d**2))) if len(d) else float("nan")}


def collision_area(a: Surface, b: Surface, T: np.ndarray, p: SurfaceMatchParams) -> float:
    """Pole kości a (po przekształceniu T), które wchodzi w kość b głębiej niż collision_mm, i odwrotnie [mm²].

    Punkt jest "w środku" kości, jeśli najbliższy jej punkt jest bliżej niż collision_search_mm,
    a punkt leży za jej powierzchnią (po przeciwnej stronie niż normalna) o więcej niż
    collision_mm. Skany są otwarte, więc to przybliżenie - wystarcza do odrzucenia
    ustawień "kość w kości".
    """
    def one_way(src: Surface, dst: Surface, M: np.ndarray) -> float:
        ps = transform_points(M, np.asarray(src.bone.cloud.points))
        pd, nd = np.asarray(dst.bone.cloud.points), np.asarray(dst.bone.cloud.normals)
        d, idx = dst.bone_tree.query(ps, distance_upper_bound=p.collision_search_mm)
        ok = np.isfinite(d)
        depth = np.einsum("ij,ij->i", pd[idx[ok]] - ps[ok], nd[idx[ok]])
        per_point = src.bone.meta.get("stats", {}).get("area_mm2", 0) / max(len(ps), 1)
        return float((depth > p.collision_mm).sum() * per_point)

    return one_way(a, b, T) + one_way(b, a, np.linalg.inv(T))


def refine_icp(src_clouds, tgt_clouds, T: np.ndarray, p: SurfaceMatchParams) -> np.ndarray:
    """ICP point-to-plane: icp_start_mm i połowa tego na rzadszych chmurach, potem 2 x sample_mm na gęstych."""
    crit = reg.ICPConvergenceCriteria(max_iteration=50)
    for dist, k in ((p.icp_start_mm, 1), (p.icp_start_mm / 2, 1), (2 * p.sample_mm, 0)):
        T = reg.registration_icp(src_clouds[k], tgt_clouds[k], dist, T,
                                 reg.TransformationEstimationPointToPlane(), crit).transformation
    return T


def match_pair(a: Surface, b: Surface, p: SurfaceMatchParams) -> dict:
    """Szuka T kładącego powierzchnię a na b (twarzą w twarz). Zwraca najlepszą z p.ransac_seeds prób.

    Returns:
        {"score", "contact_mm2", "collision_mm2", "cover_a", "cover_b", "rmse_mm",
         "T": macierz 4x4 (lista) z układu kości a do układu kości b, "seed"}.
    """
    ca, cb = a.clouds(p, flip=True), b.clouds(p, flip=False)
    dist = 1.5 * p.coarse_voxel_mm             # maks. odległość punktów uznanych za pasujące w RANSAC
    best = None
    for seed in range(p.ransac_seeds):
        o3d.utility.random.seed(seed)
        coarse = reg.registration_ransac_based_on_feature_matching(
            ca[2], cb[2], ca[3], cb[3], True, dist,
            reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             reg.CorrespondenceCheckerBasedOnDistance(dist),
             reg.CorrespondenceCheckerBasedOnNormal(np.deg2rad(p.ransac_normal_angle_deg))],
            reg.RANSACConvergenceCriteria(p.ransac_iterations, 1.0))    # 1.0 = bez wczesnego końca
        T = refine_icp(ca, cb, coarse.transformation, p)
        sc = contact_score(a, b, T, p)
        sc["collision_mm2"] = collision_area(a, b, T, p)
        sc["score"] = sc["contact_mm2"] - sc["collision_mm2"]
        if best is None or sc["score"] > best["score"]:
            best = {**sc, "T": T.tolist(), "seed": seed}
    return best


def match_all(surfaces: list[Surface], p: SurfaceMatchParams, progress=print) -> list[dict]:
    """Dopasowuje każdą parę powierzchni z różnych kości. Wynik posortowany od najlepszej pary."""
    pairs = [(i, j) for i in range(len(surfaces)) for j in range(i + 1, len(surfaces))
             if surfaces[i].bone.id != surfaces[j].bone.id]
    progress(f"Powierzchnie: {len(surfaces)}, pary do sprawdzenia: {len(pairs)}")
    results = []
    start = time.perf_counter()
    for n, (i, j) in enumerate(pairs, 1):
        a, b = surfaces[i], surfaces[j]
        r = match_pair(a, b, p)
        r.update(a=a.key, b=b.key, a_name=a.name, b_name=b.name, a_area_mm2=a.area_mm2, b_area_mm2=b.area_mm2,
                 a_manual=a.is_manual, b_manual=b.is_manual)
        results.append(r)
        if n % 10 == 0 or n == len(pairs):
            progress(f"  {n}/{len(pairs)} par ({time.perf_counter() - start:.0f} s)")
    results.sort(key=lambda r: -r["score"])
    return results


def report(surfaces: list[Surface], results: list[dict], p: SurfaceMatchParams,
           contact_params: ContactParams = ContactParams()) -> dict:
    """Słownik do zapisu w matching.json."""
    return {"params": asdict(p), "contact_params": asdict(contact_params),
            "surfaces": [{"key": s.key, "bone": s.bone.id, "bone_label": s.bone.label, "surface": s.sid,
                          "name": s.contacts.name(s.sid), "source": s.contacts.source,
                          "area_mm2": round(s.area_mm2, 1), "points": len(s.points)} for s in surfaces],
            "pairs": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in results]}

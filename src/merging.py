"""
Łączenie wielu skanów tej samej kości w jedną kość i usuwanie szumu.

Skaner łapie za każdym razem tylko część powierzchni fragmentu, a ten sam fragment
bywa skanowany kilka razy (w różnych sesjach / slotach upperjaw, lowerjaw).
Ten moduł:

1. REJESTRUJE każdą parę skanów (zgrubnie: deskryptory FPFH + RANSAC, dokładnie: ICP)
   i mierzy NAKŁADANIE: jaka część mniejszego skanu leży bliżej niż `tolerance_mm`
   od drugiego skanu, z normalnymi skierowanymi w tę samą stronę. Gładkie trzony kości
   różnych fragmentów "pasują" do siebie przypadkowo na ~5-15%, ten sam fragment
   na 35-60%. Wyniki par są zapisywane w cache, więc kolejne uruchomienia są szybkie.
2. GRUPUJE skany: nakładanie >= `min_overlap` = ta sama kość (grupy = spójne składowe
   grafu, więc dwa skany różnych końców kości łączą się przez trzeci, który obejmuje oba).
   Grupy można też podać ręcznie w settings.json ("merging.groups").
3. USTAWIA skany grupy we wspólnym układzie skanu referencyjnego (największego).
4. USUWA SZUM: oderwany kawałek skanu zostaje tylko, jeśli jest głównym kawałkiem,
   albo potwierdzają go inne skany grupy (>= `min_support` jego punktów), albo jest
   duży i leży blisko głównego. Np. połowicznie zeskanowany inny fragment leżący obok
   kości w jednym skanie nie ma potwierdzenia w pozostałych i znika.
5. SKLEJA powierzchnię: "union" = cały skan referencyjny + trójkąty pozostałych skanów
   tylko tam, gdzie czegoś brakuje (oryginalna geometria i kolory), albo "poisson" =
   jedna gładka powierzchnia z rekonstrukcji Poissona.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from data_parsing import (
    ScanFile,
    ScanParams,
    SurfaceSample,
    component_gaps,
    components,
    compute_fpfh,
    load_mesh,
    repair_mesh,
    sample_surface,
    submesh,
    to_cloud,
    voxel_keep,
)

log = logging.getLogger(__name__)
reg = o3d.pipelines.registration

FUSE_METHODS = ("union", "poisson")


@dataclass(frozen=True)
class MergeParams:
    enabled: bool = True              # false = każdy skan to osobna kość (bez łączenia)
    sample_mm: float = 0.25           # odstęp punktów używanych do porównywania skanów [mm]
    coarse_voxel_mm: float = 1.0      # rozdzielczość zgrubnej rejestracji (FPFH + RANSAC) [mm]
    tolerance_mm: float = 0.15        # bliżej niż to (i zgodne normalne) = ta sama powierzchnia
    min_overlap: float = 0.3          # nakładanie, od którego dwa skany = ta sama kość
    min_support: float = 0.25         # kawałek potwierdzony w mniejszej części = szum
    ransac_seeds: int = 3             # ile prób zgrubnej rejestracji na parę
    ransac_iterations: int = 50000
    fuse_method: str = "union"        # "union" albo "poisson"
    poisson_depth: int = 9            # tylko dla "poisson": większa = więcej szczegółów
    groups: dict | None = None        # ręczne grupy {"nazwa": ["fragment nazwy skanu", ...]}

    def registration_key(self) -> str:
        """Parametry wpływające na wynik rejestracji pary (klucz cache)."""
        return (f"s{self.sample_mm}-c{self.coarse_voxel_mm}-t{self.tolerance_mm}"
                f"-r{self.ransac_seeds}x{self.ransac_iterations}")


# -----------------
# PRZYGOTOWANIE SKANU
# -----------------

@dataclass
class PreparedScan:
    """Skan wczytany i naprawiony, z punktami do porównywania z innymi skanami."""

    file: ScanFile
    mesh: o3d.geometry.TriangleMesh       # wszystkie kawałki (szum usuwany dopiero przy łączeniu)
    tri_comp: np.ndarray                  # (T,) numer kawałka trójkąta, 0 = największy
    comp_areas: np.ndarray                # (K,) pola kawałków [mm^2]
    comp_gaps: np.ndarray                 # (K,) odległość od największego kawałka [mm]
    base_keep: np.ndarray                 # (K,) kawałek zostaje bez potwierdzenia (duży i blisko)
    sample: SurfaceSample                 # punkty na całej powierzchni (wszystkie kawałki)
    point_comp: np.ndarray                # (N,) numer kawałka każdego punktu
    checksum: str
    raw_stats: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def main_points(self) -> np.ndarray:
        """Maska punktów pewnych kawałków (używanych do rejestracji)."""
        return self.base_keep[self.point_comp]

    def clouds(self, coarse_voxel: float):
        """(chmura gęsta, chmura 2x rzadsza, chmura zgrubna, FPFH zgrubnej) - liczone raz."""
        key = ("clouds", coarse_voxel)
        if key not in self._cache:
            m = self.main_points
            pts, nrm = self.sample.points[m], self.sample.normals[m]
            dense = to_cloud(pts, nrm)
            mid_idx = voxel_keep(pts, coarse_voxel / 2)
            coarse_idx = voxel_keep(pts, coarse_voxel)
            coarse = to_cloud(pts[coarse_idx], nrm[coarse_idx])
            self._cache[key] = (dense, to_cloud(pts[mid_idx], nrm[mid_idx]), coarse,
                                compute_fpfh(coarse, 5 * coarse_voxel))
        return self._cache[key]

    def tree(self) -> cKDTree:
        if "tree" not in self._cache:
            self._cache["tree"] = cKDTree(self.sample.points[self.main_points])
        return self._cache["tree"]


def file_checksum(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def prepare_scan(file: ScanFile, scan: ScanParams, merge: MergeParams) -> PreparedScan:
    """Wczytuje skan, naprawia siatkę, dzieli na kawałki i losuje punkty powierzchni."""
    mesh = load_mesh(file.path)
    raw = {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles)}
    repair_mesh(mesh)
    tri_comp, areas = components(mesh)
    gaps = component_gaps(mesh, tri_comp)
    base_keep = (areas >= scan.min_component_area_ratio * areas[0]) & (gaps <= scan.max_component_gap_mm)
    base_keep[0] = True
    sample = sample_surface(mesh, merge.sample_mm)
    return PreparedScan(file=file, mesh=mesh, tri_comp=tri_comp, comp_areas=areas, comp_gaps=gaps,
                        base_keep=base_keep, sample=sample, point_comp=tri_comp[sample.triangle],
                        checksum=file_checksum(file.path), raw_stats=raw)


# -----------------
# REJESTRACJA PAR
# -----------------

def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def covered(points: np.ndarray, normals: np.ndarray, tree: cKDTree, target_normals: np.ndarray,
            tol: float, min_dot: float = 0.7) -> np.ndarray:
    """Maska punktów, które mają sąsiada bliżej niż `tol` o podobnie skierowanej normalnej."""
    d, idx = tree.query(points, distance_upper_bound=tol)
    ok = np.isfinite(d)
    ok[ok] &= np.einsum("ij,ij->i", normals[ok], target_normals[idx[ok]]) > min_dot
    return ok


def overlap(a: PreparedScan, b: PreparedScan, T: np.ndarray, tol: float) -> tuple[float, float]:
    """(część punktów a pokryta przez b, część punktów b pokryta przez a) przy przekształceniu a->b T."""
    ma, mb = a.main_points, b.main_points
    pa, na = a.sample.points[ma], a.sample.normals[ma]
    pb, nb = b.sample.points[mb], b.sample.normals[mb]
    fa = covered(transform_points(T, pa), na @ T[:3, :3].T, b.tree(), nb, tol).mean()
    Ti = np.linalg.inv(T)
    fb = covered(transform_points(Ti, pb), nb @ Ti[:3, :3].T, a.tree(), na, tol).mean()
    return float(fa), float(fb)


def refine_icp(src_clouds, tgt_clouds, T: np.ndarray, sample_mm: float) -> np.ndarray:
    """ICP point-to-plane od luźnego do ścisłego progu (1.5 -> 0.6 mm na rzadszych chmurach,
    potem 1 x odstęp punktów na gęstych)."""
    crit = reg.ICPConvergenceCriteria(max_iteration=40)
    for dist, k in ((1.5, 1), (0.6, 1), (max(sample_mm, 0.2), 0)):
        T = reg.registration_icp(src_clouds[k], tgt_clouds[k], dist, T,
                                 reg.TransformationEstimationPointToPlane(), crit).transformation
    return T


def register_pair(a: PreparedScan, b: PreparedScan, p: MergeParams) -> dict:
    """Szuka przekształcenia sztywnego a -> b, które najlepiej nakłada skany.

    Returns:
        {"overlap": max z obu kierunków, "overlap_a", "overlap_b", "T": macierz 4x4 (lista)}.
    """
    ca, cb = a.clouds(p.coarse_voxel_mm), b.clouds(p.coarse_voxel_mm)
    vox = p.coarse_voxel_mm
    best = None
    for seed in range(p.ransac_seeds):
        o3d.utility.random.seed(seed)
        coarse = reg.registration_ransac_based_on_feature_matching(
            ca[2], cb[2], ca[3], cb[3], True, 1.5 * vox,
            reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             reg.CorrespondenceCheckerBasedOnDistance(1.5 * vox)],
            reg.RANSACConvergenceCriteria(p.ransac_iterations, 0.999))
        T = refine_icp(ca, cb, coarse.transformation, p.sample_mm)
        fa, fb = overlap(a, b, T, p.tolerance_mm)
        if best is None or max(fa, fb) > best["overlap"]:
            best = {"overlap": max(fa, fb), "overlap_a": fa, "overlap_b": fb, "T": T.tolist()}
        if best["overlap"] >= max(0.5, 1.5 * p.min_overlap):    # pewne - nie trzeba więcej prób
            break
    return best


def pairwise(scans: list[PreparedScan], p: MergeParams, cache_file: Path | None = None,
             progress=print) -> dict[tuple[int, int], dict]:
    """Rejestruje wszystkie pary skanów (i < j). Wyniki z cache są brane bez liczenia."""
    cache = {}
    if cache_file and cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = {}
    key = p.registration_key()
    results, todo = {}, []
    for i in range(len(scans)):
        for j in range(i + 1, len(scans)):
            k = f"{scans[i].checksum}|{scans[j].checksum}|{key}"
            k_rev = f"{scans[j].checksum}|{scans[i].checksum}|{key}"
            if k in cache:
                results[i, j] = cache[k]
            elif k_rev in cache:                         # ta sama para w odwrotnej kolejności
                r = cache[k_rev]
                results[i, j] = {"overlap": r["overlap"], "overlap_a": r["overlap_b"],
                                 "overlap_b": r["overlap_a"], "T": np.linalg.inv(r["T"]).tolist()}
            else:
                todo.append((i, j, k))
    if todo:
        progress(f"Rejestracja {len(todo)} par skanów (z cache: {len(results)}) - to chwilę potrwa...")
    for n, (i, j, k) in enumerate(todo, 1):
        results[i, j] = cache[k] = register_pair(scans[i], scans[j], p)
        mark = "  <- ta sama kość" if results[i, j]["overlap"] >= p.min_overlap else ""
        progress(f"  [{n}/{len(todo)}] {scans[i].file.label}  ~  {scans[j].file.label}: "
                 f"nakładanie {results[i, j]['overlap']:.2f}{mark}")
        if cache_file and (n % 10 == 0 or n == len(todo)):       # zapis co 10 par - przerwanie nie traci wszystkiego
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache), encoding="utf-8")
    return results


def pair_T(pairs: dict, i: int, j: int) -> np.ndarray:
    """Przekształcenie skan i -> skan j."""
    if (i, j) in pairs:
        return np.asarray(pairs[i, j]["T"])
    return np.linalg.inv(np.asarray(pairs[j, i]["T"]))


def pair_overlap(pairs: dict, i: int, j: int) -> float:
    return pairs[(i, j) if (i, j) in pairs else (j, i)]["overlap"]


# -----------------
# GRUPOWANIE
# -----------------

def _matches(scan: PreparedScan, pattern: str) -> bool:
    p = pattern.lower()
    return p in scan.file.id.lower() or p in scan.file.label.lower() or p in str(scan.file.path).lower()


def group_scans(scans: list[PreparedScan], pairs: dict, p: MergeParams) -> list[dict]:
    """Dzieli skany na kości.

    Ręczne grupy z `p.groups` mają pierwszeństwo; pozostałe skany są grupowane
    automatycznie (nakładanie >= min_overlap, spójne składowe grafu).

    Returns:
        Lista {"name": nazwa ręczna albo None, "members": [indeksy skanów]}.
    """
    n = len(scans)
    assigned: dict[int, str] = {}
    groups = []
    for name, patterns in (p.groups or {}).items():
        patterns = [patterns] if isinstance(patterns, str) else list(patterns)
        members = []
        for pat in patterns:
            hit = [i for i in range(n) if _matches(scans[i], pat)]
            if not hit:
                raise ValueError(f"merging.groups[\"{name}\"]: żaden skan nie pasuje do \"{pat}\". "
                                 f"Dostępne: {', '.join(s.file.id for s in scans)}")
            members += hit
        for i in members:
            if assigned.get(i, name) != name:
                raise ValueError(f"Skan {scans[i].file.id} jest w dwóch ręcznych grupach: "
                                 f"\"{assigned[i]}\" i \"{name}\"")
            assigned[i] = name
        groups.append({"name": name, "members": sorted(set(members))})

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    free = [i for i in range(n) if i not in assigned]
    if p.enabled:
        for a in free:
            for b in free:
                if a < b and pair_overlap(pairs, a, b) >= p.min_overlap:
                    parent[find(a)] = find(b)
    auto: dict[int, list[int]] = {}
    for i in free:
        auto.setdefault(find(i), []).append(i)
    groups += [{"name": None, "members": sorted(m)} for m in auto.values()]
    return groups


# -----------------
# USTAWIENIE GRUPY I USUWANIE SZUMU
# -----------------

def align_group(scans: list[PreparedScan], members: list[int], pairs: dict, p: MergeParams,
                ref: int) -> dict[int, np.ndarray]:
    """Przekształcenia każdego skanu grupy do układu skanu `ref`.

    Kolejność dołączania: drzewo najlepszych nakładań od skanu referencyjnego
    (algorytm Prima), potem każdy skan jest dopracowany ICP względem już
    ustawionych skanów (a nie tylko jednego sąsiada).
    """
    T = {ref: np.eye(4)}
    order = [ref]
    while len(T) < len(members):
        best = max(((pair_overlap(pairs, i, j), i, j) for i in T for j in members if j not in T))
        _, i, j = best
        T[j] = T[i] @ pair_T(pairs, j, i)
        order.append(j)

    for j in order[1:]:
        placed = [i for i in order[:order.index(j)]]
        pts = np.vstack([transform_points(T[i], scans[i].sample.points[scans[i].main_points]) for i in placed])
        nrm = np.vstack([scans[i].sample.normals[scans[i].main_points] @ T[i][:3, :3].T for i in placed])
        target = to_cloud(pts, nrm)
        target_mid = target.voxel_down_sample(2 * p.sample_mm)
        src = scans[j].clouds(p.coarse_voxel_mm)
        T[j] = refine_icp(src, (target, target_mid), T[j], p.sample_mm)
    return T


def component_support(scans: list[PreparedScan], members: list[int], T: dict, p: MergeParams) -> dict:
    """Dla każdego skanu: jaka część punktów każdego kawałka leży na innym skanie grupy.

    Returns:
        {indeks_skanu: tablica (K,) ułamków w [0, 1]}; dla grupy z jednym skanem - zera.
    """
    support = {}
    tol = 2 * p.tolerance_mm          # luźniej niż przy grupowaniu - to już ustawione skany
    for j in members:
        s = scans[j]
        ncomp = len(s.comp_areas)
        others = [i for i in members if i != j]
        if not others:
            support[j] = np.zeros(ncomp)
            continue
        pts = np.vstack([transform_points(T[i], scans[i].sample.points) for i in others])
        nrm = np.vstack([scans[i].sample.normals @ T[i][:3, :3].T for i in others])
        ok = covered(transform_points(T[j], s.sample.points), s.sample.normals @ T[j][:3, :3].T,
                     cKDTree(pts), nrm, tol, min_dot=0.5)
        hits = np.bincount(s.point_comp, weights=ok, minlength=ncomp)
        total = np.bincount(s.point_comp, minlength=ncomp)
        support[j] = hits / np.maximum(total, 1)
    return support


def decide_components(scan: PreparedScan, support: np.ndarray, p: MergeParams) -> tuple[np.ndarray, list[str]]:
    """Które kawałki skanu zostają. Zwraca (maska (K,), powód dla każdego kawałka)."""
    keep = scan.base_keep | (support >= p.min_support)
    keep[0] = True
    reasons = []
    for c in range(len(keep)):
        if c == 0:
            reasons.append("główny kawałek")
        elif support[c] >= p.min_support:
            reasons.append(f"potwierdzony przez inne skany ({support[c]:.0%})")
        elif scan.base_keep[c]:
            reasons.append("duży i blisko głównego")
        elif scan.comp_gaps[c] > 0 and scan.comp_areas[c] >= 0.05 * scan.comp_areas[0]:
            reasons.append(f"SZUM: {scan.comp_gaps[c]:.0f} mm od głównego, brak potwierdzenia")
        else:
            reasons.append("SZUM: mały, brak potwierdzenia")
    return keep, reasons


# -----------------
# SKLEJANIE POWIERZCHNI
# -----------------

def fuse_union(meshes: list[o3d.geometry.TriangleMesh], spacing: float,
               cover_mm: float) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """Suma skanów bez nakładania: pierwszy w całości, z kolejnych tylko nowe miejsca.

    Trójkąt kolejnego skanu jest dodawany, jeśli jego środek jest dalej niż `cover_mm`
    od powierzchni już dodanej. Bardzo małe dodane skrawki (wzdłuż szwów) są pomijane.

    Returns:
        (siatka, numer skanu źródłowego dla każdego trójkąta (T,)).
    """
    out = o3d.geometry.TriangleMesh(meshes[0])
    source = np.zeros(len(out.triangles), dtype=np.int32)
    covered_pts = sample_surface(out, spacing).points
    for k, m in enumerate(meshes[1:], start=1):
        v, t = np.asarray(m.vertices), np.asarray(m.triangles)
        if len(t) == 0:
            continue
        d, _ = cKDTree(covered_pts).query(v[t].mean(axis=1))
        new = submesh(m, d > cover_mm)
        if len(new.triangles) == 0:
            continue
        comp, areas = components(new)
        new = submesh(new, np.isin(comp, np.flatnonzero(areas >= 2.0)))    # skrawki < 2 mm^2
        if len(new.triangles) == 0:
            continue
        out += new
        source = np.concatenate([source, np.full(len(new.triangles), k, dtype=np.int32)])
        covered_pts = np.vstack([covered_pts, sample_surface(new, spacing).points])
    return out, source


def fuse_poisson(meshes: list[o3d.geometry.TriangleMesh], spacing: float, depth: int,
                 cover_mm: float) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """Jedna powierzchnia z rekonstrukcji Poissona (punkty bez nakładania, jak w fuse_union).

    Powierzchnia Poissona dalej niż 2 x `spacing` od punktów skanów (domknięcia
    "na wyobraźnię") jest wycinana.
    """
    pts, nrm, col = [], [], []
    for m in meshes:
        s = sample_surface(m, spacing)
        if pts:
            d, _ = cKDTree(np.vstack(pts)).query(s.points)
            s = SurfaceSample(*(a[d > cover_mm] for a in (s.points, s.normals, s.colors, s.triangle)))
        pts.append(s.points), nrm.append(s.normals), col.append(s.colors)
    P = np.vstack(pts)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        to_cloud(P, np.vstack(nrm), np.vstack(col)), depth=depth)
    d, _ = cKDTree(P).query(np.asarray(mesh.vertices))
    mesh.remove_vertices_by_mask(d > 2 * spacing)
    comp, areas = components(mesh)
    mesh = submesh(mesh, np.isin(comp, np.flatnonzero(areas >= 0.05 * areas[0])))
    return mesh, np.zeros(len(mesh.triangles), dtype=np.int32)


@dataclass
class MergedBone:
    """Wynik łączenia jednej grupy skanów (w układzie skanu referencyjnego)."""

    id: str
    label: str
    mesh: o3d.geometry.TriangleMesh
    source: np.ndarray                                  # (T,) indeks w `scan_ids` dla trójkąta
    scan_ids: list[str]                                 # kolejność = kolejność w `source`
    scan_meshes: dict[str, o3d.geometry.TriangleMesh]   # zostawione kawałki każdego skanu
    noise_meshes: dict[str, o3d.geometry.TriangleMesh]  # usunięte kawałki (szum)
    meta: dict


def merge_group(scans: list[PreparedScan], group: dict, pairs: dict, p: MergeParams) -> MergedBone:
    """Ustawia skany grupy, usuwa szum i skleja jedną powierzchnię."""
    from data_parsing import slugify

    members = group["members"]
    areas = {j: float(scans[j].comp_areas[scans[j].base_keep].sum()) for j in members}
    ref = max(members, key=lambda j: areas[j])
    T = align_group(scans, members, pairs, p, ref) if len(members) > 1 else {ref: np.eye(4)}
    support = component_support(scans, members, T, p)

    order = [ref] + sorted((j for j in members if j != ref), key=lambda j: -areas[j])
    kept_meshes, scan_meshes, noise_meshes, scan_meta = [], {}, {}, []
    for j in order:
        s = scans[j]
        keep, reasons = decide_components(s, support[j], p)
        tri_keep = keep[s.tri_comp]
        good = submesh(s.mesh, tri_keep).transform(T[j])
        kept_meshes.append(good)
        scan_meshes[s.file.id] = good
        if (~tri_keep).any():
            noise_meshes[s.file.id] = submesh(s.mesh, ~tri_keep).transform(T[j])
        others = [i for i in members if i != j]
        scan_meta.append({
            "id": s.file.id, "label": s.file.label, "source": str(s.file.path.name), "folder": s.file.folder,
            "reference": j == ref,
            "overlap_with_reference": None if j == ref else round(pair_overlap(pairs, j, ref), 3),
            "best_overlap_in_group": round(max((pair_overlap(pairs, j, i) for i in others), default=0.0), 3),
            "transform_to_bone": np.round(T[j], 6).tolist(),
            "raw": s.raw_stats,
            "components": [{"area_mm2": round(float(a), 2), "gap_mm": round(float(g), 2),
                            "support": round(float(sp), 3), "kept": bool(k), "reason": r}
                           for a, g, sp, k, r in zip(s.comp_areas, s.comp_gaps, support[j], keep, reasons)],
        })

    if len(kept_meshes) == 1:
        mesh, source = kept_meshes[0], np.zeros(len(kept_meshes[0].triangles), dtype=np.int32)
        method = "single"
    elif p.fuse_method == "poisson":
        mesh, source = fuse_poisson(kept_meshes, p.sample_mm, p.poisson_depth, 2 * p.tolerance_mm)
        method = "poisson"
    elif p.fuse_method == "union":
        mesh, source = fuse_union(kept_meshes, p.sample_mm, 2 * p.tolerance_mm)
        method = "union"
    else:
        raise ValueError(f"merging.fuse_method = {p.fuse_method!r}, dostępne: {FUSE_METHODS}")
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    ref_file = scans[ref].file
    bone_id = slugify(group["name"]) if group["name"] else ref_file.id
    label = group["name"] or ref_file.label
    scan_ids = [scans[j].file.id for j in order]
    meta = {
        "id": bone_id, "label": label, "manual_group": group["name"] is not None,
        "reference_scan": ref_file.id, "fuse_method": method, "scans": scan_meta,
        "noise_removed_mm2": round(sum(c["area_mm2"] for s in scan_meta for c in s["components"]
                                       if not c["kept"]), 2),
        "merge_params": {k: v for k, v in asdict(p).items() if k != "groups"},
    }
    return MergedBone(id=bone_id, label=label, mesh=mesh, source=source, scan_ids=scan_ids,
                      scan_meshes=scan_meshes, noise_meshes=noise_meshes, meta=meta)

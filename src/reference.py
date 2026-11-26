"""
Kość WZORCOWA zestawu: skan całej kości, z której pochodzą fragmenty - prawda do oceny dopasowania.

Na zestawie testowym (np. grzesiu_test: "Test cała kość strzałkowa" + kawałki test 1-4) cała kość pokazuje,
jak fragmenty mają być złożone na końcu. Każdy fragment jest kładziony na kości wzorcowej (ta sama kość,
ten sam skaner, więc BEZ skali): RANSAC na FPFH całych kości, potem ICP point-to-plane z malejącym promieniem;
najlepsza z kilku prób = największa część kory fragmentu leżąca na wzorcu (fitness). Gdy to mniej niż
SWEEP_BELOW (trzon wygląda podobnie na całej długości, RANSAC trafia w złe miejsce), przeszukiwany jest cały
trzon (sweep: oś na osi, oba zwroty, obrót co 30°, środek co SWEEP_STEP_AU). Z tych położeń:
    - prawdziwe ustawienie każdej pary fragmentów: T_true(a -> b) = P_b⁻¹ P_a (P = fragment -> wzorzec),
    - prawdziwe pary przełomów: powierzchnie dwóch fragmentów, które w układzie wzorca leżą na sobie
      (punkty bliżej niż TOUCH_AU) - styk w a.u.²; przełom bez partnera = brakujący kawałek albo cięcie,
    - ocena wyniku dopasowania (matching.json): czy para jest prawdziwa, błąd położenia (średnia odległość
      wierzchołków fragmentu A ustawionego przez T i przez T_true) i błąd obrotu w stopniach.
Wzorzec nie jest używany do samego dopasowania - tylko do sprawdzenia (jak "referencja" w OSTEOFORM).

Położenia są zapisywane w out/3d/<zestaw>/reference.json (cache: liczone od nowa po zmianie siatek).

    import reference
    ref, frags = reference.split_reference(load_bones(features=True), ["cała kość"])
    poses = reference.register_all(ref, frags)            # {id: {"T", "fitness", "rmse_au"}}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import contacts as ct
from data_parsing import compute_fpfh, sample_surface, to_cloud
from merging import transform_points

reg = o3d.pipelines.registration

FORMAT = "bones-smm-reference/2"
VOXEL_AU = 1.5            # odstęp punktów do RANSAC [a.u.]
FPFH_RADIUS_AU = 6.0      # promień FPFH [a.u.]
RANSAC_ITERATIONS = 1000000
SEEDS = 4                 # niezależnych prób RANSAC + ICP na fragment
FIT_AU = 0.5              # fitness: punkt fragmentu leży na wzorcu (bliżej niż to, normalne zgodne)
TOUCH_AU = 1.0            # przełomy dwóch fragmentów w układzie wzorca bliżej niż to = stykają się
POSE_OK_AU = 2.0          # para "dobrze złożona": średni błąd położenia poniżej tego [a.u.]
SWEEP_BELOW = 0.6         # gdy najlepsza próba RANSAC ma mniej kory na wzorcu - przeszukanie całego trzonu (sweep)
SWEEP_STEP_AU = 8.0       # ... środek fragmentu co tyle wzdłuż osi wzorca


def split_reference(bones: list, patterns) -> tuple[object | None, list]:
    """Kość wzorcowa (id albo nazwa zawiera któryś z tekstów, jak "exclude" w contacts) i reszta fragmentów.

    Args:
        bones:    kości zestawu.
        patterns: teksty, np. ["cała kość"] (bez wielkości liter i polskich znaków, spacja = "_").

    Returns:
        (wzorzec albo None, fragmenty bez wzorca).
    """
    probe = ct.ContactParams(exclude=tuple(patterns))
    refs = [b for b in bones if ct.is_excluded(b, probe)]
    if not refs:
        return None, list(bones)
    ref = max(refs, key=lambda b: len(b.mesh.vertices))
    return ref, [b for b in bones if b.id != ref.id]


def _clouds(bone, voxel: float):
    """(gęsta chmura do ICP i oceny, rzadka chmura + FPFH do RANSAC) całej kości."""
    dense = sample_surface(bone.mesh, voxel / 3)
    coarse = sample_surface(bone.mesh, voxel)
    c = to_cloud(coarse.points, coarse.normals)
    return to_cloud(dense.points, dense.normals), c, compute_fpfh(c, FPFH_RADIUS_AU)


def fitness(src: o3d.geometry.PointCloud, tree: cKDTree, ref_normals: np.ndarray, T: np.ndarray) -> tuple[float, float]:
    """Część punktów fragmentu (po T) leżących na wzorcu (bliżej niż FIT_AU, normalne zgodne do 45°) i ich RMSE."""
    p = transform_points(T, np.asarray(src.points))
    n = np.asarray(src.normals) @ T[:3, :3].T
    d, idx = tree.query(p, distance_upper_bound=FIT_AU)
    ok = np.isfinite(d)
    ok[ok] &= np.einsum("ij,ij->i", n[ok], ref_normals[idx[ok]]) > np.cos(np.radians(45))
    return float(ok.mean()), float(np.sqrt(np.mean(d[ok] ** 2))) if ok.any() else float("nan")


def register(frag, ref, ref_clouds=None, seeds: int = SEEDS, log=None) -> dict:
    """Kładzie fragment na kości wzorcowej: najlepsza z `seeds` prób RANSAC (FPFH) + ICP point-to-plane.

    Args:
        frag, ref:  fragment i kość wzorcowa (bones.Bone).
        ref_clouds: wynik _clouds(ref, VOXEL_AU) - podany, żeby nie liczyć go dla każdego fragmentu.
        seeds:      liczba prób.
        log:        komunikaty (None = cisza).

    Returns:
        {"T": 4x4 (lista) fragment -> wzorzec, "fitness": część punktów fragmentu na wzorcu (kora; przełomy
         nie leżą na powierzchni wzorca, więc 1.0 jest niemożliwe), "rmse_au", "seed"}.
    """
    rd, rc, rf = ref_clouds or _clouds(ref, VOXEL_AU)
    fd, fc, ff = _clouds(frag, VOXEL_AU)
    tree, rn = cKDTree(np.asarray(rd.points)), np.asarray(rd.normals)
    dist = 1.5 * VOXEL_AU
    best = None
    for seed in range(seeds):
        o3d.utility.random.seed(seed)
        r = reg.registration_ransac_based_on_feature_matching(
            fc, rc, ff, rf, True, dist, reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(dist),
             reg.CorrespondenceCheckerBasedOnNormal(np.radians(35))],
            reg.RANSACConvergenceCriteria(RANSAC_ITERATIONS, 1.0))
        T = r.transformation
        for radius in (3.0, 1.5, 0.75, 0.4):
            T = reg.registration_icp(fd, rd, radius, T, reg.TransformationEstimationPointToPlane(),
                                     reg.ICPConvergenceCriteria(max_iteration=60)).transformation
        fit, rmse = fitness(fd, tree, rn, T)
        if log:
            log(f"      próba {seed + 1}: na wzorcu {fit:.0%} punktów, RMSE {rmse:.2f} a.u.")
        if best is None or fit > best["fitness"]:
            best = {"T": np.round(T, 6).tolist(), "fitness": round(fit, 4), "rmse_au": round(rmse, 4), "seed": seed}
    if best["fitness"] < SWEEP_BELOW:
        swept = sweep(frag, ref, fd, fc, rd, rc, tree, rn, log)
        if swept["fitness"] > best["fitness"]:
            best = swept
    return best


def sweep(frag, ref, fd, fc, rd, rc, tree, rn, log=None) -> dict:
    """Przeszukanie całego trzonu wzorca, gdy RANSAC pasuje słabo (trzon wygląda podobnie na całej długości).

    Oś fragmentu na osi wzorca (oba zwroty), obrót co 30°, środek co SWEEP_STEP_AU wzdłuż wzorca; z każdego
    startu ICP na rzadkich chmurach, potem 3 najlepsze dokładnie (gęste chmury).

    Returns:
        Jak register ("seed" = "przeszukanie").
    """
    from matching_surfaces import _rotation, _rotation_between

    fax, rax = ct.bone_axis(frag.mesh), ct.bone_axis(ref.mesh)
    coarse_tree, coarse_n = cKDTree(np.asarray(rc.points)), np.asarray(rc.normals)
    starts = []
    for flip in (1.0, -1.0):
        R0 = _rotation_between(flip * fax["d"], rax["d"])
        for tw in range(0, 360, 30):
            R = _rotation(rax["d"], np.radians(tw)) @ R0
            for s in np.arange(rax["lo"] + 10, rax["hi"] - 10 + 1e-9, SWEEP_STEP_AU):
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = R, rax["c"] + s * rax["d"] - R @ fax["c"]
                for radius in (4.0, 2.0, 1.0):
                    T = reg.registration_icp(fc, rc, radius, T, reg.TransformationEstimationPointToPlane(),
                                             reg.ICPConvergenceCriteria(max_iteration=30)).transformation
                starts.append((fitness(fc, coarse_tree, coarse_n, T)[0], T))
    starts.sort(key=lambda x: -x[0])
    best = None
    for _, T in starts[:3]:
        for radius in (1.5, 0.75, 0.4):
            T = reg.registration_icp(fd, rd, radius, T, reg.TransformationEstimationPointToPlane(),
                                     reg.ICPConvergenceCriteria(max_iteration=60)).transformation
        fit, rmse = fitness(fd, tree, rn, T)
        if best is None or fit > best["fitness"]:
            best = {"T": np.round(T, 6).tolist(), "fitness": round(fit, 4), "rmse_au": round(rmse, 4),
                    "seed": "przeszukanie"}
    if log:
        log(f"      przeszukanie trzonu: na wzorcu {best['fitness']:.0%} punktów, RMSE {best['rmse_au']:.2f} a.u.")
    return best


def register_all(ref, frags: list, out_dir: Path | None = None, log=print) -> dict:
    """Położenia wszystkich fragmentów na wzorcu, z cache out/3d/<zestaw>/reference.json.

    Liczone od nowa dla fragmentu, gdy zmieniła się jego siatka albo siatka wzorca (contacts.mesh_fingerprint).

    Args:
        ref:     kość wzorcowa.
        frags:   fragmenty.
        out_dir: folder wyników zestawu (None = config.DATA_OUT).
        log:     komunikaty o postępie (None = cisza).

    Returns:
        {id fragmentu: {"T", "fitness", "rmse_au", "seed"}}.
    """
    import config

    path = Path(out_dir or config.DATA_OUT) / "reference.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    ref_key = ct.mesh_fingerprint(ref.mesh)
    if data.get("format") != FORMAT or data.get("reference", {}).get("mesh") != ref_key:
        data = {"format": FORMAT, "reference": {"id": ref.id, "mesh": ref_key}, "poses": {}}
    ref_clouds = None
    for f in frags:
        key = ct.mesh_fingerprint(f.mesh)
        if data["poses"].get(f.id, {}).get("mesh") == key:
            continue
        if log:
            log(f"   {f.label}: kładę na wzorcu ({ref.label})...")
        ref_clouds = ref_clouds or _clouds(ref, VOXEL_AU)
        data["poses"][f.id] = {**register(f, ref, ref_clouds, log=log), "mesh": key}
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return {k: v for k, v in data["poses"].items() if any(f.id == k for f in frags)}


def true_transform(poses: dict, a: str, b: str) -> np.ndarray:
    """Prawdziwe ustawienie fragmentu a w układzie fragmentu b (z położeń na wzorcu): P_b⁻¹ P_a."""
    return np.linalg.inv(np.asarray(poses[b]["T"])) @ np.asarray(poses[a]["T"])


def true_contacts(surfaces: list, poses: dict) -> list[dict]:
    """Które przełomy naprawdę się stykają: powierzchnie różnych fragmentów leżące na sobie w układzie wzorca.

    Args:
        surfaces: powierzchnie (matching_surfaces.Surface) fragmentów z położeniem w `poses`.
        poses:    wynik register_all.

    Returns:
        [{"a", "b": klucze powierzchni, "contact_au2": pole styku (mniejsze z obu stron), "cover_a",
          "cover_b"}] od największego styku; tylko pary ze stykiem >= 5% mniejszej powierzchni.
    """
    placed = {s.key: transform_points(np.asarray(poses[s.bone.id]["T"]), s.points)
              for s in surfaces if s.bone.id in poses}
    trees = {k: cKDTree(v) for k, v in placed.items()}
    by_key = {s.key: s for s in surfaces}
    out = []
    keys = sorted(placed)
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            if by_key[ka].bone.id == by_key[kb].bone.id:
                continue
            fa = float(np.isfinite(trees[kb].query(placed[ka], distance_upper_bound=TOUCH_AU)[0]).mean())
            fb = float(np.isfinite(trees[ka].query(placed[kb], distance_upper_bound=TOUCH_AU)[0]).mean())
            area = min(fa * by_key[ka].area_au2, fb * by_key[kb].area_au2)
            if area >= 0.05 * min(by_key[ka].area_au2, by_key[kb].area_au2):
                out.append({"a": ka, "b": kb, "contact_au2": round(area, 1), "cover_a": round(fa, 3),
                            "cover_b": round(fb, 3)})
    return sorted(out, key=lambda r: -r["contact_au2"])


def pose_error(bones: dict, poses: dict, r: dict) -> dict:
    """Błąd ustawienia pary z matching.json względem wzorca.

    Args:
        bones: {id: Bone} (do wierzchołków fragmentu A).
        poses: wynik register_all.
        r:     para z matching.json ("a", "b", "T": a -> b).

    Returns:
        {"err_au": średnia odległość wierzchołków A ustawionych przez T i przez prawdę, "rot_deg": kąt
         obrotu między T a prawdą, "move_au": przesunięcie środka A między T a prawdą}.
    """
    a, b = r["a"].rsplit("#", 1)[0], r["b"].rsplit("#", 1)[0]
    T, Tt = np.asarray(r["T"]), true_transform(poses, a, b)
    v = np.asarray(bones[a].mesh.vertices)
    va, vt = transform_points(T, v), transform_points(Tt, v)
    D = np.linalg.inv(Tt) @ T
    rot = float(np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1))))
    return {"err_au": round(float(np.linalg.norm(va - vt, axis=1).mean()), 2), "rot_deg": round(rot, 1),
            "move_au": round(float(np.linalg.norm(va.mean(0) - vt.mean(0))), 2)}


# -----------------
# PORÓWNANIE CAŁOŚCI: złożona grupa fragmentów na kości wzorcowej
# -----------------

def group_mesh(frags: dict, members: list[str], poses: dict) -> o3d.geometry.TriangleMesh:
    """Jedna siatka z fragmentów grupy ustawionych wg złożenia (poses: id -> 4x4 w układzie grupy)."""
    out = o3d.geometry.TriangleMesh()
    for b in members:
        m = o3d.geometry.TriangleMesh(frags[b].mesh)
        m.transform(np.asarray(poses[b]))
        out += m
    out.compute_vertex_normals()
    return out


def compare_group(ref, frags: dict, members: list[str], assembly_poses: dict, own_poses: dict,
                  log=None) -> dict:
    """Kładzie CAŁĄ złożoną grupę na kości wzorcowej i porównuje z położeniami fragmentów kładzionych osobno.

    Args:
        ref:            kość wzorcowa.
        frags:          {id: Bone} fragmentów.
        members:        id fragmentów grupy.
        assembly_poses: położenia w układzie grupy (matching.json -> "assembly" -> "poses").
        own_poses:      położenia na wzorcu każdego fragmentu osobno (register_all).
        log:            komunikaty (None = cisza).

    Returns:
        {"T": 4x4 grupa -> wzorzec, "fitness": część kory całej grupy na wzorcu, "rmse_au",
         "fragments": {id: {"fitness": część kory fragmentu na wzorcu w złożeniu, "own_fitness": osobno,
                            "shift_au": średnia odległość wierzchołków między położeniem w złożeniu a osobnym}}}.
    """
    from types import SimpleNamespace

    whole = SimpleNamespace(id="+".join(members), label=" + ".join(frags[b].label for b in members),
                            mesh=group_mesh(frags, members, assembly_poses))
    res = register(whole, ref, _clouds(ref, VOXEL_AU), log=log)
    G = np.asarray(res["T"])
    rd, _, _ = _clouds(ref, VOXEL_AU)
    tree, rn = cKDTree(np.asarray(rd.points)), np.asarray(rd.normals)
    out = {}
    for b in members:
        fd, _, _ = _clouds(frags[b], VOXEL_AU)
        Tb = G @ np.asarray(assembly_poses[b])
        v = np.asarray(frags[b].mesh.vertices)
        shift = float(np.linalg.norm(transform_points(Tb, v) - transform_points(np.asarray(own_poses[b]["T"]), v),
                                     axis=1).mean()) if b in own_poses else float("nan")
        out[b] = {"fitness": round(fitness(fd, tree, rn, Tb)[0], 4),
                  "own_fitness": own_poses.get(b, {}).get("fitness"), "shift_au": round(shift, 2)}
    return {"T": res["T"], "fitness": res["fitness"], "rmse_au": res["rmse_au"], "fragments": out}


def whole_figure(ref, frags: dict, groups: list[tuple[list[str], dict]], assembly_poses: dict, view):
    """Figura: kość wzorcowa (szara, półprzezroczysta) i na niej złożone grupy (kolory fragmentów).

    Args:
        ref:            kość wzorcowa.
        frags:          {id: Bone}.
        groups:         [(id fragmentów grupy, wynik compare_group)].
        assembly_poses: położenia w układach grup (matching.json -> "assembly" -> "poses").
        view:           visualization_3d.View3DParams.

    Returns:
        plotly Figure.
    """
    import plotly.graph_objects as go

    from visualization_3d import ASSEMBLY_COLORS, FONT, LIGHTING, _f32, _ijk, decimate

    light = dict(lighting=LIGHTING, lightposition=dict(x=-1e4, y=-2e4, z=3e4))
    d = decimate(ref.mesh, 60000)
    v = np.asarray(d.vertices)
    traces = [go.Mesh3d(x=_f32(v[:, 0]), y=_f32(v[:, 1]), z=_f32(v[:, 2]), **_ijk(np.asarray(d.triangles), len(v)),
                        color="#9aa5b1", opacity=0.35, name=f"wzorzec: {ref.label}", showlegend=True,
                        hovertemplate=f"<b>{ref.label}</b> (wzorzec)<extra></extra>", **light)]
    k = 0
    for members, res in groups:
        G = np.asarray(res["T"])
        for b in members:
            d = decimate(frags[b].mesh, 30000)
            v = transform_points(G @ np.asarray(assembly_poses[b]), np.asarray(d.vertices))
            f = res["fragments"][b]
            traces.append(go.Mesh3d(x=_f32(v[:, 0]), y=_f32(v[:, 1]), z=_f32(v[:, 2]),
                                    **_ijk(np.asarray(d.triangles), len(v)),
                                    color=ASSEMBLY_COLORS[k % len(ASSEMBLY_COLORS)], name=frags[b].label,
                                    showlegend=True, **light,
                                    hovertemplate=f"<b>{frags[b].label}</b>: kora na wzorcu {f['fitness']:.0%} "
                                                  f"(osobno {f['own_fitness'] or 0:.0%})<extra></extra>"))
            k += 1
    axis = dict(backgroundcolor="#eef2f5", showticklabels=False, title="")
    fig = go.Figure(traces)
    fig.update_layout(paper_bgcolor=view.background, font=FONT, margin=dict(l=0, r=0, t=10, b=0), height=720,
                      legend=dict(x=0.99, y=0.97, xanchor="right", bgcolor="rgba(255,255,255,0.85)"),
                      scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data"))
    return fig

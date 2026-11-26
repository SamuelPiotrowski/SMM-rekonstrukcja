"""Geometric diagnostics and cautious atlas suggestions for assembled groups."""

from functools import lru_cache
from itertools import combinations, product
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

from fracture_matching import sample_surface
from osteo_engine_v6 import moved

MODELS = Path(__file__).resolve().parents[2] / "01_WZORCE_KOSCI"
ATLAS = MODELS / "Zenodo_Long_Bones" / "OBJ"
BONE_HUB = MODELS / "BoneHub" / "visible_human_3d_models" / "CT" / "Mesh" / "01_Male"
NAMES = {"Ulna": "Kość łokciowa", "Humerus": "Kość ramienna",
         "Femur": "Kość udowa", "Tibia": "Kość piszczelowa",
         "Radius": "Kość promieniowa", "Fibula": "Kość strzałkowa",
         "Clavicle": "Obojczyk", "Scapula": "Łopatka", "Hip": "Kość miedniczna",
         "Sacrum": "Kość krzyżowa"}

def closed_surface(mesh):
    """Signed distance is available only for watertight meshes."""
    preview = mesh.simplify_quadric_decimation(24000) if len(mesh.triangles) > 24000 else mesh
    if not preview.is_watertight():
        return None
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(preview))
    return scene


def inspect_groups(meshes, poses, groups, tolerance):
    """Estimate contact and local penetration at the final poses using surface samples."""
    samples = {}
    for group in groups:
        if len(group) < 2:
            continue
        for key in group:
            mesh = meshes[int(key) - 1][1]
            local, normals, _ = sample_surface(mesh, 2600, np.random.default_rng(int(key) + 41))
            world = moved(local, poses[key])
            samples[key] = (local, world, normals @ poses[key][:3, :3].T, cKDTree(world),
                            closed_surface(mesh), np.linalg.inv(poses[key]))
    overlays = {key: {"contact": [], "collision": []} for key in samples}
    summaries = []
    for group in groups:
        pairs = []
        for a, b in combinations(group, 2):
            separation = []
            depths = []
            counts = []
            near_counts = []
            methods = []
            for key, other in ((a, b), (b, a)):
                local, points, n, _, _, _ = samples[key]
                _, target, target_n, tree, scene, inverse = samples[other]
                dist, nearest = tree.query(points)
                normal = target_n[nearest]
                opposed = np.einsum("ij,ij->i", n, normal) < -.6
                signed = np.einsum("ij,ij->i", points - target[nearest], normal)
                near = dist <= 4 * tolerance
                if scene is not None:
                    signed = scene.compute_signed_distance(o3d.core.Tensor(
                        moved(points, inverse).astype(np.float32)), nsamples=3).numpy()
                    collision = signed < -tolerance
                    methods.append("signed")
                else:
                    collision = near & (signed < -tolerance)
                    methods.append("local")
                contact = (dist <= tolerance) & opposed & ~collision
                counts.append((int(contact.sum()), int(collision.sum())))
                near_counts.append(int((near | collision).sum()))
                separation.extend(dist[contact])
                depths.extend(-signed[collision])
                for name, mask in (("contact", contact), ("collision", collision)):
                    indices = np.flatnonzero(mask)
                    stride = max(1, int(np.ceil(len(indices) / 450)))
                    overlays[key][name].extend(local[indices[::stride]].tolist())
            pairs.append({
                "a": a, "b": b, "near_samples": sum(near_counts),
                "contact_samples": sum(x[0] for x in counts), "collision_samples": sum(x[1] for x in counts),
                "contact_coverage": [counts[i][0] / len(samples[key][0]) for i, key in enumerate((a, b))],
                "gap_rmse": float(np.sqrt(np.mean(np.square(separation)))) if separation else None,
                "collision_p95": float(np.percentile(depths, 95)) if depths else None,
                "method": "signed" if all(method == "signed" for method in methods) else "local",
            })
        summaries.append(pairs)
    return summaries, overlays


def normalise(p):
    p = p - p.mean(0)
    _, basis = np.linalg.eigh(p.T @ p)
    if np.linalg.det(basis) < 0:
        basis[:, 0] *= -1
    p = p @ basis
    return p / max(np.ptp(p, axis=0).max(), 1.e-12)


@lru_cache(maxsize=1)
def templates():
    paths = [(path.stem.split("-")[-1].split("_")[0], path) for path in sorted(ATLAS.glob("ABH074-*.obj"))]
    additional = {"Radius", "Fibula", "Clavicle", "Scapula", "Hip", "Sacrum"}
    paths.extend((kind, path) for path in sorted(BONE_HUB.rglob("*.stl"))
                 if (kind := path.stem.split("_")[0].title()) in additional)
    output = []
    for kind, path in paths:
        if kind in NAMES:
            mesh = trimesh.load(path, process=False, force="mesh", skip_materials=True)
            if not len(mesh.faces):
                continue
            points, _ = trimesh.sample.sample_surface(mesh, 1800, seed=22)
            points = normalise(points)
            output.append((kind, points, cKDTree(points)))
    return output


def atlas_suggestion(meshes, poses, group):
    if len(group) < 2:
        return None
    points = normalise(np.concatenate([
        moved(sample_surface(meshes[int(key) - 1][1], 350, np.random.default_rng(int(key)))[0], poses[key])
        for key in group
    ]))
    scores = {}
    for kind, reference, tree in templates():
        for signs in product((-1, 1), repeat=3):
            if np.prod(signs) != 1:
                continue
            transformed = points * signs
            forward = tree.query(transformed)[0]
            reverse = cKDTree(transformed).query(reference)[0]
            score = float(.55 * np.quantile(forward, .70) + .45 * np.quantile(reverse, .70))
            scores[kind] = min(scores.get(kind, float("inf")), score)
    if not scores:
        return None
    ranked = sorted(scores, key=scores.get)
    first = ranked[0]
    second = scores[ranked[1]] if len(ranked) > 1 else scores[first]
    return {"suggestion": NAMES[first], "clear": scores[first] < .070 and second - scores[first] > .012,
            "distance": scores[first]}


def reference_group(meshes, poses, groups, template):
    """Find the assembled group with strongest two-way support from a complete scan."""
    rng = np.random.default_rng(71)
    reference = normalise(sample_surface(template, 3600, rng)[0])
    tree = cKDTree(reference)
    flips = [signs for signs in product((-1, 1), repeat=3) if np.prod(signs) == 1]
    ranked = []
    for group in groups:
        if len(group) < 2:
            continue
        parts = [meshes[int(key) - 1][1] for key in group]
        areas = np.array([part.get_surface_area() for part in parts])
        if areas.sum() <= 0:
            continue
        points = normalise(np.concatenate([
            moved(sample_surface(part, max(20, round(3600 * share)), rng)[0], poses[key])
            for key, part, share in zip(group, parts, areas / areas.sum())
        ]))
        fit = []
        for signs in flips:
            aligned = points * signs
            forward = tree.query(aligned)[0]
            reverse = cKDTree(aligned).query(reference)[0]
            fit.append(float(max(np.quantile(forward, .80), np.quantile(reverse, .80))))
        ranked.append((min(fit), group))
    ranked.sort()
    if not ranked or ranked[0][0] > .04 or (len(ranked) > 1 and ranked[1][0] - ranked[0][0] < .01):
        raise ValueError("Pełny skan nie pozwolił jednoznacznie wybrać złożonej grupy.")
    return ranked[0][1]

"""Adapter between szpont's vertex labels and OSTEOFORM's rigid registration."""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from contacts import triangle_labels

sys.path.append(str(Path(__file__).resolve().parents[2]))
from app_v6 import pair, prepare  # noqa: E402
from osteo_engine_v6 import moved, unit  # noqa: E402


def sample_surface(mesh, count, rng, mask=None):
    indices = np.asarray(mesh.triangles)
    if mask is not None:
        indices = indices[mask]
    triangles = np.asarray(mesh.vertices)[indices]
    a, b, c = triangles.transpose(1, 0, 2)
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1)
    if not len(area) or not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError("Zaznaczenie nie obejmuje powierzchni siatki. Zamaluj pełny obszar przełomu.")
    selected = rng.choice(len(area), size=count, p=area / area.sum())
    u, v = rng.random((2, count))
    fold = u + v > 1
    u[fold], v[fold] = 1 - u[fold], 1 - v[fold]
    a, b, c = a[selected], b[selected], c[selected]
    points = a + u[:, None] * (b - a) + v[:, None] * (c - a)
    return points, unit(cross[selected]), float(area.sum() / 2)


def prepare_fragment(bone, labels, surface, seed):
    mesh = bone.mesh  # preserve szpont's coordinates and winding
    rng = np.random.default_rng(seed)
    full, normals, _ = sample_surface(mesh, 2800, rng)
    mask = triangle_labels(mesh, labels) == surface
    patch, patch_normals, area = sample_surface(mesh, 1800, rng, mask)
    return (full, normals, [(patch, patch_normals)]), area


def display_mesh(mesh, limit=4200, *, include_colors=False):
    if limit is not None and len(mesh.triangles) > limit:
        mesh = mesh.simplify_quadric_decimation(limit)
    vertices = np.asarray(mesh.vertices)
    if limit is not None:
        vertices = vertices.round(5)
    payload = {"vertices": vertices.tolist(), "triangles": np.asarray(mesh.triangles).tolist()}
    if include_colors and mesh.has_vertex_colors():
        rgb = np.rint(np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
        payload["vertex_colors"] = ["#%02x%02x%02x" % tuple(color) for color in rgb]
    return payload


def contact_distances(a, na, b, nb, transform, tolerance):
    b = moved(b, transform)
    nb = nb @ transform[:3, :3].T
    distances = []
    fractions = []
    for source, ns, target, nt in ((a, na, b, nb), (b, nb, a, na)):
        d, i = cKDTree(target).query(source)
        contact = (d <= tolerance) & (np.einsum("ij,ij->i", ns, nt[i]) < -.866)
        distances.append(d[contact])
        fractions.append(float(contact.mean()))
    distances = np.concatenate(distances)
    return distances, fractions


def match_surfaces(a, labels_a, surface_a, b, labels_b, surface_b):
    """Return ranked B→A matrices. A remains fixed in the scan's original units."""
    if a.id == b.id:
        raise ValueError("Wybierz dwa różne fragmenty.")
    first, area_a = prepare_fragment(a, labels_a, surface_a, 17)
    second, area_b = prepare_fragment(b, labels_b, surface_b, 42)
    patch_a, normals_a = first[2][0]
    patch_b, normals_b = second[2][0]
    voxel = min(np.linalg.norm(np.ptp(p, axis=0)) for p in (patch_a, patch_b)) / 33
    if not np.isfinite(voxel) or voxel <= 0:
        raise ValueError("Zaznaczony przełom jest zbyt mały.")
    tolerance = 1.5 * voxel
    work = prepare({"A": first, "B": second}, voxel)
    if min(len(work[name][2][0][1]) for name in work) < 30:
        raise ValueError("Za mało próbek. Powiększ zaznaczenie przełomu.")

    ranked = pair("A", "B", work, voxel, "roi")
    result = []
    for candidate in ranked[:6]:
        matrix = np.asarray(candidate["T"])
        distances, (fraction_a, fraction_b) = contact_distances(
            patch_a, normals_a, patch_b, normals_b, matrix, tolerance
        )
        result.append({
            "matrix": candidate["T"], "eligible": candidate["eligible"],
            "score": candidate["score"], "penetration": candidate["penetration"],
            "contact_a": fraction_a, "contact_b": fraction_b,
            "contact_area": (fraction_a * area_a + fraction_b * area_b) / 2,
            "contact_rmse": float(np.sqrt(np.mean(distances ** 2))) if len(distances) else None,
            "contact_p95": float(np.percentile(distances, 95)) if len(distances) else None,
        })
    return {
        "a": {"label": a.label, **display_mesh(a.mesh), "patch": patch_a[::3].tolist()},
        "b": {"label": b.label, **display_mesh(b.mesh), "patch": patch_b[::3].tolist()},
        "tolerance": tolerance,
        "candidates": result,
    }


def evaluate_reference(result, bone_a, bone_b, poses):
    """Evaluate after ranking. Both sets of poses are expressed relative to A."""
    if bone_a not in poses or bone_b not in poses:
        return
    a = np.asarray(poses[bone_a])
    b = np.asarray(poses[bone_b])
    reference = np.linalg.solve(a, b)
    points = np.asarray(result["b"]["vertices"])
    target = moved(points, reference)
    for candidate in result["candidates"]:
        matrix = np.asarray(candidate["matrix"])
        displacement = moved(points, matrix) - target
        rotation = Rotation.from_matrix(matrix[:3, :3] @ reference[:3, :3].T).magnitude()
        candidate["reference"] = {
            "pose_rmse_mm": float(np.sqrt(np.mean(np.sum(displacement ** 2, axis=1)))),
            "rotation_deg": float(np.degrees(rotation)),
        }

"""Build a local, reproducible assembly example from the complete fibula scan."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from bones import read_mesh
from case_inspection import inspect_groups
from case_solver import wire_parts
from osteo_engine_v6 import moved, rigid


def cut_surface(vertices, faces):
    """Keep every original triangle and find the common border of each adjacent part."""
    centered = vertices - vertices.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    direction = axes[0] * (1 if axes[0, 0] >= 0 else -1)
    along = centered @ direction
    cuts = along.min() + np.ptp(along) * np.array([.25, .50, .75])
    labels = np.searchsorted(cuts, along[faces].mean(axis=1))
    pieces = [faces[labels == i] for i in range(4)]
    seams = []
    for left, right in zip(pieces, pieces[1:]):
        common = np.intersect1d(left, right)
        edges = left[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        edges = edges[np.isin(edges, common).all(axis=1)]
        _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
        border = edges[counts[inverse] == 1]
        if len(border) < 6:
            raise ValueError("Za mało punktów na granicy podziału skanu.")
        seams.append(border)
    return pieces, seams


def parts_from_scan(mesh):
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    pieces, seams = cut_surface(vertices, np.asarray(mesh.triangles))
    rng = np.random.default_rng(43)
    parts, inputs = [], []
    for i, piece in enumerate(pieces):
        ids, faces = np.unique(piece, return_inverse=True)
        points = vertices[ids]
        rgb = colors[ids]
        triangles = faces.reshape(-1, 3)
        borders = []
        if i:
            borders.append(seams[i - 1])
        if i < len(seams):
            borders.append(seams[i][:, ::-1])
        for border in borders:
            # Add the same cut surface on both sides, with opposite winding.
            local = np.searchsorted(ids, border)
            cap = np.c_[local, np.full(len(border), len(points))]
            points = np.vstack((points, vertices[np.unique(border)].mean(axis=0)))
            rgb = np.vstack((rgb, [.73, .65, .53]))
            triangles = np.vstack((triangles, cap))
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(rng.uniform(-.5, .5, 3)).as_matrix()
        pose[:3, 3] = -pose[:3, :3] @ points.mean(axis=0)
        scan = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(moved(points, pose)),
                                        o3d.utility.Vector3iVector(triangles))
        scan.vertex_colors = o3d.utility.Vector3dVector(rgb)
        scan.compute_triangle_normals()
        parts.append((f"Odłamek {i + 1}", scan))
        inputs.append(pose)
    return parts, inputs, seams


def prepare(source, output):
    mesh = read_mesh(source)
    if not mesh.has_vertex_colors():
        raise ValueError("Przykład potrzebuje źródłowego PLY z zapisanymi kolorami.")
    parts, inputs, seams = parts_from_scan(mesh)
    points = np.asarray(mesh.vertices)
    poses = {"1": np.linalg.inv(inputs[0])}
    for i, border in enumerate(seams, 1):
        shared = points[np.unique(border)]
        local_a, local_b = moved(shared, inputs[i - 1]), moved(shared, inputs[i])
        poses[str(i + 1)] = poses[str(i)] @ rigid(local_a, local_b)
    group = [tuple(poses)]
    max_rmse, max_rotation = 0., 0.
    for (label, part), initial, calculated in zip(parts, inputs, poses.values()):
        sample = np.asarray(part.vertices)[::8]
        relative = calculated @ initial
        delta = moved(sample, relative) - sample
        max_rmse = max(max_rmse, np.sqrt(np.mean(np.einsum("ij,ij->i", delta, delta))))
        max_rotation = max(max_rotation, np.degrees(Rotation.from_matrix(relative[:3, :3]).magnitude()))
        print(f"{label}, {len(part.triangles)} trójkątów", flush=True)
    if max_rmse > 1e-5 or max_rotation > 1e-5:
        raise ValueError("Złożenie odbiega od kontrolnego położenia we wspólnym skanie.")

    tolerance = .85
    summaries, overlays = inspect_groups(parts, poses, group, tolerance)
    preview = wire_parts(parts, poses, group, overlays, detail=18000)
    output.mkdir(parents=True, exist_ok=True)
    for part in preview:
        key = part["id"]
        name = f"fibula-fragment-{key}.ply"
        scan = parts[int(key) - 1][1]
        if not o3d.io.write_triangle_mesh(str(output / name), scan, write_vertex_normals=False):
            raise OSError(f"Nie udało się zapisać {name}.")
        url = f"/api/demo/fibula/scan/{key}"
        part["uploaded_scan"] = {"label": name, "url": url, "download": url + "?download=1"}
    report = {
        "title": "Nasze skany · Kość strzałkowa",
        "case_note": (
            "Kość strzałkowa to długa, cienka kość po zewnętrznej stronie podudzia. "
            "W próbie kontrolnej cztery odłamki wydzielono cyfrowo z pełnego skanu i złożono wzdłuż wspólnych krawędzi."
        ),
        "bone_type": {"name": "Kość strzałkowa", "source": f"Oznaczenie skanu {source.name}"},
        "reference_check": {"rmse": float(max_rmse), "rotation_deg": float(max_rotation)},
        "reference_scan_url": "/scan-gallery",
        "parts": preview, "total": len(parts), "placed": len(parts), "groups": 1, "pairs": len(seams),
        "diagnostics": summaries[0], "tolerance": tolerance,
    }
    pending = output / "result.pending"
    pending.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    pending.replace(output / "result.json")
    print(f"RMSE {max_rmse:.3g}, błąd obrotu {max_rotation:.3g}°, dane {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Pełny skan PLY z kolorami")
    parser.add_argument("--out", type=Path, default=PROJECT / "out" / "fibula-demo")
    args = parser.parse_args()
    prepare(args.source, args.out)

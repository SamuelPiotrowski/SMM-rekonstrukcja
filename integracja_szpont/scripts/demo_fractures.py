"""Prepare two synthetic scans and their marked contact surfaces."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

LOCAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOCAL_ROOT / "src"))

from bones import Bone, read_mesh, write_mesh  # noqa: E402
from contacts import Contacts, save_contacts  # noqa: E402


def make_fragment(side, incomplete=False):
    size = 31
    grid = np.linspace(-14, 14, size)
    y, z = np.meshgrid(grid, grid, indexing="ij")
    x = (1.8 * np.sin(0.41 * y + 0.22 * z) + 1.4 * np.cos(0.54 * z - 0.19 * y)
         + 0.012 * y * z + 1.1 * np.sin(0.75 * y - 0.70 * z))
    patch = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    n = len(patch)
    back = patch.copy()
    back[:, 0] = side * 12
    vertices = np.vstack((patch, back))
    faces = []
    for i in range(size - 1):
        for j in range(size - 1):
            p = i * size + j
            faces.extend(((p, p + size, p + size + 1), (p, p + size + 1, p + 1)))
    cap = np.array(faces)
    if side > 0:
        cap = cap[:, ::-1]
    boundary = (list(range(size)) + list(range(2 * size - 1, n, size))
                + list(range(n - 2, n - size - 1, -1))
                + list(range(n - 2 * size, 0, -size)))
    wall = []
    for p, q in zip(boundary, boundary[1:] + boundary[:1]):
        wall.extend(((p, q, p + n), (q, q + n, p + n)))
    if side < 0:
        wall = np.asarray(wall)[:, ::-1]
    complete = np.vstack((cap, cap[:, ::-1] + n, wall))
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(complete)
    )
    if not mesh.orient_triangles():
        raise ValueError("Nie można orientować ścian przykładowej siatki.")
    oriented = np.asarray(mesh.triangles)
    a, b, c = vertices[oriented].transpose(1, 0, 2)
    if np.einsum("ij,ij->", a, np.cross(b, c)) < 0:
        oriented = oriented[:, ::-1]
    keep = np.ones(len(complete), dtype=bool)
    if incomplete:
        centers = patch[cap].mean(axis=1)
        keep[:len(cap)] = ~((centers[:, 1] < -4) & (centers[:, 2] > 1))
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(oriented[keep]),
    )
    return mesh, patch


def example(loss=False):
    a, patch_a = make_fragment(-1)
    b, patch_b = make_fragment(1, incomplete=loss)
    scatter = np.eye(4)
    scatter[:3, :3] = Rotation.from_euler("xyz", [31, -27, 62], degrees=True).as_matrix()
    scatter[:3, 3] = [34, -18, 15]
    b.transform(scatter)
    patch_b = patch_b @ scatter[:3, :3].T + scatter[:3, 3]
    kind = "ubytek" if loss else "czysty"
    return [(f"demo-{kind}-a", "Fragment A", a, patch_a),
            (f"demo-{kind}-b", "Fragment B", b, patch_b)], np.linalg.inv(scatter)


def write_fragment(out, bone_id, label, mesh, patch=None):
    target = out / "bones" / bone_id
    target.mkdir(parents=True, exist_ok=True)
    mesh_file = target / "mesh.ply"
    write_mesh(mesh_file, mesh)
    (target / "meta.json").write_text(json.dumps({"label": label}, ensure_ascii=False), encoding="utf-8")
    loaded = read_mesh(mesh_file)
    if patch is not None:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(patch).query(np.asarray(loaded.vertices))
        labels = (distances < 1e-4).astype(np.uint8)
        save_contacts(loaded, Contacts(bone_id, labels, {1: "Przełom"}))
    return Bone(bone_id, label, loaded, np.zeros(len(loaded.triangles), dtype=np.int32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss", action="store_true", help="usuń część powierzchni ze skanu B")
    args = parser.parse_args()
    out = LOCAL_ROOT / "out" / ("demo-ubytek" if args.loss else "demo-czysty")
    fragments, reference = example(args.loss)
    for fragment in fragments:
        target = out / "bones" / fragment[0] / "mesh.ply"
        if target.exists():
            raise SystemExit(f"Przykład już przygotowany: {out}")
    for fragment in fragments:
        write_fragment(out, *fragment)
    poses = {fragments[0][0]: np.eye(4).tolist(), fragments[1][0]: reference.tolist()}
    (out / "reference_poses.json").write_text(json.dumps(poses), encoding="utf-8")
    print(f"Gotowy przykład: {out}")


if __name__ == "__main__":
    main()

"""Import selected PENGWIN fragments into szpont without prelabelling their contact surfaces."""

import argparse
import json
import re

import numpy as np
import open3d as o3d

from demo_fractures import LOCAL_ROOT, write_fragment


PENGWIN = (LOCAL_ROOT.parent / "02_FRAGMENTY_ZLAMANE" / "PENGWIN"
           / "PENGWIN26_task3_clinical_fractures_train" / "mesh")
GROUPS = {"sacrum": (1, 100), "left": (101, 200), "right": (201, 300)}


def read_fragment_groups(path, lower, upper):
    """The dataset stores globally indexed vertices and named numeric OBJ groups."""
    vertices, groups = [], {}
    active = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            tag, _, value = line.strip().partition(" ")
            if tag == "v":
                vertices.append([float(x) for x in value.split()[:3]])
            elif tag == "g":
                key = value.split()[0] if value.split() else ""
                active = key if key.isdecimal() and lower <= int(key) <= upper else None
            elif tag == "f" and active is not None:
                indices = [int(x.split("/", 1)[0]) for x in value.split()]
                indices = [i - 1 if i > 0 else len(vertices) + i for i in indices]
                groups.setdefault(active, []).extend(
                    (indices[0], indices[j], indices[j + 1]) for j in range(1, len(indices) - 1)
                )
    vertices = np.asarray(vertices)
    meshes = {}
    for key, faces in sorted(groups.items(), key=lambda item: int(item[0])):
        unique, local = np.unique(np.asarray(faces), return_inverse=True)
        meshes[key] = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices[unique]),
                                               o3d.utility.Vector3iVector(local.reshape(-1, 3)))
    return meshes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="002")
    parser.add_argument("--bone", choices=GROUPS, default="right")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]{3}", args.case):
        parser.error("Identyfikator przypadku składa się z trzech cyfr.")
    source = PENGWIN / args.case
    meshes = read_fragment_groups(source / "peripelvic-fracture-fragments.obj", *GROUPS[args.bone])
    if len(meshes) < 2:
        raise SystemExit("Wybrana kość ma w tym przypadku mniej niż dwa fragmenty.")
    output = LOCAL_ROOT / "out" / f"pengwin-{args.case}-{args.bone}"
    ids = {number: f"{output.name}-{number}" for number in meshes}
    if any((output / "bones" / bone_id / "mesh.ply").exists() for bone_id in ids.values()):
        raise SystemExit(f"Przypadek już przygotowany: {output}")
    for number, mesh in meshes.items():
        write_fragment(output, ids[number], f"PENGWIN {args.case} · fragment {number}", mesh)
    reference = json.loads((source / "plan_pl_gt.json").read_text(encoding="utf-8"))
    poses = {ids[item["fragment_id"]]: item["transformation"]
             for item in reference if item["fragment_id"] in ids}
    (output / "reference_poses.json").write_text(json.dumps(poses), encoding="utf-8")
    print(f"Zaimportowano {len(meshes)} fragmenty: {output}")


if __name__ == "__main__":
    main()

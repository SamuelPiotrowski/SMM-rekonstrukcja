"""Browse the fracture meshes already stored on this computer."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import open3d as o3d

from bones import Bone

DATA = Path(__file__).resolve().parents[2] / "02_FRAGMENTY_ZLAMANE"
PENGWIN = DATA / "PENGWIN" / "PENGWIN26_task3_clinical_fractures_train" / "mesh"
FRACTURA = DATA / "FRACTURA_Bones"
GROUPS = {"sacrum": (1, 100), "left": (101, 200), "right": (201, 300)}
BONE_NAMES = {"sacrum": "kość krzyżowa", "left": "lewa kość miedniczna", "right": "prawa kość miedniczna"}
START_CASE = "pengwin-002-right"


def read_fragment_groups(path, lower, upper):
    vertices, groups = [], {}
    active = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            tag, _, value = line.strip().partition(" ")
            if tag == "v":
                vertices.append([float(x) for x in value.split()[:3]])
            elif tag == "g":
                fields = value.split()
                key = fields[0] if fields else ""
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
        meshes[key] = mesh_from_arrays(vertices[unique], local.reshape(-1, 3))
    return meshes


def mesh_from_arrays(points, faces):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(points),
                                     o3d.utility.Vector3iVector(faces))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


@lru_cache(maxsize=1)
def catalog():
    import h5py

    cases = {}
    for path in sorted(PENGWIN.glob("*/plan_pl_gt.json")):
        if not (path.parent / "peripelvic-fracture-fragments.obj").is_file():
            continue
        case = path.parent.name
        numbers = [int(item["fragment_id"]) for item in json.loads(path.read_text(encoding="utf-8"))]
        for side, (lower, upper) in GROUPS.items():
            count = sum(lower <= number <= upper for number in numbers)
            if count >= 2:
                key = f"pengwin-{case}-{side}"
                cases[key] = (dict(id=key, source="PENGWIN", label=f"{case} · {BONE_NAMES[side]}",
                                   count=count, reference=True), (path.parent, side))

    animal = FRACTURA / "Fractura" / "bone_real.hdf5"
    if not animal.is_file():
        animal = FRACTURA / "bone_real.hdf5"
    human = FRACTURA / "fractura_real-001.hdf5"
    for file_path, prefix, group in ((human, "human", "bones"), (animal, "animal", "pig_bone")):
        if not file_path.is_file():
            continue
        with h5py.File(file_path, "r") as file:
            for name, entry in file[group].items():
                if "pieces" not in entry or len(entry["pieces"]) < 2:
                    continue
                key = f"fractura-{prefix}-{name}"
                label = f"{name} · {'kości' if prefix == 'human' else 'kość zwierzęca'}"
                cases[key] = (dict(id=key, source="FRACTURA", label=label,
                                   count=len(entry["pieces"]), reference=False), (file_path, entry.name))
    return cases


def load_case(case):
    entry, location = catalog()[case]
    if entry["source"] == "PENGWIN":
        folder, side = location
        meshes = read_fragment_groups(folder / "peripelvic-fracture-fragments.obj", *GROUPS[side])
        data = json.loads((folder / "plan_pl_gt.json").read_text(encoding="utf-8"))
        poses = {f'{case}-{item["fragment_id"]}': item["transformation"] for item in data
                 if str(item["fragment_id"]) in meshes}
    else:
        import h5py

        file_path, group = location
        with h5py.File(file_path, "r") as file:
            meshes = {key: mesh_from_arrays(piece["vertices"][:], piece["faces"][:])
                      for key, piece in file[group]["pieces"].items()}
        poses = {}
    bones = [Bone(f"{case}-{key}", f"Fragment {key}", mesh, np.empty(0, dtype=np.int32),
                  meta={"stats": {"area_mm2": mesh.get_surface_area()}}) for key, mesh in meshes.items()]
    return bones, poses

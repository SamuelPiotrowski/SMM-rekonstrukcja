"""Regenerate the local FRACTURA three-piece comparison."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import open3d as o3d
import h5py
import numpy as np

from case_solver import demo_input, make_result, measure_reference, wire_parts
from fracture_library import FRACTURA


def save(directory, mode, result):
    path = directory / f"{mode}.json"
    scratch = path.with_suffix(".json.tmp")
    scratch.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    scratch.replace(path)
    print(mode, {key: val for key, val in result.items() if key not in ("parts", "sections", "diagnostics", "poses")}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-only", action="store_true")
    parser.add_argument("--open3d-only", action="store_true", help="przelicz demonstrację FPFH/RANSAC/ICP")
    args = parser.parse_args()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    folder = ROOT / "out" / "benchmark-3"
    folder.mkdir(exist_ok=True)
    if args.mixed_only:
        save(folder, "mixed", mixed_benchmark())
        return
    title = "Kość w trzech fragmentach · FRACTURA"
    meshes, labels, reference = demo_input()
    save(folder, "open3d", make_result("Składanie Open3D · trzy fragmenty", meshes,
                                       reference=reference, masks=labels, open3d_only=True))
    if args.open3d_only:
        return
    for mode, masks in (("auto", None), ("labelled", labels)):
        save(folder, mode, make_result(title, meshes, reference=reference, masks=masks))
    save(folder, "reference", dict(title="FRACTURA · ułożenie referencyjne",
                                   parts=wire_parts(meshes, reference, [tuple(reference)]),
                                   total=3, placed=3, groups=1, pairs=0, reference_view=True))
    save(folder, "mixed", mixed_benchmark())


def mixed_benchmark():
    def joined(g):
        if "mode_2/pieces" not in g or len(g["mode_2/pieces"]) != 3:
            return False
        pieces = g["mode_2/pieces"]
        # Select a benchmark where all pieces have a labelled contact in the dataset.
        # These labels are only used to choose/evaluate the fixture.
        return all((part["shared_faces"][:] >= 0).any() for part in pieces.values())

    models = []
    with h5py.File(FRACTURA / "Fractura" / "bone_synthetic.hdf5", "r") as file:
        for family in ("leg", "hip"):
            models.extend((family, key) for key, group in file[f"{family}/synthetic_fracture"].items()
                          if joined(group))
    chosen = models[:4] + [item for item in models if item[0] == "hip"][:4]
    fragments = []
    for index, (family, model) in enumerate(chosen):
        pieces, _, reference = demo_input(model, family, 120 + index)
        fragments.extend((index, mesh, reference[str(j)])
                         for j, (_, mesh) in enumerate(pieces if index < 6 else pieces[:1], 1))
    order = np.random.default_rng(317).permutation(len(fragments))
    files = [(f"Fragment {i}", fragments[j][1]) for i, j in enumerate(order, 1)]
    wanted = [{str(i) for i, j in enumerate(order, 1) if fragments[j][0] == index}
              for index in range(8)]
    result = make_result("FRACTURA · 20 wymieszanych fragmentów", files, mixed=True, progress=print)
    groups = [set(g["ids"]) for g in result["sections"] if not g.get("unassigned")]
    loose = {identifier for g in result["sections"] if g.get("unassigned") for identifier in g["ids"]}
    poses = {key: np.asarray(matrix) for key, matrix in result["poses"].items()}
    reference = {str(i): fragments[j][2] for i, j in enumerate(order, 1)}
    errors = [error for group in groups if group in wanted[:6]
              for error in measure_reference(files, poses, [sorted(group)], reference)]
    result["benchmark"] = {"correct_groups": sum(group in groups for group in wanted[:6]),
                           "expected_groups": 6, "correct_unassigned": len(loose & (wanted[6] | wanted[7])),
                           "expected_unassigned": 2, "false_groups": sum(group not in wanted[:6] for group in groups),
                           "worst_pose_rmse_percent": max((e["rmse_percent"] for e in errors), default=None),
                           "worst_rotation_deg": max((e["angle_deg"] for e in errors), default=None)}
    print("Wynik grupowania:", result["benchmark"], flush=True)
    return result


if __name__ == "__main__":
    main()

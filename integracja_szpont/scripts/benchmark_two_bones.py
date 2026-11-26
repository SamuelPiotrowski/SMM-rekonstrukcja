"""Mix the femur fixture and the fibula scan and compare their recovered poses."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark_real import ROOT, measure, preview, write_report

import numpy as np
import open3d as o3d

from bones import read_mesh
from case_solver import outward_mesh, solve_case
from prepare_fibula_demo import parts_from_scan


CASE = "udowa-strzalkowa"


def load_inputs(femur, fibula_scan, fibula_parts, seed):
    femur_reference = json.loads((femur / "reference" / "ground_truth.json").read_text(encoding="utf-8"))
    femur_files = sorted((femur / "input" / "ply").glob("fragment_*.ply"))
    fibula_files = sorted(fibula_parts.glob("fibula-fragment-*.ply"))
    if len(femur_files) != 6 or len(fibula_files) != 4:
        raise ValueError("Potrzeba sześciu PLY udowej i czterech przygotowanych PLY strzałkowej.")
    derived, input_poses, _ = parts_from_scan(read_mesh(fibula_scan))
    entries = [(path, outward_mesh(read_mesh(path)), np.asarray(femur_reference[path.stem]), 0)
               for path in femur_files]
    for path, (_, expected_mesh), pose in zip(fibula_files, derived, input_poses):
        mesh = read_mesh(path)
        if (np.asarray(mesh.vertices).shape != np.asarray(expected_mesh.vertices).shape
                or not np.allclose(mesh.vertices, expected_mesh.vertices, rtol=0., atol=1.e-4)):
            raise ValueError("Odłamki strzałkowej nie odpowiadają podanemu pełnemu skanowi.")
        entries.append((path, outward_mesh(mesh), np.linalg.inv(pose), 1))
    order = np.random.default_rng(seed).permutation(len(entries))
    sources, meshes, reference = [], [], {}
    expected = [[], []]
    for i, index in enumerate(order, 1):
        path, mesh, pose, bone = entries[index]
        name = f"fragment_{i:02d}"
        sources.append((name + ".ply", path))
        meshes.append((name, mesh))
        reference[str(i)] = pose
        expected[bone].append(str(i))
    return sources, meshes, reference, expected


def run(args):
    sources, meshes, reference, expected = load_inputs(
        args.femur, args.fibula_scan, args.fibula_parts, args.seed)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(output / "dwie-kosci-10-fragmentow.zip", "w", compression=ZIP_DEFLATED) as files:
        for name, path in sources:
            files.write(path, arcname=name)
    started = perf_counter()
    poses, groups, links, comparisons, _ = solve_case(meshes, lambda message: print(message, flush=True), mixed=True)
    elapsed = perf_counter() - started
    metrics = measure(meshes, poses, groups, reference, expected)
    # Compare models in the same orientation, aligning each recovered group independently.
    display_poses = {}
    areas = {str(i): mesh.get_surface_area() for i, (_, mesh) in enumerate(meshes, 1)}
    for group in groups:
        anchor = max(group, key=areas.get)
        alignment = reference[anchor] @ np.linalg.inv(poses[anchor])
        display_poses.update((key, alignment @ poses[key]) for key in group)
    summary = {
        "test": CASE, "title": "Kość udowa i kość strzałkowa · 10 fragmentów",
        "reference_kind": "local",
        "source_note": (
            "Sześć fragmentów pochodzi z syntetycznego modelu kości udowej. Cztery odłamki wydzielono "
            "cyfrowo z pełnego skanu strzałkowej. Wymieszane PLY mają neutralne nazwy. "
            "Podział na kości i położenia referencyjne służą do oceny po obliczeniu dopasowania."
        ),
        "groups": groups, "expected": expected, "fragments": [name for name, _ in meshes],
        "comparisons": comparisons, "seconds": round(elapsed, 1),
        "links": [[edge["a"], edge["b"]] for edge in links],
        "predicted_meshes": preview(meshes, display_poses, groups),
        "reference_meshes": preview(meshes, reference, expected),
        **metrics,
    }
    (output / f"{CASE}.json").write_text(
        json.dumps(summary, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    write_report(output, case_ids=(CASE,))
    print(json.dumps({"groups": groups, "expected": expected, "seconds": summary["seconds"], **metrics}),
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("femur", type=Path, help="Katalog femur_6_fragments_3D")
    parser.add_argument("fibula_scan", type=Path, help="Pełny skan PLY użyty do przygotowania odłamków")
    parser.add_argument("--fibula-parts", type=Path, default=ROOT / "out" / "fibula-demo")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out" / "test-udowa-strzalkowa")
    parser.add_argument("--seed", type=int, default=41)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    run(parser.parse_args())

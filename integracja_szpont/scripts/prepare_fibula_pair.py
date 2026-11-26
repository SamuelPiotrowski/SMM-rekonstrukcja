"""Build the local test of two fibulae from independently shuffled fragments."""

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

os.environ.setdefault("OMP_NUM_THREADS", "4")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np

from benchmark_real import measure, preview, write_report
from assembly_layout import combined_layout
from bones import read_mesh
from case_solver import make_result
from prepare_fibula_demo import parts_from_scan


CASE = "dwie-strzalkowe"
METHODS = {"common": "wspólna triangulacja przełomu", "grid": "przeciwne powierzchnie i brak przenikania wnętrz na siatce",
           "near": "zgodne krawędzie i styk w układzie wejściowym", "surface": "zgodność powierzchni i normalnych"}


def prepare(synthetic, scan, directory):
    originals = sorted((PROJECT / "out" / "fibula-demo").glob("fibula-fragment-*.ply"))
    generated = sorted((synthetic / "input" / "ply").glob("fragment_*.ply"))
    if len(originals) != 4 or len(generated) != 7:
        raise ValueError("Zestaw potrzebuje czterech odłamków pełnego skanu i siedmiu z fibula_fracture_v2.")
    paths = originals + generated
    order = np.random.default_rng(77).permutation(len(paths))
    names = [f"fragment_{i:02d}" for i in range(1, len(order) + 1)]
    meshes = [(name, read_mesh(paths[j])) for name, j in zip(names, order)]
    directory.mkdir(parents=True, exist_ok=True)
    with ZipFile(directory / "dwie-strzalkowe-11-odlamkow.zip", "w", compression=ZIP_DEFLATED) as archive:
        for name, j in zip(names, order):
            archive.write(paths[j], arcname=name + ".ply")

    start = perf_counter()
    result = make_result("Test nr 2 · Dwie kości strzałkowe", meshes, mixed=True, upload=True, identify_bones=False,
                         progress=lambda message: print(message, flush=True))
    seconds = round(perf_counter() - start, 1)
    groups = [section["ids"] for section in result["sections"]]
    expected = [[str(i) for i, j in enumerate(order, 1) if (j < len(originals)) == from_scan]
                for from_scan in (True, False)]
    group_sets = {frozenset(group): index for index, group in enumerate(expected)}
    if {frozenset(group) for group in groups} != group_sets.keys() or any(s.get("unassigned") for s in result["sections"]):
        raise ValueError(f"Algorytm zwrócił grupy {groups}, oczekiwano {expected}.")

    derived, initial, _ = parts_from_scan(read_mesh(scan))
    for path, (_, mesh) in zip(originals, derived):
        if not np.allclose(np.asarray(read_mesh(path).vertices), np.asarray(mesh.vertices), rtol=0, atol=1.e-4):
            raise ValueError("Podany pełny skan nie odpowiada przygotowanym odłamkom.")
    truth = json.loads((synthetic / "reference" / "ground_truth.json").read_text(encoding="utf-8"))
    poses = {key: np.asarray(matrix) for key, matrix in result["poses"].items()}
    reference = {str(i): np.linalg.inv(initial[j]) if j < len(initial) else np.asarray(truth[paths[j].stem])
                 for i, j in enumerate(order, 1)}
    metrics = measure(meshes, poses, groups, reference, expected)
    if max(error["rmse"] for error in metrics["errors"]) > .05:
        raise ValueError(f"Ułożenie odłamków odbiega od referencji: {metrics['errors']}")

    result["sections"].sort(key=lambda section: group_sets[frozenset(section["ids"])])
    details = []
    for section in result["sections"]:
        ids = set(section["ids"])
        index = group_sets[frozenset(ids)]
        kind = ("Kość strzałkowa A · skan", "Kość strzałkowa B · model")[index]
        section["bone_type"] = {"name": kind, "source": "Skład zestawu potwierdzony w porównaniu z referencją"}
        links = [link for link in result["links"] if set(link["ids"]) <= ids]
        methods = ", ".join(METHODS[method] for method in sorted({link["method"] for link in links}))
        details.append({"label": kind, "id": section["ids"][0],
                        "reason": f"Potwierdzone połączenia w grupie · {len(links)}",
                        "detail": f"Podstawa dopasowania · {methods}. Wybrane połączenia dotyczą fragmentów tej grupy."})
        for part in section["parts"]:
            key = part["id"]
            url = f"/api/demo/fibula-pair/scan/{key}"
            part["uploaded_scan"] = {"label": names[int(key) - 1] + ".ply", "url": url, "download": url + "?download=1"}
    result["combined_layout"] = combined_layout(result["sections"])
    result["separation"] = details
    result["case_note"] = "Jedenaście wymieszanych odłamków tworzy dwie grupy kości tego samego typu."
    temporary = directory / "result.pending"
    temporary.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(directory / "result.json")

    visible = {}
    for group in groups:
        anchor = max(group, key=lambda key: meshes[int(key) - 1][1].get_surface_area())
        alignment = reference[anchor] @ np.linalg.inv(poses[anchor])
        visible.update((key, alignment @ poses[key]) for key in group)
    report = {"test": CASE, "title": result["title"], "reference_kind": "local",
              "source_note": ("Kość A to cztery cyfrowo wydzielone odłamki pełnego skanu. Kość B ma siedem "
                              "odłamków z syntetycznego zestawu fibula_fracture_v2. Referencja służy do oceny."),
              "groups": groups, "expected": expected, "fragments": names, "seconds": seconds,
              "predicted_meshes": preview(meshes, visible, groups),
              "reference_meshes": preview(meshes, reference, expected), **metrics}
    (directory / f"{CASE}.json").write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    write_report(directory, case_ids=(CASE,))
    print(json.dumps({"groups": groups, "expected": expected, "seconds": seconds, **metrics}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("synthetic", type=Path, help="Katalog fibula_fracture_v2")
    parser.add_argument("scan", type=Path, help="Źródłowy pełny skan PLY kości A")
    parser.add_argument("--directory", type=Path, default=PROJECT / "out" / "test-dwie-strzalkowe")
    args = parser.parse_args()
    prepare(args.synthetic, args.scan, args.directory)

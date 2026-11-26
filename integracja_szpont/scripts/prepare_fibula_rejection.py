"""Build the local case of a fibula mixed with three femur shaft fragments."""

import argparse
import json
import os
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

os.environ.setdefault("OMP_NUM_THREADS", "4")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np

from benchmark_real import measure
from bones import read_mesh
from case_solver import make_result
from prepare_fibula_demo import parts_from_scan


TITLE = "Test nr 3 · Kość strzałkowa i obce odłamki"


def prepare(femur, scan, output):
    fibula = sorted((PROJECT / "out" / "fibula-demo").glob("fibula-fragment-*.ply"))
    if len(fibula) != 4:
        raise ValueError("Przygotuj najpierw cztery odłamki strzałkowej ze skanu.")
    # The reference positions show that 01, 02 and 04 are closest to the center of the shaft.
    paths = fibula + [femur / "input" / "ply" / f"fragment_{index:02d}.ply" for index in (1, 2, 4)]
    order = np.random.default_rng(87).permutation(len(paths))
    names = [f"fragment_{i:02d}" for i in range(1, len(paths) + 1)]
    meshes = [(name, read_mesh(paths[index])) for name, index in zip(names, order)]
    template = read_mesh(scan)
    result = make_result(TITLE, meshes, upload=True, target_template=("Kość strzałkowa", template),
                         progress=lambda message: print(message, flush=True))

    # Check the known source of each scan only after the solver has decided where it belongs.
    group = result["sections"][0]["ids"]
    excluded = result["target"]["excluded"]
    expected = {str(i) for i, index in enumerate(order, 1) if index < len(fibula)}
    foreign = set(result["poses"]) - expected
    if set(group) != expected or {entry["id"] for entry in excluded} != foreign:
        raise ValueError(f"Otrzymano grupę {group}, oczekiwano {sorted(expected)}.")
    if any(bool(set(link["ids"]) & expected) and bool(set(link["ids"]) & foreign) for link in result["links"]):
        raise ValueError("Wynik zawiera błędne połączenie między kośćmi.")
    if any(not entry["detail"].startswith("Najlepsza zbadana para") for entry in excluded):
        raise ValueError("Nie zbadano przełomów co najmniej jednego odłożonego odłamka.")
    _, initial, _ = parts_from_scan(template)
    reference = {str(i): np.linalg.inv(initial[index]) for i, index in enumerate(order, 1) if str(i) in expected}
    poses = {key: np.asarray(pose) for key, pose in result["poses"].items()}
    errors = measure(meshes, poses, [group], reference, [expected])["errors"]
    if max(error["rmse"] for error in errors) > .05:
        raise ValueError(f"Ułożenie strzałkowej odbiega od wzorca: {errors}")
    result["reference_check"] = {"rmse": max(error["rmse"] for error in errors),
                                 "rotation_deg": max(error["angle"] for error in errors)}
    for section in result["sections"]:
        for part in section["parts"]:
            url = f'/api/demo/fibula-rejection/scan/{part["id"]}'
            part["uploaded_scan"] = {"label": part["label"] + ".ply", "url": url, "download": url + "?download=1"}
    for entry in excluded:
        index = order[int(entry["id"]) - 1]
        entry["source"] = f"Zestaw testowy · trzon kości udowej ({paths[index].stem})"

    output.mkdir(parents=True, exist_ok=True)
    pending_zip = output / "strzalkowa-z-obcymi-odlamkami.pending"
    with ZipFile(pending_zip, "w", ZIP_DEFLATED) as archive:
        for name, index in zip(names, order):
            archive.write(paths[index], name + ".ply")
    pending_zip.replace(pending_zip.with_suffix(".zip"))
    pending = output / "result.pending"
    pending.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    pending.replace(pending.with_suffix(".json"))
    print(json.dumps({"accepted": group, "excluded": excluded, "reference": result["reference_check"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("femur", type=Path, help="Katalog femur_6_fragments_3D")
    parser.add_argument("scan", type=Path, help="Pełny skan PLY strzałkowej")
    parser.add_argument("--out", type=Path, default=PROJECT / "out" / "test-obce-odlamki")
    args = parser.parse_args()
    prepare(args.femur, args.scan, args.out)

"""Build the case of a missing middle fibula fragment and one foreign fragment."""

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

from bones import read_mesh
from incomplete_assembly import incomplete_result
from prepare_fibula_demo import parts_from_scan
from osteo_engine_v6 import moved


def prepare(scan, foreign, output):
    original = sorted((PROJECT / "out" / "fibula-demo").glob("fibula-fragment-*.ply"))
    if len(original) != 4:
        raise ValueError("Przygotuj najpierw cztery odłamki strzałkowej ze skanu.")
    # Keep the second fragment out of the input and interleave the other geometry.
    entries = [(foreign, None), (original[2], 2), (original[0], 0), (original[3], 3)]
    meshes = [(f"fragment_{index:02d}", read_mesh(path)) for index, (path, _) in enumerate(entries, 1)]
    template = read_mesh(scan)
    result = incomplete_result("Test nr 4 · Brakujący odłamek", meshes, template,
                               progress=lambda message: print(message, flush=True))
    expected = {str(i) for i, (_, part) in enumerate(entries, 1) if part is not None}
    rejected = result["target"]["excluded"]
    if set(result["poses"]) != expected or [item["id"] for item in rejected] != ["1"]:
        raise ValueError("Program pomylił odłamki strzałkowej z obcym odłamkiem.")
    if {frozenset(pair) for pair in result["links"]} != {frozenset(("2", "4"))}:
        raise ValueError("Wynik powinien mieć tylko potwierdzony styk po jednej stronie luki.")
    if not rejected[0]["detail"].startswith("Najlepsza zbadana para"):
        raise ValueError("Brakuje pomiaru przełomu obcego odłamka.")
    parts, initial, _ = parts_from_scan(template)
    for key in expected:
        part = entries[int(key) - 1][1]
        vertices = np.asarray(meshes[int(key) - 1][1].vertices)[::8]
        fitted = moved(vertices, np.asarray(result["poses"][key]))
        known = moved(vertices, np.linalg.inv(initial[part]))
        if np.sqrt(np.mean(np.sum((fitted - known) ** 2, axis=1))) > .05:
            raise ValueError("Odłamek ustawiono w niewłaściwym miejscu wzorca.")
    line = np.asarray(result["gap"]["points"])
    axis = line[-1] - line[0]
    axis /= np.linalg.norm(axis)
    withheld = moved(np.asarray(parts[1][1].vertices), np.linalg.inv(initial[1])) @ axis
    if not np.quantile(withheld, .1) < line[1] @ axis < np.quantile(withheld, .9):
        raise ValueError("Luka nie wskazuje położenia pominiętego fragmentu.")

    rejected[0]["source"] = "Zestaw testowy · odłamek trzonu kości udowej"
    for section in result["sections"]:
        for part in section["parts"]:
            url = f'/api/demo/fibula-gap/scan/{part["id"]}'
            part["uploaded_scan"] = {"url": url, "download": url + "?download=1", "label": part["label"] + ".ply"}
    output.mkdir(parents=True, exist_ok=True)
    zip_pending = output / "strzalkowa-z-ubytkiem.pending"
    with ZipFile(zip_pending, "w", ZIP_DEFLATED) as archive:
        for (name, _), (path, _) in zip(meshes, entries):
            archive.write(path, name + ".ply")
    zip_pending.replace(zip_pending.with_suffix(".zip"))
    pending = output / "result.pending"
    pending.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    pending.replace(pending.with_suffix(".json"))
    print(json.dumps({"placed": result["placed"], "gap": result["gap"], "rejected": rejected},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path, help="Pełny skan PLY strzałkowej")
    parser.add_argument("foreign", type=Path, help="Odłamek kości udowej w PLY")
    parser.add_argument("--out", type=Path, default=PROJECT / "out" / "test-brakujacy-odlamek")
    args = parser.parse_args()
    prepare(args.scan, args.foreign, args.out)

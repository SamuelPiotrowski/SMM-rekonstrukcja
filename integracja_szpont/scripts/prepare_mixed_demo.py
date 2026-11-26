"""Save the automatic two-bone test result for a quick local demonstration."""

import argparse
import json
import os
import sys
from pathlib import Path
from zipfile import ZipFile

os.environ.setdefault("OMP_NUM_THREADS", "4")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from case_solver import make_result
from case_upload import load_scan
from assembly_layout import combined_layout


def save_demo(directory, result):
    result["combined_layout"] = combined_layout(result["sections"])
    result["case_note"] = "Zapisany wynik automatycznego składania. Obie kości składają się w jednym widoku."
    pending = directory / "demo-result.pending"
    pending.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    output = pending.with_suffix(".json")
    pending.replace(output)
    print(output)


def prepare(directory):
    title = "Test nr 2 · Wymieszane kości"
    report = json.loads((directory / "udowa-strzalkowa.json").read_text(encoding="utf-8"))
    names = [f"{label}.ply" for label in report["fragments"]]
    known = dict(zip(map(frozenset, report["expected"]), ("Kość udowa", "Kość strzałkowa")))
    with ZipFile(directory / "dwie-kosci-10-fragmentow.zip") as archive:
        meshes = [(name.removesuffix(".ply"), load_scan(name, archive.read(name))) for name in names]
    result = make_result(title, meshes, mixed=True, upload=True,
                         progress=lambda message: print(message, flush=True))
    if {frozenset(section["ids"]) for section in result["sections"]} != known.keys():
        raise ValueError("Automatyczne składanie nie odtworzyło obu grup z próby kontrolnej.")
    for section in result["sections"]:
        section["bone_type"] = {"name": known[frozenset(section["ids"])],
                                "source": "Zestaw testowy i sprawdzenie względem referencji"}
        for part in section["parts"]:
            fragment = part["id"]
            url = f"/api/demo/mixed/scan/{fragment}"
            part["uploaded_scan"] = {"url": url, "download": url + "?download=1",
                                     "label": names[int(fragment) - 1]}
    save_demo(directory, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=PROJECT / "out" / "test-udowa-strzalkowa")
    prepare(parser.parse_args().directory)

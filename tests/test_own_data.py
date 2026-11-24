"""
Test ogólności: dane w formacie "własnych zdjęć", inne niż syntetyczny przypadek 1.

    - inny rozmiar obrazów (900 x 700) i skala (0.2 mm/piksel),
    - ciemna kość na JASNYM tle (sprawdza foreground = "auto"),
    - pliki JPG, końce przełomów klikane w PIKSELACH,
    - brak pliku prawdy (truth) - pipeline działa bez oceny.

Prawda jest użyta wyłącznie na końcu testu do sprawdzenia wyniku.
Uruchomienie (z głównego folderu projektu):
    python tests/test_own_data.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from approximation import mm_to_pixels  # noqa: E402
from geometry import rotation, transform  # noqa: E402
from pipeline import run_case  # noqa: E402
from synthetic import SyntheticParams, bone_fragments  # noqa: E402

SHAPE = (700, 900)    # (wysokość, szerokość)
SCALE = 0.2           # mm / piksel
ANGLES = (-50.0, 15.0, 110.0, -160.0)


def make_photos(folder: Path) -> dict:
    """Zapisuje JPG-i i annotations.json (piksele). Zwraca prawdziwe pary przełomów."""
    rng = np.random.default_rng(7)
    fragments, _ = bone_fragments(SyntheticParams())
    annot = {"case": "test_wlasne", "title": "Test: format własnych zdjęć", "mmPerPixel": SCALE,
             "endpointUnits": "px", "fragments": []}
    seams = {}
    for i, ((polygon, rois), angle) in enumerate(zip(fragments, np.deg2rad(ANGLES))):
        fid = f"frag{i}"
        offset = rng.uniform(-15, 15, 2)  # fragment nie na środku zdjęcia
        tr = -rotation(angle) @ polygon.mean(axis=0) + offset
        px = mm_to_pixels(transform(polygon, angle, tr), SHAPE, SCALE)
        img = Image.new("L", SHAPE[::-1], 235)
        ImageDraw.Draw(img).polygon([tuple(p) for p in px], fill=60)
        arr = np.asarray(img, dtype=float) + rng.normal(0, 6, SHAPE)
        Image.fromarray(np.uint8(np.clip(arr, 0, 255))).save(folder / f"{fid}.jpg", quality=95)
        entry = {"id": fid, "image": f"{fid}.jpg", "rois": []}
        for k, roi in enumerate(rois):
            ends = mm_to_pixels(transform(roi["endpoints"], angle, tr), SHAPE, SCALE).round(1)
            rid = f"{fid}_{k + 1}"
            entry["rois"].append({"id": rid, "endpoints": ends.tolist()})
            seams[rid] = roi["truth_seam"]
        annot["fragments"].append(entry)
    (folder / "annotations.json").write_text(json.dumps(annot, indent=2), encoding="utf-8")
    return seams


def test_own_data_format():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        seams = make_photos(folder)
        case = run_case(folder / "annotations.json", folder / "wyniki", make_3d=True, verbose=True)
        accepted = [c for c in case["candidates"] if c["accepted"]]
        correct = [c for c in accepted if seams[c["arcA"]] == seams[c["arcB"]]]
        print(f"przyjęte: {len(accepted)}, poprawne: {len(correct)}, grupy: {len(case['assembly']['components'])}")
        for name in ("case.json", "candidates.csv", "assembly.png", "input.png", "assembly_3d.html"):
            assert (folder / "wyniki" / name).exists(), name
        assert len(accepted) == 3 and len(correct) == 3, "oczekiwano 3 poprawnych połączeń"
        assert len(case["assembly"]["components"]) == 1


if __name__ == "__main__":
    test_own_data_format()
    print("OK")

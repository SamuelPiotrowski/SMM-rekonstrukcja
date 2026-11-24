"""
Cały przebieg dla jednego przypadku: adnotacje -> analiza -> dopasowanie -> złożenie -> wyniki.

    from pipeline import run_case
    case = run_case("bone_photos/annotations.json", "out/moja_kosc")

Działa tak samo dla danych syntetycznych (z truth.json) i własnych (bez prawdy).
Parametry domyślne pochodzą z settings.json (config.py); można podać własne.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import approximation
import matching
import results
import visualization_3d


def run_case(annotation_path, out_dir, truth_path=None, *,
             seg: approximation.SegmentationParams | None = None,
             approx: approximation.ApproximationParams | None = None,
             match: matching.MatchParams | None = None,
             view: visualization_3d.View3DParams | None = None,
             make_3d: bool = True, open_browser: bool = False, verbose: bool = True) -> dict:
    """Analizuje jeden plik adnotacji i zapisuje wszystkie wyniki do `out_dir`.

    Args:
        annotation_path: plik adnotacji (format w approximation.py).
        out_dir: folder wyników (tworzony, jeśli nie istnieje).
        truth_path: opcjonalny truth.json - wtedy liczone są miary względem prawdy.
        seg, approx, match, view: parametry; None = z settings.json.
        make_3d: zapisz assembly_3d.html.
        open_browser: otwórz wizualizację 3D po zapisie.

    Returns:
        słownik przypadku (to samo, co w case.json).
    """
    import config  # wczytuje settings.json dopiero tutaj - moduły działają też bez niego

    seg = seg or config.segmentation_params()
    approx = approx or config.approximation_params()
    match = match or config.match_params()
    view = view or config.view3d_params()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    def log(msg):
        if verbose:
            print(f"[{time.perf_counter() - start:6.1f} s] {msg}", flush=True)

    # 2. analiza obrazów - nie zna prawdy
    case = approximation.analyse_annotations(annotation_path, seg, approx)
    for f in case["fragments"]:
        f["image"] = results.relative_path(f["image"], out_dir)
    log(f"{case['id']}: {len(case['fragments'])} fragmenty, "
        f"{sum(len(f['arcs']) for f in case['fragments'])} przełomy")

    # 3-4. dopasowanie i złożenie - nie zna prawdy
    case["candidates"] = matching.find_candidates(case["fragments"], match)
    case["assembly"] = matching.assemble(case["fragments"], case["candidates"], match)
    matching.assembly_metrics(case)
    log(f"{len(case['candidates'])} kandydatów, przyjęte: "
        + ", ".join(f"{c['label']} ({c['metrics']['score']:.1f})" for c in case["candidates"] if c["accepted"]))

    # 5. ocena - dopiero po ustaleniu złożenia
    truth = None
    if truth_path and Path(truth_path).exists():
        truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
        matching.evaluate_truth(case, truth)
        m = case["assembly"]["metrics"]
        log(f"prawda: poprawne krawędzie {m['correctEdges']}/{m['selectedEdges']}, IoU {m['iou']}")

    case["thresholds"] = {"distance": match.tolerance_mm, "minLength": match.min_contact_mm,
                          "maxCollisionRatio": match.max_collision_ratio}
    case["notes"] = list(results.NOTES)

    # 6. wyniki
    results.save_all(case, out_dir, truth)
    if make_3d:
        html = visualization_3d.save_html(case, out_dir / "assembly_3d.html", view, open_browser)
        case["assembly3d"] = html.name
        results.save_case(case, out_dir)
    log(f"zapisano wyniki w {out_dir}")
    return case

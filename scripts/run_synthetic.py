"""
Powtarzalne badanie na danych syntetycznych: 3 przypadki x 4 fragmenty.

Generuje obrazy (src/synthetic.py), analizuje je tak samo jak własne zdjęcia
(src/pipeline.py) i porównuje z prawdą. Wyniki: out/synthetic/<przypadek>/
oraz zbiorcze out/synthetic/study_data.json/.js i *_results.csv.

Użycie (z głównego folderu projektu):
    python scripts/run_synthetic.py                 # generuje obrazy i analizuje
    python scripts/run_synthetic.py --reuse-inputs  # analizuje istniejące obrazy
    python scripts/run_synthetic.py --cases clean --open
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402

import config  # noqa: E402
import results  # noqa: E402
import synthetic  # noqa: E402
from pipeline import run_case  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reuse-inputs", action="store_true", help="nie generuj obrazów od nowa")
    parser.add_argument("--cases", nargs="+", default=list(config.SYNTHETIC_CASES), choices=["clean", "noise", "loss"])
    parser.add_argument("--no-3d", action="store_true", help="bez pliku assembly_3d.html")
    parser.add_argument("--open", action="store_true", help="otwórz wizualizacje 3D w przeglądarce")
    args = parser.parse_args()

    params = config.synthetic_params()
    root = config.DATA_OUT / "synthetic"
    start, cases = time.perf_counter(), []
    for cid in args.cases:
        dest = root / cid
        if not args.reuse_inputs or not (dest / "annotations.json").exists():
            synthetic.create_case(cid, dest, params)
        cases.append(run_case(dest / "annotations.json", dest, dest / "truth.json",
                              make_3d=not args.no_3d, open_browser=args.open))
    meta = {"title": "Studia przypadku: od obrazu do złożenia fragmentów", "synthetic": True,
            "seed": params.seed, "units": "mm", "angles": "rad",
            "coordinates": "x w prawo, y w górę. Kandydat mapuje b do a. Pozy w układzie fragmentu-kotwicy.",
            "pythonVersion": sys.version.split()[0], "numpyVersion": np.__version__,
            "scipyVersion": scipy.__version__, "matplotlibVersion": matplotlib.__version__,
            "buildSeconds": round(time.perf_counter() - start, 2), "cases": len(cases)}
    results.save_study(cases, root, meta)
    print(f"Gotowe w {time.perf_counter() - start:.1f} s. Wyniki: {root}")


if __name__ == "__main__":
    main()

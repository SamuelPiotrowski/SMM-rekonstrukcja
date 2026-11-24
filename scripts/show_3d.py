"""
Ponowna wizualizacja 3D z gotowego case.json (bez liczenia od nowa).

Użycie (z głównego folderu projektu):
    python scripts/show_3d.py out/synthetic/clean/case.json
    python scripts/show_3d.py out/moja_kosc/case.json --thickness 6

Wygląd (grubość, liczba klatek, ...) - settings.json, sekcja "visualization_3d".
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku

import config  # noqa: E402
from results import load_case  # noqa: E402
from visualization_3d import save_html  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_json")
    parser.add_argument("--thickness", type=float, help="grubość brył [mm]")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    params = config.view3d_params()
    if args.thickness:
        params = replace(params, thickness_mm=args.thickness)
    path = Path(args.case_json)
    out = save_html(load_case(path), path.parent / "assembly_3d.html", params, open_browser=not args.no_open)
    print(f"Zapisano {out}")


if __name__ == "__main__":
    main()

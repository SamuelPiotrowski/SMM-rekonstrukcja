"""
Analiza WŁASNYCH zdjęć fragmentów kości.

Przygotowanie (raz):
    1. wrzuć zdjęcia fragmentów do bone_photos/ (jeden fragment na zdjęcie),
    2. python scripts/annotate_photos.py --mm-per-pixel 0.1
       (albo --ruler-mm 10 i kliknij linijkę na każdym zdjęciu)
       -> powstaje bone_photos/annotations.json

Analiza:
    python scripts/run_own_data.py                       # bone_photos/annotations.json
    python scripts/run_own_data.py inny/annotations.json # dowolny plik adnotacji
    python scripts/run_own_data.py --truth truth.json    # jeśli znasz prawdziwe położenia

Wyniki: out/<case>/ (case.json, CSV, PNG, assembly_3d.html).
Progi dopasowania (np. min_contact_mm dla małych kości) - settings.json, sekcja "matching".
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku

import config  # noqa: E402
from approximation import load_annotations  # noqa: E402
from pipeline import run_case  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotations", nargs="?", default=str(config.DATA_IN / "annotations.json"))
    parser.add_argument("--out", help="folder wyników (domyślnie out/<case>)")
    parser.add_argument("--truth", help="opcjonalny truth.json do oceny")
    parser.add_argument("--no-3d", action="store_true", help="bez pliku assembly_3d.html")
    parser.add_argument("--no-open", action="store_true", help="nie otwieraj przeglądarki")
    args = parser.parse_args()

    path = Path(args.annotations)
    if not path.exists():
        sys.exit(f"Brak pliku {path}. Najpierw: python scripts/annotate_photos.py --mm-per-pixel <skala>")
    out = Path(args.out) if args.out else config.DATA_OUT / load_annotations(path)["case"]
    run_case(path, out, args.truth, make_3d=not args.no_3d,
             open_browser=config.OPEN_BROWSER and not args.no_open and not args.no_3d)


if __name__ == "__main__":
    main()

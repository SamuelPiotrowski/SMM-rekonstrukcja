"""
Klikanie końców przełomów na własnych zdjęciach -> annotations.json.

Użycie (z głównego folderu projektu):
    python scripts/annotate_photos.py --mm-per-pixel 0.1       # znana skala zdjęć
    python scripts/annotate_photos.py --ruler-mm 10            # skala z linijki na każdym zdjęciu
    python scripts/annotate_photos.py inny_folder --case piszczel --mm-per-pixel 0.08

Dla każdego zdjęcia: klikaj PARAMI końce przełomów (2 kliknięcia = 1 przełom),
Enter = następne zdjęcie, prawy przycisk / Backspace = cofnij.
Potem: python scripts/run_own_data.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku

import config  # noqa: E402
from annotate import annotate_folder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", nargs="?", default=str(config.DATA_IN), help="folder ze zdjęciami fragmentów")
    parser.add_argument("--case", help="nazwa przypadku (domyślnie nazwa folderu)")
    scale = parser.add_mutually_exclusive_group(required=True)
    scale.add_argument("--mm-per-pixel", type=float, help="skala zdjęć [mm/piksel]")
    scale.add_argument("--ruler-mm", type=float, help="długość odcinka linijki klikanego na każdym zdjęciu [mm]")
    args = parser.parse_args()
    out = annotate_folder(args.folder, args.case, args.mm_per_pixel, args.ruler_mm)
    print(f"Zapisano {out}. Teraz: python scripts/run_own_data.py {out}")


if __name__ == "__main__":
    main()

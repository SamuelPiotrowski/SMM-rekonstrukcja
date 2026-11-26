"""
KROK 2: ręczne oznaczanie powierzchni styku (przełomów) na kościach - w przeglądarce.

Otwiera stronę z kością w 3D: pędzel, gumka i "różdżka" (zaznacza obszar aż do ostrej
krawędzi), kilka powierzchni na kość, cofanie (Ctrl+Z), zapis (Ctrl+S).
Oznaczenia trafiają do contacts/<id_kości>.json (format: src/contacts.py).

Użycie (z głównego folderu projektu, po scripts/prepare_data.py):
    python scripts/mark_contacts.py
    python scripts/mark_contacts.py --port 8800      # inny port
    python scripts/mark_contacts.py --no-open        # bez otwierania przeglądarki

Za pierwszym razem potrzebny jest internet (pobranie biblioteki three.js, potem jest w cache).
Zakończenie: Ctrl+C w tym oknie.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku

from contact_tool import serve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, help="port serwera (0 wybiera wolny port automatycznie)")
    parser.add_argument("--no-open", action="store_true", help="nie otwieraj przeglądarki")
    parser.add_argument("--out-dir", type=Path, help="folder przygotowanych siatek (z podfolderem bones)")
    parser.add_argument("--start-library", action="store_true", help="otwórz animację i bibliotekę fragmentów")
    args = parser.parse_args()
    serve(port=args.port, open_browser=not args.no_open, out_dir=args.out_dir,
          start_library=args.start_library)


if __name__ == "__main__":
    main()

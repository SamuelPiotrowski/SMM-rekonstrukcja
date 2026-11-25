"""
KROK 2: ręczne oznaczanie powierzchni styku (przełomów) na kościach - w przeglądarce.

Otwiera stronę z kością w 3D: pędzel, gumka i "różdżka" (zaznacza obszar aż do ostrej
krawędzi), kilka powierzchni na kość, cofanie (Ctrl+Z), zapis (Ctrl+S).
Oznaczenia trafiają do contacts/<id_kości>.json (format: src/contacts.py).

Kość bez ręcznych oznaczeń otwiera się od razu z USTAWIENIEM AUTOMATYCZNYM (wynik algorytmu
z sekcji "contacts" w settings.json) - wystarczy je poprawić i zapisać (Ctrl+S). Strona pisze
wtedy "ustawienie automatyczne"; niezapisane nie staje się ręcznym oznaczeniem.

Użycie (z głównego folderu projektu, po scripts/prepare_data.py):
    python scripts/mark_contacts.py
    python scripts/mark_contacts.py --port 8800      # inny port
    python scripts/mark_contacts.py --no-open        # bez otwierania przeglądarki

Za pierwszym razem potrzebny jest internet (pobranie biblioteki three.js, potem jest w cache).
Zakończenie: Ctrl+C w tym oknie.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku

from contact_tool import serve  # noqa: E402


def main() -> None:
    """Czyta argumenty i uruchamia serwer narzędzia (contact_tool.serve) - działa do Ctrl+C."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, help="port serwera (domyślnie contacts.port z settings.json)")
    parser.add_argument("--no-open", action="store_true", help="nie otwieraj przeglądarki")
    args = parser.parse_args()
    serve(port=args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()

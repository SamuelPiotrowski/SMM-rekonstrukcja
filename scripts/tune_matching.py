"""
Podgląd dopasowania powierzchni styku z SUWAKAMI ustawień - w przeglądarce.

Pokazuje pary z out/3d/matching.json (wynik scripts/match_surfaces.py): kość A ustawiona
na kości B. Po prawej suwaki ustawień z settings.json, sekcja "matching". Po każdej
zmianie suwaka przeliczana jest TYLKO pokazana para:
    - "Szukanie ustawienia" (próbkowanie, FPFH, RANSAC, ICP) - dopasowanie od nowa, kilka sekund,
    - "Ocena" (styk, kolizja) - tylko nowa ocena obecnego ustawienia, od razu.
Lista par po lewej pokazuje stary i nowy wynik. Nic nie jest zapisywane; zmienione
ustawienia strona wypisuje gotowe do wklejenia w settings.json - potem uruchom
match_surfaces.py, żeby przeliczyć wszystkie pary.

Użycie (z głównego folderu projektu, po scripts/match_surfaces.py):
    python scripts/tune_matching.py
    python scripts/tune_matching.py --port 8800      # inny port
    python scripts/tune_matching.py --no-open        # bez otwierania przeglądarki

Działa bez internetu. Zakończenie: Ctrl+C w tym oknie. Kod: src/match_tool.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ... i w komunikatach błędów (SystemExit)

from match_tool import serve  # noqa: E402


def main() -> None:
    """Czyta argumenty i uruchamia serwer podglądu (match_tool.serve) - działa do Ctrl+C."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, help="port serwera (domyślnie matching.port z settings.json)")
    parser.add_argument("--no-open", action="store_true", help="nie otwieraj przeglądarki")
    args = parser.parse_args()
    serve(port=args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()

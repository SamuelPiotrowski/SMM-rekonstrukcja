"""
KROK 6: podgląd wyników dopasowania (out/3d/<zestaw>/matching.json) - jak show_bones.py dla prepare_data.py.
Liczy je scripts/match_surfaces.py; tutaj tylko oglądanie (i próbowanie ustawień na jednej parze).

Dwa tryby:
    domyślny - strona z SUWAKAMI (lokalny serwer, opis niżej),
    --html   - statyczna strona out/3d/<zestaw>/matching.html: najlepsze pary w 3D + macierz wyników
               wszystkich par; jeden plik, można go komuś wysłać.
Oba ostrzegają, gdy matching.json jest nieaktualny (inne ustawienia w settings.json albo
zmienione oznaczenia) - wtedy uruchom ponownie match_surfaces.py.

Strona z suwakami pokazuje pary z out/3d/<zestaw>/matching.json (wynik scripts/match_surfaces.py): kość A ustawiona
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
    python scripts/tune_matching.py --html           # statyczna strona out/3d/<zestaw>/matching.html

Działa bez internetu. Zakończenie: Ctrl+C w tym oknie. Kod: src/match_tool.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ... i w komunikatach błędów (SystemExit)

import config  # noqa: E402
from match_tool import serve, write_html  # noqa: E402


def main() -> None:
    """Czyta argumenty i zapisuje stronę HTML (--html) albo uruchamia serwer z suwakami (do Ctrl+C)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, help="port serwera (domyślnie matching.port z settings.json)")
    parser.add_argument("--no-open", action="store_true", help="nie otwieraj przeglądarki")
    parser.add_argument("--html", action="store_true", help="tylko statyczna strona out/3d/<zestaw>/matching.html (bez serwera)")
    config.add_dataset_argument(parser)
    args = parser.parse_args()
    config.use_dataset(args.dataset)
    if args.html:
        write_html(open_browser=not args.no_open)
    else:
        serve(port=args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()

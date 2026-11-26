"""
KROK 1: przygotowanie danych - skany zestawu (scanned_bones/<zestaw>/ z podfolderami) -> kości
w out/3d/<zestaw>/bones/.

Łączy skany tej samej kości w jedną, usuwa szum (oderwane kawałki niepotwierdzone przez
inne skany), liczy chmurę punktów i FPFH. Szczegóły: src/pipeline.py, src/merging.py.
Kość, która zmieniła nazwę (np. zmieniona nazwa folderu ze skanem), zachowuje oznaczenia
styku - są przenoszone na nową nazwę.

Użycie (z głównego folderu projektu):
    python scripts/prepare_data.py                     # zestaw z settings.json ("dataset")
    python scripts/prepare_data.py --dataset inny      # inny zestaw (podfolder scanned_bones/)

Pierwsze uruchomienie porównuje każdą parę skanów (kilka minut, tylko gdy łączenie
skanów jest włączone); wyniki są zapisywane w out/3d/<zestaw>/cache/, więc kolejne
uruchomienia trwają kilkanaście sekund.
Potem:
    python scripts/show_merging.py     # sprawdź, jak połączyły się skany
    python scripts/show_bones.py       # obejrzyj kości
    python scripts/mark_contacts.py    # oznacz powierzchnie styku
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ... i w komunikatach błędów (SystemExit)

import config  # noqa: E402
from pipeline import prepare_all  # noqa: E402


def main() -> None:
    """Czyta argumenty, ustawia zestaw i przygotowuje wszystkie jego kości (pipeline.prepare_all)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_dataset_argument(parser)
    args = parser.parse_args()
    config.use_dataset(args.dataset)
    if not config.DATA_IN.is_dir():
        raise SystemExit(f"Brak folderu ze skanami zestawu \"{config.DATASET}\": {config.DATA_IN}")
    prepare_all()


if __name__ == "__main__":
    main()

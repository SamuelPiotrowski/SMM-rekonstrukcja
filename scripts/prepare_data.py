"""
KROK 1: przygotowanie danych - skany z DATA_IN (scanned_bones/ z podfolderami) -> kości w out/3d/bones/.

Łączy skany tej samej kości w jedną, usuwa szum (oderwane kawałki niepotwierdzone przez
inne skany), liczy chmurę punktów i FPFH. Szczegóły: src/pipeline.py, src/merging.py.

Użycie (z głównego folderu projektu):
    python scripts/prepare_data.py

Pierwsze uruchomienie porównuje każdą parę skanów (kilka minut); wyniki są zapisywane
w out/3d/cache/, więc kolejne uruchomienia trwają kilkanaście sekund.
Potem:
    python scripts/show_merging.py     # sprawdź, jak połączyły się skany
    python scripts/show_bones.py       # obejrzyj kości
    python scripts/mark_contacts.py    # oznacz powierzchnie styku
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku

from pipeline import prepare_all  # noqa: E402

if __name__ == "__main__":
    prepare_all()

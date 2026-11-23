"""
Przygotowanie danych: wszystkie skany z DATA_IN (scanned_bones_light/) -> DATA_OUT (out/).

Dla każdego skanu: wczytanie, czyszczenie, chmura punktów, FPFH, zapis do
out/<slug>/, a na końcu out/summary.csv. Funkcje są w src/data_parsing.py,
tu są tylko ich wywołania.

Użycie (z głównego folderu projektu):
    python scripts/prepare_data.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import open3d as o3d  # noqa: E402

from config import DATA_IN, DATA_OUT, PREFERRED_FORMATS  # noqa: E402
from data_parsing import find_scans, prepare_fragment, save_fragment, write_summary  # noqa: E402

log = logging.getLogger(__name__)


def main() -> None:
    o3d.utility.random.seed(0)  # powtarzalne losowanie punktów -> identyczne chmury przy każdym uruchomieniu

    scans = find_scans()
    if not scans:
        raise FileNotFoundError(f"Brak skanów ({', '.join(PREFERRED_FORMATS)}) w {DATA_IN}")
    log.info("Znaleziono %d skanów w %s", len(scans), DATA_IN)

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    fragments = []
    for name, path in scans.items():
        frag = prepare_fragment(name, path)
        target = save_fragment(frag)
        raw, clean = frag.meta["raw"], frag.meta["clean"]
        log.info(
            "  %s: %d -> %d trójkątów, usunięte komponenty: %s, chmura: %d pkt -> %s",
            frag.meta["part"], raw["triangles"], clean["triangles"],
            frag.meta["removed_components_area_mm2"] or "brak", frag.meta["cloud_points"],
            target.relative_to(DATA_OUT.parent),
        )
        fragments.append(frag)

    summary = write_summary(fragments)
    log.info("Podsumowanie: %s", summary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

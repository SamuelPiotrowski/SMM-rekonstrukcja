"""
Wizualizacja łączenia skanów: które skany trafiły do której kości i jak się na siebie nałożyły.

Strona ma dwie części:
    1. 3D - skany każdej kości nałożone na siebie (każdy w innym kolorze), usunięty szum
       na czerwono. Lista na dole przełącza kości.
    2. Macierz nakładania wszystkich par skanów - ramki = skany połączone w jedną kość.
       Jeśli dwa skany powinny być razem, a nie są (albo odwrotnie): settings.json,
       "merging" -> "min_overlap" albo ręczne "groups".

Użycie (z głównego folderu projektu, po scripts/prepare_data.py):
    python scripts/show_merging.py

Wynik: out/3d/<zestaw>/merging.html (otwiera się sam).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ... i w komunikatach błędów (SystemExit)

import config  # noqa: E402
from bones import load_bones  # noqa: E402
from visualization_3d import load_report, merging_figure, overlap_figure, write_page  # noqa: E402


def main() -> None:
    """Wczytuje kości i merging.json, buduje obie figury i zapisuje out/3d/<zestaw>/merging.html.

    Raises:
        SystemExit: brak przygotowanych kości albo merging.json (nie uruchomiono prepare_data.py).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-open", action="store_true", help="nie otwieraj przeglądarki")
    config.add_dataset_argument(parser)
    args = parser.parse_args()
    config.use_dataset(args.dataset)

    bones = load_bones()
    report = load_report(config.DATA_OUT)
    if not bones or report is None:
        raise SystemExit(config.no_bones_message())
    params = config.view3d_params()
    merged = sum(b.n_scans > 1 for b in bones)
    noise = sum(b.meta.get("noise_removed_au2", 0) for b in bones)
    thr = report["params"]["min_overlap"]
    lead = (f"{len(report['scans'])} skanów → <b>{len(bones)} kości</b>; {merged} z nich powstało z kilku skanów. "
            f"Usunięty szum: {noise:.0f} a.u.². Dwa skany są uznawane za tę samą kość, gdy nakładanie ≥ {thr} "
            f"(część mniejszego skanu leżąca na drugim, z tolerancją {report['params']['tolerance_au']} a.u.).")
    sections = [
        ("Skany nałożone na siebie", "Każdy skan w innym kolorze, czerwony = usunięty szum. "
         "Kość wybierasz z listy w lewym dolnym rogu. Kliknięcie w legendzie ukrywa skan.",
         merging_figure(bones, params)),
        ("Nakładanie par skanów", f"Niebieskie pola (≥ {thr}) = ta sama kość. Ramki = skany połączone w jedną kość.",
         overlap_figure(report, params)),
    ]
    out = write_page(config.DATA_OUT / "merging.html", "Łączenie skanów", lead, sections, params,
                     open_browser=config.OPEN_BROWSER and not args.no_open)
    print(f"Zapisano {out}")


if __name__ == "__main__":
    main()

"""
Lista zestawów danych i ich stan: ile skanów, ile przygotowanych kości, ile ręcznych oznaczeń.

Zestaw = podfolder o tej samej nazwie w scanned_bones/, out/3d/ i contacts/ (foldery
z settings.json, sekcja "paths"). Domyślny zestaw: settings.json -> "dataset";
w każdym skrypcie jednorazowo: --dataset <nazwa>.

Użycie (z głównego folderu projektu):
    python scripts/datasets.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ... i w komunikatach błędów (SystemExit)

import config  # noqa: E402
from data_parsing import find_scans  # noqa: E402


def main() -> None:
    """Wypisuje tabelę zestawów (domyślny oznaczony *) i podpowiedź, jak wybrać inny."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    names = config.list_datasets()
    default = config.SETTINGS.get("dataset") or ""
    if not names:
        print(f"Brak zestawów. Utwórz folder {config.SCANS_ROOT.name}/<nazwa zestawu>/ i wrzuć do niego skany.")
        return
    print(f"{'':2}{'zestaw':24s}{'skany':>7s}{'kości':>7s}{'ręczne':>8s}  folder skanów")
    for name in names:
        data_in, data_out, contacts = config.dataset_dirs(name)
        scans = len(find_scans(data_in, config.scan_params())[0]) if data_in.is_dir() else 0
        bones = len(list((data_out / "bones").glob("*/meta.json")))
        manual = len(list(contacts.glob("*.json")))
        mark = "* " if name == default else "  "
        where = config._rel(data_in) if data_in.is_dir() else "(brak skanów - tylko wyniki)"
        print(f"{mark}{name:24s}{scans:7d}{bones:7d}{manual:8d}  {where}")
    if default and default not in names:
        print(f"\nUWAGA: domyślnego zestawu \"{default}\" (settings.json -> \"dataset\") nie ma na liście.")
    print("\n* = domyślny (settings.json -> \"dataset\"). Inny jednorazowo: python scripts/<skrypt>.py --dataset <nazwa>")


if __name__ == "__main__":
    main()

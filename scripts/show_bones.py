"""
Wizualizacja kości w przeglądarce: wszystkie kości "na stole" + oznaczone powierzchnie styku.

Użycie (z głównego folderu projektu, po scripts/prepare_data.py):
    python scripts/show_bones.py                    # kolory skanera
    python scripts/show_bones.py --colors fragment  # każda kość w innym kolorze
    python scripts/show_bones.py --colors fpfh      # kolor wg kształtu (FPFH)
    python scripts/show_bones.py --auto             # powierzchnie z algorytmu zamiast ręcznych
    python scripts/show_bones.py 3 strzalkowa       # tylko kość nr 3 i kości z "strzalkowa" w nazwie

Wynik: out/3d/bones.html (otwiera się sam; wygląd: settings.json, sekcja "visualization_3d").
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku

import config  # noqa: E402
from bones import load_bones, select_bones  # noqa: E402
from contacts import load_contacts  # noqa: E402
from visualization_3d import COLOR_MODES, bones_figure, save_figure  # noqa: E402


def main() -> None:
    """Wczytuje wybrane kości i ich oznaczenia, buduje figurę i zapisuje HTML (domyślnie out/3d/bones.html)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("only", nargs="*", help="numery albo fragmenty nazw kości (domyślnie wszystkie)")
    parser.add_argument("--colors", choices=COLOR_MODES, default="scan", help="kolorowanie kości")
    parser.add_argument("--auto", action="store_true", help="pokaż powierzchnie z algorytmu (detect_contacts.py)")
    parser.add_argument("--out", default=None, help="plik wynikowy (domyślnie out/3d/bones.html)")
    parser.add_argument("--no-open", action="store_true", help="nie otwieraj przeglądarki")
    args = parser.parse_args()

    all_bones = load_bones(features=args.colors == "fpfh")
    if not all_bones:
        raise SystemExit("Brak przygotowanych kości - uruchom najpierw: python scripts/prepare_data.py")
    try:
        bones = select_bones(all_bones, args.only)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    numbers = [all_bones.index(b) + 1 for b in bones]
    source = "auto" if args.auto else "manual"
    contacts = {b.id: load_contacts(b.id, b.mesh, source) for b in bones}
    marked = sum(c is not None and c.labels.any() for c in contacts.values())
    print(f"Kości: {len(bones)}, z oznaczonymi powierzchniami styku ({source}): {marked}")

    params = config.view3d_params()
    fig = bones_figure(bones, contacts, args.colors, params, numbers)
    out = Path(args.out) if args.out else config.DATA_OUT / "bones.html"
    save_figure(fig, out, params, open_browser=config.OPEN_BROWSER and not args.no_open)
    print(f"Zapisano {out}")


if __name__ == "__main__":
    main()

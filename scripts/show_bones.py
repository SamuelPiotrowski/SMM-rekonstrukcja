"""
Wizualizacja kości w przeglądarce: wszystkie kości "na stole" + oznaczone powierzchnie styku.

Użycie (z głównego folderu projektu, po scripts/prepare_data.py):
    python scripts/show_bones.py                    # kolory skanera
    python scripts/show_bones.py --colors fragment  # każda kość w innym kolorze
    python scripts/show_bones.py --colors fpfh      # kolor wg kształtu (FPFH)
    python scripts/show_bones.py --source auto      # powierzchnie z algorytmu zamiast ręcznych (krócej: --auto)
    python scripts/show_bones.py --source manual-auto  # ręczne, a dla kości bez ręcznych - z algorytmu
    python scripts/show_bones.py 3 strzalkowa       # tylko kość nr 3 i kości z "strzalkowa" w nazwie

Które powierzchnie styku (domyślnie): settings.json, sekcja "visualization_3d", klucz "source"
("manual" / "auto" / "manual-auto"); --source / --auto zmienia to jednorazowo.
W trybach "auto" i "manual-auto" brakujący albo nieaktualny wynik algorytmu jest liczony
i zapisywany w contacts/auto/ (jak w match_surfaces.py).

Wynik: out/3d/bones.html (otwiera się sam; wygląd: settings.json, sekcja "visualization_3d").
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku

import config  # noqa: E402
from bones import load_bones, select_bones  # noqa: E402
from matching_surfaces import SOURCES, bone_contacts  # noqa: E402
from visualization_3d import COLOR_MODES, bones_figure, save_figure  # noqa: E402


def main() -> None:
    """Wczytuje wybrane kości i ich oznaczenia, buduje figurę i zapisuje HTML (domyślnie out/3d/bones.html).

    Raises:
        SystemExit: zły "source" w settings.json, brak przygotowanych kości albo wzorzec
                    nie pasuje do żadnej kości.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("only", nargs="*", help="numery albo fragmenty nazw kości (domyślnie wszystkie)")
    parser.add_argument("--colors", choices=COLOR_MODES, default="scan", help="kolorowanie kości")
    parser.add_argument("--source", choices=SOURCES, default=None,
                        help="skąd powierzchnie styku (domyślnie z settings.json, \"visualization_3d\" -> \"source\")")
    parser.add_argument("--auto", action="store_true", help="to samo co --source auto")
    parser.add_argument("--out", default=None, help="plik wynikowy (domyślnie out/3d/bones.html)")
    parser.add_argument("--no-open", action="store_true", help="nie otwieraj przeglądarki")
    args = parser.parse_args()

    params = config.view3d_params()
    source = "auto" if args.auto else args.source or params.source
    if source not in SOURCES:
        raise SystemExit(f"settings.json, \"visualization_3d\" -> \"source\" = {source!r}, "
                         f"dostępne: {', '.join(SOURCES)}")

    all_bones = load_bones(features=args.colors == "fpfh")
    if not all_bones:
        raise SystemExit("Brak przygotowanych kości - uruchom najpierw: python scripts/prepare_data.py")
    try:
        bones = select_bones(all_bones, args.only)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    numbers = [all_bones.index(b) + 1 for b in bones]
    contact_params = config.contact_params()
    contacts, origins = {}, {}
    for b in bones:
        contacts[b.id], origin = bone_contacts(b, source, contact_params)
        origins[origin] = origins.get(origin, 0) + 1
    marked = sum(c is not None for c in contacts.values())
    print(f"Kości: {len(bones)}, z powierzchniami styku ({source}): {marked}  "
          f"[{', '.join(f'{k}: {v}' for k, v in origins.items())}]")

    fig = bones_figure(bones, contacts, args.colors, params, numbers)
    out = Path(args.out) if args.out else config.DATA_OUT / "bones.html"
    save_figure(fig, out, params, open_browser=config.OPEN_BROWSER and not args.no_open)
    print(f"Zapisano {out}")


if __name__ == "__main__":
    main()

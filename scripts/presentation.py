"""
PREZENTACJA: jeden przenośny plik HTML z wynikami JEDNEJ metody - łączy prepare_data / show_merging,
show_bones (+ porównanie z ręcznymi) i tune_matching --html.

Spójność: wszystko na stronie pochodzi z jednej metody, jednego zestawu i obecnych ustawień
(settings.json + settings.json zestawu). Metoda = skąd powierzchnie styku:
    manual     - ręczne oznaczenia (contacts/<zestaw>/<id>.json),
    roughness / features / atlas - algorytm wykrywania (sekcja "contacts", "method" ustawiane na tę metodę).
Nic nie jest przeliczane - strona czyta gotowe wyniki. Gdy czegoś brakuje albo policzono to z innymi
ustawieniami, zamiast tej części jest ramka PRZELICZ z gotową komendą, np.:
    python scripts/prepare_data.py                     # kości (krok 1)
    python scripts/match_surfaces.py --method atlas    # powierzchnie + dopasowanie tej metody (kroki 2-3)
Każda metoda ma osobny zapis (out/3d/<zestaw>/methods/<metoda>/), więc wyniki kilku metod
mogą istnieć naraz - nie nadpisują się.

Użycie (z głównego folderu projektu):
    python scripts/presentation.py                     # metoda z settings.json (matching.source / contacts.method)
    python scripts/presentation.py --method features   # wybrana metoda
    python scripts/presentation.py --all               # osobny plik dla każdej metody (linki między nimi)
    python scripts/presentation.py -d grzesiu_gab      # inny zestaw
    python scripts/presentation.py --steps powierzchnie porownanie   # tylko wybrane kroki
    python scripts/presentation.py --no-matching       # wszystko poza dopasowaniem końcowym

Kroki (domyślnie settings.json -> "presentation" -> "steps"): dane, powierzchnie, porownanie, dopasowanie.
Bez "dopasowanie" nie trzeba match_surfaces.py - wystarczą powierzchnie styku (ręczne albo
python scripts/detect_contacts.py --method <algorytm>); plik ma wtedy w nazwie kroki.

Wynik: raporty/prezentacje/<zestaw>/prezentacja_<metoda>.html (wszystkie kroki) albo prezentacja_<metoda>_<kroki>.html - plotly.js w środku pliku: działa bez internetu
i bez serwera, można go wysłać albo otworzyć na innym komputerze. Kod: src/presentation.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ... i w komunikatach błędów (SystemExit)

import config  # noqa: E402
import presentation  # noqa: E402


def default_method() -> str:
    """Metoda z settings.json: "manual" przy matching.source = "manual", inaczej contacts.method."""
    return presentation.current_method() or config.contact_params().method


def main() -> None:
    """Czyta argumenty i zapisuje prezentację wybranej metody (albo wszystkich)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=presentation.methods(), default=None,
                        help="metoda (domyślnie z settings.json)")
    parser.add_argument("--all", action="store_true", help="osobna prezentacja dla każdej metody")
    parser.add_argument("--steps", nargs="+", default=None, metavar="KROK",
                        help=f"kroki strony: {', '.join(presentation.STEPS)} (domyślnie z settings.json)")
    parser.add_argument("--no-matching", action="store_true",
                        help="bez dopasowania końcowego (kroki z settings.json / --steps minus \"dopasowanie\")")
    parser.add_argument("--out", default=None, help="plik wynikowy (tylko dla jednej metody)")
    parser.add_argument("--no-open", action="store_true", help="nie otwieraj przeglądarki")
    config.add_dataset_argument(parser)
    args = parser.parse_args()
    config.use_dataset(args.dataset)
    if args.all and (args.method or args.out):
        raise SystemExit("--all tworzy plik dla każdej metody - bez --method i --out")
    try:
        steps = presentation.parse_steps(args.steps) if args.steps else presentation.default_steps()
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if args.no_matching:
        steps = tuple(s for s in steps if s != "dopasowanie")
        if not steps:
            raise SystemExit("--no-matching: nie zostaje żaden krok")

    todo = presentation.methods() if args.all else [args.method or default_method()]
    opened = False
    for method in todo:
        print(f"Prezentacja: zestaw \"{config.DATASET}\", metoda: {presentation.method_label(method)}, "
              f"kroki: {', '.join(steps)}")
        path, status = presentation.build(method, Path(args.out) if args.out else None, steps=steps)
        print(f"Zapisano {path}")
        if config.OPEN_BROWSER and not args.no_open and not opened and (not args.all or all(status.values())):
            import webbrowser
            webbrowser.open(path.resolve().as_uri())
            opened = True


if __name__ == "__main__":
    main()

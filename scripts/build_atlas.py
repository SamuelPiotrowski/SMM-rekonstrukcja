"""
Baza modeli KOMPLETNYCH kości (atlas anatomiczny): indeksowanie plików modeli i lista bazy.

Pliki modeli (.ply / .obj / .stl, jeden plik = jedna kość, podfolder = typ kości) leżą w atlas/
(settings.json "paths" -> "atlas"). Skrypt zamienia każdy model w mały wpis bazy w atlas_index/
("paths" -> "atlas_index", ~1 MB na kość): tylko zewnętrzna powierzchnia (bez ścian kanału
szpikowego), jej punkty, własności do sita (długość, pole, histogramy krzywizn) i mała siatka
podglądu. Wykrywanie korzysta TYLKO z bazy - oryginały można potem usunąć albo trzymać gdzie
indziej; model bez pliku zostaje w bazie (usuwa go dopiero --prune).
Liczone są tylko nowe i zmienione pliki (suma kontrolna); plik o zmienionej nazwie jest
rozpoznawany po zawartości. To samo robi automatycznie algorytm "atlas" (detect_contacts.py,
identify_bones.py), więc ten skrypt jest potrzebny głównie do obejrzenia bazy i do --prune.

Użycie (z głównego folderu projektu):
    python scripts/build_atlas.py                     # dolicz nowe / zmienione modele, pokaż bazę
    python scripts/build_atlas.py --force             # wszystkie modele z plikami od nowa
    python scripts/build_atlas.py --prune             # usuń z bazy modele, których plików już nie ma
    python scripts/build_atlas.py --atlas D:/modele   # inny folder z plikami modeli
Ustawienia: settings.json, sekcja "atlas" (reference_scale = ile a.u. skanu to 1 jednostka modelu).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ... i w komunikatach błędów (SystemExit)

import atlas  # noqa: E402
import config  # noqa: E402


def main() -> None:
    """Uzupełnia bazę o nowe / zmienione modele (atlas.sync_index) i wypisuje jej zawartość."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="policz wszystkie modele z plikami od nowa")
    parser.add_argument("--prune", action="store_true", help="usuń z bazy modele, których plików już nie ma")
    config.add_atlas_argument(parser)
    args = parser.parse_args()
    source, index = config.use_atlas(args.atlas, args.atlas_index)
    p = config.atlas_params()

    print(f"Pliki modeli: {config._rel(source)}/   baza: {config._rel(index)}/")
    if not source.is_dir():
        print(f"  (brak folderu {config._rel(source)}/ - nowe modele: wrzuć tam pliki .ply / .obj / .stl kompletnych kości)")
    rep = atlas.sync_index(p, source, index, force=args.force, prune=args.prune, log=print)
    for key, what in (("new", "nowe"), ("changed", "zmienione"), ("renamed", "zmiana nazwy pliku"),
                      ("removed", "usunięte z bazy")):
        if rep[key]:
            print(f"  {what}: {', '.join(rep[key])}")
    data = atlas.read_index(index)
    models = data["models"]
    if not models:
        print("Baza jest pusta.")
        return
    print(f"\nBaza: {len(models)} modeli (skala: 1 jednostka modelu = {p.reference_scale:g} a.u. skanu, "
          f"zakres ×{p.scale_range[0]:g}-{p.scale_range[1]:g})")
    print(f"  {'nr':>3}  {'model':<34} {'typ':<14} {'długość':>8} {'pole':>9} {'zewn.':>6} {'punkty':>8} {'w bazie':>8}  plik")
    total = 0
    tag = atlas.index_tag(p)
    for nr, (mid, e) in enumerate(sorted(models.items()), 1):
        info = e["info"]
        size = (index / f"{mid}.npz").stat().st_size if (index / f"{mid}.npz").exists() else 0
        total += size
        has_file = (source / e["file"]).exists()
        note = e["file"] if has_file else f"{e['file']} - brak pliku (model tylko w bazie)"
        if e.get("tag") != tag:
            note += " - INNE USTAWIENIA INDEKSU" + (" (policz: --force)" if has_file else " (brak pliku do przeliczenia)")
        print(f"  {nr:3d}. {e['name'][:34]:<34} {(e.get('group') or '-')[:14]:<14} {info['length']:8.1f} "
              f"{info['area']:9.0f} {info['outer_share']:6.0%} {info['points']:8d} {size / 1e6:7.2f}M  {note}")
    print(f"  razem w bazie: {total / 1e6:.1f} MB (długość i pole w jednostkach modelu; zewn. = część powierzchni "
          "widoczna z zewnątrz - reszta to np. ściany kanału szpikowego, pomijane)")
    if rep["only_index"]:
        print(f"  {len(rep['only_index'])} modeli bez pliku zostaje w bazie (usunięcie: --prune)")
    print("\nDalej: python scripts/identify_bones.py   (z której kości pochodzi każdy fragment)\n"
          "       python scripts/detect_contacts.py --method atlas")


if __name__ == "__main__":
    main()

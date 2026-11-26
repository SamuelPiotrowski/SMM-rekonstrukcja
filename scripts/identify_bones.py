"""
Rozpoznawanie kości: z której kompletnej kości z bazy (atlas anatomiczny) pochodzi każdy fragment,
z prawdopodobieństwami, oraz podział fragmentu na korę i przełom z kontekstu anatomicznego.

Dla każdego fragmentu (src/atlas.py):
    1. analiza lokalna (szorstkość, rozrzut normalnych, gęstość siatki, wgłębienia - kanał szpikowy),
    2. sito własności (rozmiar, histogramy krzywizn) - wyklucza kości, które na pewno nie pasują,
       zanim zacznie się kosztowne dopasowanie,
    3. dopasowanie pozostałych (RANSAC + ICP ze skalą, też odbicie lustrzane = druga strona ciała),
    4. rozkład prawdopodobieństwa modeli z możliwością "żadna z bazy" (niska pewność),
    5. kora = leży na powierzchni modelu, przełom = wnętrze kości (też kanał szpikowy).

Użycie (z głównego folderu projektu, po scripts/prepare_data.py; baza: scripts/build_atlas.py):
    python scripts/identify_bones.py              # wszystkie kości zestawu
    python scripts/identify_bones.py 3 whole      # tylko kość nr 3 i kości z "whole" w nazwie
    python scripts/identify_bones.py --no-open    # bez otwierania przeglądarki
Wyniki (out/3d/<zestaw>/):
    atlas.json  - rankingi wszystkich kości (prawdopodobieństwa, skale, ustawienia na modelach, wynik sita),
    atlas.csv   - to samo jako tabela do Excela,
    atlas.html  - mapa prawdopodobieństw + fragment na najbardziej prawdopodobnym modelu w 3D.
Dopasowania są w cache (out/3d/<zestaw>/cache/atlas/), więc drugie uruchomienie trwa sekundy;
liczone od nowa po zmianie bazy, siatki kości albo ustawień (settings.json, sekcja "atlas").
Powierzchnie styku z atlasu do dopasowania fragmentów: python scripts/detect_contacts.py --method atlas
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ... i w komunikatach błędów (SystemExit)

import atlas  # noqa: E402
import config  # noqa: E402
import contacts as ct  # noqa: E402
from bones import load_bones, select_bones  # noqa: E402


def main() -> None:
    """Rozpoznaje wybrane kości, wypisuje rankingi i zapisuje atlas.json / atlas.csv / atlas.html.

    Raises:
        SystemExit: brak przygotowanych kości, pusta baza modeli albo wzorzec nie pasuje do żadnej kości.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("only", nargs="*", help="numery albo fragmenty nazw kości (domyślnie wszystkie)")
    parser.add_argument("--top", type=int, default=3, help="ile najbardziej prawdopodobnych kości wypisać (domyślnie 3)")
    parser.add_argument("--no-open", action="store_true", help="nie otwieraj przeglądarki")
    config.add_dataset_argument(parser)
    config.add_atlas_argument(parser)
    args = parser.parse_args()
    config.use_dataset(args.dataset)
    config.use_atlas(args.atlas, args.atlas_index)

    all_bones = load_bones()
    if not all_bones:
        raise SystemExit(config.no_bones_message())
    try:
        bones = select_bones(all_bones, args.only)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    p, cp, sfp = config.atlas_params(), config.contact_params(), config.surface_feature_params()
    try:
        db = atlas.load_atlas(p)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    print(f"Baza: {len(db.models)} modeli ({config._rel(config.ATLAS_INDEX)}/), kości: {len(bones)}")

    items = []
    for b in bones:
        nr = all_bones.index(b) + 1
        t = time.perf_counter()
        m = atlas.cached_match(b, db, p, cp, sfp)
        try:
            found = ct.detect_contacts(b, "atlas", cp)
            plain = ct.detect_contacts(b, "features", cp)
        except ValueError as e:         # np. zła nazwa cechy w "feature_keys"
            raise SystemExit(str(e)) from None
        how = "z cache" if m.cached else f"{time.perf_counter() - t:.0f} s"
        lines = m.ranking_lines(args.top)
        print(f"  {nr:2d}. {b.label}: {lines[0]}  ({how})")
        for line in lines[1:]:
            print(f"      {line}")
        areas = ", ".join(f"{s['area_au2']:.0f}" for s in ct.surface_stats(b.mesh, found.labels)) or "-"
        print(f"      przełomy (atlas): [{areas}] a.u.²; atlas pewność {m.confidence:.0%}"
              + ("" if m.confidence >= p.min_confidence else " - za mała, tylko analiza lokalna"))
        items.append({"nr": nr, "bone": b, "match": m, "labels_atlas": found.labels, "labels_features": plain.labels})

    jpath, cpath = atlas.write_reports(config.DATA_OUT, items, db, p)
    page = atlas.write_page(config.DATA_OUT / "atlas.html", items, db, p, config.view3d_params(),
                            open_browser=config.OPEN_BROWSER and not args.no_open)
    print(f"Zapisano {config._rel(jpath)}, {config._rel(cpath)}, {config._rel(page)}")
    print(f"Powierzchnie styku z atlasu (zapis w contacts/{config.DATASET}/auto/): "
          f"python scripts/detect_contacts.py --method atlas{config.dataset_flag()}")


if __name__ == "__main__":
    main()

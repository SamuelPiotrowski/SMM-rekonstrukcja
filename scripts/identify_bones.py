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
    python scripts/identify_bones.py --shared-scale   # wspólna skala zestawu (zapis w ustawieniach zestawu)
WSPÓLNA SKALA: wszystkie fragmenty jednej osoby mają tę samą skalę względem modelu. Skrypt zawsze
wypisuje skalę zestawu (mediana ważona pewnością i długością fragmentu), zgodność fragmentów i te,
które dopasowały się w innej skali (inna osoba albo fragment położony na modelu w złym miejscu).
Czy ustawić wspólną skalę - settings.json, "atlas" -> "one_person" (osobno dla zestawu w jego settings.json):
    true   - zestaw od jednej osoby: zawsze,
    false  - różne osoby: nigdy (i usuwa zapisaną wcześniej),
    "auto" - nie wiadomo: tylko gdy co najmniej 60% wagi fragmentów zgadza się co do skali (±10%).
Wspólna skala jest zapisywana w scanned_bones/<zestaw>/settings.json ("atlas" -> "model_scales"),
a wszystko jest dopasowywane od nowa tylko w zakresie ± shared_scale_tolerance (10%) - tak samo potem
algorytm "atlas" i dopasowanie fragmentów. --shared-scale = ustaw teraz bez względu na zgodność.
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
    parser.add_argument("--shared-scale", action="store_true",
                        help="ustaw wspólną skalę zestawu teraz (bez względu na one_person i zgodność)")
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

    items = identify(bones, all_bones, db, p, cp, sfp, args.top)
    shared = atlas.shared_scales([(it["match"], atlas.fragment_length(it["bone"].mesh), it["bone"].label)
                                  for it in items], p)
    for mid, v in shared.items():
        out = ", ".join(f"{lbl} ×{sc / p.reference_scale:.2f}" for lbl, sc in v["outliers"])
        now = p.model_scales.get(mid)
        print(f"Wspólna skala zestawu: {v['name']} ×{v['scale'] / p.reference_scale:.2f} (mediana z "
              f"{len(v['votes'])} fragmentów, zgodność {v['agree']:.0%})" + (f"; inna skala: {out}" if out else "")
              + (f" - w ustawieniach zestawu: ×{now / p.reference_scale:.2f}" if now else ""))
    one = p.one_person
    scales = {mid: v["scale"] for mid, v in shared.items()
              if args.shared_scale or one is True or (one == "auto" and v["agree"] >= atlas.SHARED_MIN_AGREE)}
    if one is False and p.model_scales and not args.shared_scale:
        scales, why = {}, "one_person = false (różne osoby) - usuwam wspólną skalę"
    else:
        why = ("--shared-scale" if args.shared_scale else "one_person = true (jedna osoba)" if one is True
               else "one_person = \"auto\": fragmenty zgodne co do skali")
    if scales != p.model_scales and (scales or one is False):
        path = config.save_dataset_settings("atlas", {"model_scales": scales})
        config.use_dataset(config.DATASET)
        p = config.atlas_params()
        print(f"{why}: zapisano w {config._rel(path)} (\"atlas\" -> \"model_scales\"). Dopasowuję od nowa"
              + (f" ze skalą ±{p.shared_scale_tolerance:.0%}:" if scales else ":"))
        items = identify(bones, all_bones, db, p, cp, sfp, args.top)
    elif shared and not scales and one == "auto":
        print("one_person = \"auto\": skale fragmentów się nie zgadzają (zgodność < "
              f"{atlas.SHARED_MIN_AGREE:.0%}) - albo różne osoby, albo fragmenty niepewnie położone na modelu "
              "(np. trzon kości długiej); bez wspólnej skali. Jedna osoba na pewno: \"one_person\": true")

    jpath, cpath = atlas.write_reports(config.DATA_OUT, items, db, p)
    page = atlas.write_page(config.DATA_OUT / "atlas.html", items, db, p, config.view3d_params(),
                            open_browser=config.OPEN_BROWSER and not args.no_open)
    print(f"Zapisano {config._rel(jpath)}, {config._rel(cpath)}, {config._rel(page)}")
    print(f"Powierzchnie styku z atlasu (zapis w contacts/{config.DATASET}/auto/): "
          f"python scripts/detect_contacts.py --method atlas{config.dataset_flag()}")


def identify(bones, all_bones, db, p, cp, sfp, top: int) -> list[dict]:
    """Dopasowuje każdą kość do bazy (z cache), wypisuje ranking i przełomy.

    Returns:
        [{"nr", "bone", "match", "labels_atlas", "labels_features"}] - do raportów atlas.json / atlas.html.

    Raises:
        SystemExit: zła nazwa cechy w ustawieniach wykrywania.
    """
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
        lines = m.ranking_lines(top)
        print(f"  {nr:2d}. {b.label}: {lines[0]}  ({how})")
        for line in lines[1:]:
            print(f"      {line}")
        areas = ", ".join(f"{s['area_au2']:.0f}" for s in ct.surface_stats(b.mesh, found.labels)) or "-"
        print(f"      przełomy (atlas): [{areas}] a.u.²; atlas pewność {m.confidence:.0%}"
              + ("" if m.confidence >= p.min_confidence else " - za mała, tylko analiza lokalna"))
        items.append({"nr": nr, "bone": b, "match": m, "labels_atlas": found.labels, "labels_features": plain.labels})
    return items


if __name__ == "__main__":
    main()

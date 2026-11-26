"""
KROK 5: dopasowanie fragmentów kości po POWIERZCHNIACH STYKU (przełomach): RANSAC + ICP na samych przełomach.

Tylko liczy i zapisuje wynik (out/3d/<zestaw>/matching.json). Oglądanie - jak show_bones.py dla prepare_data.py:
    python scripts/tune_matching.py          # podgląd z suwakami ustawień
    python scripts/tune_matching.py --html   # statyczna strona out/3d/<zestaw>/matching.html

SKĄD DANE - trzy tryby:
    manual (domyślny) - tylko RĘCZNIE oznaczone powierzchnie z contacts/<zestaw>/<id kości>.json
                        (narzędzie: python scripts/mark_contacts.py). Kości bez nich są pomijane.
    auto              - tylko powierzchnie wykryte przez ALGORYTM (settings.json, "contacts" -> "method").
                        Brane z contacts/<zestaw>/auto/<id kości>.json (cache), a gdy pliku nie ma albo jest
                        nieaktualny (inny algorytm, inne ustawienia, zmieniona siatka kości) -
                        liczone i zapisywane tam od razu. Uwaga: bazowy "roughness" jest słaby
                        i jego powierzchnie dają głównie przypadkowe pary.
    manual-auto       - ręczne, a dla kości bez ręcznych - z algorytmu (jak w "auto").
    Powierzchnie mniejsze niż min_area_au2 (20 a.u.²) są pomijane. Potrzebne są co najmniej
    2 kości z powierzchniami. Konsola wypisuje, skąd wzięto powierzchnie każdej kości.

    Jak zmienić tryb:
      - na stałe: settings.json, sekcja "matching", klucz "source": "manual" / "auto" / "manual-auto",
      - jednorazowo: python scripts/match_surfaces.py --source auto   (ma pierwszeństwo przed settings.json).

CO ROBI
    Każda powierzchnia styku (osobno, nawet kilka na jednej kości) jest dopasowywana do
    każdej powierzchni INNEJ kości:
      1. punkty powierzchni co sample_au z normalnymi (kierunek "na zewnątrz kości"),
      2. normalne powierzchni A są ODWRACANE - dwa przełomy stykają się twarzą w twarz
         (wypukłość na A = wklęsłość na B), więc po odwróceniu A "wygląda jak" B,
      3. RANSAC na deskryptorach kształtu FPFH (zgrubnie) + ICP (dokładnie) -> przekształcenie
         sztywne A -> B (obrót + przesunięcie); ransac_seeds prób, wygrywa najlepsza,
      4. ocena ustawienia:
           styk     - pole, na którym powierzchnie leżą bliżej niż tolerance_au z przeciwnymi
                      normalnymi (mniejsze z pól obu stron) [a.u.²],
           kolizja  - pole, na którym CAŁE kości wchodzą w siebie głębiej niż collision_au [a.u.²],
           RMSE     - średnia kwadratowa odległość punktów w styku [a.u.],
           WYNIK    = styk - kolizja (+ premia z atlasu); im więcej, tym lepiej.
    Wszystkie ustawienia (z opisami): settings.json, sekcja "matching"; algorytm wykrywania: "contacts".
    Liczba par rośnie kwadratowo: N powierzchni -> ~N²/2 par, każda ~1-3 s.

ATLAS ANATOMICZNY (settings.json, "matching" -> "use_atlas": true; jednorazowo --use-atlas / --no-use-atlas)
    Z bazy kompletnych kości (scripts/build_atlas.py) wiadomo, z której kości pochodzi każdy fragment
    i jak na niej leży (to samo, co liczy identify_bones.py i algorytm "atlas" - wspólny cache).
    To zawęża dopasowanie na trzy sposoby:
      - pary fragmentów na pewno z RÓŻNYCH kości są pomijane (atlas_skip_different),
      - fragmenty leżące na tym samym modelu dostają START z atlasu (ich położenie względem siebie
        wynika z ustawień na modelu) - z niego ICP; gdy wyjdzie dobry styk (atlas_enough_cover),
        RANSAC jest pomijany, więc para liczy się w ułamku sekundy,
      - PREMIA: styk liczy się do atlas_boost (25%) mocniej, gdy to ta sama kość.
    Konsola wypisuje najbardziej prawdopodobną kość każdego fragmentu i liczbę pominiętych par.

JAK CZYTAĆ WYNIK
    - Dobra para: duży styk (dziesiątki-setki a.u.²), kolizja ~0, RMSE ~0.2 a.u., a w 3D
      fragmenty układają się w jedną kość (osie mniej więcej w jednej linii).
    - Przypadkowe pary też mają trochę styku (gładkie fragmenty "pasują" lokalnie),
      liczy się wyraźna przewaga nad resztą. Na naszych danych (ręczne oznaczenia):
      pary prawdziwe 118-179 a.u.², najlepsza przypadkowa ~55 a.u.².

Użycie (z głównego folderu projektu, po scripts/prepare_data.py i mark_contacts.py):
    python scripts/match_surfaces.py                      # wszystkie kości, tryb z settings.json
    python scripts/match_surfaces.py whole_scan 3         # tylko wybrane kości (numer albo fragment nazwy)
    python scripts/match_surfaces.py --source manual-auto # jednorazowo inny tryb
    python scripts/match_surfaces.py --no-use-atlas       # jednorazowo bez atlasu (porównanie)

Wynik: out/3d/<zestaw>/matching.json - wszystkie pary: wynik, styk, kolizja, RMSE, macierz przekształcenia T
       (4x4, układ kości A -> układ kości B, jak w out/3d/<zestaw>/bones/<id>/mesh.ply), użyte ustawienia.
Kod: src/matching_surfaces.py.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ... i w komunikatach błędów (SystemExit)

import config  # noqa: E402
import matching_surfaces as ms  # noqa: E402
from bones import load_bones, select_bones  # noqa: E402


def main() -> None:
    """Wybiera powierzchnie styku, dopasowuje każdą parę, wypisuje najlepsze i zapisuje matching.json.

    Raises:
        SystemExit: zły "source" w settings.json, brak przygotowanych kości, wzorzec nie pasuje
                    do żadnej kości albo mniej niż 2 kości mają powierzchnie styku.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("only", nargs="*", help="numery albo fragmenty nazw kości (domyślnie wszystkie)")
    parser.add_argument("--source", choices=ms.SOURCES, default=None,
                        help="skąd powierzchnie styku (domyślnie z settings.json, \"matching\" -> \"source\")")
    parser.add_argument("--use-atlas", action=argparse.BooleanOptionalAction, default=None,
                        help="kontekst z atlasu anatomicznego (domyślnie z settings.json, \"matching\" -> \"use_atlas\")")
    config.add_dataset_argument(parser)
    config.add_atlas_argument(parser)
    args = parser.parse_args()
    config.use_dataset(args.dataset)
    config.use_atlas(args.atlas, args.atlas_index)

    params = config.matching_params()
    if args.source:
        params = replace(params, source=args.source)
    if args.use_atlas is not None:
        params = replace(params, use_atlas=args.use_atlas)
    if params.source not in ms.SOURCES:
        raise SystemExit(f"settings.json, \"matching\" -> \"source\": {params.source!r}, "
                         f"dostępne: {', '.join(ms.SOURCES)}")
    contact_params = config.contact_params()
    all_bones = load_bones(features=True)
    if not all_bones:
        raise SystemExit(config.no_bones_message())
    try:
        bones = select_bones(all_bones, args.only)
    except ValueError as e:
        raise SystemExit(str(e)) from None

    algo = f" (algorytm: {contact_params.method})" if params.source != "manual" else ""
    print(f"Powierzchnie styku: {ms.SOURCE_NAMES[params.source]}{algo}"
          + (", z atlasem anatomicznym" if params.use_atlas else ""))
    start = time.perf_counter()
    surfaces, info = ms.extract_surfaces(bones, params, contact_params, log=print)
    for b in bones:
        mine = [s for s in surfaces if s.bone.id == b.id]
        areas = ", ".join(f"{s.area_au2:.0f}" for s in mine)
        what = f"{len(mine)} pow. [{areas}] a.u.²" if mine else "POMINIĘTA"
        print(f"  {all_bones.index(b) + 1:2d}. {b.label}: {what} - {info[b.id]}")
    if len({s.bone.id for s in surfaces}) < 2:
        hint = {"manual": "oznacz je: python scripts/mark_contacts.py",
                "auto": "algorytm nic nie znalazł - zmień ustawienia w settings.json, sekcja \"contacts\"",
                "manual-auto": "oznacz je: python scripts/mark_contacts.py"}[params.source]
        raise SystemExit(f"Za mało kości z powierzchniami styku (potrzebne co najmniej 2) - {hint}")

    results = ms.match_all(surfaces, params)
    print(f"Gotowe w {time.perf_counter() - start:.0f} s. Najlepsze pary (wynik = styk - kolizja, a.u.²):")
    for k, r in enumerate(results[:10], 1):
        extra = (f", atlas: ta sama kość {r['atlas_same']:.0%}, start {r['start']}"
                 if "atlas_same" in r else "")
        print(f"  {k:2d}. {r['score']:6.0f}  {r['a_name']}  <->  {r['b_name']}   "
              f"styk {r['contact_au2']:.0f}, kolizja {r['collision_au2']:.0f}, RMSE {r['rmse_au']:.2f} a.u.{extra}")
    if params.use_atlas and results:
        quick = sum(not r.get("ransac", True) for r in results)
        print(f"Atlas: {quick} z {len(results)} par bez RANSAC (wystarczył start z atlasu)")

    path = config.DATA_OUT / "matching.json"
    rep = ms.report(surfaces, results, params, contact_params)
    path.write_text(json.dumps(ms.json_safe(rep), indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Zapisano {path}")
    print("Obejrzyj: python scripts/tune_matching.py (suwaki) albo python scripts/tune_matching.py --html (plik)")

if __name__ == "__main__":
    main()

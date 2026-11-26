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
           WYNIK    = styk - kolizja (x waga osi, + premia z atlasu), domyślnie jako UDZIAŁ w średnim
                      polu obu przełomów [%] ("score_mode": "dice") - część przełomu, która nic nie
                      dotyka, obniża wynik, a duże fragmenty nie wygrywają samym polem;
                      "score_mode": "area" = dawny wynik w a.u.²; im więcej, tym lepiej.
    Wszystkie ustawienia (z opisami): settings.json, sekcja "matching"; algorytm wykrywania: "contacts".
    Liczba par rośnie kwadratowo: N powierzchni -> ~N²/2 par, każda ~1-3 s.

ATLAS ANATOMICZNY (settings.json, "matching" -> "use_atlas": true; jednorazowo --use-atlas / --no-use-atlas)
    Z bazy kompletnych kości (scripts/build_atlas.py) wiadomo, z której kości pochodzi każdy fragment
    i jak na niej leży (to samo, co liczy identify_bones.py i algorytm "atlas" - wspólny cache).
    To zawęża dopasowanie na trzy sposoby:
      - pary fragmentów na pewno z RÓŻNYCH kości są pomijane (atlas_skip_different),
      - fragmenty leżące na tym samym modelu dostają START z atlasu (ich położenie względem siebie
        wynika z ustawień na modelu) - z niego ICP; gdy wyjdzie dobry styk (start_enough_cover),
        RANSAC jest pomijany, więc para liczy się w ułamku sekundy,
      - PREMIA: styk liczy się do atlas_boost (25%) mocniej, gdy to ta sama kość.
    Konsola wypisuje najbardziej prawdopodobną kość każdego fragmentu i liczbę pominiętych par.

KOŚCI DŁUGIE: OSIE W JEDNEJ LINII (settings.json, "matching" -> "axis_check": true)
    Kawałki trzonu łączą się koniec do końca, wzdłuż jednej osi. Gdy oba fragmenty są wydłużone:
      - złożenie pod kątem (> axis_max_deg) albo z osiami rozsuniętymi (koniec do boku, obok siebie)
        traci styk - przy dwukrotności progu styk w ogóle się nie liczy,
      - dodatkowe STARTY wzdłuż osi: A przedłuża B, przełom na przełomie, obrót wokół osi co 30°.
    PROPONOWANE ZŁOŻENIE: od najlepszej pary, każdy przełom i każda para kości najwyżej raz
    ("chosen" w matching.json) - wypisane osobno pod listą najlepszych par.

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
    python scripts/match_surfaces.py --method features    # jednorazowo inna metoda (algorytm features, tryb auto)
    python scripts/match_surfaces.py --no-use-atlas       # jednorazowo bez atlasu (porównanie)
    python scripts/match_surfaces.py --mode plane         # jednorazowo tryb testowy: przybliżenia płaszczyzną

Po pełnym przebiegu (wszystkie kości, tryb manual albo auto) wynik trafia do ZAPISU METODY
out/3d/<zestaw>/methods/<metoda>/ (manual, roughness, features, atlas - każda osobno, nie nadpisują się).
Z niego czytają tune_matching.py, compare_reference.py i presentation.py - po przełączeniu metody
w settings.json (albo --method) nic nie trzeba przeliczać, jeśli ta metoda była już policzona.

Wynik: out/3d/<zestaw>/methods/<metoda>/matching.json, a ostatni przebieg (każdy, także wybrane kości
       i manual-auto) również w out/3d/<zestaw>/matching.json (podgląd: tune_matching.py --last) -
       wszystkie pary: wynik, styk, kolizja, RMSE, macierz przekształcenia T
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
import presentation  # noqa: E402
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
    parser.add_argument("--mode", choices=ms.MODES, default=None,
                        help="points = RANSAC + ICP na punktach, plane = test: przybliżenia płaszczyzną "
                             "(domyślnie z settings.json, \"matching\" -> \"mode\")")
    parser.add_argument("--method", choices=presentation.methods(), default=None,
                        help="metoda: manual = ręczne oznaczenia, inaczej algorytm wykrywania (ustawia też "
                             "--source: manual / auto); wynik trafia też do zapisu metody dla presentation.py")
    parser.add_argument("--use-atlas", action=argparse.BooleanOptionalAction, default=None,
                        help="kontekst z atlasu anatomicznego (domyślnie z settings.json, \"matching\" -> \"use_atlas\")")
    config.add_dataset_argument(parser)
    config.add_atlas_argument(parser)
    args = parser.parse_args()
    config.use_dataset(args.dataset)
    config.use_atlas(args.atlas, args.atlas_index)

    params = config.matching_params()
    contact_params = config.contact_params()
    if args.method:
        if args.source and args.source != ("manual" if args.method == "manual" else "auto"):
            raise SystemExit(f"--method {args.method} i --source {args.source} wykluczają się "
                             "(metoda manual = --source manual, algorytm = --source auto)")
        params, contact_params = presentation.method_settings(args.method)
    if args.source:
        params = replace(params, source=args.source)
    if args.use_atlas is not None:
        params = replace(params, use_atlas=args.use_atlas)
    if args.mode:
        params = replace(params, mode=args.mode)
    if params.source not in ms.SOURCES:
        raise SystemExit(f"settings.json, \"matching\" -> \"source\": {params.source!r}, "
                         f"dostępne: {', '.join(ms.SOURCES)}")
    all_bones = load_bones(features=True)
    if not all_bones:
        raise SystemExit(config.no_bones_message())
    try:
        bones = select_bones(all_bones, args.only)
    except ValueError as e:
        raise SystemExit(str(e)) from None

    algo = f" (algorytm: {contact_params.method})" if params.source != "manual" else ""
    print(f"Powierzchnie styku: {ms.SOURCE_NAMES[params.source]}{algo}"
          + (", z atlasem anatomicznym" if params.use_atlas else "")
          + (", TRYB \"plane\" (przybliżenia płaszczyzną)" if params.mode == "plane" else "")
          + "; przy powierzchniach: pole (odchylenie od płaszczyzny [a.u.] - spiłowane cięcie ~0.1)")
    start = time.perf_counter()
    surfaces, info = ms.extract_surfaces(bones, params, contact_params, log=print)
    for b in bones:
        mine = [s for s in surfaces if s.bone.id == b.id]
        areas = ", ".join(f"{s.area_au2:.0f} (odchyl. {s.fit(params)['rms_au']:.2f})" for s in mine)
        what = f"{len(mine)} pow. [{areas}] a.u.²" if mine else "POMINIĘTA"
        print(f"  {all_bones.index(b) + 1:2d}. {b.label}: {what} - {info[b.id]}")
    if len({s.bone.id for s in surfaces}) < 2:
        hint = {"manual": "oznacz je: python scripts/mark_contacts.py",
                "auto": "algorytm nic nie znalazł - zmień ustawienia w settings.json, sekcja \"contacts\"",
                "manual-auto": "oznacz je: python scripts/mark_contacts.py"}[params.source]
        raise SystemExit(f"Za mało kości z powierzchniami styku (potrzebne co najmniej 2) - {hint}")

    results = ms.match_all(surfaces, params)
    assembly = ms.assemble(results, params, surfaces)

    def line(k: int, r: dict) -> str:
        """Jedna para do konsoli: wynik, nazwy, styk, kolizja, RMSE, osie, atlas, start."""
        extra = f", osie {r['axis_deg']:.0f}° / {r['axis_offset']:.1f} promienia" if "axis_deg" in r else ""
        extra += f", atlas: ta sama kość {r['atlas_same']:.0%}" if "atlas_same" in r else ""
        return (f"  {k:2d}. {r['score']:6.0f}{r.get('score_unit', '')}  {r['a_name']}  <->  {r['b_name']}   "
                f"styk {r['contact_au2']:.0f} a.u.² ({100 * r.get('dice', 0):.0f}% przełomów), "
                f"kolizja {r['collision_au2']:.0f}, RMSE {r['rmse_au']:.2f} a.u.{extra}, start {r.get('start', '?')}")

    how = ("udział w średnim polu obu przełomów [%]" if params.score_mode == "dice" else "pole [a.u.²]")
    print(f"Gotowe w {time.perf_counter() - start:.0f} s. Najlepsze pary (wynik = styk x waga osi "
          f"(x premia z atlasu) - kolizja, jako {how}):")
    for k, r in enumerate(results[:10], 1):
        print(line(k, r))
    chosen = assembly["chosen"]
    print(f"Proponowane złożenie (każdy przełom i każda para kości najwyżej raz, wynik >= "
          f"{params.assembly_min_score:g}): {len(chosen)} par")
    for k, r in enumerate(chosen[:12], 1):
        print(line(k, r))
    used = {key for r in chosen for key in (r["a"], r["b"])}
    free = [s.name for s in surfaces if s.key not in used]
    if free:
        print(f"Przełomy bez pary ({len(free)}): {'; '.join(free)}")
        print("    -> może brakuje kawałka pomiędzy, przełom jest źle wykryty albo to nie przełom")
    if results:
        quick = sum(not r.get("ransac", True) for r in results)
        print(f"Bez RANSAC (wystarczył start z osi / z atlasu): {quick} z {len(results)} par")

    path = config.DATA_OUT / "matching.json"
    rep = ms.report(surfaces, results, params, contact_params, assembly)
    path.write_text(json.dumps(ms.json_safe(rep), indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Zapisano {path}")
    method = presentation.method_key(params.source, contact_params.method)
    flag = config.dataset_flag()
    if method is None:
        print("Zapis metody pominięty: tryb \"manual-auto\" miesza ręczne z algorytmem")
        view = f" --last{flag}"
    elif len(bones) < len(all_bones):
        print("Zapis metody pominięty: dopasowano tylko wybrane kości (pełny wynik metody, jeśli był, zostaje)")
        view = f" --last{flag}"
    else:
        snap = presentation.save_snapshot(method, rep, all_bones, params, contact_params)
        print(f"Zapis metody \"{method}\" (osobny dla każdej metody - inne metody się nie nadpisują): {snap}")
        print(f"Prezentacja: python scripts/presentation.py --method {method}{flag}")
        view = f" --method {method}{flag}"
    print(f"Obejrzyj: python scripts/tune_matching.py{view} (suwaki, dodaj --html = plik)")

if __name__ == "__main__":
    main()

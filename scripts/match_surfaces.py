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
    Powierzchnie mniejsze niż min_area_mm2 (20 mm²) są pomijane. Potrzebne są co najmniej
    2 kości z powierzchniami. Konsola wypisuje, skąd wzięto powierzchnie każdej kości.

    Jak zmienić tryb:
      - na stałe: settings.json, sekcja "matching", klucz "source": "manual" / "auto" / "manual-auto",
      - jednorazowo: python scripts/match_surfaces.py --source auto   (ma pierwszeństwo przed settings.json).

CO ROBI
    Każda powierzchnia styku (osobno, nawet kilka na jednej kości) jest dopasowywana do
    każdej powierzchni INNEJ kości:
      1. punkty powierzchni co sample_mm z normalnymi (kierunek "na zewnątrz kości"),
      2. normalne powierzchni A są ODWRACANE - dwa przełomy stykają się twarzą w twarz
         (wypukłość na A = wklęsłość na B), więc po odwróceniu A "wygląda jak" B,
      3. RANSAC na deskryptorach kształtu FPFH (zgrubnie) + ICP (dokładnie) -> przekształcenie
         sztywne A -> B (obrót + przesunięcie); ransac_seeds prób, wygrywa najlepsza,
      4. ocena ustawienia:
           styk     - pole, na którym powierzchnie leżą bliżej niż tolerance_mm z przeciwnymi
                      normalnymi (mniejsze z pól obu stron) [mm²],
           kolizja  - pole, na którym CAŁE kości wchodzą w siebie głębiej niż collision_mm [mm²],
           RMSE     - średnia kwadratowa odległość punktów w styku [mm],
           WYNIK    = styk - kolizja; im więcej, tym lepiej.
    Wszystkie ustawienia (z opisami): settings.json, sekcja "matching"; algorytm wykrywania: "contacts".
    Liczba par rośnie kwadratowo: N powierzchni -> ~N²/2 par, każda ~1-3 s.

JAK CZYTAĆ WYNIK
    - Dobra para: duży styk (dziesiątki-setki mm²), kolizja ~0, RMSE ~0.2 mm, a w 3D
      fragmenty układają się w jedną kość (osie mniej więcej w jednej linii).
    - Przypadkowe pary też mają trochę styku (gładkie fragmenty "pasują" lokalnie),
      liczy się wyraźna przewaga nad resztą. Na naszych danych (ręczne oznaczenia):
      pary prawdziwe 118-179 mm², najlepsza przypadkowa ~55 mm².

Użycie (z głównego folderu projektu, po scripts/prepare_data.py i mark_contacts.py):
    python scripts/match_surfaces.py                      # wszystkie kości, tryb z settings.json
    python scripts/match_surfaces.py whole_scan 3         # tylko wybrane kości (numer albo fragment nazwy)
    python scripts/match_surfaces.py --source manual-auto # jednorazowo inny tryb

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
    config.add_dataset_argument(parser)
    args = parser.parse_args()
    config.use_dataset(args.dataset)

    params = config.matching_params()
    if args.source:
        params = replace(params, source=args.source)
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
    print(f"Powierzchnie styku: {ms.SOURCE_NAMES[params.source]}{algo}")
    start = time.perf_counter()
    surfaces, info = ms.extract_surfaces(bones, params, contact_params)
    for b in bones:
        mine = [s for s in surfaces if s.bone.id == b.id]
        areas = ", ".join(f"{s.area_mm2:.0f}" for s in mine)
        what = f"{len(mine)} pow. [{areas}] mm²" if mine else "POMINIĘTA"
        print(f"  {all_bones.index(b) + 1:2d}. {b.label}: {what} - {info[b.id]}")
    if len({s.bone.id for s in surfaces}) < 2:
        hint = {"manual": "oznacz je: python scripts/mark_contacts.py",
                "auto": "algorytm nic nie znalazł - zmień ustawienia w settings.json, sekcja \"contacts\"",
                "manual-auto": "oznacz je: python scripts/mark_contacts.py"}[params.source]
        raise SystemExit(f"Za mało kości z powierzchniami styku (potrzebne co najmniej 2) - {hint}")

    results = ms.match_all(surfaces, params)
    print(f"Gotowe w {time.perf_counter() - start:.0f} s. Najlepsze pary (wynik = styk - kolizja, mm²):")
    for k, r in enumerate(results[:10], 1):
        print(f"  {k:2d}. {r['score']:6.0f}  {r['a_name']}  <->  {r['b_name']}   "
              f"styk {r['contact_mm2']:.0f}, kolizja {r['collision_mm2']:.0f}, RMSE {r['rmse_mm']:.2f} mm")

    path = config.DATA_OUT / "matching.json"
    rep = ms.report(surfaces, results, params, contact_params)
    path.write_text(json.dumps(ms.json_safe(rep), indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Zapisano {path}")
    print("Obejrzyj: python scripts/tune_matching.py (suwaki) albo python scripts/tune_matching.py --html (plik)")

if __name__ == "__main__":
    main()

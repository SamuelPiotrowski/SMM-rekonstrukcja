"""
Porównanie cech: ręcznie oznaczone powierzchnie styku (przełomy) vs reszta kości (kora).

Dla każdej kości z ręcznymi oznaczeniami (contacts/<zestaw>/<id>.json) liczy w każdym punkcie
szorstkość, rozrzut normalnych, wypukłość, gęstość siatki, kolor i ułożenie względem osi kości,
a potem sprawdza, jak dobrze każda cecha odróżnia przełom od kory (AUC) - i ile daje
połączenie wszystkich cech naraz. Podpowiedź do pisania algorytmu wykrywania (src/contacts.py).

Użycie (z głównego folderu projektu, po scripts/prepare_data.py i mark_contacts.py):
    python scripts/compare_surfaces.py              # wszystkie kości z ręcznymi oznaczeniami
    python scripts/compare_surfaces.py whole_scan   # tylko wybrane kości

Wynik: out/3d/<zestaw>/surface_features.html (otwiera się sam) + out/3d/<zestaw>/surface_features.csv (Excel).
Ustawienia: settings.json, sekcja "surface_features".
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ... i w komunikatach błędów (SystemExit)

import config  # noqa: E402
import surface_features as sf  # noqa: E402
from bones import load_bones, select_bones  # noqa: E402
from contacts import load_contacts  # noqa: E402
from visualization_3d import write_page  # noqa: E402


def main() -> None:
    """Liczy cechy kości z ręcznymi oznaczeniami, wypisuje ranking cech i zapisuje stronę HTML + CSV.

    Raises:
        SystemExit: brak przygotowanych kości, wzorzec nie pasuje albo żadna kość nie ma oznaczeń.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("only", nargs="*", help="numery albo fragmenty nazw kości (domyślnie wszystkie)")
    parser.add_argument("--no-open", action="store_true", help="nie otwieraj przeglądarki")
    config.add_dataset_argument(parser)
    args = parser.parse_args()
    config.use_dataset(args.dataset)

    all_bones = load_bones()
    if not all_bones:
        raise SystemExit(config.no_bones_message())
    try:
        bones = select_bones(all_bones, args.only)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    params = config.surface_feature_params()
    view = config.view3d_params()

    comps = []
    for b in bones:
        c = load_contacts(b.id, b.mesh, "manual")
        if c is None or not c.labels.any():
            continue
        start = time.perf_counter()
        comp = sf.compare_bone(b, c, params)
        comps.append(comp)
        va = comp.areas
        print(f"  {all_bones.index(b) + 1:2d}. {b.label}: przełom {va[comp.marked].sum():.0f} mm², "
              f"kora {va[comp.cortex].sum():.0f} mm² ({time.perf_counter() - start:.1f} s)")
    if not comps:
        raise SystemExit("Żadna z wybranych kości nie ma ręcznych oznaczeń - python scripts/mark_contacts.py")

    rows = sf.summary(comps, params)
    model = sf.combined_model(comps, params)
    print(f"\nCechy od najlepiej odróżniającej przełom od kory ({len(comps)} kości; AUC 0.5 = nic nie mówi):")
    for r in rows:
        flag = "" if r["consistent"] else "   (na różnych kościach w różne strony)"
        print(f"  {r['name']:28s} AUC {r['auc_mean']:.2f} [{r['auc_min']:.2f}-{r['auc_max']:.2f}]  "
              f"mediana przełom {r['median_marked']:.3g} / kora {r['median_cortex']:.3g}  -> {r['direction']}{flag}")
    if model:
        a = list(model["auc"].values())
        iou = list(model["iou"].values())
        print(f"  {'WSZYSTKIE CECHY RAZEM':28s} AUC {sum(a) / len(a):.2f} [{min(a):.2f}-{max(a):.2f}]  "
              f"(regresja logistyczna, zostaw-jedną-kość; IoU punktów {sum(iou) / len(iou):.2f})")

    out_dir = config.DATA_OUT
    csv_path = sf.write_csv(out_dir / "surface_features.csv", comps, params)
    best = rows[0]
    lead = (f"Ręcznie oznaczone powierzchnie styku (przełomy) porównane z resztą kości (korą) na "
            f"<b>{len(comps)}</b> kościach. Dla każdej cechy <b>AUC</b> = szansa, że losowy punkt przełomu ma "
            f"większą wartość niż losowy punkt kory: 0.5 = cecha nic nie mówi, 1 albo 0 = idealnie rozdziela. "
            f"Najlepsza pojedyncza cecha: <b>{best['name']}</b> (AUC {best['auc_mean']:.2f}). "
            f"Pominięte: pas {params.boundary_margin_mm:g} mm przy brzegu skanu i {params.label_band_mm:g} mm "
            f"po obu stronach granicy oznaczenia.")
    if model:
        a = list(model["auc"].values())
        lead += (f" Wszystkie cechy razem (regresja logistyczna uczona na pozostałych kościach): "
                 f"<b>AUC {sum(a) / len(a):.2f}</b>.")
    sections = [
        ("Ranking cech", "Słupek = średnie AUC po kościach, wąsy = najgorsza i najlepsza kość. Różowe: cecha "
         "większa na przełomie, niebieskie: mniejsza. Najedź myszą po opis cechy i mediany.",
         sf.ranking_figure(rows, model, comps, view)),
        ("AUC na każdej kości i powierzchni", "Kolumna „razem” = model łączący cechy, oceniany na kości, "
         "której nie widział przy uczeniu. Powierzchnie kości z kilkoma przełomami są też osobno.",
         sf.auc_heatmap_figure(comps, model, rows, params, view)),
        ("Rozkłady: przełom vs kora", "Pudełko = 25-75 percentyl z medianą, wąsy = 5-95 percentyl (ważone polem). "
         "Numery kości jak na liście w mapie AUC (od góry).", sf.distributions_figure(comps, rows, params, view)),
    ]
    if model:
        sections.append(("Co wykorzystuje model łączący cechy", "Współczynniki regresji logistycznej na "
                         "wszystkich kościach (cechy standaryzowane). Cechy podobne do siebie (np. szorstkość "
                         "w różnych promieniach) dzielą się wagą, więc pojedyncze wartości czytaj ostrożnie.",
                         sf.weights_figure(model, params, view)))
    sections.append(("Cechy na kości w 3D", "Lista po lewej: kość, po prawej: cecha. Różowa linia = ręcznie "
                     "oznaczony przełom. „Połączenie cech” = prawdopodobieństwo przełomu z modelu uczonego "
                     "bez tej kości. Kolory wspólne dla wszystkich kości (2-98 percentyl).",
                     sf.features_3d_figure(comps, model, params, view)))
    html = write_page(out_dir / "surface_features.html", "Przełom vs kora: cechy powierzchni", lead, sections,
                      view, open_browser=config.OPEN_BROWSER and not args.no_open)
    print(f"\nZapisano {html}\n        {csv_path}")


if __name__ == "__main__":
    main()

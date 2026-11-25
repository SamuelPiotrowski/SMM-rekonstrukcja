"""
Automatyczne wykrywanie powierzchni styku (algorytm) + porównanie z ręcznymi oznaczeniami.

Algorytmy (src/contacts.py, nowe dodaje się dekoratorem @detector):
    roughness - bazowy: przełom = szorstka powierzchnia (słaby),
    features  - szorstkość + rozrzut normalnych + gęstość siatki naraz, bez żadnych oznaczeń:
                przełom = wyraźnie ponad typową wartość tej samej kości (lepszy od roughness).

Użycie (z głównego folderu projektu, po scripts/prepare_data.py):
    python scripts/detect_contacts.py                      # wszystkie kości, algorytm z settings.json
    python scripts/detect_contacts.py --method features    # wybrany algorytm
    python scripts/detect_contacts.py 3 5                  # tylko kości nr 3 i 5

Algorytm i jego ustawienia: settings.json, sekcja "contacts" ("method", "rough_threshold", ...).
Wyniki: contacts/<zestaw>/auto/<id>.json - zawsze liczone od nowa (ręczne oznaczenia contacts/<zestaw>/<id>.json
NIE są ruszane). Te same pliki czyta dopasowanie (match_surfaces.py w trybie auto / manual-auto).
Jeśli kość ma ręczne oznaczenia, wypisuje IoU / precyzję / czułość algorytmu względem nich.
Podgląd: python scripts/show_bones.py --auto
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ... i w komunikatach błędów (SystemExit)

import config  # noqa: E402
import contacts as ct  # noqa: E402
from bones import load_bones, select_bones  # noqa: E402


def main() -> None:
    """Uruchamia algorytm na wybranych kościach, zapisuje wyniki i porównuje je z ręcznymi oznaczeniami.

    Wyniki trafiają do contacts/<zestaw>/auto/<id>.json (zawsze liczone od nowa, bez cache).

    Raises:
        SystemExit: brak przygotowanych kości albo wzorzec nie pasuje do żadnej kości.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("only", nargs="*", help="numery albo fragmenty nazw kości (domyślnie wszystkie)")
    parser.add_argument("--method", choices=list(ct.DETECTORS), default=None, help="algorytm")
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
    params = config.contact_params()
    method = args.method or params.method
    print(f"Algorytm: {method}, kości: {len(bones)}")

    scores = []
    for b in bones:
        nr = all_bones.index(b) + 1
        start = time.perf_counter()
        try:
            found = ct.detect_contacts(b, method, params)
        except ValueError as e:         # np. zła nazwa cechy w "feature_keys"
            raise SystemExit(str(e)) from None
        ct.save_contacts(b.mesh, found, ct.contacts_path(b.id, "auto"))
        stats = ct.surface_stats(b.mesh, found.labels)
        areas = ", ".join(f"{s['area_mm2']:.0f}" for s in stats) or "-"
        line = f"  {nr:2d}. {b.label}: {len(stats)} pow. [{areas}] mm2 ({time.perf_counter() - start:.1f} s)"
        manual = ct.load_contacts(b.id, b.mesh, "manual")
        if manual is not None and manual.labels.any():
            r = ct.compare(b.mesh, found.labels, manual.labels)
            scores.append(r)
            line += f"  | vs ręczne: IoU {r['iou']:.2f}, precyzja {r['precision']:.2f}, czułość {r['recall']:.2f}"
        print(line)
    if scores:
        mean = {k: sum(s[k] for s in scores) / len(scores) for k in ("iou", "precision", "recall")}
        print(f"Średnio na {len(scores)} kościach z ręcznymi oznaczeniami: IoU {mean['iou']:.2f}, "
              f"precyzja {mean['precision']:.2f}, czułość {mean['recall']:.2f}")
    else:
        print("Brak ręcznych oznaczeń do porównania (python scripts/mark_contacts.py).")
    print("Podgląd: python scripts/show_bones.py --auto")


if __name__ == "__main__":
    main()

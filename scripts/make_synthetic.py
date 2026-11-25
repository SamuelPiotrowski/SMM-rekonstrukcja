"""
Tworzy osobny zestaw danych z syntetycznymi kośćmi 3D pociętymi na fragmenty (z pewną prawdą).

Każda kość to "rurka" z nasadami, cięta chropowatymi, skośnymi przełomami na kilka fragmentów;
każdy fragment zapisywany jest jak osobny skan (losowo obrócony i przesunięty, z szumem skanera).
Szczegóły: src/synthetic_bones.py, ustawienia: settings.json -> "synthetic".

Zapisuje:
    scanned_bones/<nazwa>/frag_XX/frag_XX.ply   skany fragmentów (numery wymieszane między kośćmi)
    scanned_bones/<nazwa>/ground_truth.json     prawda: kości, pary fragmentów, położenia
    scanned_bones/<nazwa>/settings.json         ustawienia zestawu (łączenie skanów wyłączone)
    contacts/<nazwa>/frag_XX.json               prawdziwe przełomy jako "ręczne" oznaczenia styku
    out/3d/<nazwa>/synthetic.html               podgląd: kości złożone, ten sam kolor przełomu = para

Użycie (z głównego folderu projektu):
    python scripts/make_synthetic.py                       # zestaw "synthetic"
    python scripts/make_synthetic.py syn_b --seed 7 --bones 10
    python scripts/make_synthetic.py --force               # nadpisz wcześniej wygenerowany zestaw
    python scripts/make_synthetic.py --prepare             # od razu przygotuj kości (prepare_data)
    python scripts/make_synthetic.py --preset chip         # gotowy układ -> zestaw "synthetic_chip":
                                                           # 1 kość, 4 części: zwykły przełom + płaski odłamek
Potem (jeśli bez --prepare):
    python scripts/prepare_data.py --dataset synthetic
    python scripts/show_bones.py --dataset synthetic
    python scripts/detect_contacts.py --dataset synthetic  # algorytm vs prawdziwe przełomy
    python scripts/match_surfaces.py --dataset synthetic
"""

import argparse
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ... i w komunikatach błędów (SystemExit)

import config  # noqa: E402
import synthetic_bones as sb  # noqa: E402


def clear_previous(name: str, force: bool) -> None:
    """Usuwa poprzednio WYGENEROWANY zestaw `name` (tylko z --force); prawdziwych skanów nie rusza.

    Raises:
        SystemExit: zestaw istnieje, a nie podano --force, albo to nie jest zestaw syntetyczny.
    """
    data_in, data_out, contacts = config.dataset_dirs(name)
    if not data_in.exists():
        return
    gt = sb.load_ground_truth(data_in)
    if gt is None or gt.get("format") != sb.FORMAT:
        raise SystemExit(f"Folder {config._rel(data_in)} istnieje i nie jest zestawem syntetycznym "
                         f"(brak {sb.GROUND_TRUTH}) - wybierz inną nazwę.")
    if not force:
        raise SystemExit(f"Zestaw \"{name}\" już istnieje ({config._rel(data_in)}). Nadpisanie: --force, "
                         "inny zestaw: podaj inną nazwę.")
    shutil.rmtree(data_in)
    for fid in gt.get("fragments", {}):
        (contacts / f"{fid}.json").unlink(missing_ok=True)
    if (contacts / "auto").is_dir():
        shutil.rmtree(contacts / "auto")        # wyniki algorytmu (cache) dla starych fragmentów


def main() -> None:
    """Czyta argumenty, generuje zestaw, zapisuje go i pokazuje podgląd."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", default=None,
                        help="nazwa nowego zestawu (podfolder scanned_bones/), domyślnie: synthetic "
                             "(z --preset: synthetic_<preset>)")
    parser.add_argument("--preset", choices=sorted(sb.PRESETS), default=None,
                        help="gotowy układ kości (synthetic_bones.PRESETS), np. chip = 1 kość: zwykły przełom "
                             "+ płaski odłamek między dwoma przełomami")
    parser.add_argument("--seed", type=int, default=None, help="ziarno losowania (domyślnie z settings.json)")
    parser.add_argument("--bones", type=int, default=None, help="liczba kości (domyślnie z settings.json)")
    parser.add_argument("--force", action="store_true", help="nadpisz wcześniej wygenerowany zestaw o tej nazwie")
    parser.add_argument("--no-contacts", action="store_true",
                        help="bez prawdziwych przełomów w contacts/<nazwa>/ (np. żeby oznaczać ręcznie)")
    parser.add_argument("--no-preview", action="store_true", help="bez podglądu HTML")
    parser.add_argument("--prepare", action="store_true", help="od razu uruchom prepare_data dla tego zestawu")
    args = parser.parse_args()

    name = (args.name or (f"synthetic_{args.preset}" if args.preset else "synthetic")).strip().strip("/\\")
    params = config.synthetic_params()
    if args.preset:
        params = replace(params, **sb.PRESETS[args.preset])
    if args.seed is not None:
        params = replace(params, seed=args.seed)
    if args.bones is not None:
        params = replace(params, n_bones=args.bones)
    clear_previous(name, args.force)

    start = time.perf_counter()
    synth = sb.generate(params)
    data_in, data_out, contacts = config.dataset_dirs(name)
    gt = sb.write_dataset(synth, data_in, None if args.no_contacts else contacts)
    print(f"Zestaw \"{name}\": {len(synth.bones)} kości, {len(synth.fragments)} fragmentów, "
          f"{len(gt['pairs'])} par ({time.perf_counter() - start:.1f} s, seed {params.seed})")
    for b in synth.bones:
        chips = sum(c == "chip" for c in b["cuts"])
        extra = f", płaskie odłamki: {chips}" if chips else ""
        print(f"  {b['id']}: {b['length_au']:.0f} a.u., promień {b['radius_au']:.1f} a.u.{extra} -> "
              f"{' | '.join(b['fragments'])}")
    areas = sorted(p["area_au2"] for p in gt["pairs"])
    if areas:
        print(f"  pola przełomów: {areas[0]:.0f}-{areas[-1]:.0f} a.u.² (mediana {areas[len(areas) // 2]:.0f})")
    print(f"Skany: {config._rel(data_in)}/, prawda: {config._rel(data_in / sb.GROUND_TRUTH)}")
    if not args.no_contacts:
        print(f"Prawdziwe przełomy (jako ręczne oznaczenia): {config._rel(contacts)}/")

    if not args.no_preview:
        from visualization_3d import save_figure
        path = save_figure(sb.preview_figure(synth), data_out / "synthetic.html", config.view3d_params(),
                           open_browser=config.OPEN_BROWSER, title=f"Syntetyczne kości - {name}")
        print(f"Podgląd: {config._rel(path)}")

    if args.prepare:
        from pipeline import prepare_all
        config.use_dataset(name)
        prepare_all()
    flag = "" if name == (config.SETTINGS.get("dataset") or "") else f" --dataset {name}"
    print("\nDalej:" + ("" if args.prepare else f"\n    python scripts/prepare_data.py{flag}") +
          f"\n    python scripts/show_bones.py{flag}"
          f"\n    python scripts/detect_contacts.py{flag}     # algorytm vs prawdziwe przełomy"
          f"\n    python scripts/match_surfaces.py{flag}")


if __name__ == "__main__":
    main()

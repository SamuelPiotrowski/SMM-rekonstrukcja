"""
PORÓWNANIE Z KOŚCIĄ WZORCOWĄ: jak naprawdę mają być złożone fragmenty i jak dobrze zrobiło to dopasowanie.

Zestaw testowy ma skan całej kości (domyślnie kość z listy "exclude" sekcji "contacts", np. "Test cała kość
strzałkowa") - wzór końcowego złożenia. Skrypt (src/reference.py):
    1. kładzie każdy fragment na kości wzorcowej (RANSAC + ICP, bez skali - ta sama kość i skaner) i wypisuje,
       jaka część fragmentu leży na wzorcu (kora; przełomy nie leżą, więc 100% jest niemożliwe),
    2. wypisuje PRAWDZIWE PARY PRZEŁOMÓW: powierzchnie styku dwóch fragmentów, które w układzie wzorca leżą
       na sobie (z trybu "source" - ręczne / algorytm), i przełomy bez partnera (brakujący kawałek, cięcie),
    3. ocenia wynik dopasowania metody (out/3d/<zestaw>/methods/<metoda>/matching.json, metoda z settings.json
       albo --method; --last = ostatni przebieg out/3d/<zestaw>/matching.json): dla najlepszych par
       i proponowanego złożenia - czy para jest prawdziwa i jaki jest błąd ustawienia (średnia odległość
       wierzchołków, obrót, przesunięcie).
Położenia na wzorcu są w cache: out/3d/<zestaw>/reference.json.

PRAWDA OD CZŁOWIEKA (scanned_bones/<zestaw>/truth.json, {"pairs": [["test 1", "test 4"], ...]} - nazwy albo id
fragmentów): gdy plik jest, to on decyduje, które pary fragmentów są prawdziwe (wzorzec bywa tylko kością tego samego
typu, nie tym samym egzemplarzem - wtedy jego położenia nie są prawdą); błąd ustawienia z wzorca jest wtedy pomijany.

Użycie (z głównego folderu projektu, po match_surfaces.py):
    python scripts/compare_reference.py
    python scripts/compare_reference.py --source manual           # prawdziwe pary z ręcznych oznaczeń
    python scripts/compare_reference.py --method atlas            # oceń wynik metody atlas (bez przeliczania)
    python scripts/compare_reference.py --reference "cała kość"   # inna nazwa kości wzorcowej
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ... i w komunikatach błędów (SystemExit)

import config  # noqa: E402
import matching_surfaces as ms  # noqa: E402
import presentation  # noqa: E402
import reference  # noqa: E402
from bones import load_bones  # noqa: E402
from config import read_json  # noqa: E402


def human_truth(frags: list) -> set | None:
    """Prawdziwe pary fragmentów z scanned_bones/<zestaw>/truth.json ({"pairs": [[nazwa, nazwa], ...]}) albo None.

    Args:
        frags: fragmenty zestawu (nazwy w pliku = label albo id).

    Returns:
        {frozenset((id, id))} albo None, gdy pliku nie ma.

    Raises:
        SystemExit: nazwa z pliku nie pasuje do żadnego fragmentu.
    """
    path = config.DATA_IN / "truth.json"
    if not path.exists():
        return None
    ids = {**{f.label: f.id for f in frags}, **{f.id: f.id for f in frags}}
    out = set()
    for pair in read_json(path).get("pairs", []):
        missing = [x for x in pair if x not in ids]
        if missing:
            raise SystemExit(f"{path}: nie ma fragmentu {missing} - dostępne: {', '.join(f.label for f in frags)}")
        out.add(frozenset(ids[x] for x in pair))
    return out


def main() -> None:
    """Kładzie fragmenty na wzorcu, wypisuje prawdziwe pary przełomów i ocenia matching.json.

    Raises:
        SystemExit: brak kości wzorcowej albo przygotowanych kości.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=ms.SOURCES, default=None,
                        help="skąd przełomy do prawdziwych par (domyślnie z settings.json, \"matching\" -> \"source\")")
    parser.add_argument("--reference", action="append", default=None, metavar="TEKST",
                        help="nazwa kości wzorcowej (część nazwy; domyślnie lista \"exclude\" sekcji \"contacts\")")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--method", choices=presentation.methods(), default=None,
                       help="oceń wynik tej metody (out/3d/<zestaw>/methods/<metoda>/; domyślnie metoda z settings.json)")
    which.add_argument("--last", action="store_true",
                       help="oceń ostatni przebieg out/3d/<zestaw>/matching.json zamiast zapisu metody")
    which.add_argument("--matching", default=None, metavar="PLIK", help="oceń wynik dopasowania z tego pliku")
    config.add_dataset_argument(parser)
    args = parser.parse_args()
    config.use_dataset(args.dataset)

    cp = config.contact_params()
    p = replace(config.matching_params(), use_atlas=False)
    if args.source:
        p = replace(p, source=args.source)
    elif args.method == "manual":            # ocena wyniku z ręcznych przełomów -> prawdziwe pary też z ręcznych
        p = replace(p, source="manual")
    if args.method is None and not args.last and not args.matching and p.source == "manual":
        args.method = "manual"               # --source manual -> oceń zapis metody "manual", nie metody z settings.json
    bones = load_bones(features=True)
    if not bones:
        raise SystemExit(config.no_bones_message())
    ref, frags = reference.split_reference(bones, args.reference or cp.exclude)
    if ref is None:
        raise SystemExit("Brak kości wzorcowej - podaj --reference \"<część nazwy>\" (np. \"cała kość\")")
    print(f"Kość wzorcowa: {ref.label}; fragmenty: {', '.join(f.label for f in frags)}")
    poses = reference.register_all(ref, frags)
    for f in frags:
        q = poses[f.id]
        print(f"  {f.label}: na wzorcu {q['fitness']:.0%} punktów (RMSE {q['rmse_au']:.2f} a.u.)"
              + ("  <- SŁABO, położenie niepewne" if q["fitness"] < 0.5 else ""))

    surfaces, _ = ms.extract_surfaces(frags, p, cp)
    truth = reference.true_contacts(surfaces, poses)
    names = {s.key: s.name for s in surfaces}
    print(f"\nPRAWDZIWE PARY PRZEŁOMÓW wg wzorca (powierzchnie: {ms.SOURCE_NAMES[p.source]}):")
    for t in truth:
        print(f"  {names[t['a']]}  <->  {names[t['b']]}   styk {t['contact_au2']:.0f} a.u.² "
              f"(pokrycie {t['cover_a']:.0%} / {t['cover_b']:.0%})")
    paired = {k for t in truth for k in (t["a"], t["b"])}
    alone = [s.name for s in surfaces if s.key not in paired]
    if alone:
        print(f"  bez partnera na wzorcu: {'; '.join(alone)}  (brakujący kawałek, cięcie albo nie przełom)")
    true_keys = {frozenset((t["a"], t["b"])) for t in truth}
    true_bones = {frozenset((t["a"].rsplit("#", 1)[0], t["b"].rsplit("#", 1)[0])) for t in truth}
    human = human_truth(frags)
    if human is not None:
        labels = {f.id: f.label for f in frags}
        print(f"\nPRAWDA z {config._rel(config.DATA_IN / 'truth.json')}: "
              + ", ".join(" - ".join(labels[b] for b in sorted(pr)) for pr in human)
              + "  (ma pierwszeństwo przed wzorcem; błąd ustawienia z wzorca pominięty)")

    if args.matching:
        path = Path(args.matching)
        if not path.exists():
            print(f"\nBrak {path}")
            return
    else:
        try:
            path, _, notes = presentation.find_matching(args.method, args.last)
        except SystemExit as e:
            print(f"\n{e}")
            return
        for n in notes:
            print(f"UWAGA: {n}")
    rep = read_json(path)
    by_bone = {b.id: b for b in frags}
    pairs = [r for r in rep["pairs"] if r["a"].rsplit("#", 1)[0] in poses and r["b"].rsplit("#", 1)[0] in poses]
    print(f"\nOCENA {config._rel(path)} (metoda: {presentation.method_label(presentation.report_method(rep))}, "
          f"para dobrze złożona = błąd < {reference.POSE_OK_AU:g} a.u.):")

    def verdict(r: dict) -> str:
        """Opis pary: prawdziwa / sąsiednie fragmenty / fałszywa + błąd ustawienia."""
        bones_pair = frozenset((r["a"].rsplit("#", 1)[0], r["b"].rsplit("#", 1)[0]))
        if human is not None:
            return ("PRAWDZIWA" if bones_pair in human else "fałszywa") + (
                f": osie {r['axis_deg']:.0f}°, kolizja {r['collision_au2']:.0f} a.u.²" if "axis_deg" in r else "")
        e = reference.pose_error(by_bone, poses, r)
        kind = ("PRAWDZIWA" if frozenset((r["a"], r["b"])) in true_keys else
                "sąsiednie fragmenty, inne przełomy" if bones_pair in true_bones else "fałszywa")
        ok = " DOBRZE" if kind == "PRAWDZIWA" and e["err_au"] < reference.POSE_OK_AU else ""
        return (f"{kind}{ok}: błąd {e['err_au']:.1f} a.u., obrót {e['rot_deg']:.0f}°, przesunięcie {e['move_au']:.1f} a.u.")

    for k, r in enumerate(pairs[:8], 1):
        print(f"  {k}. {r['score']:5.0f}{r.get('score_unit', '')}  {r['a_name']} <-> {r['b_name']}: {verdict(r)}")
    chosen = [r for r in pairs if r.get("chosen")]
    print(f"Proponowane złożenie ({len(chosen)} par):")
    for r in chosen:
        print(f"  {r['a_name']} <-> {r['b_name']}: {verdict(r)}")
    if human is not None:
        good = sum(frozenset((r["a"].rsplit("#", 1)[0], r["b"].rsplit("#", 1)[0])) in human for r in chosen)
        print(f"Wynik: {good} z {len(human)} prawdziwych par fragmentów w proponowanym złożeniu, "
              f"{len(chosen) - good} par fałszywych.")
    else:
        good = sum(frozenset((r["a"], r["b"])) in true_keys
                   and reference.pose_error(by_bone, poses, r)["err_au"] < reference.POSE_OK_AU for r in chosen)
        print(f"Wynik: {good} z {len(truth)} prawdziwych par dobrze złożonych w proponowanym złożeniu, "
              f"{len(chosen) - good} par złych / niedokładnych.")
    compare_whole(ref, by_bone, rep, poses)


def compare_whole(ref, by_bone: dict, rep: dict, poses: dict) -> None:
    """PORÓWNANIE CAŁOŚCI: każda złożona grupa (matching.json -> "assembly") kładziona na wzorcu jako jeden obiekt;
    wypisuje dopasowanie kory całej grupy i każdego fragmentu, zapisuje out/3d/<zestaw>/reference.html."""
    asm = rep.get("assembly") or {}
    members: dict = {}
    for b, g in asm.get("groups", {}).items():
        if b in by_bone:
            members.setdefault(g, []).append(b)
    groups = [sorted(m) for m in members.values() if len(m) >= 2]
    if not groups:
        print("\nPORÓWNANIE CAŁOŚCI: w złożeniu nie ma grupy z co najmniej 2 fragmentami "
              "(albo matching.json bez \"assembly\" - przelicz match_surfaces.py z assembly_beam > 0)")
        return
    print("\nPORÓWNANIE CAŁOŚCI (złożona grupa kładziona na wzorcu jako jeden obiekt):")
    done = []
    for m in groups:
        res = reference.compare_group(ref, by_bone, m, asm["poses"], poses)
        done.append((m, res))
        print(f"  grupa {' + '.join(by_bone[b].label for b in m)}: na wzorcu {res['fitness']:.0%} kory "
              f"(RMSE {res['rmse_au']:.2f} a.u.)")
        for b in m:
            f = res["fragments"][b]
            print(f"    {by_bone[b].label}: w złożeniu {f['fitness']:.0%} kory na wzorcu (osobno {f['own_fitness']:.0%}), "
                  f"położenie różni się od osobnego o {f['shift_au']:.1f} a.u.")
    alone = sorted(b for b in by_bone if not any(b in m for m in groups))
    if alone:
        print(f"  poza złożeniem: {', '.join(by_bone[b].label for b in alone)}")
    from visualization_3d import write_page

    page = write_page(config.DATA_OUT / "reference.html", "Porównanie złożenia z kością wzorcową",
                      f"Złożone grupy (kolory) położone na kości wzorcowej {ref.label} (szara). Kursor na fragmencie: "
                      "jaka część jego kory leży na wzorcu w złożeniu i osobno.",
                      [("Złożenie na wzorcu", "", reference.whole_figure(ref, by_bone, done, asm["poses"],
                                                                          config.view3d_params()))],
                      config.view3d_params(), open_browser=config.OPEN_BROWSER)
    print(f"Zapisano {config._rel(page)}")


if __name__ == "__main__":
    main()

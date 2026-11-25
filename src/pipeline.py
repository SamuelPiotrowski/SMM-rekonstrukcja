"""
Cały etap przygotowania danych: skany -> kości.

    from pipeline import prepare_all
    bones = prepare_all()

Kroki: wyszukanie skanów (z podfolderami) -> wczytanie i naprawa -> rejestracja par
(cache w DATA_OUT/cache/) -> grupowanie skanów tej samej kości -> łączenie i usuwanie
szumu -> chmura punktów i FPFH -> zapis do DATA_OUT/bones/<id>/, merging.json, summary.csv.
Parametry domyślne pochodzą z settings.json (config.py); można podać własne.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d

import bones as bones_mod
import merging
from data_parsing import ScanParams, find_scans


def prepare_all(data_in: Path | None = None, out_dir: Path | None = None, *,
                scan: ScanParams | None = None, merge: merging.MergeParams | None = None,
                features: bones_mod.FeatureParams | None = None, verbose: bool = True) -> list[bones_mod.Bone]:
    """Przetwarza wszystkie skany z `data_in` i zapisuje kości do `out_dir`.

    Kolejne kroki opisuje docstring modułu. Zapisuje: out_dir/bones/<id>/ (każda kość),
    out_dir/cache/pairs.json (wyniki rejestracji par - kolejne uruchomienia są szybkie),
    out_dir/merging.json i out_dir/summary.csv. Foldery kości z poprzednich uruchomień,
    których już nie ma, są usuwane.

    Args:
        data_in:  folder ze skanami (domyślnie DATA_IN z settings.json).
        out_dir:  folder wyników (domyślnie DATA_OUT z settings.json).
        scan:     ustawienia wyszukiwania skanów (domyślnie sekcja "scans").
        merge:    ustawienia łączenia skanów (domyślnie sekcja "merging").
        features: ustawienia chmury i FPFH (domyślnie sekcja "features").
        verbose:  wypisuj postęp w konsoli (z czasem od startu).

    Returns:
        Lista zapisanych kości (z chmurą i FPFH), posortowana po id - jak w load_bones.

    Raises:
        FileNotFoundError: brak folderu `data_in` albo brak w nim skanów.
    """
    import config

    data_in = Path(data_in or config.DATA_IN)
    out_dir = Path(out_dir or config.DATA_OUT)
    scan = scan or config.scan_params()
    merge = merge or config.merge_params()
    features = features or config.feature_params()
    start = time.perf_counter()

    def log(msg):
        """Wypisuje komunikat z czasem od startu (tylko gdy verbose).

        Args:
            msg: tekst komunikatu.
        """
        if verbose:
            print(f"[{time.perf_counter() - start:6.1f} s] {msg}", flush=True)

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)   # bez ostrzeżeń RANSAC
    files, excluded = find_scans(data_in, scan)
    if not files:
        raise FileNotFoundError(f"Brak skanów ({', '.join(scan.preferred_formats)}) w {data_in} "
                                "(szukane także w podfolderach)")
    log(f"Znaleziono {len(files)} skanów w {data_in}")
    for p in excluded:
        log(f"  pominięty (pasuje do scans.exclude): {p.relative_to(data_in)}")

    prepared = []
    for f in files:
        prepared.append(merging.prepare_scan(f, scan, merge))
        s = prepared[-1]
        extra = f", {len(s.comp_areas)} kawałki" if len(s.comp_areas) > 1 else ""
        log(f"  wczytano {f.label}: {len(s.mesh.triangles)} trójkątów, "
            f"{s.comp_areas.sum():.0f} mm2{extra}")

    if merge.enabled or merge.groups:
        pairs = merging.pairwise(prepared, merge, out_dir / "cache" / "pairs.json", progress=log)
    else:
        pairs = {}
    groups = merging.group_scans(prepared, pairs, merge)
    log(f"Łączenie: {len(groups)} kości z {len(prepared)} skanów")
    merged_all = [merging.merge_group(prepared, g, pairs, merge) for g in groups]
    order = sorted(range(len(groups)), key=lambda k: merged_all[k].id)   # ta sama kolejność co load_bones
    groups = [groups[k] for k in order]

    result = []
    for nr, k in enumerate(order, 1):
        merged = merged_all[k]
        bone = bones_mod.finalize_bone(merged, features)
        bones_mod.save_bone(bone, merged, out_dir)
        result.append(bone)
        m = bone.meta
        names = ", ".join(s["label"] + (" (ref.)" if s["reference"] and len(m["scans"]) > 1 else "")
                          for s in m["scans"])
        noise = f", usunięty szum: {m['noise_removed_mm2']:.0f} mm2" if m["noise_removed_mm2"] else ""
        log(f"  {nr:2d}. {bone.label}: {len(m['scans'])} skan(y) [{names}] -> "
            f"{m['stats']['area_mm2']:.0f} mm2{noise}")
        for s in m["scans"]:
            for c in s["components"]:
                if not c["kept"]:
                    log(f"      {s['label']}: usunięto kawałek {c['area_mm2']:.0f} mm2 - {c['reason']}")
        weak = [s for s in m["scans"] if not s["reference"] and s["best_overlap_in_group"] < merge.min_overlap]
        for s in weak:
            log(f"      UWAGA: {s['label']} słabo pasuje do reszty grupy (nakładanie "
                f"{s['best_overlap_in_group']:.2f}) - sprawdź: python scripts/show_merging.py")

    removed = bones_mod.remove_stale(out_dir, {b.id for b in result})
    if removed:
        log(f"Usunięto stare wyniki: {', '.join(removed)}")
    write_merging_report(prepared, pairs, groups, result, merge, out_dir)
    summary = bones_mod.write_summary(result, out_dir)
    log(f"Gotowe: {len(result)} kości w {out_dir / 'bones'}, tabela: {summary}")
    return result


def write_merging_report(prepared, pairs, groups, result, merge, out_dir: Path) -> Path:
    """Zapisuje out_dir/merging.json - dane do wizualizacji łączenia (scripts/show_merging.py).

    Zawartość:
        "params": {"min_overlap", "tolerance_mm"},
        "scans":  [{"id", "label", "path", "bone": id kości, do której trafił skan}],
        "pairs":  [{"a", "b": id skanów, "overlap": nakładanie}],
        "bones":  [{"id", "label", "scans": [id skanów]}].

    Args:
        prepared: wszystkie skany (merging.prepare_scan).
        pairs:    wyniki rejestracji par {(i, j): wynik} (merging.pairwise).
        groups:   grupy skanów w tej samej kolejności co `result`.
        result:   zapisane kości.
        merge:    użyte ustawienia łączenia.
        out_dir:  folder wyników.

    Returns:
        Ścieżka do merging.json.
    """
    scan_bone = {}
    for g, b in zip(groups, result):
        for i in g["members"]:
            scan_bone[prepared[i].file.id] = b.id
    report = {
        "params": {"min_overlap": merge.min_overlap, "tolerance_mm": merge.tolerance_mm},
        "scans": [{"id": s.file.id, "label": s.file.label, "path": str(s.file.path),
                   "bone": scan_bone.get(s.file.id)} for s in prepared],
        "pairs": [{"a": prepared[i].file.id, "b": prepared[j].file.id, "overlap": round(r["overlap"], 4)}
                  for (i, j), r in sorted(pairs.items())],
        "bones": [{"id": b.id, "label": b.label, "scans": [s["id"] for s in b.meta["scans"]]} for b in result],
    }
    path = out_dir / "merging.json"
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    prepare_all()

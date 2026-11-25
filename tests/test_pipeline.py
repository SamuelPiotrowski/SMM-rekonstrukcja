"""
Test całego przygotowania danych na SZTUCZNYCH skanach (nie potrzebuje prawdziwych skanów).

Tworzy w folderze tymczasowym:
    - fragment A (kula ~25 mm z losowymi wybrzuszeniami) zeskanowany 3 razy, za każdym
      razem częściowo, z innej strony i w innym układzie współrzędnych (obrót + przesunięcie),
      w podfolderach o TYCH SAMYCH nazwach plików (jak kolejne sesje Medita), jeden głębiej,
    - do jednego skanu A dołożony szum: oderwany kawałek 60 mm od fragmentu,
    - fragment B (inny kształt) zeskanowany raz,
    - skan "occlusion" (A i B razem) - ma być pominięty,
    - kopię skanu A1 jako .stl - ma być wybrany .ply.
Sprawdza: wyszukiwanie w podfolderach, grupowanie (A razem, B osobno), usunięcie szumu,
zapis / odczyt kości, oznaczenia styku (zapis, odczyt, przeniesienie na inną siatkę),
algorytm wykrywania (zapis w contacts/auto/, cache), dopasowanie powierzchni styku
(tryby manual / auto / manual-auto, odnalezienie drugiej strony przełomu), przeliczanie jednej
pary w podglądzie z suwakami (match_tool), budowę wizualizacji, zestawy danych oraz przeniesienie
oznaczeń po zmianie nazwy folderu ze skanem i po wyłączeniu łączenia skanów.

Uruchomienie (z głównego folderu projektu):
    python tests/test_pipeline.py
"""

import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

import config  # noqa: E402
import contacts as ct  # noqa: E402
import matching_surfaces as ms  # noqa: E402
import surface_features as sf  # noqa: E402
from bones import FeatureParams, load_bones  # noqa: E402
from data_parsing import ScanParams, find_scans  # noqa: E402
from match_tool import MatchState, matching_page  # noqa: E402
from merging import MergeParams  # noqa: E402
from pipeline import prepare_all  # noqa: E402
from visualization_3d import (View3DParams, bones_figure, match_matrix_figure, matching_figure,  # noqa: E402
                              merging_figure)


def bumpy_sphere(seed: int, radius: float = 12.0) -> trimesh.Trimesh:
    """Kula z losowymi wybrzuszeniami - kształt z cechami, które da się dopasować.

    Args:
        seed:   ziarno - inne ziarno = inny kształt (inny "fragment kości").
        radius: promień kuli przed wybrzuszeniami [mm] (30 wybrzuszeń i wgłębień, każde do ±2 mm).

    Returns:
        Zamknięta siatka trimesh (icosphere, ~82 tys. trójkątów), środek w (0, 0, 0).
    """
    rng = np.random.default_rng(seed)
    m = trimesh.creation.icosphere(subdivisions=6, radius=1.0)
    u = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    r = np.full(len(u), radius)
    for _ in range(30):
        c = rng.normal(size=3)
        c /= np.linalg.norm(c)
        a, s = rng.uniform(-2.0, 2.0), rng.uniform(0.15, 0.45)
        r += a * np.exp(-np.sum((u - c) ** 2, axis=1) / (2 * s * s))
    return trimesh.Trimesh(u * r[:, None], m.faces, process=False)


def partial_scan(mesh: trimesh.Trimesh, view: np.ndarray, seed: int) -> trimesh.Trimesh:
    """Część powierzchni widoczna z kierunku `view`, w losowym układzie współrzędnych.

    Zostają trójkąty po stronie `view` (trochę ponad połowę kuli), więc dwa skany z różnych
    stron częściowo się pokrywają - jak prawdziwe skany tego samego fragmentu.
    Potem losowy obrót i przesunięcie do ±30 mm (ziarno `seed`).

    Args:
        mesh: cały fragment (np. z bumpy_sphere), środek w (0, 0, 0).
        view: kierunek "skanera" (nie musi być jednostkowy).
        seed: ziarno losowego układu współrzędnych skanu.

    Returns:
        Nowa siatka (otwarta - z brzegiem); `mesh` się nie zmienia.
    """
    view = np.asarray(view, dtype=float) / np.linalg.norm(view)
    centers = mesh.triangles_center / np.linalg.norm(mesh.triangles_center, axis=1, keepdims=True)
    part = mesh.submesh([np.flatnonzero(centers @ view > -0.15)], append=True)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = q, rng.uniform(-30, 30, size=3)
    part.apply_transform(T)
    return part


def save(mesh: trimesh.Trimesh, path: Path, color=(214, 196, 168)) -> None:
    """Zapisuje siatkę z jednolitym kolorem wierzchołków (jak skaner) i tworzy foldery.

    Args:
        mesh:  siatka (nie jest zmieniana - kolor dostaje kopia).
        path:  plik docelowy; rozszerzenie wybiera format (.ply, .stl).
        color: kolor RGB 0-255 wszystkich wierzchołków.
    """
    mesh = mesh.copy()
    mesh.visual.vertex_colors = np.tile([*color, 255], (len(mesh.vertices), 1)).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def make_scans(root: Path) -> None:
    """Tworzy w `root` sztuczne skany opisane w docstringu modułu (A1-A3 + szum, B, occlusion, kopia .stl).

    Args:
        root: folder, który będzie DATA_IN testu (podfoldery są tworzone).
    """
    a, b = bumpy_sphere(1), bumpy_sphere(2, radius=10.0)
    a1 = partial_scan(a, [1, 0, 0], 11)
    a2 = partial_scan(a, [0.3, 1, 0.2], 12)
    a3 = partial_scan(a, [-0.4, 0.4, 1], 13)
    noise = trimesh.creation.icosphere(subdivisions=4, radius=4.0)
    noise.apply_translation(a2.vertices.mean(axis=0) + [60.0, 0, 0])
    save(a1, root / "01.01.2026-test" / "01.01.2026-test-upperjaw.ply")
    save(a1, root / "01.01.2026-test" / "01.01.2026-test-upperjaw.stl")
    save(partial_scan(b, [0, 0, 1], 21), root / "01.01.2026-test" / "01.01.2026-test-lowerjaw.ply")
    save(trimesh.util.concatenate([a2, noise]), root / "01.01.2026-test_(1)" / "01.01.2026-test-upperjaw.ply")
    save(a3, root / "01.01.2026-test_(2)" / "glebiej" / "01.01.2026-test-upperjaw.ply")
    save(trimesh.util.concatenate([a1, partial_scan(b, [1, 0, 0], 22)]),
         root / "01.01.2026-test" / "01.01.2026-test-occlusion.ply")


def check(cond: bool, msg: str) -> None:
    """Wypisuje "OK" albo "BŁĄD" z opisem; przy błędzie kończy test.

    Args:
        cond: wynik sprawdzenia.
        msg:  opis tego, co jest sprawdzane.

    Raises:
        SystemExit: kod 1, gdy `cond` jest fałszywe.
    """
    print(("  OK   " if cond else "  BŁĄD ") + msg)
    if not cond:
        raise SystemExit(1)


def main() -> None:
    """Wszystkie sprawdzenia po kolei, w folderze tymczasowym (usuwany na końcu)."""
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        scans_dir, out_dir = tmp / "skany", tmp / "out"
        make_scans(scans_dir)

        print("Wyszukiwanie skanów")
        files, excluded = find_scans(scans_dir, ScanParams())
        check(len(files) == 4, f"4 skany (te same nazwy w różnych folderach, podfolder głębiej): {len(files)}")
        check(len(excluded) == 1 and "occlusion" in excluded[0].name, "skan occlusion pominięty")
        check(all(f.path.suffix == ".ply" for f in files), "z kilku formatów wybrany .ply")
        check(len({f.id for f in files}) == 4, "unikalne nazwy skanów")

        print("Przygotowanie (łączenie, szum)")
        bones = prepare_all(scans_dir, out_dir, scan=ScanParams(), merge=MergeParams(),
                            features=FeatureParams(), verbose=False)
        check(len(bones) == 2, f"2 kości (A z 3 skanów, B z 1): {len(bones)}")
        a = max(bones, key=lambda x: x.n_scans)
        b = min(bones, key=lambda x: x.n_scans)
        check(a.n_scans == 3 and b.n_scans == 1, f"skany w kościach: A={a.n_scans}, B={b.n_scans}")
        check(a.meta["noise_removed_mm2"] > 100, f"usunięty szum (kawałek 60 mm dalej): {a.meta['noise_removed_mm2']:.0f} mm2")
        single = max(s["raw"]["triangles"] for s in a.meta["scans"])
        check(a.meta["stats"]["area_mm2"] > 1.3 * min(
            sum(c["area_mm2"] for c in s["components"] if c["kept"]) for s in a.meta["scans"]),
            "połączona kość ma większe pole niż pojedynczy skan")
        check(np.ptp(np.asarray(a.mesh.vertices), axis=0).max() < 35, "kość A bez oddalonego szumu (wymiar < 35 mm)")
        check(single > 0, "statystyki surowych skanów w meta")

        print("Ponowne uruchomienie (cache par)")
        t = time.perf_counter()
        prepare_all(scans_dir, out_dir, scan=ScanParams(), merge=MergeParams(), features=FeatureParams(),
                    verbose=False)
        check(time.perf_counter() - t < 60, f"z cache: {time.perf_counter() - t:.1f} s")

        print("Wczytanie kości")
        loaded = load_bones(out_dir, features=True)
        check(len(loaded) == 2 and all(x.fpfh is not None and x.fpfh.shape[1] == 33 for x in loaded),
              "2 kości z chmurą i FPFH (N, 33)")

        print("Powierzchnie styku")
        bone = [x for x in loaded if x.id == a.id][0]
        v = np.asarray(bone.mesh.vertices)
        labels = np.zeros(len(v), dtype=np.uint8)
        labels[v[:, 0] > np.percentile(v[:, 0], 85)] = 1
        labels[v[:, 1] < np.percentile(v[:, 1], 10)] = 2
        path = tmp / "contacts" / f"{bone.id}.json"
        ct.save_contacts(bone.mesh, ct.Contacts(bone.id, labels, {1: "przełom A", 2: "przełom B"}), path)
        back = ct.load_contacts(bone.id, bone.mesh, path=path)
        check(np.array_equal(back.labels, labels) and back.names[1] == "przełom A" and not back.remapped,
              "zapis i odczyt etykiet")
        smaller = bone.mesh.simplify_quadric_decimation(len(bone.mesh.triangles) // 3)
        moved = ct.load_contacts(bone.id, smaller, path=path)
        w = np.asarray(smaller.vertices)
        truth = np.zeros(len(w), dtype=np.uint8)
        truth[w[:, 0] > np.percentile(v[:, 0], 85)] = 1
        truth[w[:, 1] < np.percentile(v[:, 1], 10)] = 2
        check(moved.remapped and (moved.labels == truth).mean() > 0.95,
              f"przeniesienie na zmienioną siatkę: zgodność {(moved.labels == truth).mean():.3f}")
        stats = ct.surface_stats(bone.mesh, labels)
        check(len(stats) == 2 and all(s["area_mm2"] > 0 and s["max_diameter_mm"] > 0 for s in stats),
              "statystyki powierzchni (pole, średnica)")
        r = ct.compare(bone.mesh, labels, labels)
        check(abs(r["iou"] - 1) < 1e-9, "ocena: te same etykiety -> IoU = 1")
        root = tmp / "contacts"
        other = [x for x in loaded if x.id != bone.id][0]
        for name in ct.DETECTORS:
            found = ct.detect_contacts(bone, name, ct.ContactParams())
            check(len(found.labels) == len(v), f"algorytm \"{name}\" działa ({len(found.surface_ids)} pow.)")
        n = 1000
        fake = {"spread_2": np.r_[np.full(800, 0.02), np.full(200, 0.3)], "edge_mm": np.r_[np.full(800, 0.3), np.full(200, 0.15)]}
        fake = {k: x * np.random.default_rng(0).uniform(0.9, 1.1, n) for k, x in fake.items()}
        sc = ct.bone_scores(fake, np.ones(n), ("spread_2", "edge_mm"))
        check(abs(np.median(sc[:800])) < 0.5 and sc[800:].min() > 3,
              "features: wynik ~0 na typowej (korze), wysoki tam, gdzie większy rozrzut i gęstsza siatka")
        try:
            ct.bone_scores({"brightness": np.ones(n)}, np.ones(n), ("brightness",))
            check(False, "features: cecha bez stałego kierunku -> błąd")
        except ValueError:
            check(True, "features: cecha bez stałego kierunku (jasność) -> czytelny błąd")

        print("Wynik algorytmu: zapis w contacts/auto/ i cache")
        cp = ct.ContactParams(rough_threshold=0.02)                 # niski próg - na gładkiej kuli też coś znajdzie
        first, computed1 = ct.auto_contacts(other, cp, root)
        again, computed2 = ct.auto_contacts(other, cp, root)
        changed, computed3 = ct.auto_contacts(other, replace(cp, rough_threshold=0.03), root)
        check((root / "auto" / f"{other.id}.json").exists(), "wynik algorytmu w contacts/auto/<id>.json")
        check(computed1 and not computed2 and np.array_equal(first.labels, again.labels),
              "drugi raz z cache (bez liczenia)")
        check(computed3, "inne ustawienia algorytmu -> liczone od nowa")

        print("Dopasowanie powierzchni styku")
        mp = ms.SurfaceMatchParams(ransac_iterations=200000)
        modes = {}
        for source in ms.SOURCES:
            surfs, info = ms.extract_surfaces(loaded, replace(mp, source=source), cp, root)
            modes[source] = {b_id: {s.contacts.source for s in surfs if s.bone.id == b_id} for b_id in info}
        check(modes["manual"][bone.id] == {"manual"} and not modes["manual"][other.id],
              "tryb manual: tylko ręczne, kość bez nich pominięta")
        check(modes["auto"][bone.id] == {"auto:roughness"}, "tryb auto: tylko z algorytmu (także gdy są ręczne)")
        check(modes["manual-auto"][bone.id] == {"manual"} and modes["manual-auto"][other.id] <= {"auto:roughness"},
              "tryb manual-auto: ręczne, a gdy brak - z algorytmu")
        # druga strona przełomu = ta sama powierzchnia obrócona, inaczej próbkowana, z szumem, normalne przeciwne
        surfs, _ = ms.extract_surfaces([bone], mp, cp, root)
        a_s = surfs[0]
        rng = np.random.default_rng(3)
        R = trimesh.transformations.random_rotation_matrix(rng.random(3))
        R[:3, 3] = [40.0, -15.0, 8.0]
        keep = rng.random(len(a_s.points)) < 0.8
        b_pts = a_s.points[keep]
        b_s = ms.Surface(bone=a_s.bone, contacts=a_s.contacts, sid=99, area_mm2=0.8 * a_s.area_mm2,
                         points=b_pts @ R[:3, :3].T + R[:3, 3] + rng.normal(0, 0.05, b_pts.shape),
                         normals=-(a_s.normals[keep] @ R[:3, :3].T), bone_tree=a_s.bone_tree)
        r = ms.match_pair(a_s, b_s, replace(mp, collision_mm=1e6))   # kolizja bez sensu: B to ta sama kość
        err = np.linalg.norm(a_s.points @ np.asarray(r["T"])[:3, :3].T + np.asarray(r["T"])[:3, 3]
                             - (a_s.points @ R[:3, :3].T + R[:3, 3]), axis=1).mean()
        check(err < 0.3, f"odnalezione przekształcenie drugiej strony przełomu: błąd {err:.3f} mm")

        print("Wizualizacje")
        fig = bones_figure(loaded, {bone.id: back}, "fpfh", View3DParams())
        check(len(fig.data) > 4, "figura kości (FPFH + powierzchnie styku)")
        fig = merging_figure(loaded, View3DParams())
        check(len(fig.data) >= 4, "figura łączenia skanów (3 skany + szum)")
        surfs, _ = ms.extract_surfaces(loaded, replace(mp, source="manual-auto", ransac_seeds=1), cp, root)
        results = ms.match_all(surfs, replace(mp, source="manual-auto", ransac_seeds=1), progress=lambda _: None)
        rep = ms.report(surfs, results, mp, cp)
        fig = matching_figure(results, {s.bone.id: s.bone for s in surfs}, {s.bone.id: s.contacts for s in surfs})
        check(len(results) > 0 and len(fig.data) >= 2 and len(match_matrix_figure(rep).data) == 1,
              f"figury dopasowania ({len(results)} par)")

        print("Porównanie cech przełomu i kory (surface_features)")
        one = np.ones(1)
        check(sf.weighted_auc(np.array([2.0]), np.array([1.0]), one, one) == 1.0
              and sf.weighted_auc(np.array([1.0]), np.array([1.0]), one, one) == 0.5, "AUC: większy = 1, remis = 0.5")
        ow = np.asarray(other.mesh.vertices)
        other_c = ct.Contacts(other.id, (ow[:, 0] > np.percentile(ow[:, 0], 80)).astype(np.uint8))
        sfp = sf.SurfaceFeatureParams()
        comps = [sf.compare_bone(bone, back, sfp), sf.compare_bone(other, other_c, sfp)]
        n_feat = len(sf.feature_info(sfp))
        check(len(comps[0].stats) == 3 * n_feat and comps[0].marked.any() and comps[0].cortex.any()
              and not (comps[0].marked & comps[0].cortex).any(),
              f"cechy kości: {n_feat} cech x (wszystkie + 2 powierzchnie), przełom i kora rozłączne")
        model = sf.combined_model(comps, sfp)
        rows = sf.summary(comps, sfp)
        check(set(model["auc"]) == {bone.id, other.id} and len(rows) == n_feat, "model łączący cechy + ranking")
        check(len(sf.features_3d_figure(comps, model, sfp, View3DParams(triangles_total=20000)).data) == 4
              and len(sf.distributions_figure(comps, rows, sfp).data) == 2 * n_feat, "figury porównania cech")

        print("Podgląd z suwakami (match_tool)")
        rep["params"]["source"] = "manual-auto"
        (out_dir / "matching.json").write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
        tool = MatchState(out_dir, cp, View3DParams(triangles_total=20000), root)
        top = rep["pairs"][0]
        same = tool.match({"a": top["a"], "b": top["b"], "params": {}, "mode": "score", "T": top["T"]})
        check(abs(same["score"] - top["score"]) < 1e-3, "ta sama para i ustawienia -> ten sam wynik oceny")
        looser = tool.match({"a": top["a"], "b": top["b"], "params": {"tolerance_mm": 2.0}, "mode": "score",
                             "T": top["T"]})
        check(looser["contact_mm2"] >= same["contact_mm2"], "większa tolerancja styku -> styk nie mniejszy")
        full = tool.match({"a": top["a"], "b": top["b"], "params": {"sample_mm": 0.5, "ransac_iterations": 20000},
                           "mode": "full"})
        check(len(full["T"]) == 4 and full["mode"] == "full", f"dopasowanie pary od nowa ({full['seconds']} s)")
        g = tool.geometry(top["a"])
        check(len(g["positions"]) > 0 and len(g["surf_indices"]) > 0, "geometria do strony (siatka + powierzchnia)")
        check(tool.warnings(tool.params) == [] and len(tool.warnings(replace(tool.params, tolerance_mm=9.0))) == 1,
              "ostrzeżenie, gdy settings.json różni się od matching.json")
        page = matching_page(tool, out_dir / "matching.html")
        check(page.stat().st_size > 1000, "strona HTML z zapisanego matching.json (tune_matching.py --html)")

        print("Zestawy danych i zmiana nazwy kości")
        check(config.dataset_dirs("x")[0] == config.SCANS_ROOT / "x"
              and config.dataset_dirs("x")[2] == config.CONTACTS_ROOT / "x", "foldery zestawu = podfoldery")
        try:
            config.use_dataset("nie_ma_takiego_zestawu_123")
            check(False, "nieistniejący zestaw -> błąd")
        except SystemExit as e:
            check("Nie ma zestawu" in str(e), "nieistniejący zestaw -> czytelny błąd z listą zestawów")
        check(config.DATASET != "nie_ma_takiego_zestawu_123", "po błędzie zestaw bez zmian")
        check(any(f.id == "01.01.2026-test_1" and f.label == "test_(1)" for f in files),
              "jedyny skan w folderze: nazwa kości = nazwa folderu")
        check(all(np.abs(np.asarray(x.mesh.vertices).mean(axis=0)).max() < 1e-6 for x in loaded),
              "kości wyśrodkowane (także z jednego skanu)")
        for f in root.glob("*.json"):
            f.unlink()
        ct.save_contacts(bone.mesh, ct.Contacts(bone.id, labels), root / f"{bone.id}.json", ct.scan_frames(bone))
        ct.save_contacts(other.mesh, other_c, root / f"{other.id}.json", ct.scan_frames(other))
        area_a = ct.vertex_areas(bone.mesh)[labels > 0].sum()
        (scans_dir / "01.01.2026-test").rename(scans_dir / "nowa_nazwa")
        renamed = prepare_all(scans_dir, out_dir, contacts_dir=root, scan=ScanParams(), merge=MergeParams(),
                              features=FeatureParams(), verbose=False)
        new_ids = {x.id for x in renamed}
        check(other.id not in new_ids and "nowa_nazwa-lowerjaw" in new_ids, f"nowe nazwy kości: {sorted(new_ids)}")
        files_now = {f.stem for f in root.glob("*.json")}
        check(files_now <= new_ids and len(files_now) == 2, f"oznaczenia przeniesione na nowe nazwy: {sorted(files_now)}")
        nb = [x for x in renamed if x.id == "nowa_nazwa-lowerjaw"][0]
        got = ct.load_contacts(nb.id, nb.mesh, root=root, frames=ct.scan_frames(nb))
        check(np.array_equal(got.labels, other_c.labels), "kość z jednego skanu: etykiety bez zmian")
        na = max(renamed, key=lambda x: x.n_scans)
        got = ct.load_contacts(na.id, na.mesh, root=root, frames=ct.scan_frames(na))
        area = ct.vertex_areas(na.mesh)[got.labels > 0].sum()
        check(0.8 < area / area_a < 1.3, f"połączona kość: oznaczenia po zmianie nazwy ({area:.0f} / {area_a:.0f} mm²)")

        split = prepare_all(scans_dir, out_dir, contacts_dir=root, scan=ScanParams(),
                            merge=MergeParams(enabled=False), features=FeatureParams(), verbose=False)
        ref = [x for x in split if x.id == na.id][0]
        got = ct.load_contacts(ref.id, ref.mesh, root=root, frames=ct.scan_frames(ref))
        area = ct.vertex_areas(ref.mesh)[got.labels > 0].sum()
        check(len(split) == 4 and area > 0.2 * area_a,
              f"łączenie wyłączone (inny środek kości): oznaczenia przeniesione przez wspólny skan ({area:.0f} mm²)")

    print(f"Wszystkie testy OK ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()

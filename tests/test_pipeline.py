"""
Test całego przygotowania danych na SZTUCZNYCH skanach (nie potrzebuje prawdziwych skanów).

Tworzy w folderze tymczasowym:
    - fragment A (kula ~25 a.u. z losowymi wybrzuszeniami) zeskanowany 3 razy, za każdym
      razem częściowo, z innej strony i w innym układzie współrzędnych (obrót + przesunięcie),
      w podfolderach o TYCH SAMYCH nazwach plików (jak kolejne sesje Medita), jeden głębiej,
    - do jednego skanu A dołożony szum: oderwany kawałek 60 a.u. od fragmentu,
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
from dataclasses import asdict, replace
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
from visualization_3d import (View3DParams, bones_figure, compare_layers,match_matrix_figure, matching_figure,  # noqa: E402
                              merging_figure)


def bumpy_sphere(seed: int, radius: float = 12.0) -> trimesh.Trimesh:
    """Kula z losowymi wybrzuszeniami - kształt z cechami, które da się dopasować.

    Args:
        seed:   ziarno - inne ziarno = inny kształt (inny "fragment kości").
        radius: promień kuli przed wybrzuszeniami [a.u.] (30 wybrzuszeń i wgłębień, każde do ±2 a.u.).

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
    Potem losowy obrót i przesunięcie do ±30 a.u. (ziarno `seed`).

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
        check(a.meta["noise_removed_au2"] > 100, f"usunięty szum (kawałek 60 a.u. dalej): {a.meta['noise_removed_au2']:.0f} a.u.²")
        single = max(s["raw"]["triangles"] for s in a.meta["scans"])
        check(a.meta["stats"]["area_au2"] > 1.3 * min(
            sum(c["area_au2"] for c in s["components"] if c["kept"]) for s in a.meta["scans"]),
            "połączona kość ma większe pole niż pojedynczy skan")
        check(np.ptp(np.asarray(a.mesh.vertices), axis=0).max() < 35, "kość A bez oddalonego szumu (wymiar < 35 a.u.)")
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
        check(len(stats) == 2 and all(s["area_au2"] > 0 and s["max_diameter_au"] > 0 for s in stats),
              "statystyki powierzchni (pole, średnica)")
        r = ct.compare(bone.mesh, labels, labels)
        check(abs(r["iou"] - 1) < 1e-9, "ocena: te same etykiety -> IoU = 1")
        root = tmp / "contacts"
        other = [x for x in loaded if x.id != bone.id][0]
        for name in [d for d in ct.DETECTORS if d != "atlas"]:     # "atlas" potrzebuje bazy: tests/test_atlas.py
            found = ct.detect_contacts(bone, name, ct.ContactParams())
            check(len(found.labels) == len(v), f"algorytm \"{name}\" działa ({len(found.surface_ids)} pow.)")
        n = 1000
        fake = {"spread_2": np.r_[np.full(800, 0.02), np.full(200, 0.3)], "edge_au": np.r_[np.full(800, 0.3), np.full(200, 0.15)]}
        fake = {k: x * np.random.default_rng(0).uniform(0.9, 1.1, n) for k, x in fake.items()}
        sc = ct.bone_scores(fake, np.ones(n), ("spread_2", "edge_au"))
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
        b_s = ms.Surface(bone=a_s.bone, contacts=a_s.contacts, sid=99, area_au2=0.8 * a_s.area_au2,
                         points=b_pts @ R[:3, :3].T + R[:3, 3] + rng.normal(0, 0.05, b_pts.shape),
                         normals=-(a_s.normals[keep] @ R[:3, :3].T), bone_tree=a_s.bone_tree)
        r = ms.match_pair(a_s, b_s, replace(mp, collision_au=1e6))   # kolizja bez sensu: B to ta sama kość
        err = np.linalg.norm(a_s.points @ np.asarray(r["T"])[:3, :3].T + np.asarray(r["T"])[:3, 3]
                             - (a_s.points @ R[:3, :3].T + R[:3, 3]), axis=1).mean()
        check(err < 0.3, f"odnalezione przekształcenie drugiej strony przełomu: błąd {err:.3f} a.u.")

        print("Dopasowanie z atlasem anatomicznym (rankingi i ustawienia na modelu podane wprost)")

        def ctx(*cands, unknown=0.0):
            """Kontekst jak z ms.atlas_context: (model, odbicie, p, T model -> fragment[, podfolder])."""
            return {"p_unknown": unknown, "candidates": [
                {"model": c[0], "group": c[4] if len(c) > 4 else "", "mirror": c[1], "label": c[0], "p": c[2],
                 "rel_scale": 1.0, "T": np.asarray(c[3]).tolist()} for c in cands]}

        def similarity(s, rot, t):
            """Macierz 4x4 x -> s rot x + t."""
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = s * rot, t
            return M

        def rot_deg(axis, deg):
            """Obrót o `deg` stopni wokół osi `axis`."""
            return trimesh.transformations.rotation_matrix(np.deg2rad(deg), axis)[:3, :3]

        TA = similarity(1.1, rot_deg([1, 2, 0], 70), [5.0, -3.0, 2.0])
        mid = np.linalg.inv(TA) @ np.append(a_s.points.mean(axis=0), 1)          # środek A w układzie modelu
        stretch = similarity(1.15 / 1.1, np.eye(3), (1 - 1.15 / 1.1) * mid[:3])  # inna skala B na modelu
        # B na modelu: prawda (R x A) z błędem dopasowania do modelu: inna skala, 4° obrotu wokół środka
        # powierzchni, ~1 a.u. przesunięcia
        c_true = R[:3, :3] @ a_s.points.mean(axis=0) + R[:3, 3]
        err_B = similarity(1.0, rot_deg([0, 0, 1], 4), [0.7, -0.5, 0.5])
        err_B[:3, 3] += c_true - err_B[:3, :3] @ c_true
        TB = err_B @ R @ TA @ stretch
        a_s.atlas = ctx(("kość 1", False, 0.9, TA), ("kość 2", False, 0.08, np.eye(4)), unknown=0.02)
        b_s.atlas = ctx(("kość 1", False, 0.85, TB), ("kość 1", True, 0.1, np.eye(4)), unknown=0.05)
        ap = replace(mp, collision_au=1e6, use_atlas=True, ransac_iterations=200)    # RANSAC praktycznie bez szans
        starts = ms.atlas_starts(a_s, b_s, ap)
        true_pts = a_s.points @ R[:3, :3].T + R[:3, 3]

        def pose_err(T):
            """Średni błąd położenia punktów A po przekształceniu T względem prawdy [a.u.]."""
            T = np.asarray(T)
            return float(np.linalg.norm(a_s.points @ T[:3, :3].T + T[:3, 3] - true_pts, axis=1).mean())

        check(len(starts) == 1 and starts[0][0] == "kość 1" and abs(starts[0][1] - 0.765) < 1e-9
              and 0.3 < pose_err(starts[0][2]) < 5,
              f"start z atlasu tylko ze wspólnego modelu i strony (p 0.9 x 0.85), przybliżony: "
              f"błąd {pose_err(starts[0][2]):.2f} a.u.")
        t = time.perf_counter()
        r = ms.match_pair(a_s, b_s, ap)
        check(r["start"] == "atlas: kość 1" and r["ransac"] is False and pose_err(r["T"]) < 0.3,
              f"ICP ze startu z atlasu, bez RANSAC: błąd {pose_err(r['T']):.3f} a.u. ({time.perf_counter() - t:.1f} s)")
        same, diff = ms.atlas_relation(a_s, b_s)
        mean_area = 0.5 * (a_s.area_au2 + b_s.area_au2)
        area_r = ms.evaluate(a_s, b_s, np.asarray(r["T"]), replace(ap, score_mode="area"))
        check(abs(same - 0.9 * 0.85) < 1e-9 and abs(diff - (0.98 * 0.95 - same)) < 1e-9
              and abs(r["score"] - 100 * (r["contact_au2"] * (1 + ap.atlas_boost * same) - r["collision_au2"])
                      / mean_area) < 1e-6 and r["score_unit"] == "%"
              and abs(area_r["score"] - (r["contact_au2"] * (1 + ap.atlas_boost * same) - r["collision_au2"])) < 1e-6,
              f"P(ta sama kość) {same:.2f}, różne {diff:.2f}; wynik {r['score']:.0f}% (premia +{r['atlas_bonus']:.0f}), "
              f"jako pole {area_r['score']:.0f} a.u.²")
        r_off = ms.evaluate(a_s, b_s, np.asarray(r["T"]), replace(ap, use_atlas=False))
        check("atlas_same" not in r_off and abs(r_off["score"] - (r["score"] - r["atlas_bonus"])) < 1e-6,
              "bez use_atlas: wynik bez premii i bez pól atlasu")
        no_start = ms.match_pair(a_s, b_s, replace(ap, atlas_starts=0, ransac_seeds=1))
        check(no_start["start"] == "RANSAC" and no_start["ransac"], "atlas_starts = 0 -> zwykły RANSAC")
        c_s = ms.Surface(bone=other, contacts=a_s.contacts, sid=7, area_au2=a_s.area_au2, points=a_s.points,
                         normals=a_s.normals, bone_tree=a_s.bone_tree)
        b_other = ms.Surface(bone=other, contacts=b_s.contacts, sid=8, area_au2=b_s.area_au2, points=b_s.points,
                             normals=b_s.normals, bone_tree=b_s.bone_tree)
        b_other.atlas = b_s.atlas
        kinds = {"inny model": ctx(("kość 2", False, 0.99, np.eye(4))),
                 "druga strona": ctx(("kość 1", True, 0.99, np.eye(4))),
                 "niepewny": ctx(("kość 2", False, 0.4, np.eye(4)), unknown=0.6),
                 "podfolder": ctx(("kość 1b", True, 0.99, np.eye(4), "typ"))}
        a_ref = ms.Surface(bone=a_s.bone, contacts=a_s.contacts, sid=9, area_au2=1.0, points=a_s.points,
                           normals=a_s.normals, atlas=ctx(("kość 1", False, 0.99, np.eye(4))))
        a_grp = ms.Surface(bone=a_s.bone, contacts=a_s.contacts, sid=9, area_au2=1.0, points=a_s.points,
                           normals=a_s.normals, atlas=ctx(("kość 1a", False, 0.99, np.eye(4), "typ")))
        skipped = {}
        for name, k in kinds.items():
            c_s.atlas = k
            pairs, sk = ms.split_pairs([a_grp if name == "podfolder" else a_ref, c_s], ap)
            skipped[name] = bool(sk) and not pairs
        check(skipped == {"inny model": True, "druga strona": True, "niepewny": False, "podfolder": False},
              "pomijane pary na pewno różnych kości: inny model, druga strona; nie: niepewny, ten sam podfolder")
        c_s.atlas = kinds["inny model"]
        check(not ms.split_pairs([a_ref, c_s], replace(ap, use_atlas=False))[1]
              and not ms.split_pairs([a_ref, c_s], replace(ap, atlas_skip_different=1.01))[1],
              "bez use_atlas albo atlas_skip_different > 1 - nic nie jest pomijane")
        rep_atlas = ms.report([a_ref, c_s, b_other], [], ap)
        check(set(rep_atlas["atlas"]) == {a_s.bone.id, other.id} and len(rep_atlas["skipped"]) == 1
              and len(match_matrix_figure(rep_atlas).data) == 1,
              "matching.json: konteksty atlasu i pominięte pary (× w macierzy)")
        a_s.atlas = None

        print("Kości długie: osie w jednej linii (walce 60 x r 5, przełomy na końcach)")
        from types import SimpleNamespace

        import open3d as o3d
        from scipy.spatial import cKDTree

        def piece(name, height=60.0, radius=5.0, sphere=False):
            """Sztuczny fragment: walec wzdłuż z (albo kula) jako obiekt z .mesh, jak kość."""
            t = trimesh.creation.icosphere(3, radius) if sphere else trimesh.creation.cylinder(radius, height, 48)
            m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(t.vertices), o3d.utility.Vector3iVector(t.faces))
            m.compute_vertex_normals()
            cloud = o3d.geometry.PointCloud(m.vertices)
            cloud.normals = m.vertex_normals
            return SimpleNamespace(id=name, label=name, mesh=m, cloud=cloud, meta={})

        def cap(bone, z, n_up):
            """Powierzchnia styku = tarcza na wysokości z (normalne n_up = na zewnątrz kości)."""
            ang, rad = rng.random(400) * 2 * np.pi, 4.5 * np.sqrt(rng.random(400))
            pts = np.column_stack([rad * np.cos(ang), rad * np.sin(ang), np.full(400, z)])
            return ms.Surface(bone=bone, contacts=None, sid=1, points=pts, normals=np.tile([0, 0, n_up], (400, 1)),
                              area_au2=60.0, bone_tree=cKDTree(np.asarray(bone.mesh.vertices)))

        def move(R=np.eye(3), t=(0, 0, 0)):
            """Macierz 4x4 z obrotu R i przesunięcia t."""
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = R, t
            return M

        ax_p = replace(mp, axis_check=True)
        top, bottom = cap(piece("A"), 30.0, 1.0), cap(piece("B"), -30.0, -1.0)
        good = ms.axis_fit(top, bottom, move(t=(0, 0, -60)), ax_p)      # A pod B, koniec do końca
        side = ms.axis_fit(top, bottom, move(rot_deg([1, 0, 0], 90), (0, 30, -30)), ax_p)  # koniec do boku
        along = ms.axis_fit(top, bottom, move(t=(10, 0, 0)), ax_p)      # obok siebie, równolegle
        check(good["axis_weight"] == 1.0 and good["axis_deg"] < 1 and good["axis_offset"] < 0.05
              and side["axis_weight"] == 0.0 and side["axis_deg"] > 89 and along["axis_weight"] == 0.0
              and along["axis_offset"] > 1.9,
              f"ocena osi: koniec do końca waga 1, koniec do boku {side['axis_deg']:.0f}° -> 0, "
              f"obok siebie {along['axis_offset']:.1f} promienia -> 0")
        starts = ms.axis_starts_for(top, bottom, ax_p)
        ends = [ms.transform_points(T, np.zeros((1, 3)))[0] for _, T in starts]   # środek A po ustawieniu
        check(len(starts) == ax_p.axis_starts
              and all(ms.axis_fit(top, bottom, T, ax_p)["axis_weight"] == 1.0 for _, T in starts)
              and all(abs(e[2] + 60) < 1e-6 and np.hypot(e[0], e[1]) < 1.0 for e in ends),
              f"{len(starts)} startów wzdłuż osi: A przedłuża B (środek A na osi, 60 a.u. za przełomem B)")
        mid = cap(piece("C"), 0.0, 1.0)
        check(len(ms.axis_starts_for(mid, bottom, ax_p)) == 2 * ax_p.axis_starts
              and not ms.axis_starts_for(cap(piece("K", sphere=True), 4.0, 1.0), bottom, ax_p)
              and ms.axis_fit(cap(piece("K", sphere=True), 4.0, 1.0), bottom, np.eye(4), ax_p) is None
              and not ms.axis_starts_for(top, bottom, replace(ax_p, axis_check=False)),
              "przełom w środku boku -> obie strony; kula (niewydłużona) i axis_check = false -> bez osi")
        scored = ms.evaluate(top, bottom, move(rot_deg([1, 0, 0], 90), (0, 30, -30)), ax_p)
        check(scored["axis_weight"] == 0.0 and abs(scored["score"] + scored["collision_au2"]) < 1e-9,
              "styk pod złym kątem nie liczy się do wyniku")

        print("Proponowane złożenie (każdy przełom i para kości raz)")
        res = [{"a": "X#1", "b": "Y#1", "score": 100}, {"a": "X#1", "b": "Z#1", "score": 90},
               {"a": "X#2", "b": "Y#2", "score": 80}, {"a": "X#2", "b": "Z#1", "score": 70},
               {"a": "Y#2", "b": "Z#2", "score": -5}]
        chosen = ms.assemble(res)["chosen"]
        check([r["score"] for r in chosen] == [100, 70] and [r["chosen"] for r in res] == [True, False, False, True, False],
              "zachłannie od najlepszej: zajęty przełom, ta sama para kości i wynik <= 0 odpadają")

        print("Przybliżenie przełomu płaszczyzną / funkcją kwadratową (contacts.fit_surface)")
        grid = np.stack(np.meshgrid(np.linspace(-5, 5, 30), np.linspace(-5, 5, 30)), -1).reshape(-1, 2)
        tilt = rot_deg([1, 1, 0], 30)
        flat_pts = np.column_stack([grid, rng.normal(0, 0.05, len(grid))]) @ tilt.T + [3, -2, 7]
        bowl_pts = np.column_stack([grid, 0.08 * (grid ** 2).sum(1)]) @ tilt.T
        up = np.tile(tilt[:, 2], (len(grid), 1))
        f_flat = ct.fit_surface(flat_pts, up)
        f_plane, f_quad = ct.fit_surface(bowl_pts, up), ct.fit_surface(bowl_pts, up, mode="quadric")
        check(abs(f_flat["normal"] @ tilt[:, 2]) > 0.999 and f_flat["rms_au"] < 0.08 and f_quad["rms_au"] < 0.02
              and f_plane["rms_au"] > 0.3 and np.abs(ct.surface_residual(f_quad, bowl_pts)).max() < 0.05,
              f"płaszczyzna: normalna i odchylenie {f_flat['rms_au']:.2f} a.u. (szum 0.05); misa: płaszczyzna "
              f"{f_plane['rms_au']:.2f}, funkcja kwadratowa {f_quad['rms_au']:.3f} a.u.")

        print("Po wykrywaniu: rozdzielenie końców i odcięcie kory (contacts.refine_surfaces)")
        tube = trimesh.creation.cylinder(5.0, 60.0, 48)
        tv, tf = trimesh.remesh.subdivide_to_size(tube.vertices, tube.faces, 1.0)
        tube_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tv), o3d.utility.Vector3iVector(tf))
        tube_mesh.compute_vertex_normals()
        # "test 3": oba końce (z rękawem kory 12 a.u.) połączone paskiem kory wzdłuż boku = jedna powierzchnia
        one = ((np.abs(tv[:, 2]) > 18) | (tv[:, 0] > 4.6)).astype(np.uint8)
        rp = replace(cp, surface_fit="plane", split_ends=True, fit_tolerance_au=3.0, min_surface_au2=15.0)
        ref, refined = ct.refine_surfaces(tube_mesh, one, rp)
        fits = refined["fits"]
        zs = [tv[ref == k, 2] for k in (1, 2)]
        check(ref.max() == 2 and all(np.abs(z).min() > 26.5 for z in zs) and np.sign(zs[0].mean()) != np.sign(zs[1].mean())
              and all(abs(abs(fits[k]["normal"][2]) - 1) < 0.01 and fits[k]["rms_au"] < 1.0 for k in ("1", "2")),
              "jedna powierzchnia przez całą kość -> dwa końce, bez rękawa kory (zostaje do 3 a.u. od twarzy), "
              "płaszczyzny prostopadłe do osi")
        check(np.array_equal(ct.refine_surfaces(tube_mesh, one, replace(rp, surface_fit="none", split_ends=False))[0], one),
              "surface_fit = none, split_ends = false -> etykiety bez zmian")
        # głowa kości: górny koniec rozszerzony (promień x1.4 w ostatnich 8 a.u.), przełom na dolnym (zwężonym) końcu;
        # ostra krawędź kory: wąski pasek wzdłuż trzonu (normalne w bok, płaski)
        hv = tv.copy()
        grow = np.clip((hv[:, 2] - 22.0) / 8.0, 0, 1)
        hv[:, :2] *= (1 + 0.4 * grow)[:, None]
        hv[:, :2] *= (1 - 0.3 * np.clip((-hv[:, 2] - 22.0) / 8.0, 0, 1))[:, None]
        head_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(hv), o3d.utility.Vector3iVector(tf))
        head_mesh.compute_vertex_normals()
        ang = np.arctan2(hv[:, 1], hv[:, 0])
        lab3 = np.zeros(len(hv), np.uint8)
        lab3[hv[:, 2] > 27] = 1                                     # głowa (szorstka u prawdziwej kości)
        lab3[hv[:, 2] < -27] = 2                                    # przełom na zwężonym końcu
        lab3[(np.abs(ang) < 0.12) & (np.abs(hv[:, 2]) < 15)] = 3    # krawędź wzdłuż trzonu
        hp = replace(cp, surface_fit="none", min_surface_au2=5.0, head_min_ratio=1.1, crest_max_axial=0.2)
        out3, info3 = ct.refine_surfaces(head_mesh, lab3, hp)
        whys = sorted(d["why"].split(" (")[0] for d in info3["dropped"])
        check(out3.max() == 1 and hv[out3 == 1, 2].max() < -26 and whys == ["głowa kości", "ostra krawędź kory"],
              f"głowa kości (szerszy koniec) i ostra krawędź kory odrzucone, przełom na zwężonym końcu zostaje: {whys}")

        print("Płaskie cięcia, tryb \"plane\" i próg złożenia")
        rough_top = cap(piece("A"), 30.0, 1.0)
        rough_top.points[:, 2] += rng.normal(0, 0.8, len(rough_top.points))
        flat_p = replace(mp, axis_check=True, flat_max_rms_au=0.25)
        check(top.fit(flat_p)["rms_au"] < 0.05 and rough_top.fit(flat_p)["rms_au"] > 0.4,
              f"spiłowany koniec: odchylenie {top.fit(flat_p)['rms_au']:.2f} a.u., przełom {rough_top.fit(flat_p)['rms_au']:.2f} a.u.")
        pl = ms.match_pair(top, bottom, replace(ax_p, mode="plane", plane_steps=12))
        Tpl = np.asarray(pl["T"])
        check(pl["start"].startswith("płaszczyzna") and pl["ransac"] is False and pl["cover_a"] > 0.9
              and np.allclose(Tpl[:3, 3] + Tpl[:3, :3] @ top.points.mean(0), bottom.points.mean(0), atol=1e-6)
              and ms.axis_fit(top, bottom, Tpl, ax_p)["axis_weight"] == 1.0,
              f"tryb plane: płaszczyzna A na płaszczyźnie B, obrysy pokryte w {pl['cover_a']:.0%}, osie w linii")
        res = [{"a": "X#1", "b": "Y#1", "score": 40}, {"a": "X#2", "b": "Z#1", "score": 8}]
        check([r["score"] for r in ms.assemble(res, replace(mp, assembly_min_score=10.0))["chosen"]] == [40]
              and not res[1]["chosen"],
              "złożenie: para słabsza niż assembly_min_score zostaje bez pary (może brakuje kawałka)")

        print("Prostowanie osi, starty z płaszczyzn, jakość, filtr pól (pomysły z OSTEOFORM)")
        tilt5 = move(rot_deg([1, 0, 0], 5), (0.5, -0.3, -60))
        locked = ms.axis_locked_icp(top, bottom, tilt5, ax_p)
        check(ms.axis_fit(top, bottom, tilt5, ax_p)["axis_deg"] > 4.5 and ms.axis_fit(top, bottom, locked, ax_p)["axis_deg"] < 0.1
              and ms.contact_score(top, bottom, locked, ax_p)["cover_a"] > 0.8,
              "prostowanie: oś A na osi B (5° -> 0°), styk po dociśnięciu ICP z obrotem wokół osi")
        ps = ms.plane_starts_for(top, bottom, replace(ax_p, plane_starts=24, plane_starts_keep=3))
        check(len(ps) == 3 and all(lbl.startswith("płaszczyzna") for lbl, _ in ps)
              and not ms.plane_starts_for(top, bottom, ax_p),
              "starty płaszczyzna na płaszczyźnie: 3 najlepsze z 24 obrotów; plane_starts = 0 -> bez")
        q = ms.evaluate(top, bottom, move(t=(0, 0, -60)), replace(ax_p, quality_factor=True))
        check(0 < q["quality"] <= 1 and q["angle_deg"] < 1, f"jakość: kąt normalnych {q['angle_deg']:.1f}°, czynnik {q['quality']:.2f}")
        big = ms.Surface(bone=piece("D"), contacts=None, sid=1, points=bottom.points, normals=bottom.normals, area_au2=200.0)
        sk = ms.split_pairs([top, big], replace(ax_p, max_area_ratio=2.0))[1]
        check(len(sk) == 1 and "pola" in sk[0]["why"] and not ms.split_pairs([top, big], ax_p)[1],
              f"filtr pól: {sk[0]['why']}")

        print("Składanie całych grup: wiązka z kontrolą kolizji (OSTEOFORM)")

        def tube_bone(name):
            """Fragment = rura 60 x r 5 (gęsta siatka) z chmurą i KD-drzewem, jak kość."""
            c = o3d.geometry.PointCloud(tube_mesh.vertices)
            c.normals = tube_mesh.vertex_normals
            return SimpleNamespace(id=name, label=name, mesh=tube_mesh, cloud=c, meta={})

        tb = {k: tube_bone(k) for k in "ABC"}
        tsurf = [ms.Surface(bone=tb[k], contacts=None, sid=1, points=top.points, normals=top.normals, area_au2=60.0,
                            bone_tree=cKDTree(tv)) for k in "ABC"]
        e1 = {"a": "A#1", "b": "B#1", "score": 50.0, "T": move(t=(0, 0, -60)).tolist()}       # A pod B
        e2 = {"a": "B#2", "b": "C#1", "score": 40.0, "T": move(t=(0, 0, -60)).tolist()}       # B pod C
        e4 = {"a": "C#1", "b": "B#2", "score": 48.0, "T": move(t=(3, 0, -60)).tolist()}       # C w miejscu A
        greedy = [r["score"] for r in ms.assemble([e1, e4, e2], replace(mp, assembly_beam=0), tsurf)["chosen"]]
        beam = ms.assemble([e1, e4, e2], replace(mp, assembly_beam=12), tsurf)
        pz = {k: np.asarray(beam["poses"][k])[2, 3] for k in "ABC"}
        check(greedy == [50.0, 48.0] and sorted(r["score"] for r in beam["chosen"]) == [40.0, 50.0]
              and len(set(beam["groups"].values())) == 1 and pz["A"] < pz["B"] < pz["C"],
              f"zachłannie: {greedy} (C wchodzi w A); wiązka: A-B + B-C, jedna grupa A pod B pod C")

        print("Kości długie, tryb \"gap\": szczelina między przełomami, ciągłość kory")
        def ring(z, up):
            """400 punktów tarczy przełomu (promień 4.5) na wysokości z."""
            r, ang = 4.5 * np.sqrt(rng.random(400)), rng.random(400) * 2 * np.pi
            return np.column_stack([r * np.cos(ang), r * np.sin(ang), np.full(400, z)])

        ga = ms.Surface(bone=tb["A"], contacts=None, sid=1, points=ring(30.0, 1), normals=np.tile([0, 0, 1.0], (400, 1)),
                        area_au2=60.0, bone_tree=cKDTree(tv))
        gb = ms.Surface(bone=tb["B"], contacts=None, sid=1, points=ring(-30.0, -1), normals=np.tile([0, 0, -1.0], (400, 1)),
                        area_au2=60.0, bone_tree=cKDTree(tv))
        gp = replace(mp, axis_check=True, long_mode="gap", gap_twist_deg=30.0, gap_step_au=2.0, axis_lock=False)
        gap_true = move(t=(0, 0, -61.5))                      # A pod B, szczelina 1.5 a.u.
        strict = ms.evaluate(ga, gb, gap_true, replace(gp, long_mode="contact"))
        loose = ms.evaluate(ga, gb, gap_true, gp)
        found = ms.match_pair(ga, gb, gp)
        tz = np.asarray(found["T"])[2, 3]
        check(strict["contact_au2"] == 0 and loose["proximity"] > 0.95 and loose["continuity"] > 0.3
              and -63.0 < tz < -59.0 and found["axis_deg"] < 1.0 and ms.acceptable({**found, "a_area_au2": 60.0,
                                                                                     "b_area_au2": 60.0}, gp),
              f"szczelina 1.5 a.u.: ścisły styk 0, bliskość {loose['proximity']:.2f}, ciągłość kory "
              f"{loose['continuity']:.2f}; znalezione: A {(-tz - 60):+.1f} a.u. od B, osie {found['axis_deg']:.1f}° ({found['start']})")

        print("Atlas: wspólna skala zestawu")
        import atlas

        def votes(*cands):
            """Dopasowanie jak AtlasMatch: tylko lista kandydatów (model, p, ocena, skala)."""
            return SimpleNamespace(candidates=[{"model": m, "name": m, "p": q, "fit": f, "scale": sc}
                                               for m, q, f, sc in cands])

        frags = [(votes(("F", 0.8, 0.8, 0.84), ("F", 0.2, 0.7, 0.86)), 80.0, "długi 1"),
                 (votes(("F", 0.9, 0.85, 0.82)), 330.0, "cała kość"),
                 (votes(("F", 0.6, 0.75, 1.05), ("H", 0.3, 0.5, 0.7)), 40.0, "krótki"),
                 (votes(("F", 0.3, 0.9, 0.5), ("H", 0.6, 0.4, 0.9)), 50.0, "niepewny")]
        sh = atlas.shared_scales(frags, atlas.AtlasParams())
        bounds = atlas.scale_bounds(atlas.AtlasParams(model_scales={"F": 0.82}), "F")
        check(set(sh) == {"F"} and sh["F"]["scale"] == 0.82 and [o[0] for o in sh["F"]["outliers"]] == ["krótki"]
              and abs(sh["F"]["agree"] - 303.65 / 321.65) < 1e-3
              and len(sh["F"]["votes"]) == 3 and np.allclose(bounds, (0.738, 0.902))
              and atlas.scale_bounds(atlas.AtlasParams(), "F") == (0.75, 1.35),
              f"mediana ważona długością: ×{sh['F']['scale']}, inna skala: krótki; zakres ±10% tylko dla modelu z wpisem")

        print("Wizualizacje")
        fig = bones_figure(loaded, {bone.id: back}, "fpfh", View3DParams())
        check(len(fig.data) > 4, "figura kości (FPFH + powierzchnie styku)")
        half = ct.Contacts(bone.id, np.where(np.arange(len(back.labels)) % 2 == 0, back.labels, 0).astype(np.uint8))
        masks, score = compare_layers(bone, half, back)
        check(set(masks) == {"both", "manual_only"} and score is not None and score["precision"] > 0.99
              and 0.2 < score["recall"] < 0.8, f"porównanie ręczne vs algorytm (czułość {score['recall']:.2f})")
        fig = bones_figure(loaded, {bone.id: half}, "scan", View3DParams(), reference={bone.id: back})
        check(len(fig.data) > 4 and "ręczne vs algorytm" in fig.layout.title.text, "figura porównania")
        fig = merging_figure(loaded, View3DParams())
        check(len(fig.data) >= 4, "figura łączenia skanów (3 skany + szum)")
        surfs, _ = ms.extract_surfaces(loaded, replace(mp, source="manual-auto", ransac_seeds=1), cp, root)
        for s in surfs:                     # obie kości "z tej samej kości z bazy" - premia, start z atlasu
            s.atlas = ctx(("kość 1", False, 0.9, np.eye(4)), unknown=0.1)
        mp_atlas = replace(mp, source="manual-auto", ransac_seeds=1, use_atlas=True)
        results = ms.match_all(surfs, mp_atlas, progress=lambda _: None)
        rep = ms.report(surfs, results, mp_atlas, cp)
        check(all("atlas_same" in r and "start" in r for r in results) and len(rep["atlas"]) == 2,
              "dopasowanie wszystkich par z atlasem (premia, start, konteksty w matching.json)")
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
        check(abs(same["score"] - top["score"]) < 1e-3 and abs(same["atlas_bonus"] - top["atlas_bonus"]) < 1e-3,
              "ta sama para i ustawienia -> ten sam wynik oceny (z premią z atlasu z matching.json)")
        looser = tool.match({"a": top["a"], "b": top["b"], "params": {"tolerance_au": 2.0}, "mode": "score",
                             "T": top["T"]})
        check(looser["contact_au2"] >= same["contact_au2"], "większa tolerancja styku -> styk nie mniejszy")
        full = tool.match({"a": top["a"], "b": top["b"], "params": {"sample_au": 0.5, "ransac_iterations": 20000},
                           "mode": "full"})
        check(len(full["T"]) == 4 and full["mode"] == "full", f"dopasowanie pary od nowa ({full['seconds']} s)")
        g = tool.geometry(top["a"])
        check(len(g["positions"]) > 0 and len(g["surf_indices"]) > 0, "geometria do strony (siatka + powierzchnia)")
        check(tool.warnings(tool.params) == [] and len(tool.warnings(replace(tool.params, tolerance_au=9.0))) == 1,
              "ostrzeżenie, gdy settings.json różni się od matching.json")
        page = matching_page(tool, out_dir / "matching.html")
        check(page.stat().st_size > 1000, "strona HTML z zapisanego matching.json (tune_matching.py --html)")

        print("Prezentacja (zapis metody, spójność)")
        import presentation as pr
        check("settings" in config.read_json(out_dir / "merging.json"), "merging.json z ustawieniami prepare_data")
        pm, pcp = pr.method_settings("manual")
        rep_m = {"params": ms.json_safe(asdict(pm)), "contact_params": ms.json_safe(asdict(pcp)), "surfaces": [], "pairs": []}
        pr.save_snapshot("manual", rep_m, loaded, pm, pcp, out_dir, root)
        snap, why = pr.load_snapshot("manual", loaded, out_dir, root)
        check(snap is not None and why == [] and bone.id in snap["contacts"]
              and np.array_equal(snap["contacts"][bone.id].labels, ct.load_contacts(bone.id, bone.mesh, root=root).labels),
              f"zapis metody aktualny zaraz po zapisie, z tymi samymi oznaczeniami ({why})")
        manual_path = ct.contacts_path(bone.id, "manual", root)
        kept = manual_path.read_bytes()
        edited = ct.load_contacts(bone.id, bone.mesh, root=root)
        edited.labels[np.flatnonzero(edited.labels == 0)[:200]] = 1
        ct.save_contacts(bone.mesh, edited, manual_path)
        _, why = pr.load_snapshot("manual", loaded, out_dir, root)
        check(any("ręczne oznaczenia" in w for w in why), "zmienione ręczne oznaczenia -> przelicz")
        manual_path.write_bytes(kept)
        snap_rep = pr.snapshot_dir("manual", out_dir) / "matching.json"
        changed = config.read_json(snap_rep)
        changed["params"]["tolerance_au"] = 9.0
        snap_rep.write_text(json.dumps(changed), encoding="utf-8")
        _, why = pr.load_snapshot("manual", loaded, out_dir, root)
        check(len(why) == 1 and "matching" in why[0] and "tolerance_au" in why[0], "inne ustawienia dopasowania -> przelicz")
        _, why = pr.load_snapshot("features", loaded, out_dir, root)
        check(len(why) == 1 and "brak" in why[0], "metoda bez zapisu -> przelicz")
        check(pr.find_matching("manual", out_dir=out_dir)[0] == snap_rep
              and pr.find_matching(last=True, out_dir=out_dir)[0] == out_dir / "matching.json",
              "podgląd: zapis metody albo ostatni przebieg (--last)")
        try:
            pr.find_matching("features", out_dir=out_dir)
            check(False, "metoda bez zapisu -> błąd z komendą")
        except SystemExit as e:
            check("--method features" in str(e) and "--last" in str(e), "metoda bez zapisu -> błąd z komendą")
        snap_tool = MatchState(out_dir, pcp, View3DParams(triangles_total=20000), root, path=snap_rep)
        check(snap_tool.path == snap_rep and snap_tool.report["params"]["tolerance_au"] == 9.0,
              "podgląd wczytuje zapis metody (nie ostatni przebieg)")
        check(pr.method_key("manual-auto", "atlas") is None and pr.method_key("auto", "atlas") == "atlas",
              "manual-auto (mieszanka) nie jest metodą prezentacji")
        check(pr.parse_steps(["porównanie,powierzchnie"]) == ("powierzchnie", "porownanie")
              and pr.page_path("atlas", out_dir, ("powierzchnie", "porownanie")).name
              == "prezentacja_atlas_powierzchnie-porownanie.html"
              and pr.page_path("atlas", out_dir).name == "prezentacja_atlas.html",
              "kroki prezentacji: kolejność, polskie znaki, osobny plik bez dopasowania")
        try:
            pr.parse_steps(["foo"])
            check(False, "nieznany krok -> błąd")
        except ValueError:
            check(True, "nieznany krok -> błąd")

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
        check(0.8 < area / area_a < 1.3, f"połączona kość: oznaczenia po zmianie nazwy ({area:.0f} / {area_a:.0f} a.u.²)")

        split = prepare_all(scans_dir, out_dir, contacts_dir=root, scan=ScanParams(),
                            merge=MergeParams(enabled=False), features=FeatureParams(), verbose=False)
        ref = [x for x in split if x.id == na.id][0]
        got = ct.load_contacts(ref.id, ref.mesh, root=root, frames=ct.scan_frames(ref))
        area = ct.vertex_areas(ref.mesh)[got.labels > 0].sum()
        check(len(split) == 4 and area > 0.2 * area_a,
              f"łączenie wyłączone (inny środek kości): oznaczenia przeniesione przez wspólny skan ({area:.0f} a.u.²)")

        print("Syntetyczne kości (make_synthetic.py)")
        import synthetic_bones as sb
        from scipy.spatial import cKDTree
        synth =sb.generate(sb.SyntheticParams(n_bones=2, fragments=(3, 3), edge_au=0.5, seed=1))
        meshes = [trimesh.Trimesh(f.vertices, f.faces, process=False) for f in synth.fragments]
        check(len(synth.fragments) == 6 and all(m.is_watertight and m.is_winding_consistent and m.volume > 0
                                                for m in meshes), "6 fragmentów, siatki szczelne, normalne na zewnątrz")
        syn_in, syn_out, syn_ct = tmp / "syn" / "skany", tmp / "syn" / "out", tmp / "syn" / "contacts"
        gt = sb.write_dataset(synth, syn_in, syn_ct)
        check(len(gt["pairs"]) == 4, f"4 pary (po 2 przełomy na kość): {len(gt['pairs'])}")
        saved_in, saved_settings = config.DATA_IN, config.DATASET_SETTINGS
        config.DATA_IN, config.DATASET_SETTINGS = syn_in, config.load_settings(syn_in / sb.DATASET_SETTINGS)
        check(not config.merge_params().enabled, "settings.json zestawu wyłącza łączenie skanów")
        config.DATA_IN, config.DATASET_SETTINGS = saved_in, saved_settings
        syn_bones = {b.id: b for b in prepare_all(syn_in, syn_out, contacts_dir=syn_ct, scan=ScanParams(),
                                                  merge=MergeParams(enabled=False), features=FeatureParams(),
                                                  verbose=False)}
        p = gt["pairs"][0]
        a, b = syn_bones[p["a"]], syn_bones[p["b"]]
        la = ct.load_contacts(a.id, a.mesh, root=syn_ct, frames=ct.scan_frames(a)).labels
        lb = ct.load_contacts(b.id, b.mesh, root=syn_ct, frames=ct.scan_frames(b)).labels
        T = sb.true_transform(gt, a, b)
        pb = np.asarray(b.mesh.vertices)[lb == p["surface_b"]] @ T[:3, :3].T + T[:3, 3]
        d = cKDTree(np.asarray(a.mesh.vertices)[la == p["surface_a"]]).query(pb)[0]
        check(np.median(d) < 0.3, f"prawdziwe przełomy na przygotowanych kościach nakładają się "
                                  f"(true_transform, mediana {np.median(d):.2f} a.u.)")
        one = sb.SyntheticParams(n_bones=1, fragments=(2, 2), edge_au=0.5, seed=1)
        solid = [trimesh.Trimesh(f.vertices, f.faces, process=False) for f in sb.generate(one).fragments]
        hollow = [trimesh.Trimesh(f.vertices, f.faces, process=False)
                  for f in sb.generate(replace(one, marrow=0.5)).fragments]
        check(all(m.is_watertight and m.volume > 0 for m in hollow)
              and sum(m.volume for m in hollow) < sum(m.volume for m in solid) - 20,
              f"kanał szpikowy (marrow): wklęsły dół na przełomie, siatki szczelne "
              f"({sum(m.volume for m in solid) - sum(m.volume for m in hollow):.0f} a.u.³ mniej)")
        chip = sb.generate(replace(sb.SyntheticParams(edge_au=0.5), **sb.PRESETS["chip"]))
        thin = [f for f in chip.fragments if f.index == 2][0]
        tm = trimesh.Trimesh(thin.vertices, thin.faces, process=False)
        check(len(chip.fragments) == 4 and len(thin.surfaces) == 2 and tm.is_watertight and tm.volume > 0,
              f"preset chip: 4 części, płaski odłamek szczelny, 2 przełomy ({tm.volume:.0f} a.u.³)")
    print(f"Wszystkie testy OK ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()

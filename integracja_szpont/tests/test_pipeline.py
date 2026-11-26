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
algorytm wykrywania i budowę wizualizacji.

Uruchomienie (z głównego folderu projektu):
    python tests/test_pipeline.py
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

import contacts as ct  # noqa: E402
from bones import FeatureParams, load_bones  # noqa: E402
from data_parsing import ScanParams, find_scans  # noqa: E402
from merging import MergeParams  # noqa: E402
from pipeline import prepare_all  # noqa: E402
from visualization_3d import View3DParams, bones_figure, merging_figure  # noqa: E402


def bumpy_sphere(seed: int, radius: float = 12.0) -> trimesh.Trimesh:
    """Kula z losowymi wybrzuszeniami - kształt z cechami, które da się dopasować."""
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
    """Część powierzchni widoczna z kierunku `view`, w losowym układzie współrzędnych."""
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
    mesh = mesh.copy()
    mesh.visual.vertex_colors = np.tile([*color, 255], (len(mesh.vertices), 1)).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def make_scans(root: Path) -> None:
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
    print(("  OK   " if cond else "  BŁĄD ") + msg)
    if not cond:
        raise SystemExit(1)


def main() -> None:
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
        for name in ct.DETECTORS:
            found = ct.detect_contacts(bone, name, ct.ContactParams())
            check(len(found.labels) == len(v), f"algorytm \"{name}\" działa ({len(found.surface_ids)} pow.)")

        print("Wizualizacje")
        fig = bones_figure(loaded, {bone.id: back}, "fpfh", View3DParams())
        check(len(fig.data) > 4, "figura kości (FPFH + powierzchnie styku)")
        fig = merging_figure(loaded, View3DParams())
        check(len(fig.data) >= 4, "figura łączenia skanów (3 skany + szum)")
    print(f"Wszystkie testy OK ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()

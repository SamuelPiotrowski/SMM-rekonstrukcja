"""
Test atlasu anatomicznego (src/atlas.py, algorytm "atlas") na SZTUCZNYCH kościach - bez prawdziwych skanów.

Tworzy w folderze tymczasowym:
    - BAZĘ 4 kompletnych kości o różnym kształcie: "kość udowa" (gruba, okrągła, szerokie nasady;
      model z kanałem szpikowym w środku - wewnętrzna ściana jak w modelu z tomografii), "piszczel"
      (trójgraniasta), "kość ramienna" (spłaszczona, wygięta, z SZORSTKIM przyczepem mięśnia na korze),
      "kość strzałkowa" (cienka, wygięta). Modele są w innych jednostkach niż skany (skan = 1.1 x model).
    - FRAGMENTY "skanów" tych samych typów kości, ale INNEJ osoby (wymiary inne o kilka %, inne drobne
      nierówności, inny wzór szorstkości przyczepu): ucięte prawdziwymi przełomami z KANAŁEM SZPIKOWYM
      (na skanie wklęsły dół, nie dziura), skan częściowy (bez tylnej strony kory), losowo obrócone,
      z szumem skanera. Fragment kości strzałkowej jest odbity lustrzanie (druga strona ciała),
      dodatkowo fragment spoza bazy (kawałek kulistej kości).
Sprawdza:
    - bazę (indeks): liczona raz, potem z dysku; zmieniony plik - tylko on od nowa; zmiana nazwy pliku
      bez liczenia; działa po usunięciu plików modeli; mniejsza niż pliki; tylko zewnętrzna powierzchnia,
    - sito własności: wyklucza modele PRZED RANSAC, nigdy prawdziwej kości,
    - rozpoznanie: prawdziwa kość na 1. miejscu z p > 0.5, skala ~1.1, fragment spoza bazy -> "żadna z bazy",
    - korę vs przełom: IoU z prawdą lepsze niż "features", kanał szpikowy zaznaczony, szorstki przyczep
      mięśnia NIE jest przełomem (a dla "features" jest),
    - integrację: algorytm "atlas" w contacts.py (ranking w pliku wyniku, cache), raporty i strona HTML.

Uruchomienie (z głównego folderu projektu, ~2-3 min):
    python tests/test_atlas.py
"""

import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import atlas  # noqa: E402
import config  # noqa: E402
import contacts as ct  # noqa: E402
import synthetic_bones as sb  # noqa: E402
from bones import FeatureParams  # noqa: E402
from data_parsing import ScanParams  # noqa: E402
from merging import MergeParams, file_checksum  # noqa: E402
from pipeline import prepare_all  # noqa: E402
from visualization_3d import View3DParams  # noqa: E402

SCALE = 1.1                     # skan = SCALE x model (inne jednostki)
MARROW, MARROW_DEPTH = 0.45, 3.0
# ustawienia atlasu do testu (przez ustawienia zestawu - jak scanned_bones/<zestaw>/settings.json)
TEST_ATLAS = {"tolerance_au": 0.6, "scale_range": [0.8, 1.45], "curvature_radius_au": 2.0,
              "max_unexplained": 0.15, "ransac_iterations": 100000}


def check(cond: bool, msg: str) -> None:
    """Wypisuje "OK" albo "BŁĄD" z opisem; przy błędzie kończy test (kod 1)."""
    print(("  OK   " if cond else "  BŁĄD ") + msg)
    if not cond:
        raise SystemExit(1)


# -----------------
# KSZTAŁTY KOŚCI
# -----------------

@dataclass
class AnatomyShape(sb.BoneShape):
    """Kość z generatora (sb.BoneShape) + cechy typu kości: płaty przekroju, zwężanie, szorstki przyczep."""

    lobes: int = 0                  # płaty przekroju (3 = trójgraniasta jak piszczel)
    lobe_amp: float = 0.0           # wysokość płatów (część promienia)
    lobe_phase: float = 0.0
    taper: float = 0.0              # zmiana promienia wzdłuż kości (0.2 = jeden koniec o 20% grubszy)
    patch: tuple | None = None      # (kąt, wysokość, szerokość, FourierNoise) - szorstki przyczep mięśnia

    @property
    def max_radius(self) -> float:
        """Górne oszacowanie promienia (z płatami, zwężaniem i przyczepem)."""
        return super().max_radius + self.radius * (abs(self.lobe_amp) + abs(self.taper)) + (1.0 if self.patch else 0)

    def patch_weight(self, theta: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Jak bardzo punkt (kąt, wysokość) leży na przyczepie mięśnia (0..1)."""
        if self.patch is None:
            return np.zeros(np.shape(theta))
        th0, z0, width, _ = self.patch
        dth = np.angle(np.exp(1j * (np.asarray(theta) - th0)))
        return np.exp(-((self.radius * dth) ** 2 + (np.asarray(z) - z0) ** 2) / (2 * width ** 2))

    def radius_at(self, theta: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Promień kory (sb.BoneShape) z płatami, zwężaniem i szorstkim, lekko wypukłym przyczepem."""
        r = super().radius_at(theta, z) * (1 + self.taper * (np.asarray(z) / self.length - 0.5))
        if self.lobes:
            r = r + self.lobe_amp * self.radius * np.cos(self.lobes * (np.asarray(theta) - self.lobe_phase))
        if self.patch is not None:
            p = np.stack([self.radius * np.cos(theta), self.radius * np.sin(theta), z], axis=-1)
            r = r + self.patch_weight(theta, z) * (0.3 + self.patch[3](p))
        return r


# typy kości: wymiary w jednostkach modelu
TYPES = {
    "kość udowa": dict(length=90.0, radius=7.0, flare=0.5, ellipticity=0.05, bend=1.5, canal=True),
    "piszczel": dict(length=80.0, radius=6.0, flare=0.35, ellipticity=0.05, bend=1.0, lobes=3, lobe_amp=0.16,
                     taper=0.2),
    "kość ramienna": dict(length=70.0, radius=5.0, flare=0.6, ellipticity=0.3, bend=3.0, patch=(1.0, 0.75)),
    "kość strzałkowa": dict(length=85.0, radius=3.0, flare=0.3, ellipticity=0.2, bend=4.0),
}


def make_shape(kind: str, rng: np.random.Generator, individual: bool = False) -> AnatomyShape:
    """Kształt kości danego typu; individual = inna osoba (wymiary o kilka % inne, inne drobne nierówności)."""
    t = TYPES[kind]
    k = (1.04, 0.96, 1.15) if individual else (1.0, 1.0, 1.0)
    length, radius = t["length"] * k[0], t["radius"] * k[1]
    patch = None
    if "patch" in t:
        patch = (t["patch"][0], t["patch"][1] * length, 2.5, sb.FourierNoise(rng, 3, 0.3, (0.8, 2.0), n=24))
    return AnatomyShape(length=length, radius=radius, flare=t["flare"], ellipticity=t["ellipticity"],
                        ellipse_phase=0.4, bend=t["bend"] * k[2] * np.array([np.cos(0.3), np.sin(0.3)]),
                        bumps=sb.FourierNoise(rng, 3, 0.15, (4.0, 15.0)), lobes=t.get("lobes", 0),
                        lobe_amp=t.get("lobe_amp", 0.0), lobe_phase=0.2, taper=t.get("taper", 0.0), patch=patch)


def ends(shape: AnatomyShape) -> list:
    """Dwa końce kości (kopułki) - sb.cut_surfaces bez cięć."""
    return sb.cut_surfaces(np.random.default_rng(0), shape, [], sb.SyntheticParams())


def whole_bone(kind: str, rng: np.random.Generator) -> trimesh.Trimesh:
    """Model kompletnej kości (siatka zamknięta); kość udowa z wewnętrzną ścianą kanału szpikowego."""
    shape = make_shape(kind, rng)
    lo, hi = ends(shape)
    v, f, _ = sb.build_fragment(shape, lo, hi, 0.5)
    mesh = trimesh.Trimesh(v, f, process=False)
    if TYPES[kind].get("canal"):
        inner = AnatomyShape(length=shape.length, radius=0.45 * shape.radius, flare=0.0, ellipticity=0.0,
                             ellipse_phase=0.0, bend=shape.bend, bumps=sb.FourierNoise(rng, 3, 0.0, (4.0, 15.0)))
        a = sb.CutSurface(kind="end", z_ref=0.2 * shape.length, center=inner.axis(np.array(0.2 * shape.length)),
                          dome=-1.0, dome_radius=inner.max_radius)
        b = sb.CutSurface(kind="end", z_ref=0.8 * shape.length, center=inner.axis(np.array(0.8 * shape.length)),
                          dome=1.0, dome_radius=inner.max_radius)
        iv, iff, _ = sb.build_fragment(inner, a, b, 0.5)
        mesh = trimesh.util.concatenate([mesh, trimesh.Trimesh(iv, iff[:, [0, 2, 1]], process=False)])  # normalne do kanału
    return mesh


@dataclass
class FragmentSpec:
    """Opis fragmentu do wygenerowania."""

    name: str
    kind: str | None                # typ kości (None = spoza bazy)
    cuts: tuple = ()                # (dół, góra): "end" albo wysokość przełomu jako część długości
    mirror: bool = False


SPECS = [
    FragmentSpec("frag_udowa_koniec", "kość udowa", ("end", 0.35)),
    FragmentSpec("frag_piszczel_srodek", "piszczel", (0.3, 0.62)),
    FragmentSpec("frag_ramienna_przyczep", "kość ramienna", (0.5, "end")),
    FragmentSpec("frag_strzalkowa_lustro", "kość strzałkowa", (0.25, 0.6), mirror=True),
    FragmentSpec("frag_kulista", None),
]


def compact(v: np.ndarray, f: np.ndarray, keep: np.ndarray, *per_vertex):
    """Zostawia trójkąty `keep` i używane przez nie wierzchołki (z tablicami na wierzchołek)."""
    f = f[keep]
    used = np.unique(f)
    remap = np.full(len(v), -1)
    remap[used] = np.arange(len(used))
    return (v[used], remap[f], *[a[used] for a in per_vertex])


def make_fragment(spec: FragmentSpec, rng: np.random.Generator):
    """Fragment "skanu": (wierzchołki, trójkąty, etykiety przełomu, maska przyczepu, maska kanału) w układzie skanu."""
    if spec.kind is None:                                     # kawałek kulistej kości (spoza bazy)
        m = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
        u = m.vertices
        r = np.full(len(u), 13.0)
        for _ in range(25):
            c = rng.normal(size=3)
            c /= np.linalg.norm(c)
            r += rng.uniform(-1.5, 1.5) * np.exp(-np.sum((u - c) ** 2, axis=1) / (2 * 0.3 ** 2))
        v, f = u * r[:, None], m.faces
        keep = (v[f].mean(axis=1) @ np.array([0, 0, 1.0])) > 3.0
        v, f = compact(v, f, keep)
        lab = np.zeros(len(v), np.uint8)
        patch = canal = np.zeros(len(v), bool)
    else:
        shape = make_shape(spec.kind, rng, individual=True)
        e_lo, e_hi = ends(shape)
        fp = sb.SyntheticParams(tilt_deg=20.0, relief_au=0.8)
        cut = [e_lo if c == "end" and i == 0 else e_hi if c == "end" else sb.random_fracture(rng, shape, c * shape.length, fp)
               for i, c in enumerate(spec.cuts)]
        lo_id = 1 if cut[0].kind == "fracture" else 0
        hi_id = lo_id + 1 if cut[1].kind == "fracture" else 0
        v, f, lab = sb.build_fragment(shape, cut[0], cut[1], 0.3, rng.uniform(0, 2 * np.pi), lo_id, hi_id,
                                      MARROW, MARROW_DEPTH)
        ax = shape.axis(v[:, 2])
        theta = np.arctan2(v[:, 1] - ax[:, 1], v[:, 0] - ax[:, 0])
        patch = (lab == 0) & (shape.patch_weight(theta, v[:, 2]) > 0.3)
        canal = (lab > 0) & (np.linalg.norm(v[:, :2] - ax, axis=1) < 0.7 * MARROW * shape.radius)
        # skan częściowy: bez tylnej strony kory (przełomy skaner widzi całe)
        n = trimesh.Trimesh(v, f, process=False).face_normals
        view = np.array([np.cos(2.0), np.sin(2.0), 0.0])
        keep = ~((lab[f] == 0).all(axis=1) & (n @ view < -0.35))
        v, f, lab, patch, canal = compact(v, f, keep, lab, patch, canal)
    v = v * SCALE
    if spec.mirror:
        v, f = v * np.array([-1.0, 1, 1]), f[:, [0, 2, 1]]
    normals = trimesh.Trimesh(v, f, process=False).vertex_normals
    v = v + normals * rng.normal(0, 0.02, len(v))[:, None]
    R = Rotation.random(random_state=rng).as_matrix()
    return v @ R.T + rng.uniform(-30, 30, 3), f, lab, patch, canal


def save_labels(v, f, labels, path: Path, scan_file: Path) -> None:
    """Etykiety na wierzchołkach skanu -> plik oznaczeń (przechodzą na przygotowaną kość przez wspólny skan)."""
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f.astype(np.int32)))
    ct.save_contacts(mesh, ct.Contacts(path.stem, labels.astype(np.uint8)), path, {file_checksum(scan_file): np.eye(4).tolist()})


def main() -> None:
    """Wszystkie sprawdzenia po kolei, w folderze tymczasowym (usuwany na końcu)."""
    start = time.perf_counter()
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src, index = tmp / "atlas", tmp / "atlas_index"
        # ustawienia testu jak ustawienia zestawu; foldery atlasu na folder tymczasowy
        saved = (config.DATASET_SETTINGS, config.ATLAS_DIR, config.ATLAS_INDEX)
        config.DATASET_SETTINGS = {**config.DATASET_SETTINGS, "atlas": TEST_ATLAS}
        config.use_atlas(src, index)
        try:
            run(tmp, src, index, rng)
        finally:
            config.DATASET_SETTINGS = saved[0]
            config.use_atlas(*saved[1:])
    print(f"Wszystkie testy OK ({time.perf_counter() - start:.0f} s)")


def run(tmp: Path, src: Path, index: Path, rng: np.random.Generator) -> None:
    """Treść testu (foldery i ustawienia przygotowuje main)."""
    p = config.atlas_params()
    print("Baza modeli (indeks)")
    src.mkdir()
    for kind in TYPES:
        whole_bone(kind, rng).export(src / f"{kind}.ply")
    t = time.perf_counter()
    rep = atlas.sync_index(p, src, index, log=None)
    t_build = time.perf_counter() - t
    check(len(rep["new"]) == 4, f"4 nowe modele zindeksowane ({t_build:.1f} s)")
    t = time.perf_counter()
    rep = atlas.sync_index(p, src, index, log=None)
    check(len(rep["kept"]) == 4 and not rep["new"] and not rep["changed"] and time.perf_counter() - t < 1.0,
          f"drugi raz z indeksu, bez liczenia ({time.perf_counter() - t:.2f} s)")
    src_size = sum(f.stat().st_size for f in src.glob("*.ply"))
    idx_size = sum(f.stat().st_size for f in index.glob("*"))
    check(idx_size < 0.6 * src_size, f"baza mniejsza niż pliki modeli: {idx_size / 1e6:.2f} MB vs {src_size / 1e6:.2f} MB")
    db = atlas.load_atlas(p, index, src, log=None)
    femur = next(m for m in db.models if m.name == "kość udowa")
    sh = make_shape("kość udowa", np.random.default_rng(0))
    zz = femur.points[:, 2]
    radial = np.linalg.norm(femur.points[:, :2] - sh.axis(zz), axis=1)
    inner = ((radial < 0.6 * sh.radius) & (zz > 0.22 * sh.length) & (zz < 0.78 * sh.length)).mean()
    check(inner < 0.002 and femur.entry["info"]["outer_share"] < 0.85,
          f"model z kanałem szpikowym: zapisana tylko zewnętrzna powierzchnia (punktów w środku: {inner:.2%}, "
          f"zewnętrzna = {femur.entry['info']['outer_share']:.0%} powierzchni)")
    # zmiana pliku -> tylko on od nowa; zmiana nazwy -> bez liczenia
    shape = make_shape("kość strzałkowa", np.random.default_rng(99))
    lo, hi = ends(shape)
    v, f, _ = sb.build_fragment(replace(shape, length=88.0), lo, hi, 0.5)
    trimesh.Trimesh(v, f, process=False).export(src / "kość strzałkowa.ply")
    rep = atlas.sync_index(p, src, index, log=None)
    check(rep["changed"] == [m.id for m in db.models if m.name == "kość strzałkowa"] and len(rep["kept"]) == 3,
          f"zmieniony plik -> liczony tylko on ({rep['changed']})")
    (src / "piszczel.ply").rename(src / "piszczel prawa.ply")
    rep = atlas.sync_index(p, src, index, log=None)
    check(len(rep["renamed"]) == 1 and not rep["new"], f"zmiana nazwy pliku -> bez liczenia ({rep['renamed']})")
    (src / "piszczel prawa.ply").rename(src / "piszczel.ply")
    atlas.sync_index(p, src, index, log=None)
    (src / "_pomijany").mkdir()
    shutil.copy(src / "piszczel.ply", src / "_pomijany" / "kopia.ply")
    check(len(atlas.find_models(src)) == 4, "pliki w folderze \"_...\" pomijane")
    # baza bez plików modeli
    hidden = tmp / "modele_gdzie_indziej"
    src.rename(hidden)
    rep = atlas.sync_index(p, src, index, log=None)
    db = atlas.load_atlas(p, index, src, log=None)
    check(len(db.models) == 4 and len(rep["only_index"]) == 4,
          "baza działa bez plików modeli (modele zostają w indeksie)")
    pruned = tmp / "kopia_indeksu"
    shutil.copytree(index, pruned)
    rep = atlas.sync_index(p, src, pruned, prune=True, log=None)
    check(len(rep["removed"]) == 4 and not list(pruned.glob("*.npz")), "--prune usuwa modele bez plików")
    hidden.rename(src)

    print("Fragmenty (skany) i przygotowanie")
    scans, out, cdir, truth_dir, extra_dir = (tmp / x for x in ("skany", "out", "contacts", "prawda", "maski"))
    for spec in SPECS:
        v, f, lab, patch, canal = make_fragment(spec, rng)
        path = scans / spec.name / f"{spec.name}.ply"
        path.parent.mkdir(parents=True)
        trimesh.Trimesh(v, f, process=False).export(path)
        save_labels(v, f, lab, truth_dir / f"{spec.name}.json", path)
        save_labels(v, f, patch + 2 * canal, extra_dir / f"{spec.name}.json", path)
    bones = {b.id: b for b in prepare_all(scans, out, contacts_dir=cdir, scan=ScanParams(),
                                          merge=MergeParams(enabled=False), features=FeatureParams(), verbose=False)}
    check(len(bones) == len(SPECS), f"{len(SPECS)} fragmentów przygotowanych")

    def labels_of(root: Path, b) -> np.ndarray:
        """Etykiety z pliku oznaczeń przeniesione na przygotowaną kość."""
        return ct.load_contacts(b.id, b.mesh, root=root, frames=ct.scan_frames(b)).labels

    print("Rozpoznawanie kości (sito własności + RANSAC + ICP ze skalą)")
    cp, sfp = config.contact_params(), config.surface_feature_params()
    db = atlas.load_atlas(p, index, src, log=None)
    names = {m.id: m.name for m in db.models}
    matches, excluded = {}, 0
    for spec in SPECS:
        b = bones[spec.name]
        m = atlas.cached_match(b, db, p, cp, sfp)
        matches[spec.name] = m
        top = m.candidates[0] if m.candidates else None
        scr = {e["name"]: e["status"] for e in m.screening}
        excluded += sum(s != "ok" for s in scr.values())
        print(f"    {spec.name}: " + "\n      ".join(m.ranking_lines()) + f"  ({m.seconds:.0f} s)")
        if spec.kind is None:
            check(m.p_unknown > 0.5, f"{spec.name}: spoza bazy -> \"{atlas.UNKNOWN}\" {m.p_unknown:.0%}")
            continue
        check(scr[spec.kind] == "ok", f"{spec.name}: sito nie wyklucza prawdziwej kości")
        check(top is not None and names[top["model"]] == spec.kind and top["p"] > 0.5,
              f"{spec.name}: rozpoznana {spec.kind} ({top['p']:.0%})" if top else f"{spec.name}: brak kandydatów")
        check(abs(top["rel_scale"] / SCALE - 1) < 0.1, f"{spec.name}: skala ×{top['rel_scale']:.2f} (prawda ~×{SCALE})")
    check(excluded >= 3, f"sito wyklucza modele przed RANSAC (razem {excluded} wykluczeń)")
    t = time.perf_counter()
    again = atlas.cached_match(bones[SPECS[0].name], db, p, cp, sfp)
    check(again.cached and time.perf_counter() - t < 2, "drugi raz z cache")

    print("Kora vs przełom")
    ious = {"atlas": [], "features": []}
    for spec in SPECS:
        if spec.kind is None:
            continue
        b = bones[spec.name]
        truth, extra = labels_of(truth_dir, b), labels_of(extra_dir, b)
        la = ct.detect_contacts(b, "atlas", cp)
        lf = ct.detect_contacts(b, "features", cp)
        ra, rf = ct.compare(b.mesh, la.labels, truth), ct.compare(b.mesh, lf.labels, truth)
        ious["atlas"].append(ra["iou"])
        ious["features"].append(rf["iou"])
        va = ct.vertex_areas(b.mesh)
        canal = extra & 2 > 0
        rec_canal = va[canal & (la.labels > 0)].sum() / max(va[canal].sum(), 1e-9)
        print(f"    {spec.name}: IoU atlas {ra['iou']:.2f} (precyzja {ra['precision']:.2f}, czułość {ra['recall']:.2f}) "
              f"| features {rf['iou']:.2f} | kanał szpikowy wykryty w {rec_canal:.0%}")
        check(rec_canal > 0.8, f"{spec.name}: kanał szpikowy (wklęsły dół) = przełom ({rec_canal:.0%})")
        patch = extra & 1 > 0
        if patch.any():
            fa_a = va[patch & (la.labels > 0)].sum() / va[patch].sum()
            fa_f = va[patch & (lf.labels > 0)].sum() / va[patch].sum()
            check(fa_f > 0.3 and fa_a < 0.5 * fa_f,
                  f"{spec.name}: szorstki przyczep mięśnia - fałszywy alarm features {fa_f:.0%}, atlas {fa_a:.0%}")
        check("atlas" in la.info and la.info["atlas"]["ranking"], f"{spec.name}: ranking w wyniku algorytmu")
    ma, mf = np.mean(ious["atlas"]), np.mean(ious["features"])
    check(ma >= 0.6 and ma > mf + 0.05, f"średnie IoU z prawdą: atlas {ma:.2f}, features {mf:.2f}")

    print("Integracja (contacts, cache wyników, raporty)")
    b = bones[SPECS[0].name]
    root = tmp / "contacts_test"
    cpa = replace(cp, method="atlas")
    first, comp1 = ct.auto_contacts(b, cpa, root)
    again, comp2 = ct.auto_contacts(b, cpa, root)
    check(comp1 and not comp2 and again.info["atlas"]["ranking"][0]["model"] == first.info["atlas"]["ranking"][0]["model"],
          "wynik \"atlas\" w contacts/auto/ z rankingiem, drugi raz z cache")
    items = [{"nr": i, "bone": bones[s.name], "match": matches[s.name],
              "labels_atlas": ct.detect_contacts(bones[s.name], "atlas", cp).labels,
              "labels_features": ct.detect_contacts(bones[s.name], "features", cp).labels}
             for i, s in enumerate(SPECS, 1)]
    jpath, cpath = atlas.write_reports(out, items, db, p)
    page = atlas.write_page(out / "atlas.html", items, db, p, View3DParams(triangles_total=40000))
    check(jpath.stat().st_size > 1000 and cpath.stat().st_size > 100 and page.stat().st_size > 10000,
          "atlas.json, atlas.csv i atlas.html")
    whole_bone("piszczel", np.random.default_rng(5)).export(src / "nowa kość.ply")
    _, comp3 = ct.auto_contacts(b, cpa, root)
    check(comp3 and len(atlas.load_atlas(p, index, src, log=None).models) == 5,
          "nowy plik w bazie -> dopisany do indeksu, wynik liczony od nowa")


if __name__ == "__main__":
    main()

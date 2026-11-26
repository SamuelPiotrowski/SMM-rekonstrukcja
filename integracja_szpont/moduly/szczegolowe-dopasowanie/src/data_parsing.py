"""
Podproblemy 3-4 z README.md: wyszukanie skanów, wczytanie i naprawa siatek.

Skany leżą w DATA_IN (domyślnie scanned_bones/), także w podfolderach dowolnej
głębokości. Skaner Medit zapisuje każdą sesję jako folder, np.

    scanned_bones/
        23.11.2026-modelowanie matematyczne/
            23.11.2026-modelowanie matematyczne-lowerjaw.ply   (+ .obj, .stl)
            23.11.2026-modelowanie matematyczne-upperjaw.ply
        23.11.2026-modelowanie matematyczne_(1)/
            23.11.2026-modelowanie matematyczne-upperjaw.ply   <- ta sama nazwa pliku,
        ...                                                        ale INNY skan

Dlatego skan = (folder, nazwa pliku bez rozszerzenia). Ten sam skan zapisany
w kilku formatach (ply / obj / stl) jest wczytywany tylko raz. Najwygodniej: jeden
skan = jeden folder nazwany jak kość (np. first_scans/bone_01/...) - nazwa kości
to wtedy nazwa folderu (szczegóły: find_scans).

Zawartość:
    nazwy           - slugify, parse_name,
    wyszukiwanie    - find_scans -> lista ScanFile,
    wczytanie       - load_mesh, repair_mesh,
    kawałki skanu   - components, component_gaps, submesh, mesh_stats,
    próbkowanie     - sample_surface -> SurfaceSample, voxel_keep, to_cloud,
    deskryptory     - compute_fpfh.
Wszystkie długości są w a.u. (tak zapisuje skaner), pola w a.u.².

Uwaga: open3d na Windowsie nie otwiera ścieżek z polskimi znakami, dlatego
wczytujemy przez trimesh, a nazwy wyjściowe są zamieniane na ASCII (slug).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

log = logging.getLogger(__name__)

# np. "23.11.2026-kośc strzałkowa 3-lowerjaw"
NAME_PATTERN = re.compile(r"^(?P<date>\d{1,2}\.\d{1,2}\.\d{4})-(?P<bone>.+?)-(?P<part>[^-]+)$")
DATE_PREFIX = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}-")


@dataclass(frozen=True)
class ScanParams:
    """Ustawienia wyszukiwania i wstępnego czyszczenia skanów (settings.json, sekcja "scans").

    Oderwany kawałek skanu zostaje bez dodatkowego sprawdzania, jeśli jest duży
    (>= min_component_area_ratio x pole największego) i leży blisko głównego
    (<= max_component_gap_au). Pozostałe kawałki zostają tylko wtedy, gdy potwierdzą
    je inne skany tej samej kości (merging.decide_components) - inaczej to szum.
    """

    preferred_formats: tuple = ("ply", "obj", "stl")  # kolejność wyboru formatu tego samego skanu
    exclude: tuple = ("occlusion",)       # pomiń skany, których ścieżka zawiera któryś z tych tekstów
    min_component_area_ratio: float = 0.05  # kawałek mniejszy niż ten ułamek największego = podejrzany o szum
    max_component_gap_au: float = 3.0     # kawałek dalej od głównego niż to = podejrzany o szum (inny obiekt)


@dataclass(frozen=True)
class ScanFile:
    """Jeden plik skanu znaleziony w DATA_IN (wynik find_scans).

    Attributes:
        id:     unikalna nazwa ASCII, np. "23.11.2026-modelowanie_matematyczne_6-lowerjaw";
                używana w nazwach plików wynikowych i jako klucz w meta.json.
        label:  czytelna nazwa do wyświetlania, np. "modelowanie matematyczne_(6) / lowerjaw".
        path:   wybrany plik (.ply / .obj / .stl).
        folder: folder względem DATA_IN ("" = bezpośrednio w DATA_IN).
        part:   część skanu z nazwy pliku (u Medita: lowerjaw / upperjaw / occlusion...).
    """

    id: str
    label: str
    path: Path
    folder: str
    part: str


# -----------------
# NAZWY
# -----------------

def slugify(text: str) -> str:
    """Zamienia nazwę na bezpieczną nazwę ASCII (polskie znaki -> ASCII, spacje -> "_").

    Potrzebne, bo open3d na Windowsie nie umie zapisywać/czytać plików ze ścieżek
    z polskimi znakami. Zostają tylko litery ASCII, cyfry i znaki ". _ -". Przykład:
        '23.11.2026-kośc strzałkowa 3-lowerjaw' -> '23.11.2026-kosc_strzalkowa_3-lowerjaw'

    Args:
        text: dowolna nazwa (folderu, pliku, części skanu).

    Returns:
        Nazwa ASCII; może być pusta, jeśli w `text` nie było żadnego dozwolonego znaku.
    """
    text = text.replace("ł", "l").replace("Ł", "L")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"\s+", "_", text.strip())
    return re.sub(r"[^A-Za-z0-9._-]", "", text)


def parse_name(stem: str) -> dict:
    """Rozbija nazwę pliku ze skanera Medit "<data>-<kość>-<część>" na części.

    Przykład:
        "23.11.2026-kośc strzałkowa 3-lowerjaw"
        -> {"date": "23.11.2026", "bone": "kośc strzałkowa 3", "part": "lowerjaw"}

    Args:
        stem: nazwa pliku bez rozszerzenia.

    Returns:
        {"date", "bone", "part"}; jeśli nazwa nie pasuje do wzorca: date i bone = None,
        part = cała nazwa.
    """
    m = NAME_PATTERN.match(stem)
    if not m:
        return {"date": None, "bone": None, "part": stem}
    return m.groupdict()


# -----------------
# WYSZUKIWANIE PLIKÓW
# -----------------

def find_scans(data_dir: Path, params: ScanParams = ScanParams()) -> tuple[list[ScanFile], list[Path]]:
    """Szuka skanów w `data_dir` i wszystkich jego podfolderach.

    Pliki o tej samej nazwie (bez rozszerzenia) w TYM SAMYM folderze to jeden skan
    w kilku formatach - wybierany jest format stojący najwcześniej w
    `params.preferred_formats`. Te same nazwy w RÓŻNYCH folderach to różne skany.

    Nazwy skanów (ScanFile.id i ScanFile.label):
        - JEDYNY skan w podfolderze (np. bone_01/<cokolwiek>.ply): id = folder (ze ścieżką,
          "/" -> "-"), label = nazwa folderu bez daty - zmiana nazwy kości = zmiana nazwy folderu,
        - kilka skanów w podfolderze: id = folder + "-" + część (u Medita lowerjaw / upperjaw...),
          label = "<nazwa folderu bez daty> / <część>"; liczą się też skany pominięte (exclude),
          żeby zmiana "exclude" nie zmieniała nazw,
        - skan bezpośrednio w `data_dir`: id = nazwa pliku, label = nazwa pliku bez daty.
    Jeśli dwa id wyjdą takie same (bardzo nietypowe nazwy), drugie dostaje "_2" itd.

    Args:
        data_dir: folder ze skanami.
        params:   formaty i wzorce wykluczeń (wzorzec porównywany bez wielkości liter
                  ze ścieżką pliku względem `data_dir`).

    Returns:
        (skany posortowane po id, pominięte pliki pasujące do `params.exclude`).

    Raises:
        FileNotFoundError: jeśli `data_dir` nie istnieje.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Brak folderu ze skanami: {data_dir}")
    rank = {ext.lower(): i for i, ext in enumerate(params.preferred_formats)}
    best: dict[tuple[Path, str], Path] = {}
    for path in data_dir.rglob("*"):
        ext = path.suffix.lower().lstrip(".")
        if ext not in rank or not path.is_file():
            continue
        key = (path.parent, path.stem)
        current = best.get(key)
        if current is None or rank[ext] < rank[current.suffix.lower().lstrip(".")]:
            best[key] = path

    per_folder: dict[Path, int] = {}         # ile skanów w folderze (razem z pominiętymi)
    for parent, _ in best:
        per_folder[parent] = per_folder.get(parent, 0) + 1

    scans, excluded, used_ids = [], [], set()
    for (parent, stem), path in sorted(best.items(), key=lambda kv: str(kv[1]).lower()):
        rel = path.relative_to(data_dir)
        if any(p.lower() in str(rel).lower() for p in params.exclude):
            excluded.append(path)
            continue
        folder = "" if parent == data_dir else parent.relative_to(data_dir).as_posix()
        part = parse_name(stem)["part"]
        if folder and per_folder[parent] == 1:       # jedyny skan w folderze: nazwa = nazwa folderu
            scan_id = slugify(folder.replace("/", "-"))
            label = DATE_PREFIX.sub("", folder.split("/")[-1])
        elif folder:
            scan_id = slugify(folder.replace("/", "-")) + "-" + slugify(part)
            label = f"{DATE_PREFIX.sub('', folder.split('/')[-1])} / {part}"
        else:
            scan_id, label = slugify(stem), DATE_PREFIX.sub("", stem)
        unique, k = scan_id, 2
        while unique in used_ids:            # bardzo nietypowe nazwy mogą dać ten sam slug
            unique, k = f"{scan_id}_{k}", k + 1
        used_ids.add(unique)
        scans.append(ScanFile(id=unique, label=label, path=path, folder=folder, part=part))
    scans.sort(key=lambda s: s.id)
    return scans, excluded


# -----------------
# WCZYTYWANIE I NAPRAWA
# -----------------

def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    """Wczytuje plik ze skanem (przez trimesh, bo open3d nie otwiera ścieżek z polskimi znakami).

    Siatka jest wczytywana bez żadnych poprawek (wierzchołki nie są scalane - to robi
    repair_mesh). Brane są tylko kolory wierzchołków; tekstury są pomijane.

    Args:
        path: plik .ply / .obj / .stl.

    Returns:
        Siatka open3d z wierzchołkami w a.u. i kolorami wierzchołków w [0, 1] (jeśli są),
        jeszcze bez normalnych.

    Raises:
        ValueError: jeśli plik nie zawiera żadnych trójkątów.
    """
    tm = trimesh.load(path, process=False, force="mesh")
    if len(tm.faces) == 0:
        raise ValueError(f"{path.name}: plik nie zawiera trójkątów")
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(tm.faces, dtype=np.int32)),
    )
    if tm.visual.kind == "vertex":
        colors = np.asarray(tm.visual.vertex_colors)[:, :3] / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


def repair_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Naprawia błędy zapisu siatki (modyfikuje `mesh` w miejscu i go zwraca).

    Scala zduplikowane wierzchołki (STL zapisuje każdy trójkąt osobno), usuwa trójkąty
    zdegenerowane i zduplikowane, krawędzie nierozmaitościowe (wspólne dla >2 trójkątów)
    i nieużywane wierzchołki; liczy normalne wierzchołków i trójkątów. Celowo NIE
    wygładza ani nie upraszcza - szczegóły powierzchni złamania są potrzebne do dopasowania.

    Args:
        mesh: siatka z load_mesh.

    Returns:
        Ta sama siatka `mesh` (już naprawiona).
    """
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def components(mesh: o3d.geometry.TriangleMesh) -> tuple[np.ndarray, np.ndarray]:
    """Dzieli siatkę na oderwane kawałki (komponenty spójne).

    Skan zwykle ma jeden duży kawałek (kość) i czasem drobne oderwane kawałki: szum,
    fragment podłoża albo leżący obok inny fragment kości.

    Args:
        mesh: siatka (najlepiej po repair_mesh, inaczej STL rozpada się na pojedyncze trójkąty).

    Returns:
        (numer kawałka dla każdego trójkąta (T,), pola kawałków [a.u.^2] (K,)).
        Kawałki są ponumerowane od największego (0 = największy).
    """
    clusters, _, areas = mesh.cluster_connected_triangles()
    clusters, areas = np.asarray(clusters), np.asarray(areas)
    order = np.argsort(-areas, kind="stable")
    renumber = np.empty_like(order)
    renumber[order] = np.arange(len(order))
    return renumber[clusters], areas[order]


def component_gaps(mesh: o3d.geometry.TriangleMesh, tri_comp: np.ndarray) -> np.ndarray:
    """Odległość [a.u.] każdego kawałka od największego (kawałka 0).

    Odległość = najmniejsza odległość wierzchołka danego kawałka od wierzchołków kawałka 0.

    Args:
        mesh:     siatka.
        tri_comp: (T,) numer kawałka każdego trójkąta - z components().

    Returns:
        (K,) odległości w a.u.; kawałek 0 ma 0.
    """
    from scipy.spatial import cKDTree

    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    n = int(tri_comp.max()) + 1 if len(tri_comp) else 0
    gaps = np.zeros(n)
    if n <= 1:
        return gaps
    tree = cKDTree(v[np.unique(t[tri_comp == 0])])
    for c in range(1, n):
        gaps[c] = tree.query(v[np.unique(t[tri_comp == c])])[0].min()
    return gaps


def submesh(mesh: o3d.geometry.TriangleMesh, keep_triangles: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Kopia siatki tylko z wybranymi trójkątami (oryginał zostaje bez zmian).

    Args:
        mesh:           siatka źródłowa.
        keep_triangles: (T,) maska bool - True = trójkąt zostaje.

    Returns:
        Nowa siatka bez nieużywanych wierzchołków (numeracja wierzchołków się zmienia).
    """
    out = o3d.geometry.TriangleMesh(mesh)
    out.remove_triangles_by_mask(~np.asarray(keep_triangles, dtype=bool))
    out.remove_unreferenced_vertices()
    return out


def mesh_stats(mesh: o3d.geometry.TriangleMesh) -> dict:
    """Podstawowe statystyki siatki (zapisywane w meta.json i summary.csv).

    Args:
        mesh: siatka (może być pusta - wymiary są wtedy zerami).

    Returns:
        {"vertices", "triangles", "area_au2", "extent_au": [x, y, z]} - wymiary pudełka
        równoległego do osi układu [a.u.], zaokrąglone do 0.01.
    """
    extent = mesh.get_axis_aligned_bounding_box().get_extent() if len(mesh.vertices) else np.zeros(3)
    return {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "area_au2": round(float(mesh.get_surface_area()), 2),
        "extent_au": [round(float(e), 2) for e in extent],
    }


# -----------------
# PRÓBKOWANIE POWIERZCHNI
# -----------------

@dataclass
class SurfaceSample:
    """Równomierne punkty na powierzchni siatki (wynik sample_surface, ~1 punkt na woksel `spacing`).

    Wszystkie tablice mają tę samą długość N i tę samą kolejność punktów.
    """

    points: np.ndarray     # (N, 3)
    normals: np.ndarray    # (N, 3), jednostkowe, z trójkątów (skierowane na zewnątrz skanu)
    colors: np.ndarray     # (N, 3) w [0, 1]; szare, gdy siatka nie ma kolorów
    triangle: np.ndarray   # (N,) indeks trójkąta, z którego pochodzi punkt


def voxel_keep(points: np.ndarray, voxel: float) -> np.ndarray:
    """Przerzedzenie chmury: indeksy punktów, po jednym na każdy zajęty woksel.

    Przestrzeń jest dzielona na sześciany o boku `voxel` [a.u.]; z każdego zostaje pierwszy
    punkt (wg kolejności w `points`). Zwracane są indeksy, a nie punkty, żeby tak samo
    przerzedzić normalne, kolory itp.

    Args:
        points: (N, 3) punkty.
        voxel:  bok woksela [a.u.] ~ odstęp punktów po przerzedzeniu.

    Returns:
        Indeksy zostawionych punktów, rosnąco.
    """
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return np.sort(idx)


def sample_surface(mesh: o3d.geometry.TriangleMesh, spacing: float, density: float = 4.0,
                   seed: int = 0) -> SurfaceSample:
    """Losuje punkty równomiernie po powierzchni, potem przerzedza do ~1 punktu na woksel.

    Dzięki przerzedzeniu gęstość punktów nie zależy od gęstości oryginalnej siatki.
    Każdy punkt pamięta swój trójkąt (np. żeby wiedzieć, z którego kawałka pochodzi).
    Trójkąt jest losowany z prawdopodobieństwem proporcjonalnym do pola, punkt w nim -
    równomiernie. Normalna punktu = normalna trójkąta (z kolejności wierzchołków),
    kolor = kolory wierzchołków zmieszane wg położenia w trójkącie.

    Args:
        mesh:    siatka z co najmniej jednym trójkątem o niezerowym polu.
        spacing: rozmiar woksela [a.u.] ~ odstęp między punktami.
        density: ile punktów losować na spacing^2 powierzchni przed przerzedzeniem
                 (losowanych jest zawsze co najmniej 1000).
        seed:    ziarno losowania (stałe = powtarzalne wyniki).

    Returns:
        SurfaceSample z punktami, normalnymi, kolorami i numerami trójkątów.
    """
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    cross = np.cross(b - a, c - a)
    dbl_area = np.linalg.norm(cross, axis=1)
    total = dbl_area.sum() / 2
    rng = np.random.default_rng(seed)
    n = max(1000, int(density * total / spacing**2))
    tri = rng.choice(len(t), size=n, p=dbl_area / dbl_area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w = np.column_stack([1 - s, s * (1 - r2), s * r2])          # współrzędne barycentryczne
    pts = w[:, :1] * a[tri] + w[:, 1:2] * b[tri] + w[:, 2:] * c[tri]
    nrm = cross[tri] / np.maximum(dbl_area[tri], 1e-12)[:, None]
    if mesh.has_vertex_colors():
        col = np.asarray(mesh.vertex_colors)
        cols = (w[:, :1] * col[t[tri, 0]] + w[:, 1:2] * col[t[tri, 1]] + w[:, 2:] * col[t[tri, 2]])
    else:
        cols = np.full((n, 3), 0.8)
    keep = voxel_keep(pts, spacing)
    return SurfaceSample(pts[keep], nrm[keep], cols[keep], tri[keep])


def to_cloud(points: np.ndarray, normals: np.ndarray | None = None,
             colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    """Tablice numpy -> chmura punktów open3d.

    Args:
        points:  (N, 3) punkty.
        normals: (N, 3) normalne albo None.
        colors:  (N, 3) kolory w [0, 1] (wartości spoza są przycinane) albo None.

    Returns:
        Nowa chmura open3d (dane skopiowane jako float64).
    """
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(colors, dtype=np.float64), 0, 1))
    return cloud


def compute_fpfh(cloud: o3d.geometry.PointCloud, radius: float) -> o3d.pipelines.registration.Feature:
    """Deskryptory FPFH (33 liczby na punkt opisujące lokalny kształt w promieniu `radius`).

    Nie zależą od położenia ani obrotu, więc pozwalają znaleźć te same miejsca na dwóch
    skanach / fragmentach. Zależą od kierunku normalnych (odwrócone normalne = inny opis).

    Args:
        cloud:  chmura punktów z normalnymi.
        radius: promień sąsiedztwa [a.u.] (brane jest najwyżej 100 najbliższych sąsiadów).

    Returns:
        Obiekt open3d Feature; `.data` ma kształt (33, N), więc tablica (N, 33) to `.data.T`.

    Raises:
        ValueError: chmura nie ma normalnych.
    """
    if not cloud.has_normals():
        raise ValueError("FPFH wymaga normalnych")
    return o3d.pipelines.registration.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))

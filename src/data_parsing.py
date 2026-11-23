"""
Podproblemy 3-4 z README.md: wczytanie skanów z DATA_IN, oczyszczenie i przygotowanie
danych do algorytmu dopasowującego fragmenty.

Dla każdego skanu (jeden fragment = jeden stem pliku, np. ".../X-lowerjaw.ply")
powstaje w DATA_OUT/<slug>/:
    mesh_clean.ply  - oczyszczona siatka (z kolorami i normalnymi)
    cloud.ply       - chmura punktów z normalnymi (voxel = VOXEL_SIZE)
    fpfh.npy        - deskryptory FPFH chmury, kształt (N, 33)
    meta.json       - skąd pochodzi, statystyki, przesunięcie środka
oraz DATA_OUT/summary.csv ze wszystkimi fragmentami.

Uwaga: open3d na Windowsie nie otwiera ścieżek z polskimi znakami, dlatego
wczytujemy przez trimesh, a nazwy wyjściowe są zamieniane na ASCII (slug).

Uruchomienie całego etapu dla wszystkich skanów:
    python scripts/prepare_data.py
Funkcje z innego modułu:
    from data_parsing import load_prepared_fragments
"""

from __future__ import annotations

import csv
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from config import (
    CENTER_FRAGMENTS,
    DATA_IN,
    DATA_OUT,
    FPFH_RADIUS_FACTOR,
    MIN_COMPONENT_AREA_RATIO,
    NORMAL_RADIUS_FACTOR,
    PREFERRED_FORMATS,
    SAMPLING_DENSITY,
    VOXEL_SIZE,
)

log = logging.getLogger(__name__)

# np. "23.11.2026-kośc strzałkowa 3-lowerjaw"
NAME_PATTERN = re.compile(r"^(?P<date>\d{1,2}\.\d{1,2}\.\d{4})-(?P<bone>.+?)-(?P<part>[^-]+)$")


@dataclass
class Fragment:
    """Jeden zeskanowany fragment kości wraz ze wszystkimi danymi przygotowanymi do dopasowania.

    Attributes:
        name:   oryginalna nazwa skanu (nazwa pliku bez rozszerzenia, z polskimi znakami).
        slug:   ta sama nazwa zamieniona na ASCII - używana jako nazwa folderu w DATA_OUT.
        source: ścieżka do pliku źródłowego w DATA_IN.
        mesh:   oczyszczona siatka trójkątów (open3d), z normalnymi i ew. kolorami.
        cloud:  chmura punktów z normalnymi wylosowana z siatki (gęstość ~ VOXEL_SIZE).
        fpfh:   deskryptory FPFH dla każdego punktu chmury, tablica (N, 33).
        meta:   słownik ze statystykami przed/po czyszczeniu, przesunięciem środka itd.
                (to samo, co trafia do meta.json).
    """

    name: str                     # oryginalna nazwa (stem pliku)
    slug: str                     # nazwa ASCII używana w DATA_OUT
    source: Path
    mesh: o3d.geometry.TriangleMesh
    cloud: o3d.geometry.PointCloud | None = None
    fpfh: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


# -----------------
# WYSZUKIWANIE PLIKÓW
# -----------------

def slugify(text: str) -> str:
    """Zamienia nazwę skanu na bezpieczną nazwę ASCII, wygodniejszą do analizy i zapisu.

    Usuwa polskie znaki (ą->a, ł->l, ...), spacje zamienia na "_", a pozostałe
    znaki spoza [A-Za-z0-9._-] wyrzuca. Potrzebne, bo open3d na Windowsie
    nie umie zapisywać/czytać plików ze ścieżek z polskimi znakami.

    Args:
        text: dowolna nazwa, np. stem pliku ze skanera.

    Returns:
        Nazwa ASCII, np.
        '23.11.2026-kośc strzałkowa 3-lowerjaw' -> '23.11.2026-kosc_strzalkowa_3-lowerjaw'.
    """
    text = text.replace("ł", "l").replace("Ł", "L")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"\s+", "_", text.strip())
    return re.sub(r"[^A-Za-z0-9._-]", "", text)


def parse_name(stem: str) -> dict:
    """Rozbija nazwę pliku ze skanera na datę, nazwę kości i część skanu.

    Oczekiwany format (jak nazywa pliki skaner Medit): "<data>-<kość>-<część>",
    np. "23.11.2026-kośc strzałkowa 3-lowerjaw".

    Args:
        stem: nazwa pliku bez rozszerzenia.

    Returns:
        Słownik {"date", "bone", "part"}, np.
        {"date": "23.11.2026", "bone": "kośc strzałkowa 3", "part": "lowerjaw"}.
        Jeśli nazwa nie pasuje do wzorca: date i bone = None, part = cała nazwa.
    """
    m = NAME_PATTERN.match(stem)
    if not m:
        return {"date": None, "bone": None, "part": stem}
    return m.groupdict()


def find_scans(data_dir: Path = DATA_IN, formats=PREFERRED_FORMATS) -> dict[str, Path]:
    """Szuka plików ze skanami i wybiera dla każdego skanu jeden plik.

    Skaner zapisuje ten sam skan w kilku formatach (ply/obj/stl) o tej samej
    nazwie. Funkcja grupuje pliki po nazwie (bez rozszerzenia) i dla każdej
    grupy bierze format stojący najwcześniej na liście `formats`.

    Args:
        data_dir: folder ze skanami (domyślnie DATA_IN z config.py).
        formats:  rozszerzenia w kolejności preferencji, np. ("ply", "obj", "stl").
                  Pliki o innych rozszerzeniach są pomijane.

    Returns:
        Słownik {nazwa_skanu: ścieżka_do_wybranego_pliku}, posortowany po nazwie.
    """
    rank = {ext: i for i, ext in enumerate(formats)}
    best: dict[str, Path] = {}
    for path in sorted(data_dir.iterdir()):
        ext = path.suffix.lower().lstrip(".")
        if not path.is_file() or ext not in rank:
            continue
        current = best.get(path.stem)
        if current is None or rank[ext] < rank[current.suffix.lower().lstrip(".")]:
            best[path.stem] = path
    return best


# -----------------
# WCZYTYWANIE
# -----------------

def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    """Wczytuje plik ze skanem i zwraca go jako siatkę open3d.

    Czyta przez trimesh (bo open3d nie otwiera ścieżek z polskimi znakami),
    bez żadnej obróbki (process=False), a potem przepisuje wierzchołki,
    trójkąty i kolory wierzchołków (jeśli są) do obiektu open3d.

    Args:
        path: ścieżka do pliku .ply / .obj / .stl.

    Returns:
        Siatka TriangleMesh z wierzchołkami w mm i kolorami w zakresie [0, 1].

    Raises:
        ValueError: jeśli plik nie zawiera żadnych trójkątów (np. sama chmura punktów).
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


# -----------------
# CZYSZCZENIE
# -----------------

def mesh_stats(mesh: o3d.geometry.TriangleMesh) -> dict:
    """Liczy podstawowe statystyki siatki - do porównania przed i po czyszczeniu.

    Args:
        mesh: siatka do opisania.

    Returns:
        Słownik z kluczami:
            vertices   - liczba wierzchołków,
            triangles  - liczba trójkątów,
            area_mm2   - pole powierzchni [mm^2],
            extent_mm  - wymiary prostopadłościanu otaczającego [x, y, z] w mm,
            watertight - czy siatka jest zamknięta (bez dziur).
    """
    extent = mesh.get_axis_aligned_bounding_box().get_extent()
    return {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "area_mm2": round(mesh.get_surface_area(), 3),
        "extent_mm": [round(float(e), 3) for e in extent],
        "watertight": bool(mesh.is_watertight()),
    }


def remove_small_components(mesh: o3d.geometry.TriangleMesh, min_ratio: float) -> list[float]:
    """Usuwa małe, oderwane kawałki siatki (szum, odpryski, resztki tła ze skanu).

    Dzieli siatkę na komponenty spójne (grupy trójkątów połączonych krawędziami)
    i usuwa te, których pole jest mniejsze niż `min_ratio` * pole największego.
    Modyfikuje `mesh` w miejscu.

    Args:
        mesh:      siatka do oczyszczenia.
        min_ratio: próg względny, np. 0.05 = usuń wszystko poniżej 5% największego
                   kawałka. 0 lub mniej wyłącza usuwanie.

    Returns:
        Lista pól [mm^2] usuniętych komponentów (pusta, jeśli nic nie usunięto).
    """
    clusters, _, areas = mesh.cluster_connected_triangles()
    clusters, areas = np.asarray(clusters), np.asarray(areas)
    if len(areas) <= 1 or min_ratio <= 0:
        return []
    small = np.flatnonzero(areas < min_ratio * areas.max())
    if len(small):
        mesh.remove_triangles_by_mask(np.isin(clusters, small))
        mesh.remove_unreferenced_vertices()
    return [round(float(a), 3) for a in areas[small]]


def clean_mesh(mesh: o3d.geometry.TriangleMesh, min_component_ratio: float = MIN_COMPONENT_AREA_RATIO) -> dict:
    """Czyści siatkę z błędów skanowania (modyfikuje `mesh` w miejscu).

    Kolejno:
        1. scala zduplikowane wierzchołki (STL zapisuje każdy trójkąt osobno),
        2. usuwa zdegenerowane trójkąty (o zerowym polu),
        3. usuwa zduplikowane trójkąty,
        4. usuwa krawędzie nierozmaitościowe (wspólne dla >2 trójkątów),
        5. usuwa wierzchołki, których nie używa żaden trójkąt,
        6. usuwa małe oderwane kawałki (remove_small_components),
        7. liczy normalne wierzchołków i trójkątów.

    Celowo NIE wygładza ani nie upraszcza siatki - szczegóły powierzchni
    złamania są kluczowe dla dopasowania fragmentów.

    Args:
        mesh:                siatka do oczyszczenia.
        min_component_ratio: próg dla remove_small_components.

    Returns:
        Słownik z polami usuniętych ("removed_components_area_mm2") i pozostawionych
        ("kept_components_area_mm2", malejąco) komponentów w mm^2.
        Więcej niż jeden pozostawiony komponent = w pliku jest kilka osobnych kawałków.
    """
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    removed = remove_small_components(mesh, min_component_ratio)
    _, _, kept_areas = mesh.cluster_connected_triangles()

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return {
        "removed_components_area_mm2": removed,
        "kept_components_area_mm2": sorted((round(float(a), 3) for a in kept_areas), reverse=True),
    }


# -----------------
# PRZYGOTOWANIE DO DOPASOWANIA
# -----------------

def mesh_to_cloud(mesh: o3d.geometry.TriangleMesh, voxel: float = VOXEL_SIZE) -> o3d.geometry.PointCloud:
    """Zamienia siatkę na równomierną chmurę punktów z normalnymi.

    Najpierw losuje dużo punktów równomiernie po powierzchni (SAMPLING_DENSITY
    punktów na voxel^2), a potem przerzedza je siatką wokseli o boku `voxel`,
    tak że zostaje ~1 punkt na woksel. Dzięki temu gęstość punktów nie zależy
    od tego, jak gęsta była oryginalna siatka.

    Normalne biorą się z trójkątów siatki, więc są spójnie skierowane na zewnątrz
    kości (ważne dla FPFH i ICP point-to-plane w kroku dopasowania).

    Args:
        mesh:  oczyszczona siatka.
        voxel: odległość między punktami chmury [mm].

    Returns:
        PointCloud z punktami, normalnymi (długość 1) i kolorami, jeśli siatka je miała.
    """
    n_points = max(1000, int(SAMPLING_DENSITY * mesh.get_surface_area() / voxel**2))
    cloud = mesh.sample_points_uniformly(number_of_points=n_points, use_triangle_normal=True)
    cloud = cloud.voxel_down_sample(voxel)
    cloud.normalize_normals()
    return cloud


def compute_fpfh(cloud: o3d.geometry.PointCloud, voxel: float = VOXEL_SIZE) -> np.ndarray:
    """Liczy deskryptory FPFH dla każdego punktu chmury.

    FPFH (Fast Point Feature Histogram) to 33-liczbowy opis lokalnego kształtu
    powierzchni wokół punktu (jak zakrzywiają się normalne sąsiadów w promieniu
    voxel * FPFH_RADIUS_FACTOR). Nie zależy od położenia ani obrotu fragmentu,
    więc pozwala znaleźć pasujące miejsca na dwóch fragmentach - używa się go
    do zgrubnego dopasowania (RANSAC), zanim dopracuje się je ICP.

    Jeśli chmura nie ma normalnych, najpierw je szacuje
    (promień voxel * NORMAL_RADIUS_FACTOR).

    Args:
        cloud: chmura punktów (najlepiej z normalnymi z mesh_to_cloud).
        voxel: rozmiar woksela użyty przy tworzeniu chmury [mm].

    Returns:
        Tablica numpy (N, 33) - wiersz i opisuje i-ty punkt chmury.
    """
    if not cloud.has_normals():
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * NORMAL_RADIUS_FACTOR, max_nn=30)
        )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * FPFH_RADIUS_FACTOR, max_nn=100)
    )
    return np.asarray(fpfh.data).T


def prepare_fragment(name: str, path: Path) -> Fragment:
    """Pełne przygotowanie jednego skanu: wczytanie -> czyszczenie -> chmura -> FPFH.

    Kroki:
        1. wczytuje siatkę (load_mesh) i zapisuje statystyki "raw",
        2. czyści ją (clean_mesh); ostrzega w logu, jeśli zostało kilka osobnych kawałków,
        3. jeśli CENTER_FRAGMENTS, przesuwa fragment tak, by jego środek był w (0, 0, 0),
        4. tworzy chmurę punktów (mesh_to_cloud) i liczy FPFH (compute_fpfh),
        5. zbiera wszystkie informacje w słowniku meta.

    Nic nie zapisuje na dysk - od tego jest save_fragment.

    Args:
        name: nazwa skanu (stem pliku), np. "23.11.2026-kośc strzałkowa 3-lowerjaw".
        path: ścieżka do pliku ze skanem.

    Returns:
        Gotowy Fragment. W meta["translation_applied"] jest wektor przesunięcia -
        żeby wrócić do oryginalnych współrzędnych skanera, przesuń o jego minus.
    """
    log.info("Wczytywanie %s", path.name)
    mesh = load_mesh(path)
    raw = mesh_stats(mesh)

    cleaning = clean_mesh(mesh)
    kept = cleaning["kept_components_area_mm2"]
    if len(kept) > 1:
        log.warning(
            "  UWAGA: %s ma %d oddzielne kawałki (pow. %s mm2) - sprawdź, czy to jeden fragment",
            path.name, len(kept), kept,
        )

    center = mesh.get_center()
    if CENTER_FRAGMENTS:
        mesh.translate(-center)

    cloud = mesh_to_cloud(mesh)
    fpfh = compute_fpfh(cloud)

    meta = {
        "name": name,
        "source": path.name,
        **parse_name(name),
        "raw": raw,
        "clean": mesh_stats(mesh),
        **cleaning,
        "translation_applied": (-center).round(6).tolist() if CENTER_FRAGMENTS else [0.0, 0.0, 0.0],
        "cloud_points": len(cloud.points),
        "voxel_size_mm": VOXEL_SIZE,
        "has_colors": mesh.has_vertex_colors(),
    }
    return Fragment(name=name, slug=slugify(name), source=path, mesh=mesh, cloud=cloud, fpfh=fpfh, meta=meta)


# -----------------
# ZAPIS / ODCZYT
# -----------------

def save_fragment(frag: Fragment, out_dir: Path = DATA_OUT) -> Path:
    """Zapisuje przygotowany fragment do folderu out_dir/<slug>/.

    Tworzy pliki:
        mesh_clean.ply - oczyszczona siatka (binarny PLY),
        cloud.ply      - chmura punktów z normalnymi (binarny PLY),
        fpfh.npy       - deskryptory FPFH (N, 33),
        meta.json      - frag.meta (UTF-8, z polskimi znakami).
    Istniejące pliki są nadpisywane.

    Args:
        frag:    fragment z prepare_fragment.
        out_dir: folder wyjściowy (domyślnie DATA_OUT).

    Returns:
        Ścieżka do folderu, w którym zapisano fragment.
    """
    target = out_dir / frag.slug
    target.mkdir(parents=True, exist_ok=True)
    # open3d wymaga ścieżek ASCII - dlatego slug
    o3d.io.write_triangle_mesh(str(target / "mesh_clean.ply"), frag.mesh, write_ascii=False)
    o3d.io.write_point_cloud(str(target / "cloud.ply"), frag.cloud, write_ascii=False)
    np.save(target / "fpfh.npy", frag.fpfh)
    (target / "meta.json").write_text(json.dumps(frag.meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def write_summary(fragments: list[Fragment], out_dir: Path = DATA_OUT) -> Path:
    """Zapisuje tabelę podsumowującą wszystkie fragmenty do out_dir/summary.csv.

    Jeden wiersz na fragment: nazwa, data, kość, część, liczba trójkątów
    przed/po czyszczeniu, pole, wymiary, czy zamknięta, ile kawałków usunięto
    i ile zostało, liczba punktów chmury. Separator ";" i kodowanie utf-8-sig,
    żeby Excel od razu poprawnie otwierał plik z polskimi znakami.

    Args:
        fragments: lista przygotowanych fragmentów.
        out_dir:   folder wyjściowy (domyślnie DATA_OUT).

    Returns:
        Ścieżka do zapisanego pliku CSV.
    """
    path = out_dir / "summary.csv"
    cols = ["slug", "name", "date", "bone", "part", "raw_triangles", "clean_triangles",
            "area_mm2", "extent_x", "extent_y", "extent_z", "watertight", "removed_components",
            "kept_components", "cloud_points"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(cols)
        for fr in fragments:
            m, c = fr.meta, fr.meta["clean"]
            w.writerow([fr.slug, fr.name, m["date"], m["bone"], m["part"], m["raw"]["triangles"],
                        c["triangles"], c["area_mm2"], *c["extent_mm"], c["watertight"],
                        len(m["removed_components_area_mm2"]), len(m["kept_components_area_mm2"]),
                        m["cloud_points"]])
    return path


def load_prepared_fragments(out_dir: Path = DATA_OUT) -> list[Fragment]:
    """Wczytuje z dysku fragmenty przygotowane wcześniej przez scripts/prepare_data.py.

    Służy do kroku dopasowania - nie trzeba za każdym razem od nowa
    czyścić skanów. Szuka podfolderów out_dir zawierających meta.json.

    Args:
        out_dir: folder z wynikami (domyślnie DATA_OUT).

    Returns:
        Lista Fragmentów (siatka, chmura, FPFH, meta), posortowana po nazwie folderu.
        Pusta, jeśli nic nie zostało jeszcze przygotowane.
    """
    fragments = []
    for meta_path in sorted(out_dir.glob("*/meta.json")):
        d = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mesh = o3d.io.read_triangle_mesh(str(d / "mesh_clean.ply"))
        mesh.compute_vertex_normals()
        fragments.append(Fragment(
            name=meta["name"], slug=d.name, source=DATA_IN / meta["source"], mesh=mesh,
            cloud=o3d.io.read_point_cloud(str(d / "cloud.ply")),
            fpfh=np.load(d / "fpfh.npy"), meta=meta,
        ))
    return fragments

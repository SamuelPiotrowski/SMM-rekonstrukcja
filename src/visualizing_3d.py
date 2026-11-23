"""
Wizualizacja 3D fragmentów kości przygotowanych przez scripts/prepare_data.py.

Czyta z DATA_OUT (folder out/) to, co zapisał data_parsing.py:
    out/<slug>/mesh_clean.ply, cloud.ply, fpfh.npy, meta.json
i pokazuje wszystkie (albo wybrane) fragmenty w jednym oknie open3d.

Co można pokazać (show):
    mesh  - oczyszczona siatka (domyślnie),
    cloud - chmura punktów używana do dopasowania,
    fpfh  - chmura pokolorowana wg deskryptorów FPFH: podobny kolor = podobny
            lokalny kształt powierzchni. PCA jest liczone wspólnie dla wszystkich
            pokazanych fragmentów, więc kolory da się porównywać MIĘDZY fragmentami
            (miejsca o podobnym kolorze to kandydaci do dopasowania).

Ułożenie (layout):
    side     - fragmenty obok siebie wzdłuż osi X (domyślnie; po wyśrodkowaniu
               w data_parsing wszystkie leżą w (0, 0, 0) i nachodziłyby na siebie),
    original - oryginalne współrzędne skanera (cofa przesunięcie z meta.json).

Kolory (colors), dla mesh i cloud:
    scan     - kolory zapisane przez skaner (domyślnie),
    fragment - każdy fragment w innym kolorze (legenda wypisywana w konsoli).

open3d nie umie wyświetlać tekstu w oknie, dlatego opis fragmentów
(kolejność od lewej, kolory, statystyki) jest wypisywany w konsoli.

Uwaga: PyVista/VTK nie działa na tym komputerze (Windows blokuje DLL VTK),
dlatego wizualizacja używa wyłącznie open3d.

Test wizualizacji (ustawienia w settings.json, sekcja "view"):
    python tests/visualize.py      (każdy fragment osobno, po kolei)

Sterowanie w oknie open3d: lewy przycisk - obrót, kółko - zoom,
Ctrl + lewy / środkowy przycisk - przesuwanie, "+"/"-" - rozmiar punktów,
"R" - reset widoku, "H" - pomoc w konsoli, "Q"/Esc - zamknięcie.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from config import (
    AXES_SIZE_MM,
    CAMERA_TILT,
    CAMERA_ZOOM,
    DATA_OUT,
    VIS_BACKGROUND,
    VIS_BONE_COLOR,
    VIS_GAP_MM,
    VIS_MAX_NORMALS,
    VIS_NORMAL_COLOR,
    VIS_NORMAL_LENGTH_MM,
    VIS_POINT_SIZE,
    VIS_WINDOW_SIZE,
)
from data_parsing import Fragment, load_prepared_fragments

log = logging.getLogger(__name__)

SHOW_MODES = ("mesh", "cloud", "fpfh")
LAYOUTS = ("side", "original")
COLOR_MODES = ("scan", "fragment")

# Paleta "tab10" z nazwami do legendy w konsoli. Przy >10 fragmentach kolory się powtarzają.
PALETTE: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("niebieski", (0.122, 0.467, 0.706)),
    ("pomarańczowy", (1.000, 0.498, 0.055)),
    ("zielony", (0.173, 0.627, 0.173)),
    ("czerwony", (0.839, 0.153, 0.157)),
    ("fioletowy", (0.580, 0.404, 0.741)),
    ("brązowy", (0.549, 0.337, 0.294)),
    ("różowy", (0.890, 0.467, 0.761)),
    ("szary", (0.498, 0.498, 0.498)),
    ("oliwkowy", (0.737, 0.741, 0.133)),
    ("turkusowy", (0.090, 0.745, 0.812)),
)



# -----------------
# WYBÓR FRAGMENTÓW
# -----------------

def select_fragments(fragments: list[Fragment], patterns: list[str] | None) -> list[Fragment]:
    """Wybiera fragmenty, których nazwa zawiera którykolwiek z podanych wzorców.

    Porównanie bez rozróżniania wielkości liter, zarówno z nazwą ASCII (slug),
    jak i z oryginalną nazwą z polskimi znakami. Kolejność fragmentów zostaje zachowana.

    Args:
        fragments: wszystkie wczytane fragmenty.
        patterns:  fragmenty nazw, np. ["lowerjaw", "upperjaw"]. None / pusta lista = wszystkie.

    Returns:
        Lista wybranych fragmentów.

    Raises:
        ValueError: jeśli któryś wzorzec nie pasuje do żadnego fragmentu
                    (literówka nie powinna po cichu niczego nie pokazać).
    """
    if not patterns:
        return list(fragments)

    def matches(frag: Fragment, pattern: str) -> bool:
        p = pattern.lower()
        return p in frag.slug.lower() or p in frag.name.lower()

    unmatched = [p for p in patterns if not any(matches(f, p) for f in fragments)]
    if unmatched:
        available = ", ".join(f.slug for f in fragments) or "brak"
        raise ValueError(f"Żaden fragment nie pasuje do: {', '.join(unmatched)}. Dostępne: {available}")
    return [f for f in fragments if any(matches(f, p) for p in patterns)]


def order_fragments(fragments: list[Fragment], slugs: list[str]) -> list[Fragment]:
    """Wybiera fragmenty o DOKŁADNIE podanych nazwach (slug), w podanej kolejności.

    W przeciwieństwie do select_fragments kolejność wyniku = kolejność `slugs`,
    więc w layout "side" pierwszy podany fragment jest po lewej.

    Raises:
        ValueError: nieznana nazwa albo ta sama nazwa podana dwa razy.
    """
    by_slug = {f.slug: f for f in fragments}
    unknown = [s for s in slugs if s not in by_slug]
    if unknown:
        available = ", ".join(by_slug) or "brak"
        raise ValueError(f"Nieznane fragmenty: {', '.join(unknown)}. Dostępne: {available}")
    dupes = sorted({s for s in slugs if slugs.count(s) > 1})
    if dupes:
        raise ValueError(f"Fragment podany więcej niż raz: {', '.join(dupes)}")
    return [by_slug[s] for s in slugs]


def check_fragment(frag: Fragment, show: str) -> None:
    """Sprawdza, czy fragment ma dane potrzebne do wybranego trybu show.

    open3d przy błędzie odczytu zwraca pustą siatkę zamiast zgłosić wyjątek,
    więc bez tej kontroli uszkodzony plik dałby po prostu puste okno.

    Raises:
        ValueError: brak trójkątów / punktów / normalnych albo FPFH niezgodne z chmurą.
    """
    if show == "mesh":
        if len(frag.mesh.triangles) == 0:
            raise ValueError(f"{frag.slug}: mesh_clean.ply jest pusty albo nie dał się wczytać")
        return
    if frag.cloud is None or len(frag.cloud.points) == 0:
        raise ValueError(f"{frag.slug}: cloud.ply jest pusty albo nie dał się wczytać")
    if show == "fpfh":
        n = len(frag.cloud.points)
        if frag.fpfh is None or frag.fpfh.ndim != 2 or frag.fpfh.shape[0] != n:
            shape = None if frag.fpfh is None else frag.fpfh.shape
            raise ValueError(f"{frag.slug}: fpfh.npy ma kształt {shape}, a chmura {n} punktów - "
                             "uruchom ponownie scripts/prepare_data.py")


# -----------------
# KOLORY
# -----------------

def palette_color(i: int) -> tuple[str, tuple[float, float, float]]:
    """Zwraca (nazwa, RGB) i-tego koloru palety (z zawijaniem)."""
    return PALETTE[i % len(PALETTE)]


def fpfh_colors(fpfhs: list[np.ndarray]) -> list[np.ndarray]:
    """Zamienia deskryptory FPFH kilku fragmentów na kolory RGB.

    Rzutuje 33-wymiarowe deskryptory na 3 główne składowe (PCA) i każdą
    składową skaluje do [0, 1] (percentyle 1-99, wartości poza nimi są obcinane,
    żeby pojedyncze odstające punkty nie "spłaszczyły" kolorów reszty).

    PCA i skalowanie liczone są na WSZYSTKICH fragmentach naraz - ten sam
    deskryptor dostaje ten sam kolor niezależnie od fragmentu. Znaki składowych
    są ustalone (największa co do modułu waga dodatnia), więc wynik jest powtarzalny.

    Args:
        fpfhs: lista tablic (N_i, D) - po jednej na fragment.

    Returns:
        Lista tablic (N_i, 3) z wartościami w [0, 1], w tej samej kolejności.

    Raises:
        ValueError: jeśli tablice nie są 2D albo mają różną liczbę kolumn.
    """
    if not fpfhs:
        return []
    if any(f.ndim != 2 for f in fpfhs) or len({f.shape[1] for f in fpfhs}) != 1:
        raise ValueError("Wszystkie tablice FPFH muszą być 2D i mieć tyle samo kolumn")

    sizes = [len(f) for f in fpfhs]
    x = np.vstack(fpfhs).astype(np.float64)
    if len(x) == 0:
        return [np.zeros((0, 3)) for _ in fpfhs]

    x -= x.mean(axis=0)
    # wektory własne macierzy kowariancji (D x D) - tanie także dla setek tysięcy punktów
    _, vecs = np.linalg.eigh(x.T @ x)
    k = min(3, vecs.shape[1])
    comps = vecs[:, ::-1][:, :k]                       # eigh zwraca rosnąco
    biggest = np.argmax(np.abs(comps), axis=0)
    comps = comps * np.sign(comps[biggest, np.arange(k)])
    proj = x @ comps
    if k < 3:                                          # D < 3 - dopełnij zerami
        proj = np.hstack([proj, np.zeros((len(proj), 3 - k))])

    lo, hi = np.percentile(proj, [1, 99], axis=0)
    span = hi - lo
    span[span < 1e-12] = 1.0                           # stała składowa -> 0, bez dzielenia przez 0
    rgb = np.clip((proj - lo) / span, 0.0, 1.0)
    return np.split(rgb, np.cumsum(sizes)[:-1])


# -----------------
# GEOMETRIE
# -----------------

def fragment_geometry(frag: Fragment, show: str, color: tuple[float, float, float] | None,
                      fpfh_rgb: np.ndarray | None = None) -> o3d.geometry.Geometry3D:
    """Tworzy KOPIĘ geometrii fragmentu do pokazania (oryginał nie jest zmieniany).

    Args:
        frag:     fragment z load_prepared_fragments.
        show:     "mesh", "cloud" albo "fpfh".
        color:    jednolity kolor RGB [0, 1]; None = kolory skanera
                  (a gdy ich nie ma - VIS_BONE_COLOR). Ignorowany dla "fpfh".
        fpfh_rgb: kolory (N, 3) z fpfh_colors - wymagane dla "fpfh".

    Returns:
        TriangleMesh (dla "mesh") albo PointCloud (dla "cloud" / "fpfh").
    """
    if show not in SHOW_MODES:
        raise ValueError(f"Nieznany tryb show={show!r}, dostępne: {SHOW_MODES}")
    check_fragment(frag, show)

    if show == "mesh":
        geom = o3d.geometry.TriangleMesh(frag.mesh)
        if not geom.has_vertex_normals():
            geom.compute_vertex_normals()
        has_colors = geom.has_vertex_colors()
    else:
        geom = o3d.geometry.PointCloud(frag.cloud)
        has_colors = geom.has_colors()

    if show == "fpfh":
        if fpfh_rgb is None or len(fpfh_rgb) != len(geom.points):
            raise ValueError(f"{frag.slug}: liczba kolorów FPFH nie zgadza się z liczbą punktów")
        geom.colors = o3d.utility.Vector3dVector(np.asarray(fpfh_rgb, dtype=np.float64))
    elif color is not None:
        geom.paint_uniform_color(color)
    elif not has_colors:
        geom.paint_uniform_color(VIS_BONE_COLOR)
    return geom


def side_offsets(geometries: list[o3d.geometry.Geometry3D], gap: float) -> list[np.ndarray]:
    """Liczy przesunięcia, które układają geometrie obok siebie wzdłuż osi X.

    Pierwsza geometria zostaje na miejscu, każda kolejna zaczyna się `gap` mm
    za prawą krawędzią (max X) poprzedniej. Y i Z się nie zmieniają.

    Args:
        geometries: geometrie w kolejności od lewej do prawej.
        gap:        odstęp między prostopadłościanami otaczającymi [mm], >= 0.

    Returns:
        Lista wektorów przesunięcia (3,), po jednym na geometrię.
    """
    if gap < 0:
        raise ValueError(f"gap musi być >= 0, jest {gap}")
    offsets, cursor = [], None
    for geom in geometries:
        box = geom.get_axis_aligned_bounding_box()
        lo, hi = box.get_min_bound(), box.get_max_bound()
        dx = 0.0 if cursor is None else cursor - lo[0]
        offsets.append(np.array([dx, 0.0, 0.0]))
        cursor = hi[0] + dx + gap
    return offsets


def original_offset(frag: Fragment) -> np.ndarray:
    """Przesunięcie, które cofa wyśrodkowanie z data_parsing (powrót do współrzędnych skanera).

    data_parsing przesunął fragment o meta["translation_applied"], więc tu
    zwracamy minus tego wektora.
    """
    return -np.asarray(frag.meta["translation_applied"], dtype=np.float64)


def normals_lineset(cloud: o3d.geometry.PointCloud, length: float = VIS_NORMAL_LENGTH_MM,
                    max_count: int = VIS_MAX_NORMALS,
                    color: tuple[float, float, float] = VIS_NORMAL_COLOR) -> o3d.geometry.LineSet:
    """Tworzy odcinki pokazujące normalne chmury (co k-ty punkt, najwyżej max_count).

    Każdy odcinek zaczyna się w punkcie chmury i ma długość `length` mm
    w kierunku normalnej - widać, czy normalne są skierowane na zewnątrz kości.

    Raises:
        ValueError: jeśli chmura nie ma normalnych albo max_count < 1.
    """
    if not cloud.has_normals():
        raise ValueError("Chmura nie ma normalnych")
    if max_count < 1:
        raise ValueError(f"max_count musi być >= 1, jest {max_count}")
    pts, nrm = np.asarray(cloud.points), np.asarray(cloud.normals)
    step = max(1, math.ceil(len(pts) / max_count))
    start = pts[::step]
    end = start + nrm[::step] * length
    n = len(start)
    lines = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(np.vstack([start, end])),
        o3d.utility.Vector2iVector(np.column_stack([np.arange(n), np.arange(n) + n]).astype(np.int32)),
    )
    lines.paint_uniform_color(color)
    return lines


def build_scene(fragments: list[Fragment], show: str = "mesh", layout: str = "side",
                colors: str = "scan", gap: float = VIS_GAP_MM, normals: bool = False,
                axes: bool = False) -> tuple[list[o3d.geometry.Geometry3D], list[str]]:
    """Składa wszystkie geometrie do pokazania i opis fragmentów do konsoli.

    Args:
        fragments: fragmenty do pokazania (kolejność = kolejność od lewej w layout "side").
        show:      "mesh" / "cloud" / "fpfh".
        layout:    "side" (obok siebie wzdłuż X) albo "original" (współrzędne skanera).
        colors:    "scan" (kolory skanera) albo "fragment" (kolor z PALETTE na fragment);
                   ignorowane dla show="fpfh".
        gap:       odstęp między fragmentami w layout "side" [mm].
        normals:   czy dorysować normalne chmury punktów (odcinki).
        axes:      czy dodać układ współrzędnych w (0, 0, 0) o długości osi AXES_SIZE_MM
                   (X - czerwona, Y - zielona, Z - niebieska).

    Returns:
        (geometrie, linie_legendy). Fragmenty wejściowe nie są modyfikowane.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"Nieznany layout={layout!r}, dostępne: {LAYOUTS}")
    if colors not in COLOR_MODES:
        raise ValueError(f"Nieznany tryb colors={colors!r}, dostępne: {COLOR_MODES}")
    if not fragments:
        return [], []

    for frag in fragments:
        check_fragment(frag, show)
        if normals:
            check_fragment(frag, "cloud")

    rgb = fpfh_colors([f.fpfh for f in fragments]) if show == "fpfh" else [None] * len(fragments)
    geoms, names = [], []
    for i, (frag, frgb) in enumerate(zip(fragments, rgb)):
        name, color = palette_color(i)
        use_palette = colors == "fragment" and show != "fpfh"
        geoms.append(fragment_geometry(frag, show, color if use_palette else None, frgb))
        names.append(name if use_palette else None)

    if layout == "side":
        offsets = side_offsets(geoms, gap)
    else:
        offsets = [original_offset(f) for f in fragments]

    scene, legend = [], []
    for i, (frag, geom, off, name) in enumerate(zip(fragments, geoms, offsets, names), start=1):
        geom.translate(off)
        scene.append(geom)
        if normals:
            cloud = o3d.geometry.PointCloud(frag.cloud)
            cloud.translate(off)
            scene.append(normals_lineset(cloud))
        legend.append(describe_fragment(i, frag, name))

    if axes:
        scene.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXES_SIZE_MM))
    return scene, legend


def describe_fragment(index: int, frag: Fragment, color_name: str | None = None) -> str:
    """Jedna linia opisu fragmentu do konsoli, np.
    "[1] lowerjaw  (niebieski)  133542 trójkątów, 2985.3 mm2, 76.4 x 12.2 x 22.2 mm"."""
    m = frag.meta
    clean = m.get("clean", {})
    ext = clean.get("extent_mm")
    parts = [f"[{index}] {m.get('part') or frag.slug}"]
    if color_name:
        parts.append(f"({color_name})")
    stats = []
    if "triangles" in clean:
        stats.append(f"{clean['triangles']} trójkątów")
    if "area_mm2" in clean:
        stats.append(f"{clean['area_mm2']:.1f} mm2")
    if ext:
        stats.append(" x ".join(f"{e:.1f}" for e in ext) + " mm")
    if "cloud_points" in m:
        stats.append(f"{m['cloud_points']} pkt chmury")
    if len(m.get("kept_components_area_mm2", [])) > 1:
        stats.append(f"UWAGA: {len(m['kept_components_area_mm2'])} osobne kawałki")
    return "  ".join(parts) + "  " + ", ".join(stats)


# -----------------
# RENDEROWANIE
# -----------------

def camera_direction(geometries: list[o3d.geometry.Geometry3D],
                     tilt: float = CAMERA_TILT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dobiera kamerę tak, żeby było widać największą "ścianę" sceny.

    Domyślnie open3d patrzy wzdłuż osi Z, a kości (płaskie i długie) widać
    wtedy często z boku jako cienką kreskę. Tu kamera patrzy wzdłuż osi,
    w której scena jest najcieńsza, lekko pochylona (`tilt`) w stronę osi
    średniej, żeby było widać głębię. Najdłuższa oś jest poziomo na ekranie
    i rośnie w PRAWO (bez tego obraz bywa lustrzany i layout "side" szedłby od prawej).

    Returns:
        (lookat, front, up): środek sceny oraz jednostkowe, prostopadłe do
        siebie wektory "od środka do kamery" i "do góry ekranu".
        Prawo ekranu w open3d = cross(up, front).
    """
    box = o3d.geometry.AxisAlignedBoundingBox()
    for geom in geometries:
        box += geom.get_axis_aligned_bounding_box()
    order = np.argsort(box.get_extent(), kind="stable")    # od najmniejszego wymiaru
    axes = np.eye(3)
    thin, middle, longest = axes[order[0]], axes[order[1]], axes[order[2]]
    # cross(up, front) ma znak cross(middle, side * thin) - wybieramy stronę tak,
    # żeby wskazywał +longest
    side = 1.0 if np.dot(np.cross(middle, thin), longest) > 0 else -1.0
    front = side * thin + tilt * middle
    front /= np.linalg.norm(front)
    up = middle - np.dot(middle, front) * front               # prostopadły do front
    up /= np.linalg.norm(up)
    return box.get_center(), front, up


def _setup_visualizer(geometries: list[o3d.geometry.Geometry3D], visible: bool, title: str,
                      size: tuple[int, int], point_size: float) -> o3d.visualization.Visualizer:
    # UWAGA: Visualizer musi być zniszczony (destroy_window + zwolniony obiekt), zanim
    # powstanie następny - jeśli stary obiekt jest kasowany dopiero po utworzeniu nowego
    # okna (np. `vis = ...` w pętli), open3d psuje kontekst OpenGL i kolejne zrzuty są
    # puste. Dlatego okno żyje tylko wewnątrz render_to_file / show_interactive.
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(window_name=title, width=size[0], height=size[1], visible=visible):
        raise RuntimeError("open3d nie mógł utworzyć okna (brak OpenGL?)")
    for geom in geometries:
        vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.background_color = np.asarray(VIS_BACKGROUND, dtype=np.float64)
    opt.point_size = float(point_size)
    opt.mesh_show_back_face = True   # skany nie są zamknięte - bez tego "dziury" od spodu
    opt.light_on = True

    lookat, front, up = camera_direction(geometries)
    view = vis.get_view_control()
    view.set_lookat(lookat)
    view.set_front(front)
    view.set_up(up)
    view.set_zoom(CAMERA_ZOOM)
    return vis


def render_to_file(geometries: list[o3d.geometry.Geometry3D], path: Path,
                   size: tuple[int, int] = VIS_WINDOW_SIZE, point_size: float = VIS_POINT_SIZE) -> Path:
    """Renderuje scenę bez pokazywania okna i zapisuje obraz (PNG, JPG, ...).

    Zapis przez Pillow, a nie open3d - open3d na Windowsie nie radzi sobie
    ze ścieżkami z polskimi znakami.

    Returns:
        Ścieżka do zapisanego pliku.
    """
    if not geometries:
        raise ValueError("Brak geometrii do wyrenderowania")
    path = Path(path)
    vis = _setup_visualizer(geometries, visible=False, title="screenshot", size=size, point_size=point_size)
    try:
        vis.poll_events()
        vis.update_renderer()
        buf = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    finally:
        vis.destroy_window()
    img = (np.clip(buf, 0.0, 1.0) * 255).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return path


def show_interactive(geometries: list[o3d.geometry.Geometry3D], title: str = "Fragmenty kości",
                     size: tuple[int, int] = VIS_WINDOW_SIZE, point_size: float = VIS_POINT_SIZE) -> None:
    """Otwiera okno open3d i czeka, aż użytkownik je zamknie."""
    if not geometries:
        raise ValueError("Brak geometrii do pokazania")
    vis = _setup_visualizer(geometries, visible=True, title=title, size=size, point_size=point_size)
    try:
        vis.run()
    finally:
        vis.destroy_window()

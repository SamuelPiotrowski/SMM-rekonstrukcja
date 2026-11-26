"""
Interaktywne wizualizacje 3D kości - pliki HTML otwierane w przeglądarce (plotly).

Trzy wizualizacje:

1. KOŚCI (bones_figure; scripts/show_bones.py -> out/3d/<zestaw>/bones.html) - wszystkie kości
   ułożone "na stole", każda leży płasko (najdłuższa oś poziomo). Powierzchnie styku
   (jeśli oznaczone) są zaznaczone kolorem.
       - lista "Pokaż" - wszystkie kości albo jedna, z przybliżeniem,
       - przyciski: powierzchnie styku pokaż / ukryj, widok 3D / z góry / z boku,
       - kliknięcie kości w legendzie - ukrycie / pokazanie,
       - najechanie na kość - szczegóły (skany, pole, wymiary, powierzchnie styku).
   Kolory (color_mode): "scan" - kolory skanera, "fragment" - każda kość w innym kolorze,
   "fpfh" - kolor wg deskryptorów FPFH (podobny kolor = podobny lokalny kształt,
   porównywalny między kośćmi).
   Tryb porównania (show_bones.py --compare -> bones_compare.html): ręczne oznaczenia vs
   algorytm - zgodne / fałszywy alarm / przeoczone (COMPARE_LAYERS) + IoU, precyzja, czułość.

2. ŁĄCZENIE SKANÓW (merging_figure + overlap_figure; scripts/show_merging.py
   -> out/3d/<zestaw>/merging.html) - jak skany połączyły się w kości: każdy skan w innym
   kolorze (drobne "paski" dwóch kolorów = skany dobrze się pokrywają), usunięty szum
   na czerwono, macierz nakładania wszystkich par skanów.

3. DOPASOWANIE POWIERZCHNI STYKU (matching_figure + match_matrix_figure;
   scripts/tune_matching.py --html -> out/3d/<zestaw>/matching.html) - najlepsze pary: kość A
   ustawiona na kości B znalezionym przekształceniem, obie powierzchnie styku
   w kolorze; macierz wyników wszystkich par powierzchni.

Zapis: save_figure (jedna figura na całe okno) albo write_page (kilka figur w sekcjach
z nagłówkami). Przy embed_plotly = True plik HTML działa bez internetu i można go
po prostu komuś wysłać. Każdy widok 3D ma linijkę (src/web/ruler.js): przycisk "📏 Linijka"
na dole albo klawisz L, potem dwa kliknięcia w kość = odległość w a.u.

Wizualizacja nie zmienia wyników - czyta tylko out/3d/<zestaw>/ i contacts/.
"""

from __future__ import annotations

import json
import math
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from scipy.spatial import cKDTree

from bones import Bone, read_mesh
from config import read_json
from contacts import Contacts, compare, surface_color, surface_stats, vertex_areas

COLORS = ("#477eae", "#299784", "#ce8b3e", "#8b6ab3", "#c4566a", "#5b8f3a", "#3f6fb5", "#a7743a")
SCAN_COLORS = ("#d9c7a7", "#2b9be0", "#f2b134", "#7b61ff", "#18b0a6", "#e8508f", "#9bc53d", "#ff8a3d")
NOISE_COLOR = "#d64545"
COLOR_MODES = ("scan", "fragment", "fpfh")
FONT = dict(family="Segoe UI, Arial, sans-serif", color="#1f2a36")
LIGHTING = dict(ambient=0.55, diffuse=0.75, specular=0.12, roughness=0.65, fresnel=0.05)
RULER_JS = Path(__file__).resolve().parent / "web" / "ruler.js"     # linijka w widokach 3D (pomiar w a.u.)


@dataclass(frozen=True)
class View3DParams:
    """Wygląd wizualizacji HTML (settings.json, sekcja "visualization_3d", bez "open_browser")."""

    triangles_total: int = 240000      # trójkątów w całym HTML, dzielone między kości (więcej = większy plik)
    embed_plotly: bool = True          # True = działa bez internetu (plik ~4.5 MB większy)
    background: str = "#f4f6f8"        # kolor tła strony
    bone_color: str = "#e3d9c6"        # kolor kości, gdy skan nie ma kolorów
    gap_au: float = 10.0               # odstęp między kośćmi na stole
    source: str = "manual"             # powierzchnie styku w show_bones.py: "manual" / "auto" / "manual-auto"


# -----------------
# GEOMETRIA DO RYSOWANIA
# -----------------

def decimate(mesh: o3d.geometry.TriangleMesh, max_triangles: int) -> o3d.geometry.TriangleMesh:
    """Uproszczona kopia siatki do pokazania (oryginał bez zmian, kolory zostają).

    Upraszczanie quadric decimation zachowuje kształt, ale NIE numerację wierzchołków -
    etykiety powierzchni trzeba przenosić po najbliższym wierzchołku (cKDTree).

    Args:
        mesh:          siatka kości albo skanu.
        max_triangles: docelowa liczba trójkątów; mniejsze siatki są tylko kopiowane.

    Returns:
        Nowa siatka z normalnymi wierzchołków.
    """
    if len(mesh.triangles) <= max_triangles:
        out = o3d.geometry.TriangleMesh(mesh)
    else:
        out = mesh.simplify_quadric_decimation(max_triangles)
    out.compute_vertex_normals()
    return out


def lying_rotation(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Obrót kładący kość płasko: najdłuższa oś = X, najcieńsza = Z (do góry).

    Osie główne są liczone z co najwyżej ~20 000 wierzchołków (szybko także dla dużych siatek).

    Args:
        mesh: siatka kości.

    Returns:
        Macierz 3x3 R (wiersze = osie główne PCA, bez odbicia lustrzanego);
        użycie: (v - środek) @ R.T.
    """
    v = np.asarray(mesh.vertices)
    c = v - v.mean(axis=0)
    _, _, vt = np.linalg.svd(c[:: max(1, len(c) // 20000)], full_matrices=False)
    if np.linalg.det(vt) < 0:
        vt[2] *= -1
    return vt


def _hex(rgb: np.ndarray) -> np.ndarray:
    """Kolory RGB -> teksty "#rrggbb" (format vertexcolor w plotly).

    Args:
        rgb: (N, 3) kolory w [0, 1] (wartości spoza są przycinane).

    Returns:
        (N,) tablica tekstów "#rrggbb".
    """
    rgb = (np.clip(rgb, 0, 1) * 255).round().astype(int)
    return np.array([f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb])


def _f32(a) -> np.ndarray:
    """Tablica float32 - plotly zapisuje ją binarnie, o połowę mniej niż float64.

    Args:
        a: tablica albo lista liczb.

    Returns:
        Ta sama tablica jako float32 (bez kopii, jeśli już jest float32).
    """
    return np.asarray(a, dtype=np.float32)


def _ijk(t: np.ndarray, n_vertices: int) -> dict:
    """Indeksy trójkątów w najmniejszym typie (plotly zapisuje je binarnie - mniejszy plik).

    Args:
        t:          (T, 3) indeksy wierzchołków trójkątów.
        n_vertices: liczba wierzchołków - decyduje o typie (uint16 do 65535, inaczej int32).

    Returns:
        {"i", "j", "k"} - argumenty go.Mesh3d.
    """
    t = np.asarray(t, dtype=np.uint16 if n_vertices < 65536 else np.int32)
    return dict(i=t[:, 0], j=t[:, 1], k=t[:, 2])


def _compact(points: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tylko wierzchołki użyte przez `tri` (z przenumerowaniem).

    Args:
        points: (V, 3) wierzchołki.
        tri:    (T, 3) indeksy trójkątów w `points`.

    Returns:
        (użyte wierzchołki (V', 3), trójkąty (T, 3) z indeksami w nich).
    """
    used, inv = np.unique(tri, return_inverse=True)
    return points[used], inv.reshape(tri.shape)


def fpfh_colors(fpfhs: list[np.ndarray]) -> list[np.ndarray]:
    """Deskryptory FPFH kilku kości -> kolory RGB (3 główne składowe PCA, wspólne dla wszystkich).

    Ten sam deskryptor dostaje ten sam kolor niezależnie od kości, więc miejsca o podobnym
    kolorze na dwóch kościach mają podobny lokalny kształt. Skala: percentyle 1-99.

    Args:
        fpfhs: deskryptory (N_i, 33) każdej kości.

    Returns:
        Kolory (N_i, 3) w [0, 1] dla każdej kości, w tej samej kolejności.
    """
    sizes = [len(f) for f in fpfhs]
    x = np.vstack(fpfhs).astype(np.float64)
    x -= x.mean(axis=0)
    _, vecs = np.linalg.eigh(x.T @ x)
    comps = vecs[:, ::-1][:, :3]
    comps = comps * np.sign(comps[np.argmax(np.abs(comps), axis=0), np.arange(3)])
    proj = x @ comps
    lo, hi = np.percentile(proj, [1, 99], axis=0)
    rgb = np.clip((proj - lo) / np.where(hi - lo < 1e-12, 1, hi - lo), 0, 1)
    return np.split(rgb, np.cumsum(sizes)[:-1])


def shelf_layout(sizes: list[np.ndarray], gap: float, aspect: float = 1.8) -> list[np.ndarray]:
    """Układa prostokąty (szer., wys.) w rzędy od lewej, w kolejności listy - jak książki na półkach.

    Szerokość rzędu dobrana tak, żeby całość miała proporcje ~aspect : 1. Kolejne rzędy
    idą w dół (malejące y).

    Args:
        sizes:  (szerokość, wysokość) każdego prostokąta [a.u.].
        gap:    odstęp między prostokątami [a.u.].
        aspect: proporcje szerokość : wysokość całości.

    Returns:
        Środki (x, y) prostokątów.
    """
    total = sum((w + gap) * (h + gap) for w, h in sizes)
    row_w = max(max(w for w, _ in sizes), math.sqrt(total * aspect))
    centers, x, y, row_h = [], 0.0, 0.0, 0.0
    for w, h in sizes:
        if x > 0 and x + w > row_w:
            x, y, row_h = 0.0, y - row_h - gap, 0.0
        centers.append(np.array([x + w / 2, y - h / 2]))
        x += w + gap
        row_h = max(row_h, h)
    return centers


def _bone_hover(nr: int, bone: Bone, stats: list[dict], contacts: Contacts | None,
                compare_lines: list[str] | None = None) -> str:
    """Podpowiedź po najechaniu na kość (hovertemplate plotly, HTML).

    Zawiera: numer i nazwę, skany, pole, wymiary, usunięty szum, powierzchnie styku
    z ich kolorem, polem i średnicą (albo - w trybie porównania - wynik porównania).

    Args:
        nr:            numer kości na liście.
        bone:          kość (dane z bone.meta).
        stats:         powierzchnie styku z contacts.surface_stats; [] = nie oznaczone.
        contacts:      etykiety kości (nazwy powierzchni) albo None.
        compare_lines: linie porównania ręczne vs algorytm (zastępują listę powierzchni) albo None.

    Returns:
        Tekst HTML dla hovertemplate (z "<extra></extra>" - bez nazwy śladu obok).
    """
    m = bone.meta
    st = m.get("stats", {})
    ext = " × ".join(f"{e:.1f}" for e in sorted(st.get("extent_au", [0, 0, 0]), reverse=True))
    lines = [f"<b>{nr}. {bone.label}</b>",
             f"skany: {len(m.get('scans', []))} ({', '.join(s['label'] for s in m.get('scans', []))})",
             f"pole: {st.get('area_au2', 0):.0f} a.u.²   wymiary: {ext} a.u."]
    if m.get("noise_removed_au2"):
        lines.append(f"usunięty szum: {m['noise_removed_au2']:.0f} a.u.²")
    atl = (contacts.info or {}).get("atlas") if contacts is not None else None
    if atl:                             # wynik algorytmu "atlas": z której kości z bazy pochodzi fragment
        top = atl.get("ranking", [])[:1]
        lines.append("atlas: " + (f"{top[0]['label']} {top[0]['p']:.0%}, " if top else "")
                     + f"żadna z bazy {atl.get('p_unknown', 1.0):.0%}")
    if compare_lines is not None:
        lines += compare_lines
    elif stats:
        lines.append("<b>powierzchnie styku:</b>")
        for s in stats:
            name = contacts.name(s["id"]) if contacts else f"Powierzchnia {s['id']}"
            lines.append(f"<span style='color:{surface_color(s['id'])}'>■</span> {name}: {s['area_au2']:.0f} a.u.², "
                         f"średnica {s['max_diameter_au']:.1f} a.u.")
    else:
        lines.append("<i>powierzchnie styku: nie oznaczone</i>")
    return "<br>".join(lines) + "<extra></extra>"


COMPARE_LAYERS = (   # (klucz, kolor, nazwa) - tryb porównania ręczne vs algorytm
    ("both", "#2e9e5b", "zgodne (ręczne i algorytm)"),
    ("auto_only", "#e8622c", "tylko algorytm (fałszywy alarm)"),
    ("manual_only", "#3b6fd8", "tylko ręczne (przeoczone)"),
    ("unchecked", "#8a8f98", "algorytm (brak ręcznych do porównania)"),
)


def compare_layers(bone: Bone, predicted: Contacts | None, truth: Contacts | None) -> tuple[dict, dict | None]:
    """Podział kości na zgodne / fałszywe alarmy / przeoczenia (ręczne oznaczenia = prawda).

    Args:
        bone:      kość.
        predicted: wynik algorytmu albo None (nic nie znalazł).
        truth:     ręczne oznaczenia albo None (brak - wtedy nie ma z czym porównać).

    Returns:
        ({klucz z COMPARE_LAYERS: maska (V,) wierzchołków bone.mesh} - tylko niepuste,
         wynik contacts.compare albo None, gdy brak ręcznych oznaczeń).
    """
    n = len(bone.mesh.vertices)
    p = predicted.labels > 0 if predicted is not None else np.zeros(n, bool)
    if truth is None or not truth.labels.any():
        return ({"unchecked": p} if p.any() else {}), None
    t = truth.labels > 0
    masks = {"both": p & t, "auto_only": p & ~t, "manual_only": ~p & t}
    pred_labels = predicted.labels if predicted is not None else np.zeros(n, np.uint8)
    return {k: m for k, m in masks.items() if m.any()}, compare(bone.mesh, pred_labels, truth.labels)


# -----------------
# 1. KOŚCI NA STOLE
# -----------------

def bones_figure(bones: list[Bone], contacts: dict[str, Contacts | None] | None = None,
                 color_mode: str = "scan", params: View3DParams = View3DParams(),
                 numbers: list[int] | None = None,
                 reference: dict[str, Contacts | None] | None = None) -> go.Figure:
    """Figura plotly: wszystkie kości na stole + powierzchnie styku (opis: docstring modułu, pkt 1).

    Każda kość jest uproszczona do ~params.triangles_total / liczba kości trójkątów
    (co najmniej 5000), położona płasko (lying_rotation) i ustawiona na stole
    (shelf_layout). Powierzchnie styku to osobne nakładki uniesione 0.04 a.u. nad kością -
    dzięki temu da się je ukryć przyciskiem. Na stole jest podziałka 10 albo 50 a.u.

    Tryb porównania (podane `reference`): `contacts` = wynik algorytmu, `reference` = ręczne
    oznaczenia; zamiast kolorów powierzchni są kolory COMPARE_LAYERS (zgodne / fałszywy alarm /
    przeoczone), a podpowiedź i ramka pokazują IoU, precyzję i czułość.

    Args:
        bones:      kości (dla color_mode="fpfh" wczytane z features=True).
        contacts:   {id kości: Contacts albo None}; brak klucza = kość bez oznaczeń.
        color_mode: "scan" / "fragment" / "fpfh".
        params:     wygląd (triangles_total, bone_color, gap_au, background).
        numbers:    numery kości do wyświetlenia (domyślnie 1..N) - żeby przy wybranych
                    kościach zgadzały się z numerami z pełnej listy.
        reference:  {id kości: ręczne Contacts albo None} - włącza tryb porównania.

    Returns:
        Figura do zapisania przez save_figure.

    Raises:
        ValueError: nieznany color_mode albo pusta lista kości.
    """
    if color_mode not in COLOR_MODES:
        raise ValueError(f"Nieznany tryb kolorów {color_mode!r}, dostępne: {COLOR_MODES}")
    if not bones:
        raise ValueError("Brak kości do pokazania")
    contacts = contacts or {}
    numbers = numbers or list(range(1, len(bones) + 1))
    fp_rgb = fpfh_colors([b.fpfh for b in bones]) if color_mode == "fpfh" else [None] * len(bones)

    # ułożenie: siatka na stole, każda kość płasko (z = 0 na stole)
    shown, sizes = [], []
    per_bone = max(5000, params.triangles_total // len(bones))
    for b in bones:
        d = decimate(b.mesh, per_bone)
        R = lying_rotation(b.mesh)
        center = np.asarray(b.mesh.vertices).mean(axis=0)
        v = (np.asarray(d.vertices) - center) @ R.T
        shown.append((d, R, center, v))
        sizes.append(np.ptp(v, axis=0))
    slots = shelf_layout([s[:2] for s in sizes], params.gap_au)
    traces, groups, focus, scores = [], [], [], []
    for k, (b, (d, R, center, v), fp) in enumerate(zip(bones, shown, fp_rgb)):
        nr = numbers[k]
        offset = np.r_[slots[k], 0.0] - np.r_[(v.min(0) + v.max(0))[:2] / 2, v.min(0)[2]]
        p = v + offset
        t = np.asarray(d.triangles)
        c = contacts.get(b.id)
        stats = surface_stats(b.mesh, c.labels) if c is not None and c.labels.any() else []
        # nakładki: (maska wierzchołków b.mesh, kolor, podpowiedź)
        if reference is None:
            layers = [(c.labels == s["id"], surface_color(s["id"]),
                       f"<b>{c.name(s['id'])}</b> ({nr}. {b.label})<br>pole: {s['area_au2']:.1f} a.u.²"
                       f"<br>największa średnica: {s['max_diameter_au']:.1f} a.u.") for s in stats]
            hover = _bone_hover(nr, b, stats, c)
        else:
            masks, score = compare_layers(b, c, reference.get(b.id))
            va = vertex_areas(b.mesh)
            layers = [(masks[key], color, f"<b>{name}</b> ({nr}. {b.label})<br>pole: {va[masks[key]].sum():.1f} a.u.²")
                      for key, color, name in COMPARE_LAYERS if key in masks]
            if score is None:
                lines = ["<i>brak ręcznych oznaczeń - nie ma z czym porównać</i>"]
            else:
                scores.append(score)
                lines = [f"<b>vs ręczne:</b> IoU {score['iou']:.2f}, precyzja {score['precision']:.2f}, "
                         f"czułość {score['recall']:.2f}"]
            lines += [f"<span style='color:{color}'>■</span> {name}: {va[masks[key]].sum():.0f} a.u.²"
                      for key, color, name in COMPARE_LAYERS if key in masks]
            hover = _bone_hover(nr, b, stats, c, lines)
        color = COLORS[k % len(COLORS)]
        mesh_kw = dict(x=_f32(p[:, 0]), y=_f32(p[:, 1]), z=_f32(p[:, 2]), **_ijk(t, len(p)),
                       name=f"{nr}. {b.label}", legendgroup=b.id, showlegend=True, hovertemplate=hover,
                       lighting=LIGHTING, lightposition=dict(x=-1e4, y=-2e4, z=3e4), flatshading=False)
        if color_mode == "scan" and d.has_vertex_colors():
            mesh_kw["vertexcolor"] = _hex(np.asarray(d.vertex_colors))
        elif color_mode == "fpfh":
            _, near = cKDTree(np.asarray(b.cloud.points)).query(np.asarray(d.vertices))
            mesh_kw["vertexcolor"] = _hex(fp[near])
        else:
            mesh_kw["color"] = color if color_mode == "fragment" else params.bone_color
        idx = [len(traces)]
        traces.append(go.Mesh3d(**mesh_kw))

        # powierzchnie styku: osobne, lekko uniesione nakładki (przycisk pokaż / ukryj)
        if layers:
            _, near = cKDTree(np.asarray(b.mesh.vertices)).query(np.asarray(d.vertices))
            nrm = np.asarray(d.vertex_normals) @ R.T
            for mask, layer_color, layer_hover in layers:
                tri = t[mask[near][t].all(axis=1)]
                if not len(tri):
                    continue
                q, tri = _compact(p + 0.04 * nrm, tri)
                idx.append(len(traces))
                traces.append(go.Mesh3d(
                    x=_f32(q[:, 0]), y=_f32(q[:, 1]), z=_f32(q[:, 2]), **_ijk(tri, len(q)),
                    color=layer_color, opacity=1.0, legendgroup=b.id, showlegend=False,
                    hovertemplate=layer_hover + "<extra></extra>",
                    lighting=LIGHTING, lightposition=dict(x=-1e4, y=-2e4, z=3e4)))
        top = p.max(axis=0)
        mid = (p.min(axis=0) + p.max(axis=0)) / 2
        idx.append(len(traces))
        traces.append(go.Scatter3d(x=[mid[0]], y=[mid[1]], z=[top[2] + 2.0], mode="text", text=[f"<b>{nr}</b>"],
                                   textfont=dict(size=16, color="#1f2a36"), legendgroup=b.id, showlegend=False,
                                   hoverinfo="skip"))
        groups.append(idx)
        focus.append((p.min(axis=0), p.max(axis=0)))

    # stół, podziałka
    allp = np.vstack([np.r_[lo, hi].reshape(2, 3) for lo, hi in focus])
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    margin = 0.06 * max(hi[0] - lo[0], hi[1] - lo[1])
    lo[:2] -= margin
    hi[:2] += margin
    z_lo, z_hi = -0.5, hi[2] + 4.0
    lo[1] -= 6.0                                                    # miejsce na podziałkę
    bar = 10.0 if (hi[0] - lo[0]) < 300 else 50.0
    table = go.Mesh3d(x=[lo[0], hi[0], hi[0], lo[0]], y=[lo[1], lo[1], hi[1], hi[1]], z=[-0.05] * 4,
                      i=[0, 0], j=[1, 2], k=[2, 3], color="#dde3e9", hoverinfo="skip", showlegend=False,
                      lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0))
    scale = go.Scatter3d(x=[lo[0] + margin, lo[0] + margin + bar], y=[lo[1] + 3.0] * 2, z=[0.3, 0.3],
                         mode="lines+text", text=["", f"<b>{bar:g} a.u.</b>"], textposition="top center",
                         line=dict(color="#1f2a36", width=6), hoverinfo="skip", showlegend=False)
    traces = [table, scale] + traces
    groups = [[i + 2 for i in g] for g in groups]
    n_traces = len(traces)

    # kamera w układzie znormalizowanym sceny (jak w wizualizacji 2D)
    span = np.array([hi[0] - lo[0], hi[1] - lo[1], z_hi - z_lo])
    aspect = span / span[0]
    mid_scene = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (z_lo + z_hi) / 2])

    def camera(box_lo, box_hi, direction, up=(0, 0, 1), zoom=0.8):
        """Kamera plotly patrząca na pudełko box_lo..box_hi [a.u.] z kierunku `direction`.

        Args:
            box_lo, box_hi: rogi pudełka (x, y, z) w a.u., w układzie sceny.
            direction:      kierunek od środka pudełka do kamery (nie musi być jednostkowy).
            up:             kierunek "do góry" na ekranie.
            zoom:           odległość kamery względem rozmiaru pudełka (większa = dalej).

        Returns:
            {"center", "eye", "up"} w znormalizowanym układzie sceny plotly (scene.camera).
        """
        center = ((np.asarray(box_lo) + np.asarray(box_hi)) / 2 - mid_scene) / span * aspect
        size = max(((np.asarray(box_hi) - np.asarray(box_lo)) / span * aspect)[:2])
        eye = center + zoom * max(size, 0.08) * np.asarray(direction, dtype=float)
        return dict(center=dict(zip("xyz", center.tolist())), eye=dict(zip("xyz", eye.tolist())),
                    up=dict(zip("xyz", up)))

    whole = (lo, hi)
    views = {"3D": ((0.45, -1.0, 0.85), (0, 0, 1)), "Z góry": ((0, 0, 1.5), (0, 1, 0)),
             "Z boku": ((0, -1.6, 0.25), (0, 0, 1))}
    show_opts = [dict(label="Wszystkie kości", method="update",
                      args=[{"visible": [True] * n_traces}, {"scene.camera": camera(*whole, *views["3D"])}])]
    for k, b in enumerate(bones):
        vis = [i < 2 for i in range(n_traces)]
        for i in groups[k]:
            vis[i] = True
        show_opts.append(dict(label=f"{numbers[k]}. {b.label}", method="update",
                              args=[{"visible": vis}, {"scene.camera": camera(*focus[k], (0.35, -0.9, 0.75), zoom=1.1)}]))
    contact_idx = [i for g in groups for i in g[1:-1]]
    n_marked = sum(1 for b in bones if contacts.get(b.id) is not None and contacts[b.id].labels.any())

    info = [f"<b>Kości: {len(bones)}</b>   (oznaczone powierzchnie styku: {n_marked})",
            f"kolory: {dict(scan='skanera', fragment='osobny dla każdej kości', fpfh='FPFH (podobny kolor = podobny kształt)')[color_mode]}",
            ""]
    if reference is not None:
        info[0] = f"<b>Kości: {len(bones)}</b>   (z ręcznymi oznaczeniami: {len(scores)})"
    info += [
            "<i>lewy przycisk: obrót · kółko: zoom · prawy: przesuwanie</i>",
            "<i>najedź na kość: szczegóły · legenda: ukryj / pokaż</i>",
            "<i>📏 Linijka (na dole, klawisz L): 2 kliknięcia = odległość w a.u.</i>"]
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text="Fragmenty kości" if reference is None else "Powierzchnie styku: ręczne vs algorytm",
                   x=0.5, font=dict(size=20)),
        paper_bgcolor=params.background, font=FONT, margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(x=0.99, y=0.97, xanchor="right", bgcolor="rgba(255,255,255,0.85)", title="Kości",
                    itemclick="toggle", itemdoubleclick="toggleothers"),
        scene=dict(xaxis=dict(range=[lo[0], hi[0]], visible=False), yaxis=dict(range=[lo[1], hi[1]], visible=False),
                   zaxis=dict(range=[z_lo, z_hi], visible=False), aspectmode="manual",
                   aspectratio=dict(zip("xyz", aspect.tolist())), camera=camera(*whole, *views["3D"])),
        annotations=[dict(text="<br>".join(info), x=0.01, y=0.97, xref="paper", yref="paper", showarrow=False,
                          align="left", xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.85)",
                          bordercolor="#c9d2db", borderwidth=1, borderpad=10, font=dict(size=13))],
        updatemenus=[
            dict(type="dropdown", x=0.01, y=0.80, xanchor="left", yanchor="top", buttons=show_opts, active=0,
                 bgcolor="#ffffff", bordercolor="#c9d2db", pad=dict(t=0, b=0)),
            dict(type="buttons", direction="left", x=0.99, y=0.02, xanchor="right", yanchor="bottom",
                 showactive=True, active=0, buttons=[
                     dict(label=name, method="relayout", args=[{"scene.camera": camera(*whole, d, up)}])
                     for name, (d, up) in views.items()])])
    if reference is not None:      # legenda porównania: lewy dół, nad przyciskiem pokaż / ukryj
        legend = []
        if scores:
            mean = {key: sum(s[key] for s in scores) / len(scores) for key in ("iou", "precision", "recall")}
            legend.append(f"<b>Średnio ({len(scores)} kości): IoU {mean['iou']:.2f} · precyzja "
                          f"{mean['precision']:.2f} · czułość {mean['recall']:.2f}</b>")
        else:
            legend.append("<b>Brak ręcznych oznaczeń - nie ma z czym porównać</b>")
        legend += [f"<span style='color:{color}'>■</span> {name}" for _, color, name in COMPARE_LAYERS]
        legend += ["",
                   "<i>IoU = pokrycie wykrytego z ręcznym (wspólne / suma obu)</i>",
                   "<i>precyzja = ile z wykrytego to naprawdę styk (niska: fałszywe alarmy)</i>",
                   "<i>czułość = ile prawdziwego styku wykryto (niska: przeoczenia)</i>",
                   "<i>1.00 = idealnie, 0.00 = zupełnie źle; liczone na polu powierzchni</i>"]
        fig.add_annotation(text="<br>".join(legend), x=0.01, y=0.09, xref="paper", yref="paper", showarrow=False,
                           align="left", xanchor="left", yanchor="bottom", bgcolor="rgba(255,255,255,0.85)",
                           bordercolor="#c9d2db", borderwidth=1, borderpad=10, font=dict(size=13))
    if contact_idx:
        fig.layout.updatemenus += (dict(
            type="buttons", direction="left", x=0.01, y=0.02, xanchor="left", yanchor="bottom", showactive=True,
            active=0, buttons=[
                dict(label="Powierzchnie styku: pokaż", method="restyle", args=[{"visible": True}, contact_idx]),
                dict(label="ukryj", method="restyle", args=[{"visible": False}, contact_idx])]),)
    return fig


# -----------------
# 2. ŁĄCZENIE SKANÓW
# -----------------

def merging_figure(bones: list[Bone], params: View3DParams = View3DParams(),
                   numbers: list[int] | None = None) -> go.Figure:
    """Figura plotly: skany każdej kości nałożone na siebie (każdy w innym kolorze) + usunięty szum.

    Pokazuje tylko kości złożone z >1 skanu albo z usuniętym szumem, najpierw te z największą
    liczbą skanów; lista w lewym dolnym rogu przełącza kości. Siatki skanów są czytane
    z <folder kości>/scans/ (zapisuje je bones.save_bone).

    Args:
        bones:   kości z load_bones (muszą mieć ustawiony `folder`).
        params:  wygląd (triangles_total, background).
        numbers: numery kości do wyświetlenia (domyślnie 1..N).

    Returns:
        Figura; gdy żadna kość nie pasuje - figura z samym komunikatem.
    """
    numbers = numbers or list(range(1, len(bones) + 1))
    chosen = [(n, b) for n, b in zip(numbers, bones) if b.n_scans > 1 or b.meta.get("noise_removed_au2")]
    chosen.sort(key=lambda nb: (-nb[1].n_scans, -nb[1].meta.get("noise_removed_au2", 0)))   # ciekawsze najpierw
    traces, groups, boxes, notes = [], [], [], []
    per_scan = max(5000, params.triangles_total // max(1, sum(b.n_scans for _, b in chosen)))
    for n, b in chosen:
        idx, pts = [], []
        for s_i, s in enumerate(b.meta["scans"]):
            for noise in (False, True):
                path = b.folder / "scans" / (f"{s['id']}_noise.ply" if noise else f"{s['id']}.ply")
                if not path.exists():
                    continue
                d = decimate(read_mesh(path), per_scan)
                v, t = np.asarray(d.vertices), np.asarray(d.triangles)
                pts.append(v)
                color = NOISE_COLOR if noise else SCAN_COLORS[s_i % len(SCAN_COLORS)]
                removed = [c for c in s["components"] if not c["kept"]]
                if noise:
                    name = f"szum: {s['label']}"
                    hover = "<br>".join([f"<b>USUNIĘTY SZUM</b> ze skanu {s['label']}"] +
                                        [f"{c['area_au2']:.0f} a.u.² - {c['reason']}" for c in removed])
                else:
                    role = "skan referencyjny" if s["reference"] else f"nakładanie z ref.: {s['overlap_with_reference']:.2f}"
                    name = f"{s['label']}" + (" (ref.)" if s["reference"] else "")
                    hover = f"<b>{s['label']}</b><br>{role}<br>plik: {s['source']}"
                idx.append(len(traces))
                traces.append(go.Mesh3d(x=_f32(v[:, 0]), y=_f32(v[:, 1]), z=_f32(v[:, 2]),
                                        **_ijk(t, len(v)), color=color, name=name,
                                        showlegend=True, legendgroup=f"{b.id}-{s['id']}-{noise}",
                                        hovertemplate=hover + "<extra></extra>", lighting=LIGHTING,
                                        lightposition=dict(x=-1e4, y=-2e4, z=3e4), visible=not groups))
        groups.append(idx)
        allp = np.vstack(pts)
        boxes.append((allp.min(axis=0), allp.max(axis=0)))
        lines = [f"<b>{n}. {b.label}</b>  ({b.n_scans} skan(y), sklejanie: {b.meta['fuse_method']})"]
        for s_i, s in enumerate(b.meta["scans"]):
            tag = "referencyjny" if s["reference"] else f"nakładanie {s['overlap_with_reference']:.2f}"
            lines.append(f"<span style='color:{SCAN_COLORS[s_i % len(SCAN_COLORS)]}'>■</span> {s['label']} ({tag})")
            for c in s["components"]:
                if not c["kept"]:
                    lines.append(f"   <span style='color:{NOISE_COLOR}'>■</span> usunięto {c['area_au2']:.0f} a.u.²: {c['reason']}")
        notes.append("<br>".join(lines))

    if not traces:
        fig = go.Figure()
        fig.add_annotation(text="Żadna kość nie powstała z kilku skanów i nie usunięto szumu.",
                           showarrow=False, font=dict(size=16))
        fig.update_layout(paper_bgcolor=params.background, font=FONT, height=300,
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

    def layout_for(k):
        """Zmiany układu (osie, proporcje, opis) po wybraniu k-tej kości z listy.

        Args:
            k: indeks kości w `chosen`.

        Returns:
            {ścieżka plotly: wartość} - drugi argument przycisku "update".
        """
        lo, hi = boxes[k]
        span = np.maximum(hi - lo, 1e-3)
        return {"scene.xaxis.range": [lo[0], hi[0]], "scene.yaxis.range": [lo[1], hi[1]],
                "scene.zaxis.range": [lo[2], hi[2]], "scene.aspectratio": dict(zip("xyz", (span / span.max()).tolist())),
                "annotations[0].text": notes[k] + HELP_MERGE}

    buttons = []
    for k, (n, b) in enumerate(chosen):
        vis = [False] * len(traces)
        for i in groups[k]:
            vis[i] = True
        buttons.append(dict(label=f"{n}. {b.label}", method="update", args=[{"visible": vis}, layout_for(k)]))
    first = layout_for(0)
    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor=params.background, font=FONT, margin=dict(l=0, r=0, t=10, b=0), height=720,
        legend=dict(x=0.99, y=0.97, xanchor="right", bgcolor="rgba(255,255,255,0.85)", title="Skany"),
        scene=dict(xaxis=dict(range=first["scene.xaxis.range"], title="x [a.u.]", backgroundcolor="#eef2f5"),
                   yaxis=dict(range=first["scene.yaxis.range"], title="y [a.u.]", backgroundcolor="#eef2f5"),
                   zaxis=dict(range=first["scene.zaxis.range"], title="z [a.u.]", backgroundcolor="#eef2f5"),
                   aspectmode="manual", aspectratio=first["scene.aspectratio"]),
        annotations=[dict(text=first["annotations[0].text"], x=0.01, y=0.97, xref="paper", yref="paper",
                          showarrow=False, align="left", xanchor="left", yanchor="top",
                          bgcolor="rgba(255,255,255,0.88)", bordercolor="#c9d2db", borderwidth=1, borderpad=10,
                          font=dict(size=13))],
        updatemenus=[dict(type="dropdown", direction="up", x=0.01, y=0.02, xanchor="left", yanchor="bottom",
                          buttons=buttons, active=0, bgcolor="#ffffff", bordercolor="#c9d2db")])
    return fig


# Podpowiedź pod opisem kości w merging_figure.
HELP_MERGE = ("<br><br><i>drobne paski dwóch kolorów = skany leżą na sobie (dobrze)<br>"
              "jeden kolor przykrywa drugi w całości = coś się rozjechało</i>")


def overlap_figure(report: dict, params: View3DParams = View3DParams()) -> go.Figure:
    """Macierz nakładania par skanów z merging.json (wysoka wartość = ta sama kość).

    Skany są posortowane po kości, do której trafiły; ramki obejmują skany połączone
    w jedną kość. Kolor zmienia się skokowo na progu min_overlap.

    Args:
        report: zawartość merging.json (load_report).
        params: wygląd (background).

    Returns:
        Figura z mapą cieplną.
    """
    scans = report["scans"]
    ids = [s["id"] for s in scans]
    pos = {s: i for i, s in enumerate(ids)}
    n = len(ids)
    m = np.full((n, n), np.nan)
    for p in report["pairs"]:
        i, j = pos[p["a"]], pos[p["b"]]
        m[i, j] = m[j, i] = p["overlap"]
    bone_of = [s["bone"] for s in scans]
    order = sorted(range(n), key=lambda i: (str(bone_of[i]), ids[i]))
    labels = [f"{scans[i]['label']}" for i in order]
    mo = m[np.ix_(order, order)]
    thr = report["params"]["min_overlap"]
    text = [["" if np.isnan(x) else f"{x:.2f}" for x in row] for row in mo]
    fig = go.Figure(go.Heatmap(
        z=mo, x=labels, y=labels, text=text, texttemplate="%{text}", textfont=dict(size=10),
        colorscale=[[0, "#f4f6f8"], [thr * 0.99, "#c9d6e3"], [thr, "#2b9be0"], [1, "#1d4f91"]], zmin=0, zmax=1,
        colorbar=dict(title="nakładanie", tickvals=[0, thr, 1], ticktext=["0", f"próg {thr}", "1"]),
        hovertemplate="%{y}<br>%{x}<br>nakładanie: %{z:.2f}<extra></extra>", xgap=1, ygap=1))
    # ramki grup (kości)
    start = 0
    for i in range(1, n + 1):
        if i == n or bone_of[order[i]] != bone_of[order[start]]:
            if i - start > 1:
                fig.add_shape(type="rect", x0=start - 0.5, x1=i - 0.5, y0=start - 0.5, y1=i - 0.5,
                              line=dict(color="#1f2a36", width=2))
            start = i
    size = max(420, 34 * n + 220)
    fig.update_layout(paper_bgcolor=params.background, plot_bgcolor=params.background, font=FONT,
                      height=size, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(tickangle=-40, side="bottom"), yaxis=dict(autorange="reversed"))
    return fig


# -----------------
# 3. DOPASOWANIE POWIERZCHNI STYKU
# -----------------

# Kolory w matching_figure: kość A, kość B i ich powierzchnie styku.
MATCH_COLORS = {"a": "#6f9fd8", "b": "#e0c9a0", "surf_a": "#e8508f", "surf_b": "#18b0a6"}


def _surface_overlay(bone: Bone, labels: np.ndarray, sid: int, d: o3d.geometry.TriangleMesh, v: np.ndarray):
    """Nakładka powierzchni styku `sid` na uproszczonej siatce `d`.

    Etykiety z pełnej siatki kości są przenoszone na `d` po najbliższym wierzchołku;
    brany jest trójkąt, którego wszystkie 3 wierzchołki należą do powierzchni.

    Args:
        bone:   kość (pełna siatka - do etykiet).
        labels: (V,) etykiety wierzchołków bone.mesh.
        sid:    numer powierzchni.
        d:      uproszczona siatka (decimate).
        v:      wierzchołki `d` już przekształcone do pokazania (np. przesunięte).

    Returns:
        (wierzchołki, trójkąty) nakładki albo (None, None), gdy po uproszczeniu nic nie zostało.
    """
    _, near = cKDTree(np.asarray(bone.mesh.vertices)).query(np.asarray(d.vertices))
    t = np.asarray(d.triangles)
    tri = t[(labels[near][t] == sid).all(axis=1)]
    return _compact(v, tri) if len(tri) else (None, None)


def matching_figure(results: list[dict], bones: dict[str, Bone], contacts: dict[str, Contacts],
                    params: View3DParams = View3DParams(), max_shown: int = 15) -> go.Figure:
    """Najlepsze pary powierzchni: kość A ustawiona na kości B (lista wybiera parę).

    Kość B zostaje w swoim układzie, kość A jest przenoszona przekształceniem T z wyniku.
    Obok opis pary: wynik, styk, pokrycie obu stron, RMSE i kolizja.

    Args:
        results:   pary z matching_surfaces.match_all (posortowane od najlepszej).
        bones:     {id kości: Bone} - te same, których użyto do dopasowania.
        contacts:  {id kości: Contacts} - etykiety, z których wzięto powierzchnie.
        params:    wygląd (triangles_total, background).
        max_shown: ile najlepszych par dać na listę.

    Returns:
        Figura z listą par w lewym dolnym rogu.
    """
    shown = results[:max_shown]
    per_bone = max(4000, params.triangles_total // max(1, 2 * len(shown)))
    traces, groups, boxes, notes = [], [], [], []
    light = dict(lighting=LIGHTING, lightposition=dict(x=-1e4, y=-2e4, z=3e4))
    for n, r in enumerate(shown):
        idx, pts = [], []
        T = np.asarray(r["T"])
        for side in ("a", "b"):
            bone_id, sid = r[side].rsplit("#", 1)
            bone = bones[bone_id]
            d = decimate(bone.mesh, per_bone)
            v = np.asarray(d.vertices)
            if side == "a":
                v = v @ T[:3, :3].T + T[:3, 3]
            pts.append(v)
            t = np.asarray(d.triangles)
            idx.append(len(traces))
            traces.append(go.Mesh3d(x=_f32(v[:, 0]), y=_f32(v[:, 1]), z=_f32(v[:, 2]), **_ijk(t, len(v)),
                                    color=MATCH_COLORS[side], name=f"{side.upper()}: {bone.label}", showlegend=True,
                                    legendgroup=f"{n}-{side}", hovertemplate=f"<b>{bone.label}</b><extra></extra>",
                                    visible=n == 0, **light))
            nrm = np.asarray(d.vertex_normals) @ (T[:3, :3].T if side == "a" else np.eye(3))
            q, tri = _surface_overlay(bone, contacts[bone_id].labels, int(sid), d, v + 0.04 * nrm)
            if q is not None:
                idx.append(len(traces))
                traces.append(go.Mesh3d(x=_f32(q[:, 0]), y=_f32(q[:, 1]), z=_f32(q[:, 2]), **_ijk(tri, len(q)),
                                        color=MATCH_COLORS[f"surf_{side}"], name=f"powierzchnia {side.upper()}",
                                        showlegend=True, legendgroup=f"{n}-{side}s", visible=n == 0,
                                        hovertemplate=f"<b>{r[side + '_name']}</b><extra></extra>", **light))
        groups.append(idx)
        allp = np.vstack(pts)
        boxes.append((allp.min(axis=0), allp.max(axis=0)))
        notes.append("<br>".join([
            f"<b>{n + 1}. miejsce</b>   wynik: <b>{r['score']:.0f}</b> a.u.²",
            f"<span style='color:{MATCH_COLORS['surf_a']}'>■</span> A: {r['a_name']} ({r['a_area_au2']:.0f} a.u.²"
            f"{', ręczna' if r.get('a_manual', True) else ', z algorytmu'})",
            f"<span style='color:{MATCH_COLORS['surf_b']}'>■</span> B: {r['b_name']} ({r['b_area_au2']:.0f} a.u.²"
            f"{', ręczna' if r.get('b_manual', True) else ', z algorytmu'})",
            f"styk: {r['contact_au2']:.0f} a.u.²  (A pokryte w {r['cover_a']:.0%}, B w {r['cover_b']:.0%})",
            f"RMSE w styku: {r['rmse_au']:.2f} a.u." if r.get("rmse_au") is not None else "RMSE w styku: –",
            f"kolizja kości: {r['collision_au2']:.0f} a.u.²",
            *([f"atlas: ta sama kość {r['atlas_same']:.0%}, premia +{r['atlas_bonus_au2']:.0f} a.u.², "
               f"start: {r.get('start', 'RANSAC')}"] if "atlas_same" in r else []),
            "", "<i>wynik = styk - kolizja" + (" + premia z atlasu" if "atlas_same" in r else "") + "</i>"]))

    def layout_for(k):
        """Zmiany układu (sześcienne osie wokół pary, opis) po wybraniu k-tej pary z listy.

        Args:
            k: indeks pary w `shown`.

        Returns:
            {ścieżka plotly: wartość} - drugi argument przycisku "update".
        """
        lo, hi = boxes[k]
        c, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.05
        return {"scene.xaxis.range": [c[0] - half, c[0] + half], "scene.yaxis.range": [c[1] - half, c[1] + half],
                "scene.zaxis.range": [c[2] - half, c[2] + half], "annotations[0].text": notes[k]}

    buttons = []
    for k, r in enumerate(shown):
        vis = [False] * len(traces)
        for i in groups[k]:
            vis[i] = True
        buttons.append(dict(label=f"{k + 1}. {r['a_name']}  ↔  {r['b_name']}  ({r['score']:.0f})",
                            method="update", args=[{"visible": vis}, layout_for(k)]))
    first = layout_for(0)
    axis = dict(backgroundcolor="#eef2f5", showticklabels=False, title="")
    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor=params.background, font=FONT, margin=dict(l=0, r=0, t=10, b=0), height=760,
        legend=dict(x=0.99, y=0.97, xanchor="right", bgcolor="rgba(255,255,255,0.85)"),
        scene=dict(xaxis=dict(range=first["scene.xaxis.range"], **axis),
                   yaxis=dict(range=first["scene.yaxis.range"], **axis),
                   zaxis=dict(range=first["scene.zaxis.range"], **axis), aspectmode="cube"),
        annotations=[dict(text=first["annotations[0].text"], x=0.01, y=0.97, xref="paper", yref="paper",
                          showarrow=False, align="left", xanchor="left", yanchor="top",
                          bgcolor="rgba(255,255,255,0.88)", bordercolor="#c9d2db", borderwidth=1, borderpad=10,
                          font=dict(size=13))],
        # direction="up": lista na dole wykresu rozwijana w dół chowała się pod następną sekcją strony
        updatemenus=[dict(type="dropdown", direction="up", x=0.01, y=0.02, xanchor="left", yanchor="bottom",
                          buttons=buttons, active=0, bgcolor="#ffffff", bordercolor="#c9d2db")])
    return fig


def match_matrix_figure(report: dict, params: View3DParams = View3DParams()) -> go.Figure:
    """Macierz wyników wszystkich par powierzchni (puste = ta sama kość, × = pominięta przez atlas).

    Args:
        report: słownik z matching_surfaces.report (to samo, co w matching.json).
        params: wygląd (background).

    Returns:
        Figura z mapą cieplną; powierzchnie z algorytmu mają dopisek "(algorytm)".
    """
    surfs = report["surfaces"]
    keys = [s["key"] for s in surfs]
    pos = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    m = np.full((n, n), np.nan)
    for p in report["pairs"]:
        i, j = pos[p["a"]], pos[p["b"]]
        m[i, j] = m[j, i] = p["score"]
    labels = [f"{s['bone_label']} · {s['surface']}{'' if s['source'] == 'manual' else ' (algorytm)'}" for s in surfs]
    text = [["" if np.isnan(x) else f"{x:.0f}" for x in row] for row in m]
    hover = [["–" if np.isnan(x) else f"wynik: {x:.0f} a.u.²" for x in row] for row in m]
    for p in report.get("skipped", []):
        if p["a"] in pos and p["b"] in pos:
            i, j = pos[p["a"]], pos[p["b"]]
            text[i][j] = text[j][i] = "×"
            hover[i][j] = hover[j][i] = f"pominięta: atlas - różne kości ({p['atlas_diff']:.0%})"
    top = np.nanmax(m) if np.isfinite(m).any() else 1.0
    fig = go.Figure(go.Heatmap(
        z=m, x=labels, y=labels, text=text, texttemplate="%{text}", textfont=dict(size=9),
        colorscale=[[0, "#f4f6f8"], [0.5, "#8fb8de"], [1, "#1d4f91"]], zmin=0, zmax=max(top, 1.0),
        colorbar=dict(title="wynik [a.u.²]"), xgap=1, ygap=1, customdata=hover,
        hovertemplate="%{y}<br>%{x}<br>%{customdata}<extra></extra>"))
    size = max(480, 22 * n + 260)
    fig.update_layout(paper_bgcolor=params.background, plot_bgcolor=params.background, font=FONT,
                      height=size, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(tickangle=-40, side="bottom", tickfont=dict(size=10)),
                      yaxis=dict(autorange="reversed", tickfont=dict(size=10)))
    return fig


# -----------------
# ZAPIS HTML
# -----------------

# Szablon strony write_page; {{ }} = zwykłe nawiasy CSS (str.format).
PAGE = """<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{ margin: 0; background: {bg}; font-family: "Segoe UI", Arial, sans-serif; color: #1f2a36; }}
header {{ padding: 18px 24px 4px; }} h1 {{ margin: 0; font-size: 22px; }}
p.lead {{ margin: 6px 0 0; color: #6b7b8c; max-width: 980px; line-height: 1.5; }}
section {{ background: #fff; margin: 14px 24px; border-radius: 12px; border: 1px solid #dde3e9; overflow: hidden; }}
section h2 {{ font-size: 15px; margin: 0; padding: 12px 16px; border-bottom: 1px solid #dde3e9; }}
section .note {{ font-size: 13px; color: #6b7b8c; padding: 8px 16px 0; margin: 0; }}
</style>{head}</head><body>
<header><h1>{title}</h1><p class="lead">{lead}</p></header>
{sections}
</body></html>"""


def _ruler_head() -> str:
    """Znacznik <script> z linijką (src/web/ruler.js) do <head> strony - definiuje attachRuler."""
    return f"<script>{RULER_JS.read_text(encoding='utf-8')}</script>"


def _ruler_post(fig: go.Figure) -> str | None:
    """post_script dla fig.to_html: podpina linijkę, gdy figura jest 3D (None = figura 2D, bez linijki)."""
    if any(t.type in ("mesh3d", "scatter3d") for t in fig.data):
        return "attachRuler(document.getElementById('{plot_id}'));"
    return None


def _plotly_head(embed: bool) -> str:
    """Znacznik <script> z plotly.js do <head> strony write_page.

    Args:
        embed: True = cały plotly.js wklejony w stronę (~4.5 MB, działa offline),
               False = link do CDN w wersji zainstalowanego pakietu plotly.

    Returns:
        Tekst HTML znacznika <script>.
    """
    if embed:
        from plotly.offline import get_plotlyjs
        return f"<script>{get_plotlyjs()}</script>"
    import plotly
    return f'<script src="https://cdn.plot.ly/plotly-{plotly.__version__.split("+")[0]}.min.js"></script>'


def write_page(path: Path, title: str, lead: str, sections: list[tuple[str, str, go.Figure]],
               params: View3DParams = View3DParams(), open_browser: bool = False) -> Path:
    """Zapisuje stronę HTML z kilkoma figurami plotly (każda w swojej sekcji z nagłówkiem).

    Args:
        path:         plik wynikowy (foldery są tworzone).
        title:        tytuł strony i nagłówek na górze.
        lead:         opis pod tytułem (może zawierać HTML, np. <b>).
        sections:     [(nagłówek sekcji, opis albo "", figura), ...].
        params:       wygląd (background, embed_plotly).
        open_browser: otwórz stronę w przeglądarce po zapisie.

    Returns:
        Ścieżka zapisanego pliku.
    """
    config = dict(displaylogo=False, scrollZoom=True, toImageButtonOptions=dict(format="png", scale=2))
    body = []
    for heading, note, fig in sections:
        div = fig.to_html(full_html=False, include_plotlyjs=False, config=config, post_script=_ruler_post(fig))
        body.append(f"<section><h2>{heading}</h2>" + (f"<p class='note'>{note}</p>" if note else "") + f"{div}</section>")
    html = PAGE.format(title=title, lead=lead, bg=params.background, head=_plotly_head(params.embed_plotly) + _ruler_head(),
                       sections="\n".join(body))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    if open_browser:
        webbrowser.open(path.resolve().as_uri())
    return path


def save_figure(fig: go.Figure, path: Path, params: View3DParams = View3DParams(), open_browser: bool = False,
                title: str = "Fragmenty kości") -> Path:
    """Zapisuje jedną figurę jako samodzielny plik HTML na całe okno przeglądarki.

    Args:
        fig:          figura plotly.
        path:         plik wynikowy (foldery są tworzone).
        params:       wygląd (background, embed_plotly).
        open_browser: otwórz plik w przeglądarce po zapisie.
        title:        tytuł karty przeglądarki.

    Returns:
        Ścieżka zapisanego pliku.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    div = fig.to_html(full_html=False, include_plotlyjs=True if params.embed_plotly else "cdn",
                      default_height="100vh", default_width="100%",
                      config=dict(displaylogo=False, scrollZoom=True, responsive=True,
                                  toImageButtonOptions=dict(format="png", filename=path.stem, scale=2)),
                      post_script=_ruler_post(fig))
    path.write_text(f"<!doctype html><html lang='pl'><head><meta charset='utf-8'><title>{title}</title>"
                    f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
                    f"<style>html,body{{margin:0;height:100%;overflow:hidden;background:{params.background}}}</style>"
                    f"{_ruler_head()}</head><body>{div}</body></html>", encoding="utf-8")
    if open_browser:
        webbrowser.open(path.resolve().as_uri())
    return path


def load_report(out_dir: Path) -> dict | None:
    """Wczytuje raport łączenia skanów out_dir/merging.json (pipeline.write_merging_report).

    Args:
        out_dir: folder wyników (DATA_OUT).

    Returns:
        Zawartość merging.json ({"params", "scans", "pairs", ...}) albo None, gdy pliku nie ma.
    """
    p = Path(out_dir) / "merging.json"
    return read_json(p) if p.exists() else None

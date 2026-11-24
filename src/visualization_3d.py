"""
Interaktywna wizualizacja 3D złożenia - jeden plik HTML otwierany w przeglądarce.

Każdy fragment (kontur 2D z analizy) jest wyciągany w bryłę o zadanej grubości.
Animacja pokazuje drogę od fragmentów rozłożonych "na stole" (tak jak na zdjęciach)
do złożonej kości. Powierzchnie przełomów są kolorowane:
    - ten sam kolor na dwóch fragmentach = algorytm uznał, że te przełomy pasują,
    - czerwony = przełom bez przyjętej pary.

Sterowanie w przeglądarce:
    suwak / przyciski "Złóż" i "Rozłóż" - animacja,
    lewy przycisk myszy - obrót, kółko - zoom, prawy - przesuwanie,
    kliknięcie fragmentu w legendzie - ukrycie / pokazanie,
    najechanie na fragment lub przełom - szczegóły (pole, dopasowania, błędy),
    przyciski widoku - 3D / z góry / z boku.

Wizualizacja nie zmienia wyników - korzysta wyłącznie z case.json (lub słownika case).
"""
from __future__ import annotations

import webbrowser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Delaunay, cKDTree

from geometry import angle_difference, arc_length, area, interior_grid, polygon_path, resample_closed, rotation
from results import COLORS, final_poses


@dataclass
class View3DParams:
    thickness_mm: float | None = None   # grubość brył; None = automatycznie z rozmiaru fragmentów
    frames: int = 36                    # liczba klatek animacji (więcej = płynniej, większy plik)
    mesh_points: int = 260              # punkty konturu na bryłę
    lift_factor: float = 3.0            # jak wysoko (x grubość) fragmenty "przelatują" w animacji
    colors: tuple = tuple(COLORS)       # kolory fragmentów
    match_colors: tuple = ("#f2b134", "#18b0a6", "#e8508f", "#7b61ff", "#2b9be0", "#9bc53d", "#ff8a3d", "#00c2d1")
    unmatched_color: str = "#d64545"
    background: str = "#f4f6f8"
    embed_plotly: bool = True           # True = działa bez internetu (plik ~4 MB większy)


# -----------------
# BRYŁA FRAGMENTU
# -----------------

def _darker(hex_color: str, factor: float) -> str:
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float) * factor
    return "#" + "".join(f"{int(np.clip(v, 0, 255)):02x}" for v in rgb)


def triangulate_polygon(contour, interior_step: float):
    """Triangulacja wielokąta (także wklęsłego): punkty konturu + siatka wewnątrz.

    Returns:
        (punkty (P, 2) - najpierw kontur, potem wnętrze; trójkąty (T, 3) CCW)
    """
    c = np.asarray(contour)
    grid, _ = interior_grid(c, interior_step)
    if len(grid):
        d, _ = cKDTree(c).query(grid)
        grid = grid[d > 0.7 * interior_step]
    pts = np.vstack([c, grid])
    tri = Delaunay(pts).simplices
    cent = pts[tri].mean(axis=1)
    path = polygon_path(c)
    ok = path.contains_points(cent)
    # Trójkąty "przeskakujące" wcięcia konturu mają środek krawędzi poza wielokątem.
    for a, b in ((0, 1), (1, 2), (2, 0)):
        mid = 0.5 * (pts[tri[:, a]] + pts[tri[:, b]])
        ok &= path.contains_points(0.85 * mid + 0.15 * cent)
    tri = tri[ok]
    v0, v1, v2 = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    cross = (v1 - v0)[:, 0] * (v2 - v0)[:, 1] - (v1 - v0)[:, 1] * (v2 - v0)[:, 0]
    tri[cross < 0] = tri[cross < 0][:, [0, 2, 1]]
    return pts, tri


def fragment_solid(contour, thickness: float, arcs: list, n_points: int = 260):
    """Bryła fragmentu: górna i dolna podstawa + ściana boczna.

    Returns:
        słownik: "xy" (V, 2) i "z" (V,) wierzchołków w układzie lokalnym,
        "faces" (F, 3), "kind" (F,) - -2 podstawa, -1 ściana zewnętrzna,
        k >= 0 ściana przełomu arcs[k]; "ring" - kontur (n, 2).
    """
    c = resample_closed(contour, n_points)
    if area(c) < 0:
        c = c[::-1]
    n = len(c)
    spacing = arc_length(np.vstack([c, c[0]]))[-1] / n
    extent = float(np.ptp(c, axis=0).max())
    pts, tri = triangulate_polygon(c, max(4 * spacing, extent / 14))
    p = len(pts)
    half = thickness / 2
    # Osobne wierzchołki podstaw i ściany -> ostra krawędź przy oświetleniu.
    xy = np.vstack([pts, pts, c, c])
    z = np.r_[np.full(p, half), np.full(p, -half), np.full(n, half), np.full(n, -half)]
    top, bottom = tri, tri[:, [0, 2, 1]] + p
    i = np.arange(n)
    j = (i + 1) % n
    ring_top, ring_bottom = 2 * p + i, 2 * p + n + i
    wall = np.vstack([np.c_[ring_bottom, 2 * p + n + j, 2 * p + j],
                      np.c_[ring_bottom, 2 * p + j, ring_top]])
    # Który odcinek konturu należy do którego przełomu.
    mids = 0.5 * (c + c[j])
    seg_kind = np.full(n, -1)
    best = np.full(n, np.inf)
    tol = max(0.8, 1.5 * spacing)
    for k, arc in enumerate(arcs):
        d, _ = cKDTree(np.asarray(arc["points"])).query(mids)
        hit = (d < tol) & (d < best)
        seg_kind[hit], best[hit] = k, d[hit]
    faces = np.vstack([top, bottom, wall])
    kind = np.r_[np.full(len(top) + len(bottom), -2), seg_kind, seg_kind]
    return {"xy": xy, "z": z, "faces": faces, "kind": kind, "ring": c}


# -----------------
# POZY W ANIMACJI
# -----------------

def _ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def _start_layout(ids: list, sizes: dict, final_points) -> dict:
    """Środki fragmentów "na stole": siatka OBOK miejsca złożenia (po lewej), żeby
    składane fragmenty nie lądowały na tych, które jeszcze leżą."""
    cols = int(np.ceil(np.sqrt(len(ids))))
    rows = int(np.ceil(len(ids) / cols))
    cell = 1.25 * max(sizes.values())
    final_points = np.asarray(final_points)
    lo, hi = final_points.min(axis=0), final_points.max(axis=0)
    center = np.array([lo[0] - 0.2 * cell - cols * cell / 2, (lo[1] + hi[1]) / 2])
    return {fid: center + cell * np.array([k % cols - (cols - 1) / 2, (rows - 1) / 2 - k // cols])
            for k, fid in enumerate(ids)}


def _xyz(x, y, z) -> dict:
    """Współrzędne jako float32 - plik HTML jest o połowę mniejszy, dokładność ~0.01 mm wystarcza."""
    return {k: np.asarray(v, dtype=np.float32) for k, v in (("x", x), ("y", y), ("z", z))}


def assembly_order(case: dict) -> list[str]:
    """Kolejność składania: dla każdej grupy kotwica, potem fragmenty wg drzewa połączeń (BFS)."""
    accepted = [c for c in case["candidates"] if c.get("accepted")]
    order = []
    for comp in case["assembly"]["components"]:
        members = {p["id"] for p in comp["poses"]}
        queue = [comp["anchor"]]
        while queue:
            fid = queue.pop(0)
            if fid in order:
                continue
            order.append(fid)
            for c in accepted:
                if fid in (c["a"], c["b"]):
                    other = c["b"] if c["a"] == fid else c["a"]
                    if other in members and other not in order:
                        queue.append(other)
        order += [p["id"] for p in comp["poses"] if p["id"] not in order]
    return order


def _place(xy_local, c_local, theta, c_world):
    return (xy_local - c_local) @ rotation(theta).T + c_world


# -----------------
# FIGURA
# -----------------

def _match_info(case: dict, params: View3DParams):
    """Dla każdego łuku: kolor, partner i opis przyjętego dopasowania."""
    info = {}
    accepted = [c for c in case["candidates"] if c.get("accepted")]
    for k, c in enumerate(accepted):
        col = params.match_colors[k % len(params.match_colors)]
        m = c["metrics"]
        verdict = {True: " ✓ zgodne z prawdą", False: " ✗ niezgodne z prawdą"}.get(c.get("correct"), "")
        text = (f"<b>{c['arcA']} ↔ {c['arcB']}</b>{verdict}<br>wynik: {m['score']:.1f}"
                f"<br>długość styku: {m['length']:.1f} mm<br>RMSE (surowy kontur): {m.get('rawRmse') or 0:.2f} mm"
                f"<br>kolizja: {m['collisionArea']:.1f} mm²")
        info[c["arcA"]] = {"color": col, "partner": c["arcB"], "text": text, "score": m["score"]}
        info[c["arcB"]] = {"color": col, "partner": c["arcA"], "text": text, "score": m["score"]}
    return info, accepted


def build_figure(case: dict, params: View3DParams = View3DParams()) -> go.Figure:
    """Figura plotly z animacją składania fragmentów."""
    fragments = case["fragments"]
    match, accepted = _match_info(case, params)
    sizes = {f["id"]: float(np.ptp(np.asarray(f["contour"]), axis=0).max()) for f in fragments}
    thickness = params.thickness_mm or float(np.clip(
        0.25 * np.median([np.ptp(np.asarray(f["contour"]), axis=0).min() for f in fragments]), 2.0, 40.0))
    lift = params.lift_factor * thickness
    eps = 0.04 * thickness

    poses_end = final_poses(case)
    solids, c_local, c_end = {}, {}, {}
    for f in fragments:
        solids[f["id"]] = fragment_solid(f["contour"], thickness, f["arcs"], params.mesh_points)
        c_local[f["id"]] = np.asarray(f["contour"]).mean(axis=0)
        th, t = poses_end[f["id"]]
        c_end[f["id"]] = rotation(th) @ c_local[f["id"]] + t
    final_xy = np.vstack([_place(solids[f["id"]]["ring"], c_local[f["id"]], poses_end[f["id"]][0], c_end[f["id"]])
                          for f in fragments])
    c_start = _start_layout([f["id"] for f in fragments], sizes, final_xy)
    theta_end = {fid: poses_end[fid][0] for fid in c_end}

    # Składanie po kolei, w kolejności drzewa połączeń (kotwica, potem dołączane fragmenty).
    # Każdy fragment leci w swoim oknie czasu, okna zachodzą na siebie w połowie.
    order = assembly_order(case)
    rank = {fid: r for r, fid in enumerate(order)}
    window = 2.0 / (len(order) + 1)

    def state(fid, t):
        """(theta, środek, uniesienie) fragmentu w chwili t (0 = stół, 1 = złożone)."""
        e = float(_ease(np.clip((t - rank[fid] * window / 2) / window, 0.0, 1.0)))
        theta = angle_difference(theta_end[fid], 0.0) * e
        height = lift * (0.7 + 0.6 * rank[fid] / max(1, len(order) - 1))
        return theta, (1 - e) * c_start[fid] + e * c_end[fid], height * np.sin(np.pi * e)

    # --- ślady (traces): kolejność jest ważna, klatki animacji odwołują się do indeksów
    traces, dynamic = [], []   # dynamic: (indeks śladu, funkcja(e) -> dict x/y/z)
    ref = case["assembly"].get("referenceContour")
    ground_z = -thickness / 2 - 0.02 * thickness
    if ref:
        r = np.asarray(ref)
        r = np.vstack([r, r[0]])
        traces.append(go.Scatter3d(x=r[:, 0], y=r[:, 1], z=np.full(len(r), ground_z), mode="lines",
                                   line=dict(color="#8a99a8", width=3, dash="dash"),
                                   name="Kontur referencyjny", hoverinfo="skip"))

    for idx, f in enumerate(fragments):
        fid, s = f["id"], solids[f["id"]]
        base = params.colors[idx % len(params.colors)]
        face_colors = np.where(s["kind"] == -2, base, _darker(base, 0.93)).astype(object)
        for k, arc in enumerate(f["arcs"]):
            face_colors[s["kind"] == k] = match.get(arc["id"], {}).get("color", params.unmatched_color)
        lines = []
        for arc in f["arcs"]:
            mi = match.get(arc["id"])
            lines.append(f"{arc['id']} → {mi['partner']} (wynik {mi['score']:.1f})" if mi
                         else f"{arc['id']} → brak pary")
        hover = (f"<b>Fragment {fid}</b><br>pole: {f['area']:.0f} mm²<br>"
                 + "<br>".join(lines) + "<extra></extra>")

        def mesh_xyz(e, fid=fid, s=s):
            th, cw, z0 = state(fid, e)
            xy = _place(s["xy"], c_local[fid], th, cw)
            return _xyz(xy[:, 0], xy[:, 1], s["z"] + z0)

        xyz = mesh_xyz(1.0)
        dynamic.append((len(traces), mesh_xyz))
        traces.append(go.Mesh3d(**xyz, i=s["faces"][:, 0], j=s["faces"][:, 1], k=s["faces"][:, 2],
                                facecolor=face_colors, flatshading=False, name=f"Fragment {fid}",
                                legendgroup=fid, showlegend=True, hovertemplate=hover,
                                lighting=dict(ambient=0.68, diffuse=0.62, specular=0.15, roughness=0.6, fresnel=0.05),
                                lightposition=dict(x=-1e4, y=-2e4, z=3e4)))

        ring = np.vstack([s["ring"], s["ring"][0]])

        def outline_xyz(e, fid=fid, ring=ring):
            th, cw, z0 = state(fid, e)
            xy = _place(ring, c_local[fid], th, cw)
            return _xyz(xy[:, 0], xy[:, 1], np.full(len(xy), thickness / 2 + eps + z0))

        dynamic.append((len(traces), outline_xyz))
        traces.append(go.Scatter3d(**outline_xyz(1.0), mode="lines", line=dict(color=_darker(base, 0.6), width=2),
                                   legendgroup=fid, showlegend=False, hoverinfo="skip"))

        for arc in f["arcs"]:
            mi = match.get(arc["id"])
            pts = np.asarray(arc["points"])[::3]

            def arc_xyz(e, fid=fid, pts=pts):
                th, cw, z0 = state(fid, e)
                xy = _place(pts, c_local[fid], th, cw)
                return _xyz(xy[:, 0], xy[:, 1], np.full(len(xy), thickness / 2 + 2 * eps + z0))

            dynamic.append((len(traces), arc_xyz))
            traces.append(go.Scatter3d(
                **arc_xyz(1.0), mode="lines", legendgroup=fid, showlegend=False,
                line=dict(color=mi["color"] if mi else params.unmatched_color, width=7),
                hovertemplate=(mi["text"] if mi else f"<b>{arc['id']}</b><br>brak przyjętej pary")
                + f"<br>długość przełomu: {arc['length']:.1f} mm<extra>{arc['id']}</extra>"))

        def label_xyz(e, fid=fid):
            th, cw, z0 = state(fid, e)
            return _xyz([cw[0]], [cw[1]], [thickness / 2 + 3 * eps + z0])

        dynamic.append((len(traces), label_xyz))
        traces.append(go.Scatter3d(**label_xyz(1.0), mode="text", text=[f"<b>{fid}</b>"],
                                   textfont=dict(size=18, color="#1f2a36"), legendgroup=fid,
                                   showlegend=False, hoverinfo="skip"))

    # --- zakresy osi stałe dla wszystkich klatek (inaczej scena "skacze")
    steps = np.linspace(0, 1, params.frames + 1)
    all_xy = []
    for f in fragments:
        for e in steps:
            th, cw, _ = state(f["id"], e)
            all_xy.append(_place(solids[f["id"]]["ring"][::8], c_local[f["id"]], th, cw))
    if ref:
        all_xy.append(np.asarray(ref))
    all_xy = np.vstack(all_xy)
    margin = 0.06 * np.ptp(all_xy, axis=0).max()
    lo, hi = all_xy.min(axis=0) - margin, all_xy.max(axis=0) + margin
    z_lo, z_hi = ground_z - 0.5 * thickness, thickness / 2 + lift + thickness
    span = np.array([hi[0] - lo[0], hi[1] - lo[1], z_hi - z_lo])
    traces.insert(0, go.Mesh3d(x=[lo[0], hi[0], hi[0], lo[0]], y=[lo[1], lo[1], hi[1], hi[1]],
                               z=[ground_z - 0.01] * 4, i=[0, 0], j=[1, 2], k=[2, 3], color="#dde3e9",
                               opacity=1.0, hoverinfo="skip", showlegend=False, name="stół",
                               lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0)))
    dynamic = [(i + 1, fn) for i, fn in dynamic]

    # --- klatki animacji
    frames = []
    for k, e in enumerate(steps):
        frames.append(go.Frame(name=str(k), traces=[i for i, _ in dynamic],
                               data=[go.Scatter3d(**fn(e)) if not isinstance(traces[i], go.Mesh3d)
                                     else go.Mesh3d(**fn(e)) for i, fn in dynamic]))
    names = [fr.name for fr in frames]
    play = dict(frame=dict(duration=45, redraw=True), transition=dict(duration=0), mode="immediate", fromcurrent=False)

    # --- opis po lewej
    metrics = case["assembly"].get("metrics", {})
    info = [f"<b>{case.get('title', case['id'])}</b>"]
    if metrics.get("iou") is not None:
        info.append(f"IoU z referencją: <b>{metrics['iou']:.3f}</b>")
    if metrics.get("correctEdges") is not None:
        info.append(f"poprawne połączenia: <b>{metrics['correctEdges']}/{metrics['selectedEdges']}</b>")
    info.append(f"grupy po złożeniu: <b>{len(case['assembly']['components'])}</b>")
    info.append("<br><b>Przyjęte połączenia</b> (ten sam kolor = pasujące przełomy):")
    for c in accepted:
        col = match[c["arcA"]]["color"]
        mark = {True: " ✓", False: " ✗"}.get(c.get("correct"), "")
        info.append(f"<span style='color:{col}'>■</span> {c['arcA']} ↔ {c['arcB']}  "
                    f"wynik {c['metrics']['score']:.1f}{mark}")
    if not accepted:
        info.append("brak")
    info.append(f"<span style='color:{params.unmatched_color}'>■</span> przełom bez pary")
    info.append("<br><i>mysz: obrót / kółko: zoom / prawy: przesuwanie<br>najedź na fragment, by zobaczyć szczegóły</i>")

    # Kamera: współrzędne znormalizowane sceny (środek sceny = 0, szerokość osi x = 1).
    aspect = np.array([1.0, span[1] / span[0], span[2] / span[0]])
    mid = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (z_lo + z_hi) / 2])
    f_lo, f_hi = final_xy.min(axis=0), final_xy.max(axis=0)
    focus = (np.r_[(f_lo + f_hi) / 2, 0.0] - mid) / span * aspect
    dist = 1.15 * max((f_hi - f_lo) / span[:2] * aspect[:2])

    def camera(direction, up=(0, 0, 1), center=focus, d=dist):
        eye = center + d * np.asarray(direction, dtype=float)
        return dict(center=dict(zip("xyz", center.tolist())), eye=dict(zip("xyz", eye.tolist())),
                    up=dict(zip("xyz", up)))

    cameras = {
        "Złożona kość": camera((0.55, -0.95, 0.75)),
        "Z góry": camera((0, 0, 1.45), up=(0, 1, 0)),
        "Z boku": camera((0, -1.45, 0.25)),
        "Cały stół": camera((0.75, -1.25, 0.95), center=np.zeros(3), d=1.0),
    }
    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        title=dict(text=f"Rekonstrukcja 3D — {case.get('title', case['id'])}", x=0.5, font=dict(size=20)),
        paper_bgcolor=params.background, font=dict(family="Segoe UI, Arial, sans-serif", color="#1f2a36"),
        margin=dict(l=0, r=0, t=60, b=0), height=820,
        legend=dict(x=0.99, y=0.97, xanchor="right", bgcolor="rgba(255,255,255,0.8)", title="Fragmenty"),
        scene=dict(
            xaxis=dict(range=[lo[0], hi[0]], title="x [mm]", showbackground=False, gridcolor="#d5dbe1",
                       zeroline=False, tickfont=dict(size=10, color="#8a99a8")),
            yaxis=dict(range=[lo[1], hi[1]], title="y [mm]", showbackground=False, gridcolor="#d5dbe1",
                       zeroline=False, tickfont=dict(size=10, color="#8a99a8")),
            zaxis=dict(range=[z_lo, z_hi], visible=False),
            aspectmode="manual", aspectratio=dict(x=1, y=span[1] / span[0], z=span[2] / span[0]),
            camera=cameras["Złożona kość"]),
        annotations=[dict(text="<br>".join(info), x=0.01, y=0.97, xref="paper", yref="paper", showarrow=False,
                          align="left", xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.85)",
                          bordercolor="#c9d2db", borderwidth=1, borderpad=10, font=dict(size=13))],
        updatemenus=[
            dict(type="buttons", direction="left", x=0.01, y=0.02, xanchor="left", yanchor="bottom",
                 pad=dict(r=8), showactive=False, buttons=[
                     dict(label="▶ Złóż", method="animate", args=[names, play]),
                     dict(label="◀ Rozłóż", method="animate", args=[names[::-1], play]),
                     dict(label="⏸", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])]),
            dict(type="buttons", direction="left", x=0.99, y=0.02, xanchor="right", yanchor="bottom",
                 showactive=True, active=0, buttons=[
                     dict(label=name, method="relayout", args=[{"scene.camera": cam}])
                     for name, cam in cameras.items()])],
        sliders=[dict(active=len(frames) - 1, x=0.22, y=0.02, len=0.5, xanchor="left", yanchor="bottom",
                      currentvalue=dict(visible=False), pad=dict(t=0, b=0), ticklen=4, minorticklen=2,
                      steps=[dict(label=_slider_label(k, len(steps) - 1), value=str(k), method="animate",
                                  args=[[str(k)], dict(frame=dict(duration=0, redraw=True),
                                                       transition=dict(duration=0), mode="immediate")])
                             for k in range(len(steps))])])
    return fig


def _slider_label(k: int, last: int) -> str:
    """Etykieta tylko przy ~0/25/50/75/100 % - inaczej napisy nachodzą na siebie."""
    marks = {round(m * last / 100) for m in (0, 25, 50, 75, 100)}
    return f"{round(100 * k / last)}%" if k in marks else ""


def save_html(case: dict, path, params: View3DParams = View3DParams(), open_browser: bool = False) -> Path:
    """Zapisuje wizualizację do pliku HTML (i opcjonalnie otwiera w przeglądarce)."""
    path = Path(path)
    fig = build_figure(case, params)
    fig.write_html(path, include_plotlyjs=True if params.embed_plotly else "cdn", auto_play=False,
                   config=dict(displaylogo=False, scrollZoom=True,
                               toImageButtonOptions=dict(format="png", filename=path.stem, scale=2)))
    if open_browser:
        webbrowser.open(path.resolve().as_uri())
    return path

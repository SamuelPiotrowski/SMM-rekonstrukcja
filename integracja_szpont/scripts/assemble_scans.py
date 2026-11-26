"""Automatically rank and animate matches between the team's scanned fragments."""

import argparse
import html
import itertools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
LOCAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOCAL / "src"))

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

import config
import contacts as ct
import matching_surfaces as registration
from bones import load_bones
from data_parsing import sample_surface, submesh
from fracture_matching import display_mesh
from osteo_engine_v6 import refine_contact


def add_caps(bones, surfaces, parameters):
    """Coarse guesses for smooth fracture ends missed by roughness alone."""
    for bone in bones:
        points = np.asarray(bone.mesh.vertices)
        centered = points - points.mean(axis=0)
        _, basis = np.linalg.eigh(centered.T @ centered)
        axis = basis[:, -1]
        along = centered @ axis
        if np.ptp(along) <= 0:
            continue
        normals = np.asarray(bone.mesh.vertex_normals)
        markers = np.zeros(len(points), np.uint8)
        for sid, sign in enumerate((-1, 1), 201):
            coordinate = sign * along
            end = coordinate > coordinate.max() - np.clip(np.ptp(coordinate) * .22, 3.0, 9.0)
            markers[end & (sign * (normals @ axis) > .20)] = sid
        triangles = ct.triangle_labels(bone.mesh, markers)
        tree = cKDTree(np.asarray(bone.cloud.points))
        for sid in (201, 202):
            patch = submesh(bone.mesh, triangles == sid)
            if not patch.has_triangles() or patch.get_surface_area() < parameters.min_area_mm2:
                continue
            samples = sample_surface(patch, parameters.sample_mm)
            annotation = ct.Contacts(bone.id, markers, {sid: "Przypuszczalny koniec"}, source="auto:caps")
            surfaces.append(registration.Surface(bone, annotation, sid, samples.points, samples.normals,
                                                 patch.get_surface_area(), tree))


def shape(points, normals):
    center = points.mean(axis=0)
    z = normals.mean(axis=0)
    if np.linalg.norm(z) < .10:
        _, vectors = np.linalg.eigh(np.cov(points.T))
        z = vectors[:, 0]
    z /= np.linalg.norm(z)
    xy = points - center
    xy -= (xy @ z)[:, None] * z
    _, vectors = np.linalg.eigh(xy.T @ xy)
    x = vectors[:, -1]
    x -= (x @ z) * z
    x /= np.linalg.norm(x)
    return center, np.column_stack((x, np.cross(z, x), z))


def assess(a, b, matrix, p, check_collision=True):
    distances, cosines, covers = [], [], []
    for source, target, transform in ((a, b, matrix), (b, a, np.linalg.inv(matrix))):
        points = source.points @ transform[:3, :3].T + transform[:3, 3]
        normals = source.normals @ transform[:3, :3].T
        gap, nearest = target.tree().query(points)
        opposing = -np.einsum("ij,ij->i", normals, target.normals[nearest])
        good = (gap <= p.tolerance_mm) & (opposing >= np.cos(np.deg2rad(p.normal_angle_deg)))
        covers.append(float(good.mean()))
        distances.append(gap[good])
        cosines.append(opposing[good])
    good_distances = np.concatenate(distances)
    angles = np.degrees(np.arccos(np.clip(np.concatenate(cosines), -1, 1)))
    contact = min(a.area_mm2 * covers[0], b.area_mm2 * covers[1])
    trimmed = np.sort(good_distances)[:max(1, int(.8 * len(good_distances)))]
    rmse = float(np.sqrt(np.mean(trimmed ** 2))) if len(trimmed) else None
    angle = float(np.median(angles)) if len(angles) else None
    collision = registration.collision_area(a, b, matrix, p) if check_collision else 0.0
    quality = contact / (1 + rmse / p.tolerance_mm) * np.exp(-angle / 45) if rmse is not None else 0
    return {"score": float(quality - collision), "contact_mm2": float(contact),
            "overlap_mm2": collision, "coverage": covers, "trimmed_rmse_mm": rmse,
            "normal_error_deg": angle, "matrix": matrix.tolist(),
            "eligible": bool(contact >= 18 and min(covers) >= .20 and collision <= .25 * contact and quality > collision)}


def candidates(a, b, p):
    # The szpont FPFH/RANSAC hypothesis is complemented by rotations around the
    # patch normals. Only the most promising local fits proceed to trimmed ICP.
    upstream = registration.match_pair(a, b, p)
    ca, ba = shape(a.points, a.normals)
    cb, bb = shape(b.points, -b.normals)
    starts = []
    for theta in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        rotation = bb @ Rotation.from_rotvec([0, 0, theta]).as_matrix() @ ba.T
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = cb - rotation @ ca
        score = assess(a, b, matrix, p, check_collision=False)["score"]
        starts.append((score, matrix))
    starts.sort(key=lambda item: item[0], reverse=True)
    ca = a.clouds(p, flip=True)
    cb = b.clouds(p, flip=False)
    matrices = [np.asarray(upstream["T"])]
    matrices.extend(registration.refine_icp(ca, cb, matrix, p) for _, matrix in starts[:3])
    ranked = [assess(a, b, matrix, p) for matrix in matrices]
    best = max(ranked, key=lambda match: (match["eligible"], match["score"]))
    refined = refine_contact(b.points, b.normals, a.points, a.normals, np.asarray(best["matrix"]),
                             p.tolerance_mm)
    return max((best, assess(a, b, refined, p)), key=lambda match: (match["eligible"], match["score"]))


def create_figure(a, b, matrix):
    ma = display_mesh(a.mesh, 11500)
    mb = display_mesh(b.mesh, 11500)
    pa = np.asarray(ma["vertices"])
    pb = np.asarray(mb["vertices"])
    center = pb.mean(axis=0)
    end = center @ matrix[:3, :3].T + matrix[:3, 3]
    outward = end - pa.mean(axis=0)
    if np.linalg.norm(outward) < 1:
        outward = np.linalg.eigh(np.cov(pa.T))[1][:, 0]
    outward /= np.linalg.norm(outward)
    start = end + outward * max(14.0, np.ptp(pa @ outward) * .55)
    rotate = Slerp([0, 1], Rotation.from_matrix(np.stack((np.eye(3), matrix[:3, :3]))))

    def motion(points, t):
        eased = t * t * (3 - 2 * t)
        return rotate(eased).apply(points - center) + start * (1 - eased) + end * eased

    def mesh(points, triangles, color, label):
        return go.Mesh3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                         i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                         color=color, name=label, showlegend=True, hoverinfo="name",
                         lighting=dict(ambient=.57, diffuse=.91, roughness=.84, specular=.14),
                         lightposition=dict(x=130, y=-165, z=255))

    fixed = mesh(pa, np.asarray(ma["triangles"]), "#ead8b3", a.label)
    moving = mesh(motion(pb, 0), np.asarray(mb["triangles"]), "#98bddd", b.label)
    frames = []
    points_everywhere = [pa]
    for i, t in enumerate(np.linspace(0, 1, 31)):
        points = motion(pb, t)
        points_everywhere.append(points)
        frames.append(go.Frame(name=str(i), traces=[1], data=[
            go.Mesh3d(x=points[:, 0], y=points[:, 1], z=points[:, 2])
        ]))
    figure = go.Figure(data=[fixed, moving], frames=frames)
    all_points = np.concatenate(points_everywhere)
    lo, hi = all_points.min(axis=0) - 4, all_points.max(axis=0) + 4
    axes = {name: dict(range=[lo[i], hi[i]], visible=False)
            for i, name in enumerate(("xaxis", "yaxis", "zaxis"))}
    figure.update_layout(
        height=680, margin=dict(l=0, r=0, t=14, b=0), paper_bgcolor="#fff",
        legend=dict(orientation="h", y=1), scene=dict(**axes, aspectmode="data", bgcolor="#f3f6fa",
                   camera=dict(eye=dict(x=1.29, y=1.38, z=.87), projection=dict(type="orthographic"))),
        updatemenus=[dict(type="buttons", direction="left", x=.01, y=.07, buttons=[
            dict(label="Odtwórz składanie", method="animate", args=[None, {
                "mode": "immediate", "frame": {"duration": 85, "redraw": True},
                "transition": {"duration": 0}, "fromcurrent": False}]),
            dict(label="Wstrzymaj", method="animate", args=[[None], {"mode": "immediate",
                "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]),
        ])],
        sliders=[dict(x=.30, y=.07, len=.66, currentvalue=dict(prefix="Składanie: "), steps=[
            dict(label=f"{round(100*i/30)}%", method="animate", args=[[str(i)], {
                "mode": "immediate", "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0}}]) for i in range(31)])],
    )
    return figure


def write_report(path, bones, rows, source):
    by_id = {bone.id: bone for bone in bones}
    accepted = [row for row in rows if row["eligible"]]
    shown = accepted[:3] if accepted else rows[:1]
    intro = ("Przełomy wykryte automatycznie na siatkach. Wyniki są propozycjami do sprawdzenia."
             if source == "auto" else "Przełomy pochodzą z oznaczeń zespołu. Ustawienie obliczył algorytm.")
    status = ("Pokazano najwyżej ocenione hipotezy połączeń." if accepted
              else "Żadne dopasowanie nie spełniło kryteriów. Pokazana próba wymaga sprawdzenia.")
    sections = []
    for index, row in enumerate(shown):
        a, b = by_id[row["a"]], by_id[row["b"]]
        rmse = f'{row["trimmed_rmse_mm"]:.2f} mm' if row["trimmed_rmse_mm"] is not None else '—'
        figure = create_figure(b, a, np.asarray(row["matrix"]))
        sections.append(f'<section><h2>{html.escape(a.label)} + {html.escape(b.label)}</h2>'
                        f'<p>Styk {row["contact_mm2"]:.1f} mm² · przenikanie {row["overlap_mm2"]:.1f} mm²'
                        f' · RMSE styku {rmse}</p>'
                        + figure.to_html(full_html=False, include_plotlyjs=index == 0, auto_play=False,
                                         config={"responsive": True, "displaylogo": False}) + '</section>')
    css = ("body{margin:0;background:#f4f6f8;color:#1f2a36;font:16px Segoe UI,sans-serif}"
           "main{max-width:1240px;margin:auto;padding:24px}"
           "section{background:white;border:1px solid #dce4ea;border-radius:12px;margin:20px 0;padding:16px}"
           "h1,h2{margin:0 0 8px}p{margin:8px 0}")
    temporary = path.with_suffix(".html.tmp")
    alternate = "manual" if source == "auto" else "auto"
    alternate_label = "Dopasowanie z oznaczeniami zespołu" if source == "auto" else "W pełni automatyczne wykrywanie"
    temporary.write_text('<!doctype html><html lang="pl"><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width,initial-scale=1">'
                    '<title>Składanie naszych skanów</title><style>' + css +
                    '</style><main><h1>Składanie naszych skanów</h1>'
                    f'<p><a id="alternate" href="assembly-{alternate}.html">{alternate_label}</a></p>'
                    '<p id="appLinks" hidden><a href="/case-start">Wybór przypadku</a> · '
                    '<a href="/?case=our-scans">Otwórz wszystkie skany</a></p>'
                    '<script>if(location.protocol!=="file:"){document.getElementById("appLinks").hidden=false;'
                    f'document.getElementById("alternate").href="/our-scans?source={alternate}";'
                    '}</script>'
                    '<p>' + intro + ' ' + status +
                    '</p><p>Ruch zaczyna się od pozycji pokazowej. Położenie końcowe wyliczył algorytm.</p>' +
                    ''.join(sections) + '</main></html>', encoding="utf-8")
    temporary.replace(path)


def rank_pairs(rows):
    best = {}
    for row in sorted(rows, key=lambda entry: (entry["eligible"], entry["score"]), reverse=True):
        if row["a"] > row["b"]:
            row = dict(row, a=row["b"], b=row["a"], surface_a=row["surface_b"],
                       surface_b=row["surface_a"], region_a=row["region_b"], region_b=row["region_a"],
                       coverage=row["coverage"][::-1], matrix=np.linalg.inv(row["matrix"]).tolist())
        best.setdefault((row["a"], row["b"]), row)
    return list(best.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=LOCAL / "out" / "wasze-skany")
    parser.add_argument("--source", choices=("auto", "manual"), default="auto")
    args = parser.parse_args()
    bones = load_bones(args.out_dir, features=True)
    if len(bones) < 2:
        raise SystemExit("Najpierw przygotuj skany w podanym katalogu.")
    p = registration.SurfaceMatchParams(source=args.source, min_area_mm2=12,
                                        ransac_iterations=65000 if args.source == "auto" else 300000,
                                        ransac_seeds=1 if args.source == "auto" else 2)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    surfaces, info = registration.extract_surfaces(bones, p, config.contact_params())
    if args.source == "auto":
        add_caps(bones, surfaces, p)
    pairs = [(a, b) for a, b in itertools.combinations(surfaces, 2)
             if a.bone.id != b.bone.id and max(a.area_mm2, b.area_mm2) <= 4 * min(a.area_mm2, b.area_mm2)]
    print(f"{len(bones)} fragmentów, {len(surfaces)} regionów, {len(pairs)} par", flush=True)
    rows = []
    for i, (a, b) in enumerate(pairs, 1):
        row = candidates(a, b, p)
        row.update(a=a.bone.id, b=b.bone.id, surface_a=a.sid, surface_b=b.sid,
                   region_a=a.contacts.source, region_b=b.contacts.source)
        rows.append(row)
        if i % 20 == 0 or i == len(pairs):
            print(f"{i}/{len(pairs)}", flush=True)
    top = rank_pairs(rows)
    result = {"source": args.source, "detected_surfaces": info, "input_fragments": len(bones),
              "tested_region_pairs": len(pairs), "results": top}
    (args.out_dir / f"assembly-{args.source}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                             encoding="utf-8")
    write_report(args.out_dir / f"assembly-{args.source}.html", bones, top, args.source)
    for row in top[:5]:
        print(row["a"], row["b"], row["eligible"], f'{row["score"]:.2f}', flush=True)


if __name__ == "__main__":
    main()

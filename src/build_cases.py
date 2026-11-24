"""Reproducible 2D fracture study: synthetic raster inputs -> observed contours -> fits.

Run from any directory: python python/build_cases.py
No ground-truth adjacency or pose is read by analyse_case(), fit_arc_pair(),
or assemble().  Two clicked fracture-ROI endpoints are supplied annotations.
The independent truth file is read only by evaluate_truth() after reconstruction.
All geometric coordinates are millimetres, x right and y up. No reflections.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy
from scipy import ndimage as ndi
from scipy.interpolate import make_splprep
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
MM_PER_PIXEL = 0.25
IMAGE_SIZE = 512
SEED = 20261124
SIGMAS = (0.04, 0.12, 0.35)
STEP = 0.25
TOLERANCE = 0.60
MIN_CONTACT = 12.0
COLORS = ["#477eae", "#299784", "#ce8b3e", "#8b6ab3"]


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def transform(P, theta, translation):
    return np.asarray(P) @ rotation(theta).T + np.asarray(translation)


def area(P):
    return float(0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - P[:, 1] * np.roll(P[:, 0], -1)))


def polygon_path(P):
    """Append a dedicated CLOSEPOLY vertex; preserve every measured vertex."""
    P=np.asarray(P)
    return MplPath(np.vstack([P,P[0]]),closed=True)


def arc_length(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def sample_polyline(P, count=None, step=STEP):
    P = np.asarray(P, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-8]
    P = P[keep]
    s = arc_length(P)
    q = np.linspace(0, s[-1], count or max(8, int(np.ceil(s[-1] / step)) + 1))
    return np.column_stack([np.interp(q, s, P[:, j]) for j in range(2)])


def spline_curve(P, sigma, count=181):
    """Cubic B-spline, make_splprep residual upper bound s=N*sigma**2."""
    P = sample_polyline(P, step=STEP)
    u = arc_length(P)
    u /= u[-1]
    spl, _ = make_splprep(P.T, u=u, k=3, s=len(P) * sigma * sigma)
    dense_u = np.linspace(0, 1, 2500)
    dense = spl(dense_u).T
    dense_s = arc_length(dense)
    sampled_s = np.linspace(0, dense_s[-1], count)
    uq = np.interp(sampled_s, dense_s, dense_u)
    Q = spl(uq).T
    d1, d2 = spl.derivative(1)(uq).T, spl.derivative(2)(uq).T
    speed = np.linalg.norm(d1, axis=1)
    kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(speed, 1e-9) ** 3
    details={"controlPoints":spl.c.tolist(),"knots":spl.t.tolist(),"degree":int(spl.k),
             "residualRmse":float(np.sqrt(np.mean(np.sum((P-spl(u).T)**2,axis=1))))}
    return Q, sampled_s, kappa, details


def width(y):
    y = np.asarray(y)
    return 11.0 + 13.5 * np.exp(-((y - 12.0) / 12.0) ** 2) + 13.0 * np.exp(-((y - 138.0) / 12.0) ** 2) + 1.2 * np.sin(y / 19.0)


def seam(y, index):
    w = float(width(y))
    rng = np.random.default_rng(SEED + 31 * index)
    x = np.linspace(-w, w, 19)
    h = rng.uniform(-3.7, 3.7, len(x))
    h[0] = h[-1] = 0
    # Long, nonperiodic irregular teeth; different fingerprint for each seam.
    return np.c_[x, y + h]


def canonical_geometry(loss=False):
    cuts = [seam(39, 1), seam(76, 2), seam(111, 3)]
    bounds = [0, 39, 76, 111, 150]
    fragments = []
    for i in range(4):
        y0, y1 = bounds[i], bounds[i + 1]
        bottom = cuts[i - 1].copy() if i else np.array([[-width(0), 0], [width(0), 0]])
        top = cuts[i].copy() if i < 3 else np.array([[-width(150), 150], [width(150), 150]])
        if loss and i == 1:
            # Lost material on one side of A/B contact; original outline is held out.
            bottom = sample_polyline(bottom, count=190)
            bottom[:, 1] += 2.6 * np.exp(-((bottom[:, 0] - 4.0) / 2.0) ** 2)
        yr = np.linspace(y0, y1, 90)
        right = np.c_[width(yr), yr][1:-1]
        left = np.c_[-width(yr[::-1]), yr[::-1]][1:-1]
        polygon = np.vstack([bottom, right, top[::-1], left])
        rois = []
        if i:
            rois.append({"endpoints": [bottom[0].tolist(), bottom[-1].tolist()], "truth_seam": i})
        if i < 3:
            rois.append({"endpoints": [top[-1].tolist(), top[0].tolist()], "truth_seam": i + 1})
        fragments.append((polygon, rois))
    y = np.linspace(0, 150, 301)
    outline = np.vstack([np.c_[-width(y), y], np.c_[width(y[::-1]), y[::-1]]])
    if area(outline) < 0:
        outline = outline[::-1]
    return fragments, outline


def points_to_pixels(P):
    P = np.asarray(P)
    return np.c_[P[:, 0] / MM_PER_PIXEL + IMAGE_SIZE / 2, IMAGE_SIZE / 2 - P[:, 1] / MM_PER_PIXEL]


def pixels_to_points(P):
    P = np.asarray(P)
    return np.c_[(P[:, 0] - IMAGE_SIZE / 2) * MM_PER_PIXEL, (IMAGE_SIZE / 2 - P[:, 1]) * MM_PER_PIXEL]


def render_input(P, case_id, rng):
    mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    ImageDraw.Draw(mask).polygon([tuple(x) for x in points_to_pixels(P)], fill=255)
    binary = np.asarray(mask) > 127
    yy, xx = np.indices(binary.shape)
    texture = ndi.gaussian_filter(rng.normal(0, 1, binary.shape), 2)
    texture /= np.std(texture)
    bg = 24 + 9 * xx / IMAGE_SIZE + 4 * yy / IMAGE_SIZE
    fg = 218 + 6 * np.sin(xx / 83) + 4 * texture
    image = np.where(binary, fg, bg)
    sigma = 0.35 if case_id == "clean" else 0.85
    noise = 1.8 if case_id == "clean" else 12.0
    image = ndi.gaussian_filter(image, sigma) + rng.normal(0, noise, image.shape)
    return Image.fromarray(np.uint8(np.clip(image, 0, 255))), binary


def otsu(image):
    counts = np.bincount(np.asarray(image, dtype=np.uint8).ravel(), minlength=256).astype(float)
    prob = counts / counts.sum()
    cumulative = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    total = mu[-1]
    variance = (total * cumulative - mu) ** 2 / np.maximum(cumulative * (1 - cumulative), 1e-15)
    return int(np.argmax(variance))


def contour_from_mask(binary):
    # Matplotlib's contour generator applies marching squares to the observed mask.
    fig, ax = plt.subplots(figsize=(2, 2))
    cs = ax.contour(binary.astype(float), levels=[0.5])
    paths = cs.allsegs[0]
    plt.close(fig)
    pixel_path = max(paths, key=len)
    P = pixels_to_points(pixel_path[:-1])
    if area(P) < 0:
        P = P[::-1]
    return P


def extract_roi(contour, endpoint_hints):
    i, j = [int(np.argmin(np.linalg.norm(contour - h, axis=1))) for h in endpoint_hints]
    n = len(contour)
    forward = contour[(i + np.arange((j - i) % n + 1)) % n]
    other = contour[(j + np.arange((i - j) % n + 1)) % n]
    # Both candidates follow CCW. Retain shorter arc, which this dataset guarantees.
    return forward if arc_length(forward)[-1] < arc_length(other)[-1] else other


def rigid(A, B):
    """Map B -> A, no reflection. A and B have the same number of samples."""
    a, b = A.mean(axis=0), B.mean(axis=0)
    U, _, Vt = np.linalg.svd((B - b).T @ (A - a))
    D = np.diag([1, np.linalg.det(Vt.T @ U.T)])
    R = Vt.T @ D @ U.T
    t = a - R @ b
    theta = math.atan2(R[1, 0], R[0, 0])
    return theta, t


def interpolation(P, s, samples):
    return np.column_stack([np.interp(samples, s, P[:, j]) for j in range(2)])


def interior_grid(P, step=0.50):
    low, high = P.min(axis=0), P.max(axis=0)
    x = np.arange(low[0] + step / 2, high[0], step)
    y = np.arange(low[1] + step / 2, high[1], step)
    xx, yy = np.meshgrid(x, y)
    Q = np.c_[xx.ravel(), yy.ravel()]
    return Q[polygon_path(P).contains_points(Q)], step


def quality(P, Q, theta, translation, polyA, gridB, grid_step):
    TQ = transform(Q, theta, translation)
    treeQ, treeP = cKDTree(TQ), cKDTree(P)
    da, ia = treeQ.query(P)
    db, ib = treeP.query(TQ)
    all_d = np.r_[da, db]
    # Equal arc-length samples => fractions approximate physical length fractions.
    ca, cb = float(np.mean(da <= TOLERANCE)), float(np.mean(db <= TOLERANCE))
    len_a, len_b = arc_length(P)[-1], arc_length(Q)[-1]
    contact = 0.5 * (ca * len_a + cb * len_b)
    moved_grid = transform(gridB, theta, translation)
    overlap = float(np.count_nonzero(polygon_path(polyA).contains_points(moved_grid))) * grid_step ** 2
    va = np.gradient(P, axis=0)
    vb = np.gradient(TQ, axis=0)
    va /= np.maximum(np.linalg.norm(va, axis=1, keepdims=True), 1e-10)
    vb /= np.maximum(np.linalg.norm(vb, axis=1, keepdims=True), 1e-10)
    # Q is stored in original CCW direction. Matching contact tangents must oppose.
    cosine = np.clip(np.sum(va * vb[ia], axis=1), -1, 1)
    normal_error = np.degrees(np.arccos(-cosine))
    keep = da <= TOLERANCE
    normals = float(np.median(normal_error[keep])) if np.any(keep) else 180.0
    trim_n = max(3, int(0.8 * len(all_d)))
    trimmed = float(np.sqrt(np.mean(np.sort(all_d)[:trim_n] ** 2)))
    p95 = float(np.percentile(all_d, 95))
    area_min = min(abs(area(polyA)), len(gridB) * grid_step ** 2)
    ratio = overlap / max(area_min, 1e-9)
    # Transparent ranking score, NOT a calibrated probability.
    score = contact / (1 + trimmed / TOLERANCE) * np.exp(-normals / 45.0) - 7 * overlap
    if ratio > 0.025 or contact < MIN_CONTACT:
        score -= 40
    return {"rmse": float(np.sqrt(np.mean(all_d ** 2))), "trimmedRmse": trimmed,
            "median": float(np.median(all_d)), "p95": p95,
            "hausdorff": float(all_d.max()), "coverageA": 100 * ca, "coverageB": 100 * cb,
            "length": float(contact), "collisionArea": overlap, "collisionRatio": ratio,
            "normalError": normals, "score": float(score)}


def fit_arc_pair(arcA, arcB, fragmentA, fragmentB):
    """Unknown point correspondence: enumerate physically equal sliding subarcs.

    No truth IDs, reference image, or poses are passed to this function.
    Both full arcs are uniformly sampled from measured B-splines. Reverse B.
    Search sliding windows at 0.75 mm and four lengths, then score geometry.
    """
    P = np.asarray(arcA["smoothLevels"][1]["points"])
    Q = np.asarray(arcB["smoothLevels"][1]["points"])
    Pr = sample_polyline(P, step=0.20)
    Qr = sample_polyline(Q, step=0.20)
    Qrev = Qr[::-1]
    sp, sq = arc_length(Pr), arc_length(Qrev)
    length_min = min(sp[-1], sq[-1])
    guesses = []
    for fraction in [1.0, 0.90, 0.75, 0.60]:
        length = fraction * length_min
        if length < MIN_CONTACT:
            continue
        startsA = np.linspace(0, sp[-1] - length, max(1, int(np.ceil((sp[-1] - length) / 0.75)) + 1))
        startsB = np.linspace(0, sq[-1] - length, max(1, int(np.ceil((sq[-1] - length) / 0.75)) + 1))
        for startA in startsA:
            A = interpolation(Pr, sp, np.linspace(startA, startA + length, 64))
            for startB in startsB:
                B = interpolation(Qrev, sq, np.linspace(startB, startB + length, 64))
                theta, tr = rigid(A, B)
                error = float(np.sqrt(np.mean(np.sum((A - transform(B, theta, tr)) ** 2, axis=1))))
                guesses.append((error, theta, tr, A, B, length, float(startA), float(startB)))
    # Only competitive correspondence hypotheses proceed to costly overlap checks.
    guesses.sort(key=lambda x: x[0])
    chosen = []
    for g in guesses:
        if not any(abs(math.atan2(math.sin(g[1]-h[1]), math.cos(g[1]-h[1]))) < np.deg2rad(1.0)
                   and np.linalg.norm(g[2]-h[2]) < 0.40 for h in chosen):
            chosen.append(g)
        if len(chosen) >= 12:
            break
    gridB, step = fragmentB["_grid"]
    evaluated = []
    for g in chosen:
        error, theta, tr, A, B, length, sa, sb = g
        metrics = quality(Pr, Qr, theta, tr, np.asarray(fragmentA["contour"]), gridB, step)
        metrics["pairedRmse"] = error
        metrics["windowLength"] = float(length)
        metrics["startA"] = sa
        metrics["startBReversed"] = sb
        # Informational Pearson correlation, evaluated on this same window.
        def turn_sig(points):
            v = np.diff(points, axis=0)
            ang = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
            return np.diff(ang)
        a, b = turn_sig(A), turn_sig(B)
        metrics["correlation"] = float(np.corrcoef(a, b)[0, 1]) if min(np.std(a),np.std(b)) > 1e-8 else None
        evaluated.append((metrics["score"], theta, tr, A, B, metrics))
    result = max(evaluated, key=lambda x:x[0])
    _, theta, tr, A, B, metrics = result
    # Independent final distances on the unsmoothed observed ROI, not the spline.
    rawP=sample_polyline(np.asarray(arcA["points"]),step=.20)
    rawQ=transform(sample_polyline(np.asarray(arcB["points"]),step=.20),theta,tr)
    raw_distances=np.r_[cKDTree(rawQ).query(rawP)[0],cKDTree(rawP).query(rawQ)[0]]
    metrics["rawRmse"]=float(np.sqrt(np.mean(raw_distances**2)))
    metrics["rawP95"]=float(np.percentile(raw_distances,95))
    return {"id": arcA["id"] + "_" + arcB["id"], "a": fragmentA["id"], "b": fragmentB["id"],
            "arcA": arcA["id"], "arcB": arcB["id"], "matchedA": A.tolist(), "matchedB": B.tolist(),
            "evaluationA":Pr.tolist(),"evaluationB":Qr.tolist(),
            "theta": float(theta), "translation": tr.tolist(), "metrics": metrics,
            "label": arcA["id"] + " ↔ " + arcB["id"], "accepted": False,
            "hypothesesTested": len(guesses), "hypothesesVerified": len(chosen)}


def compose(Ta, Tab):
    """Ta: a->world, Tab:b->a, result b->world."""
    return (Ta[0] + Tab[0], rotation(Ta[0]) @ Tab[1] + Ta[1])


def inverse(T):
    return (-T[0], -rotation(-T[0]) @ T[1])


def assemble(fragments, candidates):
    ids = [f["id"] for f in fragments]
    parent = {i: i for i in ids}
    def find(i):
        while i != parent[i]:
            i = parent[i]
        return i
    used_arcs, selected = set(), []
    for c in sorted(candidates, key=lambda c: c["metrics"]["score"], reverse=True):
        m = c["metrics"]
        if m["score"] <= 0 or m["length"] < MIN_CONTACT or m["collisionRatio"] > 0.025:
            continue
        if c["arcA"] in used_arcs or c["arcB"] in used_arcs or find(c["a"]) == find(c["b"]):
            continue
        parent[find(c["a"])] = find(c["b"])
        used_arcs.update([c["arcA"], c["arcB"]])
        c["accepted"] = True
        selected.append(c)
    components=[]
    remaining=set(ids)
    while remaining:
        root=sorted(remaining)[0]
        poses = {root: (0.0, np.zeros(2))}
        while True:
            changed = False
            for c in selected:
                a,b = c["a"],c["b"]
                Tab = (c["theta"], np.asarray(c["translation"]))
                if a in poses and b not in poses:
                    poses[b] = compose(poses[a], Tab); changed = True
                elif b in poses and a not in poses:
                    poses[a] = compose(poses[b], inverse(Tab)); changed = True
            if not changed:
                break
        components.append({"id":f"component{len(components)+1}","anchor":root,"size":len(poses),
                           "poses":[{"id":i,"theta":float(t[0]),"translation":t[1].tolist()} for i,t in poses.items()]})
        remaining-=set(poses)
    poses=components[0]["poses"]
    return {"poses": poses,"components":components,
            "initialPoses": [{"id":i,"theta":0,"translation":[-45+90*(k%2),-45+90*(k//2)]} for k,i in enumerate(ids)],
            "selectedCandidateIds": [c["id"] for c in selected]}


def rounded(data):
    if isinstance(data, np.ndarray):
        return rounded(data.tolist())
    if isinstance(data, (np.floating, float)):
        if not np.isfinite(data):
            raise ValueError("Non-finite output")
        return round(float(data), 6)
    if isinstance(data, (np.bool_, bool)):
        return bool(data)
    if isinstance(data, (np.integer, int)):
        return int(data)
    if isinstance(data, dict):
        return {k: rounded(v) for k,v in data.items()}
    if isinstance(data, (list, tuple)):
        return [rounded(x) for x in data]
    return data


def save_json(path, data):
    path.write_text(json.dumps(rounded(data), ensure_ascii=False, indent=2), encoding="utf-8")


def create_inputs(case_id):
    dest = ROOT / "assets" / "cases" / case_id
    dest.mkdir(parents=True, exist_ok=True)
    fragments, outline = canonical_geometry(loss=case_id == "loss")
    original, _ = canonical_geometry(loss=False)
    angles = np.deg2rad([24, -37, 68, -19])
    annotations = {"case":case_id, "synthetic":True, "units":"mm", "mmPerPixel":MM_PER_PIXEL,
                   "imageSize":IMAGE_SIZE,"note":"Endpoint hints simulate two manual clicks per fracture ROI. They reveal ROIs, never adjacency or matching points.","fragments":[]}
    truth = {"case":case_id,"outline":outline.tolist(),"fragments":[],"seams":{}}
    for i, ((P, rois), ang) in enumerate(zip(fragments, angles)):
        fid = chr(65+i)
        cen = original[i][0].mean(axis=0)
        tr = -rotation(ang) @ cen
        local = transform(P, ang, tr)
        # Same random field in every case; only blur/noise amplitude and loss change.
        image, _ = render_input(local, case_id, np.random.default_rng(SEED+101*i))
        image.save(dest / (fid + ".png"))
        clicked = []
        for k, roi in enumerate(rois):
            rid = fid + str(k+1)
            endpoints = transform(roi["endpoints"], ang, tr)
            # Subpixel rounding is what a human endpoint click would imply.
            endpoints = np.round(endpoints / MM_PER_PIXEL) * MM_PER_PIXEL
            clicked.append({"id":rid,"endpoints":endpoints.tolist()})
            truth["seams"][rid] = roi["truth_seam"]
        annotations["fragments"].append({"id":fid,"image":f"assets/cases/{case_id}/{fid}.png","rois":clicked})
        inv = inverse((ang,tr))
        truth["fragments"].append({"id":fid,"theta":float(inv[0]),"translation":inv[1].tolist(),
                                    "polygon":P.tolist(),"originalPolygon":original[i][0].tolist()})
    save_json(ROOT/"data"/f"{case_id}_annotations.json", annotations)
    save_json(ROOT/"data"/f"{case_id}_truth.json", truth)
    return annotations, truth


def analyse_case(annotation_path):
    """Analysis receives ONLY image paths, pixel scale and manual ROI endpoints."""
    annot = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    if annot["imageSize"] != IMAGE_SIZE or annot["mmPerPixel"] != MM_PER_PIXEL:
        raise ValueError("This teaching script requires 512x512 images and 0.25 mm/pixel. Change its calibration functions for other data.")
    case_id = annot["case"]
    fragments = []
    for source in annot["fragments"]:
        image = Image.open(ROOT/source["image"]).convert("L")
        threshold = otsu(image)
        binary = np.asarray(image) > threshold
        labels, n = ndi.label(binary)
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        binary = labels == np.argmax(sizes)
        contour = contour_from_mask(binary)
        mpath = f"assets/cases/{case_id}/{source['id']}_mask.png"
        Image.fromarray(binary.astype(np.uint8)*255).save(ROOT/mpath)
        contour_display = sample_polyline(np.vstack([contour, contour[0]]), count=600)[:-1]
        item = {"id":source["id"],"image":source["image"],"mask":mpath,"contour":contour_display.tolist(),
                "threshold":threshold,"area":abs(area(contour)),"arcs":[],"_grid":interior_grid(contour_display)}
        for roi in source["rois"]:
            raw = extract_roi(contour, np.asarray(roi["endpoints"]))
            raw = sample_polyline(raw, step=0.20)
            levels = []
            for sigma, label in zip(SIGMAS,["Szczegóły: 0,04 mm","Roboczy: 0,12 mm","Silne: 0,35 mm"]):
                q,s,k,details = spline_curve(raw,sigma)
                levels.append({"label":label,"sigma":sigma,"points":q.tolist(),"s":s.tolist(),"curvature":k.tolist(),**details})
            item["arcs"].append({"id":roi["id"],"points":raw.tolist(),"length":float(arc_length(raw)[-1]),
                                   "endpoints":roi["endpoints"],"smoothLevels":levels})
        fragments.append(item)
    candidates = []
    for ia,fa in enumerate(fragments):
        for fb in fragments[ia+1:]:
            for aa in fa["arcs"]:
                for ab in fb["arcs"]:
                    candidates.append(fit_arc_pair(aa,ab,fa,fb))
    candidates.sort(key=lambda x:x["metrics"]["score"],reverse=True)
    assembly = assemble(fragments,candidates)
    for f in fragments:
        f.pop("_grid")
    title = {"clean":"01 / Cztery fragmenty: czysty obraz", "noise":"02 / Te same fragmenty: szum i rozmycie", "loss":"03 / Ubytek na jednym przełomie"}[case_id]
    description = {"clean":"Komplementarne, poszarpane brzegi. Dyskretyzacja obrazu powoduje niezerowy błąd nawet dla idealnego modelu.",
                   "noise":"Ten sam obiekt i położenia; zwiększony szum oraz rozmycie sprawdzają stabilność segmentacji i splajnów.",
                   "loss":"Z fragmentu B usunięto lokalny odprysk. Dopasowanie jest częściowe; szczeliny nie należy usuwać deformacją ani przesuwaniem materiału w kolizję."}[case_id]
    return {"id":case_id,"title":title,"description":description,"scale":MM_PER_PIXEL,
            "referenceImage":f"assets/cases/{case_id}/reference.png","inputImage":f"assets/cases/{case_id}/input.png",
            "maskImage":f"assets/cases/{case_id}/masks.png","assemblyImage":f"assets/cases/{case_id}/assembly.png",
            "fragments":fragments,"candidates":candidates,"assembly":assembly,
            "thresholds":{"distance":TOLERANCE,"minLength":MIN_CONTACT,"maxCollisionRatio":0.025},
            "notes":["Wszystkie obrazy są syntetyczne; nie są zdjęciami pacjenta ani rzeczywistą walidacją medyczną.",
                     "Dwa końce każdego przełomu są dostarczoną adnotacją. Ich wskazanie jest etapem ręcznym.",
                     "Wyszukiwanie nie zna sąsiedztwa ani prawdziwych transformacji. Przeszukuje odwrócone okna o tej samej długości w mm.",
                     "B-splajny są obliczone w Pythonie. Wybór skali w raporcie pokazuje gotowe warianty; nie uruchamia SciPy w przeglądarce.",
                     "Pole kolizji jest estymacją siatkową z krokiem 0,5 mm, więc ma skończoną rozdzielczość.",
                     "Wszystkie pary sąsiadów są hipotezami; wysoki wynik rankingu nie jest prawdopodobieństwem.",
                     "Globalny montaż jest zachłannym wyborem niekonfliktujących połączeń i kompozycją transformacji w drzewie."]}


def evaluate_truth(case, truth):
    """Truth is consulted only after every fit and selected assembly is fixed."""
    ftruth={x["id"]:x for x in truth["fragments"]}
    A = ftruth["A"]
    canonical_to_A = inverse((A["theta"],np.array(A["translation"])))
    truth_poses={}
    for f in case["fragments"]:
        t=ftruth[f["id"]]
        pose=compose(canonical_to_A,(t["theta"],np.array(t["translation"])))
        f["poseTruth"]={"theta":float(pose[0]),"translation":pose[1].tolist()}
        truth_poses[f["id"]]=pose
    for c in case["candidates"]:
        c["correct"] = truth["seams"][c["arcA"]] == truth["seams"][c["arcB"]]
        rel=compose(inverse(truth_poses[c["a"]]),truth_poses[c["b"]])
        c["metrics"]["angleError"] = abs(math.degrees(math.atan2(math.sin(c["theta"]-rel[0]),math.cos(c["theta"]-rel[0]))))
        c["metrics"]["translationError"] = float(np.linalg.norm(np.array(c["translation"])-rel[1]))
    reference=transform(truth["outline"],*canonical_to_A)
    case["assembly"]["referenceContour"]=reference.tolist()
    poses={p["id"]:(p["theta"],np.asarray(p["translation"])) for p in case["assembly"]["poses"]}
    allP=[transform(f["contour"],*poses[f["id"]]) for f in case["fragments"] if f["id"] in poses]
    bounds=np.vstack([reference]+allP)
    low,high=bounds.min(axis=0),bounds.max(axis=0)
    step=.35
    xx,yy=np.meshgrid(np.arange(low[0],high[0]+step,step),np.arange(low[1],high[1]+step,step))
    points=np.c_[xx.ravel(),yy.ravel()]
    refinside=polygon_path(reference).contains_points(points)
    count=np.zeros(len(points),dtype=np.uint8)
    for poly in allP:
        count += polygon_path(poly).contains_points(points)
    inside=count>0
    iou=float(np.count_nonzero(inside & refinside)/np.count_nonzero(inside | refinside))
    angle_err=[];trans_err=[]
    for fid,pose in poses.items():
        if fid=="A":
            continue
        target=truth_poses[fid]
        angle_err.append(abs(math.degrees(math.atan2(math.sin(pose[0]-target[0]),math.cos(pose[0]-target[0])))))
        trans_err.append(np.linalg.norm(pose[1]-target[1]))
    selected=[c for c in case["candidates"] if c["accepted"]]
    connected=len(case["assembly"]["components"])==1
    case["assembly"]["metrics"]={"iou":iou if connected else None,"anchoredIou":iou,
        "angleError":float(np.mean(angle_err)) if angle_err else None,
        "translationError":float(np.mean(trans_err)) if trans_err else None,
        "evaluatedFragments":len(angle_err),"components":len(case["assembly"]["components"]),
        "largestComponentSize":max(c["size"] for c in case["assembly"]["components"]),
        "correctEdges":sum(c["correct"] for c in selected),"selectedEdges":len(selected),"fragmentsPlaced":len(poses),
        "collisionArea":float(np.sum(count>1)*step**2),"gridStep":step,
        "edgePrecision":sum(c["correct"] for c in selected)/max(1,len(selected)),
        "edgeRecall":sum(c["correct"] for c in selected)/3}


def make_images(case, truth):
    dest=ROOT/"assets"/"cases"/case["id"]
    # Contact sheet of the four actual input rasters, not a vector screenshot.
    for key,name in [("image","input.png"),("mask","masks.png")]:
        sheet=Image.new("RGB",(1080,1120),"#edf3f6")
        draw=ImageDraw.Draw(sheet)
        for i,f in enumerate(case["fragments"]):
            im=Image.open(ROOT/f[key]).convert("RGB")
            x,y=20+(i%2)*540,35+(i//2)*550
            sheet.paste(im,(x,y))
            draw.text((x+8,y+5),f["id"],fill="#ffcd68",stroke_width=1)
            if key=="image":
                for arc in f["arcs"]:
                    for p in points_to_pixels(arc["endpoints"]):
                        cx,cy=p+[x,y]
                        draw.ellipse((cx-5,cy-5,cx+5,cy+5),outline="#f6b752",width=2)
            # 10 mm scale, positioned inside dark margins.
            draw.line((x+20,y+480,x+60,y+480),fill="#ffcd68",width=3)
            draw.text((x+20,y+486),"10 mm",fill="#ffcd68")
        sheet.save(dest/name)
    fig,ax=plt.subplots(figsize=(5,8))
    for i,t in enumerate(truth["fragments"]):
        p=np.asarray(t["originalPolygon"])
        ax.fill(p[:,0],p[:,1],color=COLORS[i],alpha=.75,ec="white",lw=1.2)
        cen=p.mean(0); ax.text(*cen,chr(65+i),ha="center",va="center",fontsize=18,color="white",weight="bold")
    ax.set_aspect("equal");ax.set_xlabel("x [mm]");ax.set_ylabel("y [mm]");ax.set_title("Model referencyjny — 4 fragmenty")
    ax.grid(alpha=.13);fig.tight_layout();fig.savefig(dest/"reference.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,8))
    ref=np.asarray(case["assembly"]["referenceContour"])
    ax.plot(ref[:,0],ref[:,1],"--",c="#adbac5",label="Referencja",lw=1.5)
    poses={p["id"]:p for p in case["assembly"]["poses"]}
    if len(case["assembly"]["components"])>1:
        # Display disconnected groups apart. Offset is presentation only, not inferred.
        poses={}
        for k,component in enumerate(case["assembly"]["components"]):
            for p in component["poses"]:
                p=dict(p);p["translation"]=(np.array(p["translation"])+[90*k,0]).tolist()
                poses[p["id"]]=p
    for i,f in enumerate(case["fragments"]):
        if f["id"] not in poses:continue
        t=poses[f["id"]];p=transform(f["contour"],t["theta"],t["translation"])
        ax.fill(p[:,0],p[:,1],color=COLORS[i],alpha=.8,ec="white",lw=.8,label=f["id"])
    ax.set_aspect("equal");ax.set_xlabel("x [mm]");ax.set_ylabel("y [mm]")
    iou=case['assembly']['metrics']['iou']
    ax.set_title(f"Wynik algorytmu • IoU = {iou:.3f}" if iou is not None else "Dwie osobne grupy — brak wiarygodnego połączenia")
    ax.legend(loc="upper right");ax.grid(alpha=.12);fig.tight_layout();fig.savefig(dest/"assembly.png",dpi=150);plt.close(fig)


def export_csv(cases):
    names=["case","candidate","a","b","correct","selected","angle_deg","tx_mm","ty_mm","rmse_mm","paired_rmse_mm","p95_mm","coverage_A_pct","coverage_B_pct","contact_mm","collision_mm2","score","angle_error_deg","translation_error_mm"]
    with (ROOT/"data"/"candidate_results.csv").open("w",newline="",encoding="utf-8-sig") as h:
        w=csv.writer(h);w.writerow(names)
        for case in cases:
            for c in case["candidates"]:
                m=c["metrics"]
                w.writerow([case["id"],c["id"],c["a"],c["b"],c["correct"],c["accepted"],math.degrees(c["theta"]),*c["translation"],m["rmse"],m["pairedRmse"],m["p95"],m["coverageA"],m["coverageB"],m["length"],m["collisionArea"],m["score"],m["angleError"],m["translationError"]])
    with (ROOT/"data"/"assembly_results.csv").open("w",newline="",encoding="utf-8-sig") as h:
        w=csv.writer(h);w.writerow(["case","IoU","correct_edges","selected_edges","fragments_placed","mean_angle_error_deg","mean_translation_error_mm","overlap_mm2"])
        for c in cases:
            m=c["assembly"]["metrics"]
            w.writerow([c["id"],m["iou"],m["correctEdges"],m["selectedEdges"],m["fragmentsPlaced"],m["angleError"],m["translationError"],m["collisionArea"]])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-inputs",action="store_true",help="Analyse existing PNG inputs and endpoint annotations; do not regenerate images.")
    args=parser.parse_args()
    (ROOT/"data").mkdir(exist_ok=True)
    start=time.perf_counter();cases=[]
    for cid in ["clean","noise","loss"]:
        if not args.reuse_inputs:
            create_inputs(cid)
        case=analyse_case(ROOT/"data"/f"{cid}_annotations.json")
        truth=json.loads((ROOT/"data"/f"{cid}_truth.json").read_text(encoding="utf-8"))
        evaluate_truth(case,truth)
        make_images(case,truth)
        cases.append(case)
        print(cid,json.dumps(case["assembly"]["metrics"],ensure_ascii=False),flush=True)
        print("selected",[(c["id"],round(c["metrics"]["score"],3),c["correct"]) for c in case["candidates"] if c["accepted"]],flush=True)
    data={"meta":{"title":"Studia przypadku: od obrazu do złożenia 4 fragmentów","synthetic":True,
         "seed":SEED,"units":"mm","angles":"rad","coordinates":"Cartesian x right, y up. Candidate maps b into a. Assembly and truth poses are anchored in fragment A local coordinates.",
         "pythonVersion":"3.12+","numpyVersion":np.__version__,"scipyVersion":scipy.__version__,"matplotlibVersion":matplotlib.__version__,
         "annotation":"Supplied two-endpoint hints for each fracture ROI. No adjacency or point correspondences are supplied.",
         "buildSeconds":round(time.perf_counter()-start,2),"cases":3,"fragmentsPerCase":4},"cases":cases}
    data=rounded(data)
    save_json(ROOT/"data"/"study_data.json",data)
    (ROOT/"data"/"study_data.js").write_text("window.STUDY_DATA="+json.dumps(data,ensure_ascii=False,separators=(",",":"))+";\n",encoding="utf-8")
    export_csv(cases)
    print("Written study_data.js/json and CSV. Seconds",time.perf_counter()-start,flush=True)


if __name__=="__main__":
    main()

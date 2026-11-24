"""
Etapy 3-5: dopasowanie par przełomów, odrzucenie złych, złożenie, ocena.

    3. fit_arc_pair / find_candidates - dla każdej pary łuków z różnych fragmentów
       szuka transformacji sztywnej (bez znajomości odpowiedników punktów)
       i ocenia ją geometrycznie (quality).
    4. assemble - zachłannie wybiera najlepsze niekonfliktujące połączenia
       (odrzuca: słaby wynik, kolizje, zajęty łuk, cykl) i składa pozy fragmentów.
    5. evaluate_truth - TYLKO dla danych z prawdą (np. syntetycznych): porównuje
       wynik z prawdziwymi położeniami. Własne dane nie wymagają tego kroku.

Kandydat "a <- b" przechowuje transformację mapującą fragment b do układu a.
Wynik `score` to przejrzysty ranking, a NIE skalibrowane prawdopodobieństwo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from geometry import (angle_difference, arc_length, area, compose, interior_grid, interpolate_at,
                      inverse, polygon_path, rigid_fit, sample_polyline, transform)


@dataclass
class MatchParams:
    """Progi i ustawienia wyszukiwania. Domyślne wartości dobrane dla kości ~150 mm."""
    tolerance_mm: float = 0.60          # odległość, poniżej której punkty uznaje się za stykające się
    min_contact_mm: float = 12.0        # minimalna długość styku, żeby przyjąć połączenie
    min_score: float = 6.0              # minimalny score (~styk ważony jakością [mm]); ok. połowa min_contact_mm
    max_collision_ratio: float = 0.025  # maks. pole nakładania (głębszego niż tolerance_mm) / pole mniejszego fragmentu
    spline_level: int = 1               # który poziom wygładzenia (approximation) użyć
    resample_step_mm: float = 0.20      # krok próbkowania łuków do porównań
    window_step_mm: float = 0.75        # krok przesuwania okien po łukach
    window_fractions: tuple = (1.0, 0.90, 0.75, 0.60)  # długości okien jako ułamek krótszego łuku
    window_samples: int = 64            # punkty w oknie do dopasowania SVD
    hypotheses_verified: int = 12       # ile najlepszych hipotez dostaje pełną (drogą) ocenę
    dedupe_angle_deg: float = 1.0       # hipotezy bliższe niż to uznaje się za duplikaty
    dedupe_translation_mm: float = 0.40
    grid_step_mm: float = 0.50          # krok siatki do liczenia pola kolizji
    normal_scale_deg: float = 45.0      # kara za niezgodność normalnych: exp(-błąd / skala)
    overlap_penalty: float = 7.0        # kara za każdy mm^2 nakładania
    reject_penalty: float = 40.0        # kara za przekroczenie progów (styk / kolizja)


# -----------------
# 3. OCENA I DOPASOWANIE PARY ŁUKÓW
# -----------------

def quality(p, q, theta, translation, polygon_a, grid_b, grid_step, params: MatchParams = MatchParams()) -> dict:
    """Miary dopasowania łuku p (fragment a) i łuku q (fragment b) po transformacji b -> a."""
    tq = transform(q, theta, translation)
    da, ia = cKDTree(tq).query(p)
    db, _ = cKDTree(p).query(tq)
    all_d = np.r_[da, db]
    tol = params.tolerance_mm
    # Równe odstępy po długości => ułamki punktów ~ ułamki długości.
    ca, cb = float(np.mean(da <= tol)), float(np.mean(db <= tol))
    contact = 0.5 * (ca * arc_length(p)[-1] + cb * arc_length(q)[-1])
    # Kolizja = wnętrze b wchodzące w a GŁĘBIEJ niż tolerancja. Pas tolerancji przy brzegu a
    # to strefa styku (szum konturu, dyskretna siatka) - liczenie go jako kolizji karało
    # każde poprawne połączenie proporcjonalnie do długości przełomu.
    tg = transform(grid_b, theta, translation)
    inside = tg[polygon_path(polygon_a).contains_points(tg)]
    overlap = 0.0
    if len(inside):
        edge = sample_polyline(np.vstack([polygon_a, polygon_a[:1]]), step=tol / 4)
        depth = cKDTree(edge).query(inside)[0]
        overlap = float(np.count_nonzero(depth > tol)) * grid_step ** 2
    va, vb = np.gradient(p, axis=0), np.gradient(tq, axis=0)
    va /= np.maximum(np.linalg.norm(va, axis=1, keepdims=True), 1e-10)
    vb /= np.maximum(np.linalg.norm(vb, axis=1, keepdims=True), 1e-10)
    # q jest w oryginalnym kierunku CCW; styczne stykających się brzegów są przeciwne.
    cosine = np.clip(np.sum(va * vb[ia], axis=1), -1, 1)
    normal_error = np.degrees(np.arccos(-cosine))
    keep = da <= tol
    normals = float(np.median(normal_error[keep])) if np.any(keep) else 180.0
    trimmed = float(np.sqrt(np.mean(np.sort(all_d)[:max(3, int(0.8 * len(all_d)))] ** 2)))
    ratio = overlap / max(min(abs(area(polygon_a)), len(grid_b) * grid_step ** 2), 1e-9)
    score = contact / (1 + trimmed / tol) * np.exp(-normals / params.normal_scale_deg) - params.overlap_penalty * overlap
    if ratio > params.max_collision_ratio or contact < params.min_contact_mm:
        score -= params.reject_penalty
    return {"rmse": float(np.sqrt(np.mean(all_d ** 2))), "trimmedRmse": trimmed,
            "median": float(np.median(all_d)), "p95": float(np.percentile(all_d, 95)),
            "hausdorff": float(all_d.max()), "coverageA": 100 * ca, "coverageB": 100 * cb,
            "length": float(contact), "collisionArea": overlap, "collisionRatio": ratio,
            "normalError": normals, "score": float(score)}


def _turning_signature(points):
    v = np.diff(points, axis=0)
    return np.diff(np.unwrap(np.arctan2(v[:, 1], v[:, 0])))


def fit_arc_pair(arc_a, arc_b, fragment_a, fragment_b, grid_b, params: MatchParams = MatchParams()) -> dict:
    """Najlepsze dopasowanie łuku b do łuku a bez znanych odpowiedników punktów.

    Łuk b jest odwracany (brzegi stykają się "naprzeciw"). Przesuwa okna równej
    długości [mm] po obu łukach, dla każdej pary okien liczy transformację SVD,
    a najlepsze, różne od siebie hipotezy ocenia pełną funkcją quality().
    Nie dostaje żadnej prawdy (sąsiedztwa, póz).
    """
    level = params.spline_level
    pr = sample_polyline(np.asarray(arc_a["smoothLevels"][level]["points"]), step=params.resample_step_mm)
    qr = sample_polyline(np.asarray(arc_b["smoothLevels"][level]["points"]), step=params.resample_step_mm)
    qrev = qr[::-1]
    sp, sq = arc_length(pr), arc_length(qrev)
    shorter = min(sp[-1], sq[-1])
    guesses = []
    for fraction in params.window_fractions:
        length = fraction * shorter
        if length < params.min_contact_mm:
            continue
        starts_a = np.linspace(0, sp[-1] - length, max(1, int(np.ceil((sp[-1] - length) / params.window_step_mm)) + 1))
        starts_b = np.linspace(0, sq[-1] - length, max(1, int(np.ceil((sq[-1] - length) / params.window_step_mm)) + 1))
        for start_a in starts_a:
            a = interpolate_at(pr, sp, np.linspace(start_a, start_a + length, params.window_samples))
            for start_b in starts_b:
                b = interpolate_at(qrev, sq, np.linspace(start_b, start_b + length, params.window_samples))
                theta, tr = rigid_fit(a, b)
                error = float(np.sqrt(np.mean(np.sum((a - transform(b, theta, tr)) ** 2, axis=1))))
                guesses.append((error, theta, tr, a, b, length, float(start_a), float(start_b)))

    base = {"id": arc_a["id"] + "_" + arc_b["id"], "a": fragment_a["id"], "b": fragment_b["id"],
            "arcA": arc_a["id"], "arcB": arc_b["id"], "label": arc_a["id"] + " ↔ " + arc_b["id"],
            "accepted": False, "hypothesesTested": len(guesses)}
    if not guesses:
        # Oba łuki krótsze niż min_contact_mm - para nie może być przyjęta.
        return {**base, "theta": 0.0, "translation": [0.0, 0.0], "matchedA": [], "matchedB": [],
                "evaluationA": pr.tolist(), "evaluationB": qr.tolist(), "hypothesesVerified": 0,
                "metrics": {"score": -params.reject_penalty, "length": 0.0, "collisionRatio": 0.0,
                            "rmse": None, "pairedRmse": None, "p95": None, "coverageA": 0.0,
                            "coverageB": 0.0, "collisionArea": 0.0, "rawRmse": None, "rawP95": None}}

    # Tylko konkurencyjne hipotezy idą do kosztownego liczenia kolizji.
    guesses.sort(key=lambda g: g[0])
    chosen = []
    for g in guesses:
        if not any(abs(angle_difference(g[1], h[1])) < np.deg2rad(params.dedupe_angle_deg)
                   and np.linalg.norm(g[2] - h[2]) < params.dedupe_translation_mm for h in chosen):
            chosen.append(g)
        if len(chosen) >= params.hypotheses_verified:
            break
    evaluated = []
    polygon_a = np.asarray(fragment_a["contour"])
    for error, theta, tr, a, b, length, sa, sb in chosen:
        m = quality(pr, qr, theta, tr, polygon_a, grid_b[0], grid_b[1], params)
        m.update(pairedRmse=error, windowLength=float(length), startA=sa, startBReversed=sb)
        ta, tb = _turning_signature(a), _turning_signature(b)  # informacyjna korelacja Pearsona
        m["correlation"] = float(np.corrcoef(ta, tb)[0, 1]) if min(np.std(ta), np.std(tb)) > 1e-8 else None
        evaluated.append((m["score"], theta, tr, a, b, m))
    _, theta, tr, a, b, metrics = max(evaluated, key=lambda x: x[0])

    # Niezależne odległości na surowym (niewygładzonym) łuku z konturu.
    raw_p = sample_polyline(np.asarray(arc_a["points"]), step=params.resample_step_mm)
    raw_q = transform(sample_polyline(np.asarray(arc_b["points"]), step=params.resample_step_mm), theta, tr)
    raw_d = np.r_[cKDTree(raw_q).query(raw_p)[0], cKDTree(raw_p).query(raw_q)[0]]
    metrics["rawRmse"] = float(np.sqrt(np.mean(raw_d ** 2)))
    metrics["rawP95"] = float(np.percentile(raw_d, 95))
    return {**base, "matchedA": a.tolist(), "matchedB": b.tolist(),
            "evaluationA": pr.tolist(), "evaluationB": qr.tolist(),
            "theta": float(theta), "translation": tr.tolist(), "metrics": metrics,
            "hypothesesVerified": len(chosen)}


def find_candidates(fragments: list[dict], params: MatchParams = MatchParams()) -> list[dict]:
    """Dopasowuje każdy łuk z każdym łukiem innego fragmentu. Wynik posortowany malejąco po score."""
    grids = {f["id"]: interior_grid(np.asarray(f["contour"]), params.grid_step_mm) for f in fragments}
    candidates = []
    for ia, fa in enumerate(fragments):
        for fb in fragments[ia + 1:]:
            for aa in fa["arcs"]:
                for ab in fb["arcs"]:
                    candidates.append(fit_arc_pair(aa, ab, fa, fb, grids[fb["id"]], params))
    candidates.sort(key=lambda c: c["metrics"]["score"], reverse=True)
    return candidates


# -----------------
# 4. ODRZUCANIE ZŁYCH POŁĄCZEŃ I ZŁOŻENIE
# -----------------

def is_acceptable(candidate: dict, params: MatchParams = MatchParams()) -> bool:
    """Czy kandydat w ogóle może być połączeniem (niezależnie od innych)."""
    m = candidate["metrics"]
    return (m["score"] >= params.min_score and m["length"] >= params.min_contact_mm
            and m["collisionRatio"] <= params.max_collision_ratio)


def initial_layout(fragment_ids: list[str], fragments: list[dict] | None = None) -> list[dict]:
    """Rozłożenie fragmentów "na stole" w siatce, przed złożeniem (do prezentacji)."""
    cols = max(1, math.ceil(math.sqrt(len(fragment_ids))))
    spacing = 90.0
    if fragments:
        spacing = 1.15 * max(float(np.ptp(np.asarray(f["contour"]), axis=0).max()) for f in fragments)
    return [{"id": i, "theta": 0.0,
             "translation": [spacing * (k % cols - (cols - 1) / 2), -spacing * (k // cols)]}
            for k, i in enumerate(fragment_ids)]


def assemble(fragments: list[dict], candidates: list[dict], params: MatchParams = MatchParams()) -> dict:
    """Zachłanny wybór niekonfliktujących połączeń + złożenie póz w drzewie.

    Kandydaci są przeglądani od najlepszego. Kandydat jest odrzucany, gdy:
    nie spełnia progów (is_acceptable), któryś jego łuk jest już użyty,
    albo oba fragmenty są już w tej samej grupie (cykl). Przyjęte dostają
    candidate["accepted"] = True.

    Returns:
        {"poses" (największej grupy), "components", "initialPoses", "selectedCandidateIds"}
    """
    ids = [f["id"] for f in fragments]
    parent = {i: i for i in ids}

    def find(i):
        while i != parent[i]:
            i = parent[i]
        return i

    used_arcs, selected = set(), []
    for c in sorted(candidates, key=lambda c: c["metrics"]["score"], reverse=True):
        if not is_acceptable(c, params):
            continue
        if c["arcA"] in used_arcs or c["arcB"] in used_arcs or find(c["a"]) == find(c["b"]):
            continue
        parent[find(c["a"])] = find(c["b"])
        used_arcs.update([c["arcA"], c["arcB"]])
        c["accepted"] = True
        selected.append(c)

    components, remaining = [], list(ids)
    while remaining:
        root = remaining[0]
        poses = {root: (0.0, np.zeros(2))}
        changed = True
        while changed:
            changed = False
            for c in selected:
                a, b = c["a"], c["b"]
                t_ab = (c["theta"], np.asarray(c["translation"]))
                if a in poses and b not in poses:
                    poses[b] = compose(poses[a], t_ab)
                    changed = True
                elif b in poses and a not in poses:
                    poses[a] = compose(poses[b], inverse(t_ab))
                    changed = True
        components.append({"id": f"component{len(components) + 1}", "anchor": root, "size": len(poses),
                           "poses": [{"id": i, "theta": float(t[0]), "translation": np.asarray(t[1]).tolist()}
                                     for i, t in poses.items()]})
        remaining = [i for i in remaining if i not in poses]
    components.sort(key=lambda c: -c["size"])
    return {"poses": components[0]["poses"], "anchor": components[0]["anchor"], "components": components,
            "initialPoses": initial_layout(ids, fragments),
            "selectedCandidateIds": [c["id"] for c in selected]}


def assembly_metrics(case: dict, grid_step: float = 0.35) -> dict:
    """Miary złożenia, które NIE wymagają prawdy (działają dla własnych danych)."""
    assembly = case["assembly"]
    poses = {p["id"]: (p["theta"], np.asarray(p["translation"])) for p in assembly["poses"]}
    placed = [transform(f["contour"], *poses[f["id"]]) for f in case["fragments"] if f["id"] in poses]
    bounds = np.vstack(placed)
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    xx, yy = np.meshgrid(np.arange(low[0], high[0] + grid_step, grid_step),
                         np.arange(low[1], high[1] + grid_step, grid_step))
    points = np.c_[xx.ravel(), yy.ravel()]
    count = np.zeros(len(points), dtype=np.uint8)
    for poly in placed:
        count += polygon_path(poly).contains_points(points)
    selected = [c for c in case["candidates"] if c["accepted"]]
    assembly["metrics"] = {
        "components": len(assembly["components"]),
        "largestComponentSize": max(c["size"] for c in assembly["components"]),
        "fragmentsPlaced": len(poses), "selectedEdges": len(selected),
        "collisionArea": float(np.sum(count > 1) * grid_step ** 2), "gridStep": grid_step,
        "meanSelectedScore": float(np.mean([c["metrics"]["score"] for c in selected])) if selected else None,
        "meanSelectedRawRmse": float(np.mean([c["metrics"]["rawRmse"] for c in selected])) if selected else None}
    return assembly["metrics"]


# -----------------
# 5. OCENA WZGLĘDEM PRAWDY (tylko dane z prawdą)
# -----------------

def evaluate_truth(case: dict, truth: dict, grid_step: float = 0.35) -> None:
    """Uzupełnia case o miary względem prawdy. Wywoływać dopiero PO złożeniu.

    Format `truth`: {"outline": kontur całej kości (opcjonalnie),
                     "fragments": [{"id", "theta", "translation"}]  (lokalny -> wspólny układ),
                     "seams": {id łuku: numer przełomu}}  (łuki z tym samym numerem pasują)
    Pozy są porównywane w układzie fragmentu-kotwicy złożenia.
    """
    ftruth = {x["id"]: x for x in truth["fragments"]}
    anchor = case["assembly"].get("anchor", case["fragments"][0]["id"])
    a = ftruth[anchor]
    canonical_to_anchor = inverse((a["theta"], np.array(a["translation"])))
    truth_poses = {}
    for f in case["fragments"]:
        t = ftruth[f["id"]]
        pose = compose(canonical_to_anchor, (t["theta"], np.array(t["translation"])))
        f["poseTruth"] = {"theta": float(pose[0]), "translation": pose[1].tolist()}
        truth_poses[f["id"]] = pose
    seams = truth.get("seams", {})
    for c in case["candidates"]:
        c["correct"] = seams.get(c["arcA"]) is not None and seams.get(c["arcA"]) == seams.get(c["arcB"])
        rel = compose(inverse(truth_poses[c["a"]]), truth_poses[c["b"]])
        c["metrics"]["angleError"] = abs(math.degrees(angle_difference(c["theta"], rel[0])))
        c["metrics"]["translationError"] = float(np.linalg.norm(np.array(c["translation"]) - rel[1]))

    poses = {p["id"]: (p["theta"], np.asarray(p["translation"])) for p in case["assembly"]["poses"]}
    placed = [transform(f["contour"], *poses[f["id"]]) for f in case["fragments"] if f["id"] in poses]
    iou, overlap = None, None
    if truth.get("outline"):
        reference = transform(truth["outline"], *canonical_to_anchor)
        case["assembly"]["referenceContour"] = reference.tolist()
        bounds = np.vstack([reference] + placed)
        low, high = bounds.min(axis=0), bounds.max(axis=0)
        xx, yy = np.meshgrid(np.arange(low[0], high[0] + grid_step, grid_step),
                             np.arange(low[1], high[1] + grid_step, grid_step))
        points = np.c_[xx.ravel(), yy.ravel()]
        ref_inside = polygon_path(reference).contains_points(points)
        count = np.zeros(len(points), dtype=np.uint8)
        for poly in placed:
            count += polygon_path(poly).contains_points(points)
        inside = count > 0
        iou = float(np.count_nonzero(inside & ref_inside) / np.count_nonzero(inside | ref_inside))
        overlap = float(np.sum(count > 1) * grid_step ** 2)

    angle_err, trans_err = [], []
    for fid, pose in poses.items():
        if fid == anchor:
            continue
        target = truth_poses[fid]
        angle_err.append(abs(math.degrees(angle_difference(pose[0], target[0]))))
        trans_err.append(float(np.linalg.norm(pose[1] - target[1])))
    selected = [c for c in case["candidates"] if c["accepted"]]
    n_seams = max(1, len(set(seams.values())))
    connected = len(case["assembly"]["components"]) == 1
    case["assembly"].setdefault("metrics", {}).update({
        "iou": iou if connected else None, "anchoredIou": iou,
        "angleError": float(np.mean(angle_err)) if angle_err else None,
        "translationError": float(np.mean(trans_err)) if trans_err else None,
        "evaluatedFragments": len(angle_err), "components": len(case["assembly"]["components"]),
        "largestComponentSize": max(c["size"] for c in case["assembly"]["components"]),
        "correctEdges": sum(c["correct"] for c in selected), "selectedEdges": len(selected),
        "fragmentsPlaced": len(poses), "collisionArea": overlap, "gridStep": grid_step,
        "edgePrecision": sum(c["correct"] for c in selected) / max(1, len(selected)),
        "edgeRecall": sum(c["correct"] for c in selected) / n_seams})

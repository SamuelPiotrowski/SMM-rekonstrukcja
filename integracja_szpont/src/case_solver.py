"""Solve a single set of fragments and compare the FRACTURA demo with its reference."""

import copy
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import config
import contacts as ct
import matching_surfaces as registration
from bones import Bone, load_bones
from data_parsing import load_mesh, sample_surface, submesh, to_cloud
from fracture_matching import display_mesh
from fracture_library import FRACTURA
from case_inspection import atlas_suggestion, inspect_groups, reference_group
from grid_contacts import grid_edges
from seam_matching import assemble_shared_frame
from osteo_engine_v6 import beam_assemble, moved, rigid

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from assemble_scans import add_caps, assess, candidates


def triangle_keys(mesh):
    points = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    lengths = np.linalg.norm(points[:, [1, 2, 0]] - points[:, [2, 0, 1]], axis=2)
    order = np.argsort(lengths, axis=1)
    sizes = np.take_along_axis(lengths, order, axis=1)
    keys = {}
    for i in np.flatnonzero((sizes[:, 0] > .19) & (np.diff(sizes, axis=1).min(axis=1) > .008)):
        key = tuple(np.rint(sizes[i] / .003).astype(int))
        keys[key] = i if key not in keys else None
    return points, order, keys


def shared_geometry_edges(bones, progress):
    """Find matching triangle geometry when fragments retain the same fracture tessellation."""
    surfaces = {bone.id: triangle_keys(bone.mesh) for bone in bones}
    matches = []
    for a, b in combinations(bones, 2):
        ap, ao, ak = surfaces[a.id]
        bp, bo, bk = surfaces[b.id]
        poses = []
        locations = []
        triangles = []
        center = np.asarray(b.mesh.vertices).mean(0)
        normal_a, normal_b = (np.asarray(bone.mesh.compute_triangle_normals().triangle_normals)
                             for bone in (a, b))
        for key in sorted(ak.keys() & bk.keys()):
            ai, bi = ak[key], bk[key]
            if ai is None or bi is None:
                continue
            pa, pb = ap[ai, ao[ai]], bp[bi, bo[bi]]
            pose = rigid(pa, pb)
            if normal_a[ai] @ (pose[:3, :3] @ normal_b[bi]) > -.85:
                continue
            if np.max(np.linalg.norm(moved(pb, pose) - pa, axis=1)) > .006:
                continue
            poses.append(pose)
            locations.append(pa.mean(0))
            triangles.append((pa, pb))
        if len(poses) < 5:
            continue
        stack = np.stack(poses)
        centers = stack[:, :3, :3] @ center + stack[:, :3, 3]
        support = cKDTree(centers).query_ball_point(centers, .25)
        groups = [[j for j in ids if np.linalg.norm(stack[i, :3, :3] - stack[j, :3, :3]) < .028]
                  for i, ids in enumerate(support)]
        group = max(groups, key=len)
        if len(group) < 5 or np.linalg.norm(np.ptp(np.asarray(locations)[group], axis=0)) < 3:
            continue
        # Refit all vertices of the agreeing triangles, weighting each triangle equally.
        target, source = zip(*(triangles[i] for i in group))
        pose = rigid(np.concatenate(target), np.concatenate(source))
        matches.append(dict(a=a.id, b=b.id, T=pose.tolist(), roi_a=f"shared:{b.id}",
                            roi_b=f"shared:{a.id}", score=1.0, eligible=True,
                            hypothesis_id=f"common-{a.id}-{b.id}", contact=None))
    if matches:
        progress(f"Potwierdzono wspólną geometrię na {len(matches)} parach fragmentów…")
    return matches


def demo_input(model="000001194", family="leg", seed=19):
    import h5py

    path = FRACTURA / "Fractura" / "bone_synthetic.hdf5"
    group = f"{family}/synthetic_fracture/{model}/mode_2/pieces"
    rng = np.random.default_rng(seed)
    meshes, masks, reference = [], {}, {}
    with h5py.File(path, "r") as file:
        for index, (key, piece) in enumerate(file[group].items(), 1):
            vertices, faces = piece["vertices"][:], piece["faces"][:]
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                           o3d.utility.Vector3iVector(faces))
            transform = np.eye(4)
            transform[:3, :3] = Rotation.random(random_state=rng).as_matrix()
            transform[:3, 3] = rng.uniform(-1.3, 1.3, size=3) - transform[:3, :3] @ vertices.mean(0)
            mesh.transform(transform)
            mesh.compute_vertex_normals()
            meshes.append((f"Fragment {index}", mesh))
            masks[str(index)] = piece["shared_faces"][:]
            reference[str(index)] = np.linalg.inv(transform)
    return meshes, masks, reference


def wire_parts(meshes, poses, components, overlays=None, detail=4800, coarse_poses=None, *, preserve_frame=False):
    """Separate disconnected groups in the animation, without changing their local fits."""
    detail = min(detail, max(900, 240_000 // sum(map(len, components))))
    original = {str(i): mesh for i, (_, mesh) in enumerate(meshes, 1)}
    extent = max(np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound()) for mesh in original.values())
    spacing = extent * 1.06
    cursor = 0.0
    parts = []
    for group in components:
        bounds = []
        previews = {}
        for key in group:
            preview = display_mesh(original[key], limit=detail, include_colors=True)
            vertices = np.asarray(preview["vertices"])
            final = moved(vertices, poses[key])
            bounds.append(final)
            previews[key] = (preview, vertices)
        packed = np.concatenate(bounds)
        low, high = packed.min(0), packed.max(0)
        shift = np.array([cursor - low[0], -(low[1] + high[1]) / 2, -(low[2] + high[2]) / 2])
        if preserve_frame:
            shift = np.zeros(3)
        cursor += high[0] - low[0] + spacing * .38
        for key in group:
            preview, vertices = previews[key]
            index = int(key) - 1
            position = len(parts)
            spread = np.array([position % 3 * spacing * .86, -(position // 3) * spacing * .85, 0.])
            spread -= vertices.mean(0)
            matrix = poses[key].copy()
            matrix[:3, 3] += shift - matrix[:3, :3] @ spread
            preview["vertices"] = (vertices + spread).round(6).tolist()
            marks = {name: (np.asarray(points) + spread).round(6).tolist() for name, points
                     in (overlays or {}).get(key, {}).items() if points}
            if coarse_poses is not None:
                coarse = coarse_poses[key].copy()
                coarse[:3, 3] += shift - coarse[:3, :3] @ spread
                marks["coarse_matrix"] = coarse.tolist()
            parts.append(dict(id=key, label=meshes[index][0], **preview, **marks, matrix=matrix.tolist()))
    return parts


def measure_reference(meshes, poses, components, reference):
    group = max(components, key=len)
    root = group[0]
    align = reference[root] @ np.linalg.inv(poses[root])
    extent = np.linalg.norm(np.ptp(np.concatenate([
        moved(np.asarray(meshes[int(key) - 1][1].vertices)[::8], reference[key])
        for key in group
    ]), axis=0))
    errors = []
    for key in group[1:]:
        fitted = align @ poses[key]
        ref = reference[key]
        pts = np.asarray(meshes[int(key) - 1][1].vertices)[::6]
        delta = moved(pts, fitted) - moved(pts, ref)
        rmse = np.sqrt(np.mean(np.sum(delta**2, axis=1)))
        angle = np.degrees(Rotation.from_matrix(fitted[:3, :3] @ ref[:3, :3].T).magnitude())
        errors.append(dict(fragment=meshes[int(key) - 1][0], rmse_percent=float(100 * rmse / extent),
                           angle_deg=float(angle)))
    return errors


def outward_mesh(mesh):
    vertices = np.asarray(mesh.vertices) - mesh.get_center()
    faces = np.asarray(mesh.triangles)
    a, b, c = (vertices[faces[:, column]] for column in range(3))
    volume = np.einsum("ij,ij->", a, np.cross(b, c))
    if volume >= 0 or not mesh.is_watertight():
        return mesh
    outward = copy.deepcopy(mesh)
    outward.triangles = o3d.utility.Vector3iVector(faces[:, [0, 2, 1]])
    outward.compute_triangle_normals().compute_vertex_normals()
    return outward


def mixed_match_failure(result, a, b):
    coverage = min(result["coverage"])
    if coverage < .47:
        return "Za małe pokrycie styku", f"Pokrycie słabszej strony wyniosło {coverage:.0%}. Wymagane jest 47%."
    if result["trimmed_rmse_mm"] > .48:
        return "Za duża szczelina", "Odstęp między powierzchniami przekroczył próg dla trybu mieszanego."
    if result["normal_error_deg"] > 19:
        return "Powierzchnie skierowane pod złym kątem", "Odchylenie normalnych od położenia naprzeciwko siebie przekroczyło 19°."
    if result["overlap_mm2"] > .12 * result["contact_mm2"]:
        return "Fragmenty przenikają się", "Szacowane przenikanie przekracza 12% pola styku."
    if result["contact_mm2"] < .055 * min(a.bone.meta["stats"]["area_mm2"], b.bone.meta["stats"]["area_mm2"]):
        return "Za mały obszar zetknięcia", "Styk obejmuje mniej niż 5,5% powierzchni mniejszego odłamka."
    if not result["eligible"]:
        return "Styk poniżej progu akceptacji", "Połączenie nie spełniło podstawowych warunków pola styku i przenikania."
    return None


def prepare_fragments(meshes, fracture_masks=None):
    if len(meshes) < 2:
        raise ValueError("Do złożenia potrzebne są co najmniej dwa fragmenty.")
    extent = max(np.linalg.norm(m.get_max_bound() - m.get_min_bound()) for _, m in meshes)
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError("Fragmenty mają niepoprawne wymiary.")
    scale = 72 / extent
    bones, work = [], {}
    for i, (label, source) in enumerate(meshes, 1):
        key = str(i)
        mesh = copy.deepcopy(source)
        if len(mesh.triangles) > 42000 and fracture_masks is None:
            mesh = mesh.simplify_quadric_decimation(42000)
        mesh.scale(scale, center=np.zeros(3))
        mesh.compute_vertex_normals()
        samples = sample_surface(mesh, .68, density=1.7)
        cloud = to_cloud(samples.points, samples.normals)
        area = mesh.get_surface_area()
        bones.append(Bone(key, label, mesh, np.empty(0, dtype=np.int32), cloud=cloud,
                          meta={"stats": {"area_mm2": area}}))
        work[key] = (samples.points, samples.normals, [])
    return bones, work, scale


def candidate_surfaces(bones, work, params, fracture_masks=None, *, mixed=False):
    surfaces = []
    detector = config.contact_params()
    for bone in bones:
        if fracture_masks is None:
            annotation = ct.detect_contacts(bone, "roughness", detector)
            labels = ct.triangle_labels(bone.mesh, annotation.labels)
        else:
            annotation = ct.Contacts(bone.id, np.empty(0, dtype=np.uint8), source="dataset")
            labels = fracture_masks[bone.id] + 1
        proposed = []
        for label in np.unique(labels):
            if label <= 0:
                continue
            mesh = submesh(bone.mesh, labels == label)
            area = mesh.get_surface_area()
            if area < params.min_area_mm2:
                continue
            patch = sample_surface(mesh, params.sample_mm)
            proposed.append(registration.Surface(bone, annotation, int(label), patch.points,
                                                 patch.normals, area, cKDTree(work[bone.id][0])))
        proposed.sort(key=lambda region: region.area_mm2, reverse=True)
        surfaces.extend(proposed[:3] if mixed else proposed)
    if fracture_masks is None:
        add_caps(bones, surfaces, params)
    return surfaces


def similar_area(a, b):
    return max(a.area_mm2, b.area_mm2) <= 4 * min(a.area_mm2, b.area_mm2)


def surface_shape(surface):
    spread = np.linalg.eigvalsh(np.cov(surface.points.T)).clip(min=1.e-8)
    return np.r_[np.log(surface.area_mm2), np.log(spread[:-1] / spread[-1])] * [1.3, .35, .6]


def remember_surface_match(evidence, a, b, result, failure):
    key = frozenset((a.bone.id, b.bone.id))
    score = result["score"] / min(a.area_mm2, b.area_mm2)
    if key not in evidence or score > evidence[key][0]:
        evidence[key] = (score, failure)


def match_params(open3d_only=False):
    return registration.SurfaceMatchParams(min_area_mm2=12., sample_mm=.48, coarse_voxel_mm=1.35,
                                           fpfh_radius_mm=5., tolerance_mm=.85, icp_start_mm=3.,
                                           ransac_iterations=320000 if open3d_only else 70000,
                                           ransac_seeds=3 if open3d_only else 2)


def solve_case(meshes, progress=lambda message: None, fracture_masks=None, mixed=False, open3d_only=False,
               surface_evidence=None):
    bones, work, scale = prepare_fragments(meshes, fracture_masks)
    p = match_params(open3d_only)
    exact = []
    unassigned = bones
    comparisons = 0
    if fracture_masks is None and not open3d_only:
        comparisons = len(bones) * (len(bones) - 1) // 2
        exact = shared_geometry_edges(bones, progress)
        if exact:
            poses, components, links = beam_assemble(work, exact, p.tolerance_mm, unique_roi=True, width=12)
        if mixed or not exact or len(components) > 1:
            grid = grid_edges(meshes, progress, allow_partial=mixed)
            if grid:
                progress(f"Sprawdzam położenia wyznaczone na regularnej siatce dla {len(grid)} par…")
                for edge in grid:
                    matrix = np.asarray(edge["T"])
                    matrix[:3, 3] *= scale
                    edge["T"] = matrix.tolist()
                exact.extend(grid)
                poses, components, links = beam_assemble(work, exact, p.tolerance_mm, unique_roi=True, width=12)
        if exact:
            if mixed:
                # Keep strongly established groups separate. Rank remaining independent scans.
                remaining = {group[0] for group in components if len(group) == 1}
                unassigned = [bone for bone in bones if bone.id in remaining]
                exact = links
            if len(components) == 1 or len(unassigned) < 2:
                for pose in poses.values():
                    pose[:3, 3] /= scale
                return poses, components, links, comparisons, p.tolerance_mm / scale
        if not mixed and len(bones) >= 3:
            shared = assemble_shared_frame(bones, work, exact, p.tolerance_mm, progress)
            if shared is not None:
                poses, components, links = shared
                for pose in poses.values():
                    pose[:3, 3] /= scale
                for edge in links:
                    if edge["hypothesis_id"].startswith("near-"):
                        edge["contact"] /= scale ** 2
                return poses, components, links, comparisons, p.tolerance_mm / scale
    progress("Szukam powierzchni przełomu…")
    surfaces = candidate_surfaces(unassigned, work, p, fracture_masks, mixed=mixed)
    pairs = [(a, b) for a, b in combinations(surfaces, 2)
             if a.bone.id != b.bone.id and similar_area(a, b)]
    if mixed:
        # Prefer similar-sized and similarly proportioned patches before registration.
        shapes = {surface.key: surface_shape(surface) for surface in surfaces}
        ranked = sorted(pairs, key=lambda pair: np.linalg.norm(
            shapes[pair[0].key] - shapes[pair[1].key]))
        count = {}
        pairs = []
        for a, b in ranked:
            if max(count.get(a.key, 0), count.get(b.key, 0)) >= 4:
                continue
            pairs.append((a, b))
            for surface in (a, b):
                count[surface.key] = count.get(surface.key, 0) + 1
    edges = exact.copy()
    for i, (a, b) in enumerate(pairs, 1):
        progress(f"Dopasowuję powierzchnie {i}/{len(pairs)}…")
        coarse = {}
        if open3d_only:
            match = registration.match_pair(a, b, p)
            if not match["coarse_fitness"]:
                continue
            coarse = {"coarse_T": match["coarse_T"]}
            result = assess(a, b, np.asarray(match["T"]), p)
        else:
            result = candidates(a, b, p)
        failure = mixed_match_failure(result, a, b) if mixed else None
        if surface_evidence is not None:
            remember_surface_match(surface_evidence, a, b, result, failure)
        if not result["eligible"] or failure is not None:
            continue
        edges.append(dict(a=b.bone.id, b=a.bone.id, T=result["matrix"], roi_a=b.sid, roi_b=a.sid,
                          score=(result["score"] / min(a.area_mm2, b.area_mm2)),
                          eligible=True, hypothesis_id=str(i), contact=result["contact_mm2"] / scale**2,
                          **coarse))
    progress("Sprawdzam zgodność całego złożenia…")
    poses, components, links = beam_assemble(work, edges, p.tolerance_mm, unique_roi=True, width=12)
    for pose in poses.values():
        pose[:3, 3] /= scale
    if open3d_only:
        for edge in links:
            matrix = np.asarray(edge["coarse_T"])
            matrix[:3, 3] /= scale
            edge["coarse_T"] = matrix
    return poses, components, links, comparisons + len(pairs), p.tolerance_mm / scale


def coarse_tree(poses, components, links):
    neighbours = {key: [] for key in poses}
    for link in links:
        a, b, transform = link["a"], link["b"], link["coarse_T"]
        neighbours[a].append((b, transform))
        neighbours[b].append((a, np.linalg.inv(transform)))
    result = {}
    for group in components:
        root = group[0]
        result[root] = poses[root].copy()
        stack = [root]
        while stack:
            key = stack.pop()
            for other, transform in neighbours[key]:
                if other not in result:
                    result[other] = result[key] @ transform
                    stack.append(other)
    return result


def compare_excluded_surfaces(meshes, group, progress):
    """Explain rejections without changing the fitted assembly."""
    p = match_params()
    bones, work, _ = prepare_fragments(meshes)
    surfaces = candidate_surfaces(bones, work, p, mixed=True)
    shapes = {surface.key: surface_shape(surface) for surface in surfaces}
    target = [surface for surface in surfaces if surface.bone.id in group]
    trials = []
    for bone in bones:
        if bone.id in group:
            continue
        nearest = {}
        for a in (surface for surface in surfaces if surface.bone.id == bone.id):
            for b in target:
                if not similar_area(a, b):
                    continue
                distance = np.linalg.norm(shapes[a.key] - shapes[b.key])
                if b.bone.id not in nearest or distance < nearest[b.bone.id][0]:
                    nearest[b.bone.id] = (distance, a, b)
        trials.extend((a, b) for _, a, b in sorted(nearest.values(), key=lambda row: row[0])[:4])
    evidence = {}
    for i, (a, b) in enumerate(trials, 1):
        progress(f"Sprawdzam przełomy odłożonych odłamków {i}/{len(trials)}…")
        result = candidates(a, b, p)
        remember_surface_match(evidence, a, b, result, mixed_match_failure(result, a, b))
    return evidence, len(trials)


def target_exclusions(meshes, group, links, evidence):
    labels = {str(i): label for i, (label, _) in enumerate(meshes, 1)}
    excluded = []
    for key, label in labels.items():
        if key in group:
            continue
        neighbors = sorted({edge["b"] if edge["a"] == key else edge["a"]
                            for edge in links if key in (edge["a"], edge["b"])}, key=int)
        tested = [(other, evidence[frozenset((key, other))]) for other in group
                  if frozenset((key, other)) in evidence]
        fitted = ", ".join(labels[other] for other in neighbors)
        if tested:
            other, (_, failure) = max(tested, key=lambda item: item[1][0])
            if failure is None:
                reason, detail = ("Brak potwierdzonego połączenia w złożeniu",
                                  "Styk przeszedł ocenę lokalną. W końcowym złożeniu fragment pozostał poza wybraną grupą.")
            else:
                reason, detail = failure
            detail = f"Najlepsza zbadana para z grupą wybranej kości to {labels[other]}. {detail}"
            if fitted:
                detail += f" W osobnej grupie dopasowano go z {fitted}."
        elif fitted:
            reason = "Potwierdzony styk w osobnej grupie"
            detail = f"Zgodną geometrię styku znaleziono z {fitted}. Brak potwierdzonego połączenia z grupą odniesienia."
        else:
            reason = "Brak potwierdzonego styku"
            detail = "Sprawdzone ustawienia nie dały wiarygodnego połączenia z grupą odniesienia."
        excluded.append({"id": key, "label": label, "reason": reason, "detail": detail})
    return excluded


def make_result(title, meshes, reference=None, masks=None, progress=lambda message: None, mixed=False,
                open3d_only=False, upload=False, target_anchor=None, identify_bones=True, target_template=None):
    target = target_anchor is not None or target_template is not None
    if target_anchor is not None and not 1 <= target_anchor <= len(meshes):
        raise ValueError("Fragment odniesienia musi należeć do wgranego zestawu.")
    mixed = mixed or target
    evidence = {} if target_anchor is not None else None
    if upload:
        meshes = [(label, outward_mesh(mesh)) for label, mesh in meshes]
    poses, components, links, pair_count, tolerance = solve_case(meshes, progress, fracture_masks=masks,
                                                               mixed=mixed, open3d_only=open3d_only,
                                                               surface_evidence=evidence)
    anchor = str(target_anchor) if target_anchor is not None else None
    inspected = components
    if target_template is not None:
        target_name, template = target_template
        progress("Porównuję złożone grupy z pełnym skanem odniesienia…")
        inspected = [reference_group(meshes, poses, components, template)]
        evidence, additional = compare_excluded_surfaces(meshes, inspected[0], progress)
        pair_count += additional
    elif anchor:
        target_name = meshes[target_anchor - 1][0]
        inspected = [next(group for group in components if anchor in group)]
    largest = max(map(len, inspected))
    progress("Sprawdzam styk i możliwe przenikanie w końcowych położeniach…")
    diagnostics, overlays = inspect_groups(meshes, poses, inspected, tolerance)
    sections = []
    atlas = None
    if target:
        group = inspected[0]
        excluded = target_exclusions(meshes, group, links, evidence)
        sections.append({"parts": wire_parts(meshes, poses, inspected, overlays, detail=11500),
                         "ids": list(group), "diagnostics": diagnostics[0], "atlas": None})
        if excluded:
            loose = [(entry["id"],) for entry in excluded]
            neutral = {key: np.eye(4) for key in poses}
            sections.append({"parts": wire_parts(meshes, neutral, loose, detail=6500),
                             "ids": [entry["id"] for entry in excluded], "diagnostics": [],
                             "unassigned": True, "atlas": None})
    elif mixed:
        for group, pairs in zip(components, diagnostics):
            if len(group) == 1:
                continue
            classification = None
            if identify_bones:
                progress("Porównuję złożoną grupę z atlasem kości…")
                classification = atlas_suggestion(meshes, poses, group)
            progress("Przygotowuję animację złożonej grupy…")
            sections.append({"parts": wire_parts(meshes, poses, [group], overlays, detail=11500),
                             "ids": list(group), "diagnostics": pairs, "atlas": classification})
        unmatched = [group for group in components if len(group) == 1]
        if unmatched:
            progress("Przygotowuję podgląd nieprzypisanych fragmentów…")
            sections.append({"parts": wire_parts(meshes, poses, unmatched, detail=6500),
                             "ids": [group[0] for group in unmatched], "unassigned": True,
                             "diagnostics": [], "atlas": None})
    elif upload and identify_bones and len(components) == 1:
        progress("Porównuję złożoną kość z atlasem…")
        atlas = atlas_suggestion(meshes, poses, components[0])
    coarse = coarse_tree(poses, components, links) if open3d_only else None
    detail = 24000 if upload else 11500
    report = {
        "title": title, "parts": [] if mixed else wire_parts(meshes, poses, components, overlays,
                                                           detail=detail, coarse_poses=coarse),
        "total": len(meshes), "placed": largest, "groups": len(components), "pairs": pair_count,
        "sections": sections, "diagnostics": [pair for group in diagnostics for pair in group],
        "tolerance": tolerance, "mixed": mixed, "atlas": atlas,
        "shared_frame_fit": any(edge["hypothesis_id"].startswith("near-") for edge in links),
        "poses": {key: pose.tolist() for key, pose in poses.items()},
        "links": [{"a": meshes[int(edge["a"]) - 1][0], "b": meshes[int(edge["b"]) - 1][0],
                   "ids": [edge["a"], edge["b"]], "method": edge["hypothesis_id"].split("-")[0]
                   if "-" in edge["hypothesis_id"] else "surface",
                   "contact": edge["contact"]} for edge in links],
        "reference_errors": measure_reference(meshes, poses, components, reference) if reference else None,
        "labelled_surfaces": masks is not None,
        "open3d_stages": open3d_only,
        "coarse_reference_errors": measure_reference(meshes, coarse, components, reference)
        if coarse is not None and reference else None,
    }
    if target:
        from assembly_layout import combined_layout

        report["target"] = {"anchor": anchor, "label": target_name, "excluded": excluded}
        if target_template is not None:
            report["target"]["basis"] = "Pełny skan odniesienia"
            report["bone_type"] = {"name": target_name, "source": "Zgodność z pełnym skanem odniesienia"}
            sections[0]["bone_type"] = report["bone_type"]
        report["combined_layout"] = combined_layout(sections)
    return report


def team_selection(directory):
    saved = json.loads((directory / "assembly-manual.json").read_text(encoding="utf-8"))["results"]
    if not saved:
        raise ValueError("Brak zapisanego dopasowania zespołu.")
    return saved


def team_scan_files(directory, bone_id):
    return sorted(path for path in (directory / "bones" / bone_id / "scans").glob("*.ply")
                  if not path.stem.endswith("_noise"))


def scan_payload(mesh, limit=None):
    return display_mesh(mesh, limit=limit, include_colors=True)


def team_scan_preview(directory, part_id, scan_id, full_detail=False):
    if part_id not in ("1", "2") or not scan_id.isdecimal() or len(scan_id) > 4:
        raise KeyError("ujęcie")
    selection = team_selection(directory)[0]
    key = "a" if part_id == "1" else "b"
    files = team_scan_files(directory, selection[key])
    index = int(scan_id)
    if index >= len(files):
        raise KeyError("ujęcie")
    return scan_payload(load_mesh(files[index]), None if full_detail else 16000)


def team_result(directory):
    saved = team_selection(directory)
    selected = saved[0]
    meshes = {bone.id: bone for bone in load_bones(directory)}
    a, b = (meshes[selected[key]] for key in ("a", "b"))
    parts = [(a.label, a.mesh), (b.label, b.mesh)]
    poses = {"1": np.asarray(selected["matrix"]), "2": np.eye(4)}
    group = [("1", "2")]
    summaries, overlays = inspect_groups(parts, poses, group, .5)
    preview = wire_parts(parts, poses, group, overlays, detail=11500)
    for fragment, bone in zip(preview, (a, b)):
        fragment["source_scans"] = [{"label": path.name, "url": f"/api/team-scans/{fragment['id']}/{i}"}
                                    for i, path in enumerate(team_scan_files(directory, bone.id))]
    return {"title": "Dopasowanie naszych skanów mm", "parts": preview,
            "total": 2, "placed": 2, "groups": 1, "pairs": len(saved),
            "diagnostics": summaries[0], "labelled_surfaces": True, "source_label": "zespołu"}

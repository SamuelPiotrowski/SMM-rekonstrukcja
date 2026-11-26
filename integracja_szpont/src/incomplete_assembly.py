"""Place partial scans that share exterior triangles with a complete reference."""

import numpy as np
from scipy.spatial import cKDTree

from case_solver import (compare_excluded_surfaces, prepare_fragments, shared_geometry_edges,
                         target_exclusions, wire_parts)
from assembly_layout import combined_layout
from case_inspection import inspect_groups
from osteo_engine_v6 import moved, rigid


def triangle_landmarks(mesh):
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    lengths = np.linalg.norm(triangles[:, [1, 2, 0]] - triangles[:, [2, 0, 1]], axis=2)
    order = lengths.argsort(axis=1)
    lengths = np.take_along_axis(lengths, order, axis=1)
    keep = (lengths[:, 0] > .06) & (np.diff(lengths, axis=1).min(axis=1) > .003)
    return lengths[keep], np.take_along_axis(triangles[keep], order[keep, :, None], axis=1)


def locate_fragment(mesh, tree, template_triangles):
    lengths, triangles = triangle_landmarks(mesh)
    sample = np.random.default_rng(37).choice(len(lengths), min(900, len(lengths)), replace=False)
    distances, matches = tree.query(lengths[sample], k=2, distance_upper_bound=.0002)
    unique = np.isfinite(distances[:, 0]) & ~np.isfinite(distances[:, 1])
    source = triangles[sample[unique]]
    target = template_triangles[matches[unique, 0]]
    minimum = max(20, int(.03 * len(sample)))
    if len(source) < minimum:
        return None
    poses = np.stack([rigid(a, b) for a, b in zip(target, source)])
    rotations = poses[:, :3, :3]
    centroid = np.asarray(mesh.vertices).mean(axis=0)
    centers = rotations @ centroid + poses[:, :3, 3]
    parameters = np.c_[centers / .35, rotations.reshape(-1, 9) / .025]
    consensus = cKDTree(parameters)
    best = consensus.query_ball_point(parameters, 1., return_length=True).argmax()
    links = consensus.query_ball_point(parameters[best], 1.)
    if len(links) < minimum or np.linalg.norm(np.ptp(target[links].mean(axis=1), axis=0)) < 3.:
        return None
    return rigid(target[links].reshape(-1, 3), source[links].reshape(-1, 3))


def find_interior_gap(template, meshes, poses):
    points = np.asarray(template.vertices)
    center = points.mean(axis=0)
    delta = points - center
    axis = np.linalg.eigh(delta.T @ delta)[1][:, -1]
    along = delta @ axis
    length = np.ptp(along)
    occupied = []
    for key, pose in poses.items():
        vertices = np.asarray(meshes[int(key) - 1][1].vertices)
        projection = (moved(vertices, pose) - center) @ axis
        occupied.append((projection.min(), projection.max()))
    occupied.sort()
    end = occupied[0][1]
    gaps = []
    for lo, hi in occupied[1:]:
        if lo - end > max(3., .04 * length):
            gaps.append((end, lo))
        end = max(end, hi)
    if not gaps:
        return None
    lo, hi = max(gaps, key=lambda pair: pair[1] - pair[0])
    markers = []
    for at in np.linspace(lo, hi, 3):
        near = np.abs(along - at) < max(1., .012 * length)
        local_center = points[near].mean(axis=0) if np.any(near) else center
        markers.append(local_center + (at - (local_center - center) @ axis) * axis)
    return {"length": float(hi - lo), "points": np.asarray(markers).tolist()}


def incomplete_result(title, meshes, template, progress=lambda message: None):
    target_lengths, target_triangles = triangle_landmarks(template)
    tree = cKDTree(target_lengths)
    poses = {}
    for index, (label, mesh) in enumerate(meshes, 1):
        progress(f"Porównuję zewnętrzną powierzchnię {label} z pełnym skanem…")
        pose = locate_fragment(mesh, tree, target_triangles)
        if pose is not None:
            poses[str(index)] = pose
    if len(poses) < 2:
        raise ValueError("Za mało wiarygodnych pozycji w pełnym skanie do oceny ubytku.")
    group = tuple(poses)
    gap = find_interior_gap(template, meshes, poses)
    if gap is None:
        raise ValueError("Brak potwierdzonej luki między dopasowanymi fragmentami.")
    bones, _, scale = prepare_fragments(meshes)
    potential = shared_geometry_edges([bone for bone in bones if bone.id in poses], progress)
    links = []
    for edge in potential:
        a, b = edge["a"], edge["b"]
        local = np.asarray(edge["T"])
        local[:3, 3] /= scale
        expected = np.linalg.inv(poses[a]) @ poses[b]
        if np.linalg.norm(local[:3, 3] - expected[:3, 3]) < .5 and np.linalg.norm(local[:3, :3] - expected[:3, :3]) < .025:
            links.append(edge)
    evidence, trials = compare_excluded_surfaces(meshes, group, progress)
    rejected = target_exclusions(meshes, group, [], evidence)
    tolerance = .85 / scale
    diagnostics, overlays = inspect_groups(meshes, poses, [group], tolerance)
    bone_type = {"name": "Kość strzałkowa", "source": "Zgodność z pełnym skanem odniesienia"}
    fitted = wire_parts(meshes, poses, [group], overlays, detail=11500, preserve_frame=True)
    sections = [{"parts": fitted, "ids": list(group), "diagnostics": diagnostics[0], "bone_type": bone_type}]
    if rejected:
        parked = [(item["id"],) for item in rejected]
        neutral = {item["id"]: np.eye(4) for item in rejected}
        sections.append({"parts": wire_parts(meshes, neutral, parked, detail=6500), "unassigned": True,
                         "ids": list(neutral), "diagnostics": []})
    layout = combined_layout(sections)
    anchor = fitted[0]
    placement = layout[anchor["id"]]
    offset = np.eye(4)
    offset[:3, 3] = placement["offset"]
    display = np.asarray(placement["matrix"]) @ offset @ np.linalg.inv(anchor["matrix"])
    gap["combined_points"] = moved(np.asarray(gap["points"]), display).tolist()
    gap["contacts"] = len(links)
    return {
        "title": title, "mixed": True, "total": len(meshes), "placed": len(group),
        "sections": sections, "combined_layout": layout, "gap": gap, "diagnostics": diagnostics[0],
        "tolerance": tolerance, "pairs": len(meshes) + trials,
        "target": {"label": bone_type["name"], "basis": "Pełny skan odniesienia", "excluded": rejected},
        "poses": {key: pose.tolist() for key, pose in poses.items()},
        "links": [[edge["a"], edge["b"]] for edge in links],
    }

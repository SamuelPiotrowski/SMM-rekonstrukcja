"""Additional fracture candidates for fragments close in their input frame."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr

from bones import Bone
from fracture_matching import sample_surface
from osteo_engine_v6 import beam_assemble, moved, refine_contact


@dataclass
class Fragment:
    bone: Bone
    points: np.ndarray
    normals: np.ndarray
    tree: cKDTree
    area: float
    center: np.ndarray
    ridge_points: np.ndarray
    ridge_tangents: np.ndarray
    ridge_normals: np.ndarray
    ridge_tree: cKDTree


def ridges(mesh):
    """Midpoints and tangents of edges at a change in surface orientation."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.compute_triangle_normals().triangle_normals)
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    keys = edges[:, 0].astype(np.int64) * len(vertices) + edges[:, 1]
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    adjacent = np.flatnonzero(keys[1:] == keys[:-1])
    first, second = order[adjacent], order[adjacent + 1]
    cosine = np.einsum("ij,ij->i", normals[first // 3], normals[second // 3])
    selected = np.flatnonzero(cosine < .83)
    if len(selected) > 2400:
        selected = selected[np.linspace(0, len(selected) - 1, 2400, dtype=int)]
    detected = edges[first[selected]]
    endpoints = vertices[detected]
    tangents = endpoints[:, 1] - endpoints[:, 0]
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1.e-12)
    face_normals = np.stack((normals[first[selected] // 3], normals[second[selected] // 3]), axis=1)
    return endpoints.mean(axis=1), tangents, face_normals


def prepare(bone):
    count = int(np.clip(bone.mesh.get_surface_area() * 2.6, 1800, 6200))
    points, normals, area = sample_surface(bone.mesh, count, np.random.default_rng(int(bone.id)))
    edge_points, tangents, face_normals = ridges(bone.mesh)
    return Fragment(bone, points, normals, cKDTree(points), area, bone.mesh.get_center(),
                    edge_points, tangents, face_normals, cKDTree(edge_points))


def distances(source, target, transform):
    points = moved(source.points, transform)
    normals = source.normals @ transform[:3, :3].T
    gap, nearest = target.tree.query(points)
    other_normals = target.normals[nearest]
    opposition = -np.einsum("ij,ij->i", normals, other_normals)
    signed = np.einsum("ij,ij->i", points - target.points[nearest], other_normals)
    return gap, opposition, signed


def ridge_support(a, b, matrix, tolerance):
    if not (len(a.ridge_points) and len(b.ridge_points)):
        return 0.
    supports = []
    for source, target, pose in ((b, a, matrix), (a, b, np.linalg.inv(matrix))):
        points = moved(source.ridge_points, pose)
        gaps, indices = target.ridge_tree.query(points, k=list(range(1, min(3, len(target.ridge_points)) + 1)))
        tangent = source.ridge_tangents @ pose[:3, :3].T
        alignment = np.abs(np.einsum("ij,ikj->ik", tangent, target.ridge_tangents[indices]))
        normals = source.ridge_normals @ pose[:3, :3].T
        cosine = np.einsum("ijv,iklv->ikjl", normals, target.ridge_normals[indices])
        # A seam can have opposite fracture normals and a continuous outer cortex.
        interface = np.maximum.reduce([
            np.minimum(cosine[:, :, 0, 0], -cosine[:, :, 1, 1]),
            np.minimum(cosine[:, :, 1, 1], -cosine[:, :, 0, 0]),
            np.minimum(cosine[:, :, 0, 1], -cosine[:, :, 1, 0]),
            np.minimum(cosine[:, :, 1, 0], -cosine[:, :, 0, 1]),
        ])
        matches = (gaps < tolerance) & (alignment > .75) & (interface > .55)
        supports.append(np.count_nonzero(matches.any(axis=1)))
    return min(supports) / min(len(a.ridge_points), len(b.ridge_points))


def score_pose(a, b, matrix, tolerance, nearby_area):
    ab = distances(a, b, np.linalg.inv(matrix))
    ba = distances(b, a, matrix)
    contact = []
    overlap = []
    for source, (gap, opposition, signed) in ((a, ab), (b, ba)):
        good = (gap < tolerance) & (opposition > .50)
        weights = np.maximum(0, 1 - (gap[good] / tolerance) ** 2)
        contact.append(source.area * weights.sum() / len(gap))
        overlap.append(np.mean((gap < 3 * tolerance) & (signed < -.85 * tolerance)))
    smaller = min(a.area, b.area)
    area = min(contact)
    if area < max(1.0, .003 * smaller, .10 * nearby_area) or max(overlap) > .08:
        return None
    motion = np.linalg.norm(moved(b.center, matrix) - b.center) / tolerance
    angle = Rotation.from_matrix(matrix[:3, :3]).magnitude()
    seam = ridge_support(a, b, matrix, tolerance)
    value = area / smaller + .30 * seam - .70 * max(overlap) - .025 * motion - .085 * angle
    return value, area


def pair_candidates(a, b, tolerance):
    if a.area < b.area:
        a, b = b, a
    identity = np.eye(4)
    ab = distances(a, b, identity)
    ba = distances(b, a, identity)
    nearby = [source.area * np.mean((gap < 2.7 * tolerance) & (opposition > .38))
              for source, (gap, opposition, _) in ((a, ab), (b, ba))]
    nearby_area = min(nearby)
    if nearby_area < max(1.3, .007 * b.area):
        return []
    # Keep the shared frame as a hypothesis and refine competing rotations locally.
    comparable = b.area > .36 * a.area
    angles = (-.32, .32) if comparable else (-1.57, -.65, .65, 1.57, np.pi)
    starts = [identity]
    for axis in np.eye(3):
        for angle in angles:
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            pose[:3, 3] = b.center - pose[:3, :3] @ b.center
            starts.append(pose)
    patch_a = ab[0] < 5. * tolerance
    radius = np.max(np.linalg.norm(b.points - b.center, axis=1))
    bound = max(2.5 * tolerance, min(8. * tolerance, .45 * radius))
    fits = []
    for start in starts:
        fit = refine_contact(a.points[patch_a], a.normals[patch_a], b.points, b.normals, start,
                             1.45 * tolerance, steps=14)
        for pose in (start, fit):
            shift = np.linalg.norm(moved(b.center, pose) - b.center)
            angle = Rotation.from_matrix(pose[:3, :3]).magnitude()
            if shift > bound or (comparable and angle > np.pi / 4):
                continue
            measured = score_pose(a, b, pose, tolerance, nearby_area)
            if measured is not None:
                fits.append((measured[0], pose, measured[1]))
    fits.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for value, pose, area in fits:
        if any(np.linalg.norm(matrix[:3, :3] - pose[:3, :3]) < .20 and
               np.linalg.norm(moved(b.center, matrix) - moved(b.center, pose)) < .45 * tolerance
               for _, matrix, _ in selected):
            continue
        selected.append((value, pose, area))
        if len(selected) == 2:
            break
    return [(a.bone.id, b.bone.id, value, pose, area) for value, pose, area in selected]


def nearby_edges(fragments, tolerance, progress):
    areas = {fragment.bone.id: fragment.area for fragment in fragments}
    largest_area = max(areas.values())
    pairs = list(combinations(fragments, 2))
    edges = []
    for i, (a, b) in enumerate(pairs, 1):
        gap = np.maximum(a.bone.mesh.get_min_bound() - b.bone.mesh.get_max_bound(),
                         b.bone.mesh.get_min_bound() - a.bone.mesh.get_max_bound())
        if np.linalg.norm(np.maximum(gap, 0)) > 2.7 * tolerance:
            continue
        progress(f"Sprawdzam krawędzie sąsiednich fragmentów {i}/{len(pairs)}…")
        for rank, (target, source, value, pose, area) in enumerate(pair_candidates(a, b, tolerance)):
            anchor_weight = np.sqrt(min(areas[target], areas[source]) / largest_area)
            edges.append(dict(a=target, b=source, T=pose.tolist(), roi_a=f"near:{source}", roi_b=f"near:{target}",
                              score=.58 + .30 * anchor_weight + value, eligible=True, contact=area,
                              hypothesis_id=f"near-{target}-{source}-{rank}"))
    return edges


def shared_frame_cost(bones, tolerance):
    areas = {bone.id: bone.mesh.get_surface_area() for bone in bones}
    centers = {bone.id: bone.mesh.get_center() for bone in bones}
    largest = max(areas.values())

    def penalty(group, poses):
        anchor = max(group, key=areas.get)
        frame = np.linalg.inv(poses[anchor])
        total = 0.
        for key in group:
            if key == anchor:
                continue
            pose = frame @ poses[key]
            drift = np.linalg.norm(moved(centers[key], pose) - centers[key]) / tolerance
            rotation = Rotation.from_matrix(pose[:3, :3]).magnitude()
            weight = np.sqrt(areas[key] / largest)
            total += weight * (.06 * max(drift - 2, 0) + .14 * rotation)
        return total

    return penalty


def solid_checks(bones, work, tolerance):
    """Check sampled deep overlap on closed input meshes using signed distance."""
    if not all(bone.mesh.is_edge_manifold(allow_boundary_edges=False) for bone in bones):
        return None, None
    scenes = {}
    for bone in bones:
        scene = o3d.t.geometry.RaycastingScene(nthreads=2)
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(bone.mesh))
        scenes[bone.id] = scene
    samples = {key: points[np.linspace(0, len(points) - 1, min(1500, len(points)), dtype=int)]
               for key, (points, _, _) in work.items()}

    def deep_overlap(poses):
        inverses = {key: np.linalg.inv(pose) for key, pose in poses.items()}
        world = {key: moved(points, poses[key]) for key, points in samples.items()}
        bounds = {key: (points.min(0), points.max(0)) for key, points in world.items()}
        for a, b in combinations(world, 2):
            low_a, high_a = bounds[a]
            low_b, high_b = bounds[b]
            if np.any(high_a + tolerance < low_b) or np.any(high_b + tolerance < low_a):
                continue
            for source, target in ((a, b), (b, a)):
                point = o3d.core.Tensor(moved(world[source], inverses[target]).astype(np.float32))
                signed = scenes[target].compute_signed_distance(point, nthreads=2, nsamples=3).numpy()
                if np.mean(signed < -tolerance) > .045:
                    return True
        return False

    def closest(key, points):
        query = o3d.core.Tensor(points.astype(np.float32))
        hit = scenes[key].compute_closest_points(query, nthreads=2)
        return hit["points"].numpy(), hit["primitive_normals"].numpy()

    return deep_overlap, closest


def contacts_in_group(fragments, pairs, poses, tolerance, closest):
    measurements = []
    inverse = {key: np.linalg.inv(pose) for key, pose in poses.items()}
    for a, b in pairs:
        smaller = min(fragments[a].area, fragments[b].area)
        quality, rows = [], []
        for source, target in ((a, b), (b, a)):
            part = fragments[source]
            relative = inverse[target] @ poses[source]
            points = moved(part.points, relative)
            neighbors, target_normals = closest(target, points)
            normals = part.normals @ relative[:3, :3].T
            gap = np.linalg.norm(points - neighbors, axis=1)
            opposing = -np.einsum("ij,ij->i", normals, target_normals)
            keep = (gap <= 1.7 * tolerance) & (opposing > .55)
            weights = opposing[keep] * np.exp(-.5 * (gap[keep] / tolerance) ** 2)
            density = part.area / (smaller * len(points))
            quality.append(density * weights.sum())
            indices = np.flatnonzero(keep)
            if len(indices) < 8:
                continue
            sample = np.linspace(0, len(indices) - 1, min(180, len(indices)), dtype=int)
            ids = indices[sample]
            world = moved(part.points[ids], poses[source])
            surface = moved(neighbors[ids], poses[target])
            direction = target_normals[ids] @ poses[target][:3, :3].T
            weight = np.sqrt(weights[sample] * density * len(indices) / len(sample))
            rows.append((source, target, world, surface, direction, weight))
        measurements.append((min(quality), rows))
    return measurements


def contact_step(measurements, centers, free, translation_only, tolerance):
    columns = {key: 6 * index for index, key in enumerate(free)}
    row_indices, column_indices, coefficients, residuals = [], [], [], []
    for quality, observations in measurements:
        if quality < .004:
            continue
        for source, target, points, surface, normal, weight in observations:
            offset = len(residuals)
            residual = np.einsum("ij,ij->i", points - surface, normal) / tolerance
            residuals.extend(weight * residual)
            for key, vertices, sign in ((source, points, 1), (target, surface, -1)):
                if key not in columns:
                    continue
                # An infinitesimal rigid motion about the fragment center, in units of tolerance.
                jacobian = np.c_[np.cross(vertices - centers[key], normal) / tolerance, normal]
                if key in translation_only:
                    jacobian[:, :3] = 0.
                coefficients.extend((sign * weight[:, None] * jacobian).ravel())
                row_indices.extend(np.repeat(np.arange(offset, offset + len(vertices)), 6))
                column_indices.extend(np.tile(np.arange(columns[key], columns[key] + 6), len(vertices)))
    if not coefficients:
        return None
    matrix = csr_matrix((coefficients, (row_indices, column_indices)), shape=(len(residuals), 6 * len(free)))
    step = lsmr(matrix, -np.array(residuals), damp=.12, maxiter=60)[0].reshape(-1, 6)
    limit = max(1., np.linalg.norm(step[:, :3], axis=1).max() / .09, np.linalg.norm(step[:, 3:], axis=1).max())
    return step / limit


def refine_group(fragments, nearby, links, poses, tolerance, verify, closest, penalty, progress):
    pairs = sorted({tuple(sorted((edge["a"], edge["b"]))) for edge in nearby + links})
    anchor = max(fragments, key=lambda key: fragments[key].area)
    free = [key for key in fragments if key != anchor]
    # Small pieces can have too little contact to constrain a local rotation.
    translation_only = {key for key in free if fragments[key].area < .14 * fragments[anchor].area}
    measurements = contacts_in_group(fragments, pairs, poses, tolerance, closest)
    initial = np.array([value for value, _ in measurements])
    linked = {tuple(sorted((edge["a"], edge["b"]))) for edge in links}
    preserve = np.array([pair in linked for pair in pairs])
    progress("Doprecyzowuję wspólne ułożenie na powierzchniach styku…")
    for _ in range(5):
        centers = {key: moved(part.center, poses[key]) for key, part in fragments.items()}
        step = contact_step(measurements, centers, free, translation_only, tolerance)
        if step is None:
            break
        previous = sum(value for value, _ in measurements) - .12 * penalty(tuple(fragments), poses)
        for fraction in (1., .5, .25):
            trial = dict(poses)
            for key, delta in zip(free, step * fraction):
                rotation = Rotation.from_rotvec(delta[:3]).as_matrix()
                correction = np.eye(4)
                correction[:3, :3] = rotation
                correction[:3, 3] = centers[key] - rotation @ centers[key] + tolerance * delta[3:]
                trial[key] = correction @ poses[key]
            updated = contacts_in_group(fragments, pairs, trial, tolerance, closest)
            scores = np.array([value for value, _ in updated])
            if np.any(scores[preserve] < .82 * initial[preserve]):
                continue
            if scores.sum() - .12 * penalty(tuple(fragments), trial) > previous + 1.e-5 and not verify(trial):
                poses, measurements = trial, updated
                break
        else:
            break
    final_support = {pair: value for pair, (value, _) in zip(pairs, measurements)}
    final_links = []
    for edge in links:
        a, b = edge["a"], edge["b"]
        area = min(fragments[a].area, fragments[b].area)
        final_links.append({**edge, "T": (np.linalg.inv(poses[a]) @ poses[b]).tolist(),
                            "contact": area * final_support[tuple(sorted((a, b)))]})
    return poses, final_links


def assemble_shared_frame(bones, work, exact, tolerance, progress):
    """Use nearby seams only when the input is consistent with a shared scan frame."""
    verify, closest = solid_checks(bones, work, tolerance)
    if verify is None or verify({bone.id: np.eye(4) for bone in bones}):
        return None
    fragments = {bone.id: prepare(bone) for bone in bones}
    nearby = nearby_edges(list(fragments.values()), tolerance, progress)
    if len({tuple(sorted((edge["a"], edge["b"]))) for edge in nearby + exact}) < len(bones) - 1:
        return None
    penalty = shared_frame_cost(bones, tolerance)
    poses, groups, links = beam_assemble(work, exact + nearby, tolerance, unique_roi=True, width=24,
                                        group_cost=penalty,
                                        accept=lambda poses, groups: len(groups) == 1 and not verify(poses))
    if not links:
        return None
    if all(edge["hypothesis_id"].startswith("near-") for edge in links):
        poses, links = refine_group(fragments, nearby, links, poses, tolerance, verify, closest, penalty, progress)
    return poses, groups, links

"""Exact opposing contacts for meshes generated on the same cubic sampling grid."""

from dataclasses import dataclass
from itertools import combinations, permutations, product

import numpy as np
from scipy.signal import fftconvolve


@dataclass
class GridSurface:
    basis: np.ndarray
    origin: np.ndarray
    step: float
    vertices: np.ndarray
    directions: np.ndarray


def grid_surface(mesh):
    points = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    resolution = max(np.ptp(points, axis=0)) * 1.e-6
    if not np.isfinite(resolution) or resolution <= 0 or not mesh.has_vertex_normals():
        return None
    faces = faces[::max(1, len(faces) // 40000)]
    edges = points[faces[:, [1, 2, 0]]] - points[faces]
    edges = edges.reshape(-1, 3)
    lengths = np.linalg.norm(edges, axis=1)
    bins = np.round(lengths / resolution).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    for key in values[np.argsort(counts)[-4:][::-1]]:
        mask = bins == key
        if key == 0 or mask.sum() < .10 * len(edges):
            continue
        step = np.median(lengths[mask])
        directions = edges[mask] / lengths[mask, None]
        x = directions[0]
        perpendicular = directions[np.abs(directions @ x) < 1.e-4]
        if not len(perpendicular):
            continue
        y = perpendicular[0] - x * (x @ perpendicular[0])
        y /= np.linalg.norm(y)
        basis = np.column_stack((x, y, np.cross(x, y)))
        coordinates = (points - points[0]) @ basis * (2 / step)
        rounded = np.rint(coordinates)
        if np.max(np.abs(coordinates - rounded)) > .002:
            continue
        vertices = rounded.astype(np.int32)
        for offset in product((0, 1), repeat=3):
            shifted = vertices + offset
            half_axes = shifted % 2
            # The isosurface crosses grid edges at their midpoints.
            on_edge = half_axes.sum(axis=1) == 1
            if on_edge.mean() < .99 or np.max(np.ptp(shifted, axis=0)) > 400:
                continue
            normals = np.asarray(mesh.vertex_normals)[on_edge] @ basis
            directions = (half_axes[on_edge] * np.where(normals < 0, -1, 1)).astype(np.int8)
            origin = points[0] - basis @ (np.asarray(offset) * (step / 2))
            return GridSurface(basis, origin, step, shifted[on_edge], directions)
    return None


def oriented_cells(vertices, directions):
    cells = vertices // 2
    low = cells.min(axis=0)
    offsets = cells - low
    axes = np.argmax(np.abs(directions), axis=1)
    signs = directions[np.arange(len(axes)), axes]
    grid = np.zeros((*tuple(offsets.max(axis=0) + 1), 3), dtype=np.int8)
    grid[(*offsets.T, axes)] = signs
    return grid, low, cells, axes, signs


def solid_cells(faces):
    """Recover occupied samples when crossings agree in all three directions."""
    shape = np.array(faces.shape[:3]) + 1
    solid = None
    for axis in range(3):
        crossings = np.zeros(shape, dtype=bool)
        region = [slice(0, size) for size in faces.shape[:3]]
        region[axis] = slice(1, None)
        crossings[tuple(region)] = faces[..., axis] != 0
        filled = np.logical_xor.accumulate(crossings, axis=axis)
        if np.any(np.take(filled, -1, axis=axis)):
            return None
        if solid is not None and not np.array_equal(solid, filled):
            return None
        solid = filled
    return solid


def overlapping_solid(target, source, offset):
    if target is None or source is None:
        return False
    start = np.maximum(offset, 0)
    stop = np.minimum(offset + source.shape, target.shape)
    if np.any(stop <= start):
        return False
    a = tuple(slice(lo, hi) for lo, hi in zip(start, stop))
    b = tuple(slice(lo, hi) for lo, hi in zip(start - offset, stop - offset))
    return np.any(target[a] & source[b])


def match_grids(a, b):
    min_vertices = min(len(a.vertices), len(b.vertices))
    if min_vertices < 80 or abs(a.step / b.step - 1) > 1.e-4:
        return None
    target, low_a, _, _, _ = oriented_cells(a.vertices, a.directions)
    target_solid = solid_cells(target)
    # Project the three signed face channels only to shortlist translations.
    # Each shortlisted translation must still match the same faces and opposite signs.
    weights = np.array([1., .92, .84], dtype=np.float32)
    raster_a = target @ weights
    candidates = []
    for permutation in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            rotation = np.eye(3, dtype=np.int32)[list(permutation)] * np.asarray(signs)[:, None]
            if np.linalg.det(rotation) < 0:
                continue
            source, low_b, cells, axes, normal_sign = oriented_cells(
                b.vertices @ rotation.T, b.directions @ rotation.T)
            if np.prod(np.array(target.shape[:3]) + source.shape[:3] - 1) > 6_000_000:
                continue
            source_solid = solid_cells(source) if target_solid is not None else None
            raster_b = source @ weights
            correlation = fftconvolve(raster_a, -raster_b[::-1, ::-1, ::-1])
            peaks = np.argpartition(correlation.ravel(), -4)[-4:]
            for peak in peaks:
                shift = np.array(np.unravel_index(peak, correlation.shape)) - source.shape[:3] + 1 + low_a - low_b
                matched = cells + shift - low_a
                inside = np.all((matched >= 0) & (matched < target.shape[:3]), axis=1)
                indices = np.flatnonzero(inside)
                dots = target[(*matched[indices].T, axes[indices])] * normal_sign[indices]
                support = indices[dots < 0]
                opposing = len(support)
                same = np.count_nonzero(dots > 0)
                if opposing < max(80, .035 * min_vertices) or same > .12 * opposing:
                    continue
                if np.count_nonzero(np.bincount(axes[support], minlength=3) >= .04 * opposing) < 2:
                    continue
                # Opposing faces can also score highly when two solids slide into one another.
                if overlapping_solid(target_solid, source_solid, shift + low_b - low_a):
                    continue
                candidates.append((opposing - 2 * same, rotation, shift))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, rotation, shift = candidates[0]
    if len(candidates) > 1 and candidates[1][0] > .90 * score:
        return None
    matrix = np.eye(4)
    matrix[:3, :3] = a.basis @ rotation @ b.basis.T
    matrix[:3, 3] = a.origin + a.basis @ (shift * a.step) - matrix[:3, :3] @ b.origin
    return matrix, score


def grid_edges(meshes, progress, *, allow_partial=False):
    surfaces = []
    for key, (_, mesh) in enumerate(meshes, 1):
        surface = grid_surface(mesh)
        if surface is None:
            if not allow_partial:
                return []
            continue
        surfaces.append((key, surface))
    edges = []
    pairs = list(combinations(surfaces, 2))
    for index, ((a, target), (b, source)) in enumerate(pairs, 1):
        progress(f"Szukam styków na regularnej siatce {index}/{len(pairs)}…")
        match = match_grids(target, source)
        if match is None:
            continue
        matrix, score = match
        edges.append(dict(a=str(a), b=str(b), T=matrix.tolist(), roi_a=f"grid:{b}", roi_b=f"grid:{a}",
                          score=1 + score / min(len(target.vertices), len(source.vertices)), eligible=True,
                          hypothesis_id=f"grid-{a}-{b}", contact=None))
    return edges

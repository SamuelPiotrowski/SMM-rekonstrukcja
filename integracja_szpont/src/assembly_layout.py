"""Presentation layouts that preserve the solver's rigid transforms."""

import numpy as np

from osteo_engine_v6 import moved


def combined_layout(sections):
    """Keep fitted bones intact and place excluded parts outside the assembly."""
    by_id = {}
    arranged = []
    parked = [part for section in sections if section.get("unassigned") for part in section["parts"]]
    for section in sections:
        if section.get("unassigned"):
            continue
        parts = section["parts"]
        assembled = np.concatenate([moved(np.asarray(part["vertices"]), np.asarray(part["matrix"]))
                                    for part in parts])
        centered = assembled - assembled.mean(axis=0)
        _, basis = np.linalg.eigh(centered.T @ centered)
        x, z = basis[:, -1], basis[:, 1]
        rotation = np.stack((x, np.cross(z, x), z))
        oriented = assembled @ rotation.T
        arranged.append((parts, rotation, oriented.min(axis=0), oriented.max(axis=0)))
        by_id.update((part["id"], np.asarray(part["vertices"])) for part in parts)

    longest = max(high[0] - low[0] for _, _, low, high in arranged)
    separation = .17 * longest
    total_height = sum(high[2] - low[2] for _, _, low, high in arranged) + separation * (len(arranged) - 1)
    cursor = total_height / 2
    final = {}
    for parts, rotation, low, high in arranged:
        group_transform = np.eye(4)
        group_transform[:3, :3] = rotation
        group_transform[:3, 3] = [-(low[0] + high[0]) / 2, -(low[1] + high[1]) / 2, cursor - high[2]]
        cursor -= high[2] - low[2] + separation
        final.update((part["id"], group_transform @ np.asarray(part["matrix"])) for part in parts)

    if parked:
        right = max(moved(vertices, final[key])[:, 0].max() for key, vertices in by_id.items())
        parked_vertices = [np.asarray(part["vertices"]) for part in parked]
        row_height = 1.12 * max(np.ptp(vertices, axis=0)[2] for vertices in parked_vertices)
        for index, (part, vertices) in enumerate(zip(parked, parked_vertices)):
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            key = part["id"]
            final[key] = np.eye(4)
            final[key][:3, 3] = [right + .18 * longest - low[0], -(low[1] + high[1]) / 2,
                                ((len(parked) - 1) / 2 - index) * row_height - (low[2] + high[2]) / 2]
            by_id[key] = vertices

    rng = np.random.default_rng(17)
    ids = rng.permutation(sorted(by_id, key=int))
    extent = max(np.linalg.norm(np.ptp(vertices, axis=0)) for vertices in by_id.values())
    columns = (len(ids) + 1) // 2
    layout = {}
    for index, key in enumerate(ids):
        row, column = divmod(index, columns)
        start = extent * np.array([1.11 * (column - (columns - 1) / 2),
                                    rng.uniform(-.17, .17), 1.09 * (.5 - row)])
        start[[0, 2]] += extent * rng.uniform(-.10, .10, 2)
        offset = start - by_id[key].mean(axis=0)
        undo_offset = np.eye(4)
        undo_offset[:3, 3] = -offset
        layout[key] = {"offset": offset.tolist(), "matrix": (final[key] @ undo_offset).tolist()}
    return layout

"""
Wspólna geometria 2D używana przez wszystkie moduły.

Konwencje (obowiązują w całym projekcie):
    - współrzędne w milimetrach, x w prawo, y w górę,
    - wielokąty i kontury jako tablice (N, 2), kontury zewnętrzne przeciwnie
      do ruchu wskazówek zegara (CCW, dodatnie pole),
    - transformacja sztywna T = (theta, t) działa jako  p' = R(theta) @ p + t,
      bez odbić lustrzanych,
    - kąty w radianach.
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib.path import Path as MplPath


# -----------------
# TRANSFORMACJE SZTYWNE
# -----------------

def rotation(theta: float) -> np.ndarray:
    """Macierz obrotu 2x2 o kąt theta [rad]."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def transform(points, theta: float, translation) -> np.ndarray:
    """Obraca punkty o theta i przesuwa o translation."""
    return np.asarray(points, dtype=float) @ rotation(theta).T + np.asarray(translation, dtype=float)


def compose(t_a, t_ab):
    """Złożenie transformacji. t_a: a -> świat, t_ab: b -> a. Wynik: b -> świat."""
    return (t_a[0] + t_ab[0], rotation(t_a[0]) @ np.asarray(t_ab[1]) + np.asarray(t_a[1]))


def inverse(t):
    """Transformacja odwrotna do (theta, t)."""
    return (-t[0], -rotation(-t[0]) @ np.asarray(t[1]))


def angle_difference(a: float, b: float) -> float:
    """Najkrótsza różnica kątów a - b w zakresie (-pi, pi]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def rigid_fit(target, source):
    """Najlepsza transformacja sztywna source -> target (Kabsch / SVD, bez odbić).

    Obie tablice muszą mieć tyle samo punktów, a punkt i-ty odpowiada i-temu.

    Returns:
        (theta, translation) takie, że transform(source, theta, translation) ~ target.
    """
    target, source = np.asarray(target), np.asarray(source)
    a, b = target.mean(axis=0), source.mean(axis=0)
    u, _, vt = np.linalg.svd((source - b).T @ (target - a))
    d = np.diag([1, np.linalg.det(vt.T @ u.T)])
    r = vt.T @ d @ u.T
    t = a - r @ b
    return math.atan2(r[1, 0], r[0, 0]), t


# -----------------
# WIELOKĄTY I ŁAMANE
# -----------------

def area(polygon) -> float:
    """Pole ze znakiem (wzór sznurowadłowy). Dodatnie dla CCW."""
    p = np.asarray(polygon)
    return float(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)))


def polygon_path(polygon) -> MplPath:
    """Zamknięta ścieżka matplotlib (do testów punkt-w-wielokącie)."""
    p = np.asarray(polygon)
    return MplPath(np.vstack([p, p[0]]), closed=True)


def arc_length(points) -> np.ndarray:
    """Skumulowana długość łamanej; pierwszy element = 0."""
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def interpolate_at(points, s, samples) -> np.ndarray:
    """Punkty łamanej `points` (z długościami `s`) w zadanych długościach `samples`."""
    return np.column_stack([np.interp(samples, s, points[:, j]) for j in range(2)])


def sample_polyline(points, count: int | None = None, step: float = 0.25) -> np.ndarray:
    """Równomiernie (po długości) próbkuje łamaną: `count` punktów albo co `step` mm."""
    p = np.asarray(points, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-8]
    p = p[keep]
    s = arc_length(p)
    q = np.linspace(0, s[-1], count or max(8, int(np.ceil(s[-1] / step)) + 1))
    return interpolate_at(p, s, q)


def resample_closed(contour, count: int) -> np.ndarray:
    """Równomiernie próbkuje zamknięty kontur do `count` punktów (bez powtórzenia pierwszego)."""
    c = np.asarray(contour, dtype=float)
    return sample_polyline(np.vstack([c, c[0]]), count=count + 1)[:-1]


def interior_grid(polygon, step: float = 0.5):
    """Punkty regularnej siatki leżące wewnątrz wielokąta. Zwraca (punkty, krok)."""
    p = np.asarray(polygon)
    low, high = p.min(axis=0), p.max(axis=0)
    xx, yy = np.meshgrid(np.arange(low[0] + step / 2, high[0], step),
                         np.arange(low[1] + step / 2, high[1], step))
    q = np.c_[xx.ravel(), yy.ravel()]
    return q[polygon_path(p).contains_points(q)], step

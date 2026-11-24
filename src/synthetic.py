"""
Etap 1: syntetyczne dane testowe - kość, przełomy (seam), obrazy wejściowe.

Buduje kształt kości długiej (dwie nasady + trzon), tnie go poszarpanymi
przełomami na fragmenty, każdy fragment obraca i renderuje do osobnego
obrazu z szumem. Zapisuje to w tym samym formacie, jaki mają własne dane:

    <folder>/A.png, B.png, ...
    <folder>/annotations.json  - obrazy, skala, końce przełomów (wejście analizy)
    <folder>/truth.json        - prawdziwe położenia i pary przełomów (tylko do oceny)

Dzięki temu dalsza analiza nie odróżnia danych syntetycznych od własnych.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from approximation import mm_to_pixels
from geometry import area, inverse, rotation, sample_polyline, transform
from results import save_json


@dataclass
class SyntheticParams:
    seed: int = 20261124
    mm_per_pixel: float = 0.25
    image_size: int = 512
    length_mm: float = 150.0                         # długość kości
    seam_heights_mm: tuple = (39.0, 76.0, 111.0)     # wysokości przełomów -> len + 1 fragmentów
    seam_points: int = 19                            # liczba "zębów" przełomu
    seam_amplitude_mm: float = 3.7                   # wysokość zębów
    angles_deg: tuple = (24.0, -37.0, 68.0, -19.0)   # obrót każdego fragmentu na zdjęciu
    titles: dict = field(default_factory=lambda: {
        "clean": "01 / Cztery fragmenty: czysty obraz",
        "noise": "02 / Te same fragmenty: szum i rozmycie",
        "loss": "03 / Ubytek na jednym przełomie"})
    descriptions: dict = field(default_factory=lambda: {
        "clean": "Komplementarne, poszarpane brzegi. Dyskretyzacja obrazu powoduje niezerowy błąd nawet dla idealnego modelu.",
        "noise": "Ten sam obiekt i położenia; zwiększony szum oraz rozmycie sprawdzają stabilność segmentacji i splajnów.",
        "loss": "Z fragmentu B usunięto lokalny odprysk. Dopasowanie jest częściowe; szczeliny nie należy usuwać deformacją ani przesuwaniem materiału w kolizję."})


# -----------------
# KSZTAŁT KOŚCI I PRZEŁOMY
# -----------------

def width(y, length: float = 150.0):
    """Połowa szerokości kości na wysokości y: trzon + dwie nasady + lekka falistość."""
    y = np.asarray(y)
    return (11.0 + 13.5 * np.exp(-((y - 0.08 * length) / 12.0) ** 2)
            + 13.0 * np.exp(-((y - 0.92 * length) / 12.0) ** 2) + 1.2 * np.sin(y / 19.0))


def seam(y: float, index: int, params: SyntheticParams = SyntheticParams()) -> np.ndarray:
    """Poszarpany przełom w poprzek kości na wysokości y (od lewej do prawej krawędzi).

    Każdy przełom ma inny losowy "odcisk palca" (seed zależny od index).
    """
    w = float(width(y, params.length_mm))
    rng = np.random.default_rng(params.seed + 31 * index)
    x = np.linspace(-w, w, params.seam_points)
    h = rng.uniform(-params.seam_amplitude_mm, params.seam_amplitude_mm, len(x))
    h[0] = h[-1] = 0
    return np.c_[x, y + h]


def bone_fragments(params: SyntheticParams = SyntheticParams(), loss: bool = False):
    """Wielokąty fragmentów (wspólny układ kości) i kontur całej kości.

    Args:
        loss: usuń lokalny odprysk z dolnego przełomu fragmentu B (test ubytku).

    Returns:
        (lista (wielokąt, [ {"endpoints", "truth_seam"} ]), kontur całej kości)
    """
    length = params.length_mm
    cuts = [seam(y, k + 1, params) for k, y in enumerate(params.seam_heights_mm)]
    bounds = [0.0, *params.seam_heights_mm, length]
    n = len(bounds) - 1
    fragments = []
    for i in range(n):
        y0, y1 = bounds[i], bounds[i + 1]
        bottom = cuts[i - 1].copy() if i else np.array([[-width(0, length), 0], [width(0, length), 0]])
        top = cuts[i].copy() if i < n - 1 else np.array([[-width(length, length), length], [width(length, length), length]])
        if loss and i == 1:
            # Utracony materiał po jednej stronie styku A/B; oryginał zostaje w truth.
            bottom = sample_polyline(bottom, count=190)
            bottom[:, 1] += 2.6 * np.exp(-((bottom[:, 0] - 4.0) / 2.0) ** 2)
        yr = np.linspace(y0, y1, 90)
        right = np.c_[width(yr, length), yr][1:-1]
        left = np.c_[-width(yr[::-1], length), yr[::-1]][1:-1]
        polygon = np.vstack([bottom, right, top[::-1], left])
        rois = []
        if i:
            rois.append({"endpoints": [bottom[0].tolist(), bottom[-1].tolist()], "truth_seam": i})
        if i < n - 1:
            rois.append({"endpoints": [top[-1].tolist(), top[0].tolist()], "truth_seam": i + 1})
        fragments.append((polygon, rois))
    y = np.linspace(0, length, 301)
    outline = np.vstack([np.c_[-width(y, length), y], np.c_[width(y[::-1], length), y[::-1]]])
    if area(outline) < 0:
        outline = outline[::-1]
    return fragments, outline


# -----------------
# OBRAZY
# -----------------

def render_image(polygon, noisy: bool, rng, params: SyntheticParams = SyntheticParams()):
    """Jasny fragment z teksturą na ciemnym tle; `noisy` = mocniejsze rozmycie i szum (przypadki noise i loss)."""
    size = params.image_size
    mask = Image.new("L", (size, size), 0)
    px = mm_to_pixels(polygon, (size, size), params.mm_per_pixel)
    ImageDraw.Draw(mask).polygon([tuple(x) for x in px], fill=255)
    binary = np.asarray(mask) > 127
    yy, xx = np.indices(binary.shape)
    texture = ndi.gaussian_filter(rng.normal(0, 1, binary.shape), 2)
    texture /= np.std(texture)
    background = 24 + 9 * xx / size + 4 * yy / size
    foreground = 218 + 6 * np.sin(xx / 83) + 4 * texture
    image = np.where(binary, foreground, background)
    sigma, noise = (0.85, 12.0) if noisy else (0.35, 1.8)
    image = ndi.gaussian_filter(image, sigma) + rng.normal(0, noise, image.shape)
    return Image.fromarray(np.uint8(np.clip(image, 0, 255))), binary


def create_case(case_id: str, dest: Path, params: SyntheticParams = SyntheticParams()):
    """Zapisuje obrazy, annotations.json i truth.json jednego przypadku do `dest`.

    case_id: "clean" (czysty obraz), "noise" (szum + rozmycie), "loss" (ubytek na przełomie B).

    Returns:
        (annotations, truth) - te same słowniki, które trafiły do plików.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    fragments, outline = bone_fragments(params, loss=case_id == "loss")
    original, _ = bone_fragments(params, loss=False)
    angles = np.deg2rad(params.angles_deg)
    if len(angles) < len(fragments):
        raise ValueError("angles_deg musi mieć kąt dla każdego fragmentu")
    annotations = {"case": case_id, "synthetic": True, "title": params.titles.get(case_id, case_id),
                   "description": params.descriptions.get(case_id, ""), "units": "mm",
                   "mmPerPixel": params.mm_per_pixel, "endpointUnits": "mm",
                   "note": "Końce przełomów symulują dwa ręczne kliknięcia. Nie zdradzają sąsiedztwa ani odpowiadających punktów.",
                   "fragments": []}
    truth = {"case": case_id, "outline": outline.tolist(), "fragments": [], "seams": {}}
    for i, ((polygon, rois), angle) in enumerate(zip(fragments, angles)):
        fid = chr(65 + i)
        center = original[i][0].mean(axis=0)
        tr = -rotation(angle) @ center
        local = transform(polygon, angle, tr)
        # To samo pole losowe w każdym przypadku; zmienia się tylko szum/rozmycie i ubytek.
        image, _ = render_image(local, case_id != "clean", np.random.default_rng(params.seed + 101 * i), params)
        image.save(dest / f"{fid}.png")
        clicked = []
        for k, roi in enumerate(rois):
            rid = f"{fid}{k + 1}"
            ends = transform(roi["endpoints"], angle, tr)
            # Zaokrąglenie do piksela - tak jak przy ręcznym kliknięciu.
            ends = np.round(ends / params.mm_per_pixel) * params.mm_per_pixel
            clicked.append({"id": rid, "endpoints": ends.tolist()})
            truth["seams"][rid] = roi["truth_seam"]
        annotations["fragments"].append({"id": fid, "image": f"{fid}.png", "rois": clicked})
        inv = inverse((angle, tr))
        truth["fragments"].append({"id": fid, "theta": float(inv[0]), "translation": inv[1].tolist(),
                                   "polygon": polygon.tolist(), "originalPolygon": original[i][0].tolist()})
    save_json(dest / "annotations.json", annotations)
    save_json(dest / "truth.json", truth)
    return annotations, truth

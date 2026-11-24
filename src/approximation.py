"""
Etap 2: od obrazu fragmentu do aproksymowanych krzywych przełomów.

    obraz (PNG/JPG/...) -> maska (próg Otsu) -> kontur [mm] -> łuki przełomów (ROI)
    -> B-splajny na kilku poziomach wygładzenia (+ krzywizna)

Działa na dowolnych zdjęciach: dowolny rozmiar obrazu, własna skala [mm/piksel],
jasny albo ciemny obiekt na tle (wykrywane automatycznie). Jedyna ręczna
adnotacja to dwa końce każdego przełomu (patrz scripts/annotate_photos.py).

Format pliku adnotacji (JSON):
    {
      "case": "moja_kosc",                 // nazwa przypadku (folder wyników)
      "title": "...", "description": "...", // opcjonalnie, do raportu
      "mmPerPixel": 0.1,                   // skala; można nadpisać per fragment
      "endpointUnits": "px",               // "px" (kliknięcia na obrazie) albo "mm"
      "fragments": [
        {"id": "A", "image": "A.jpg",      // ścieżka względem pliku adnotacji
         "mmPerPixel": 0.1,                // opcjonalnie
         "rois": [{"id": "A1", "endpoints": [[x1, y1], [x2, y2]],
                   "via": [x, y]}]}        // "via" opcjonalnie: punkt w środku przełomu,
      ]                                    // gdy przełom NIE jest krótszym łukiem konturu
    }

Współrzędne w pikselach: (kolumna, wiersz) od lewego górnego rogu, jak w każdym
programie graficznym. W mm: środek obrazu = (0, 0), x w prawo, y w górę.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import contourpy
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from scipy.interpolate import make_splprep

from geometry import arc_length, area, resample_closed, sample_polyline


# -----------------
# PARAMETRY
# -----------------

@dataclass
class SegmentationParams:
    """Jak oddzielić fragment od tła."""
    foreground: str = "auto"     # "auto" | "bright" | "dark" - czy kość jest jaśniejsza od tła
    blur_sigma_px: float = 0.0   # rozmycie Gaussa przed progowaniem (zdjęcia z szumem/teksturą)
    fill_holes: bool = False     # wypełnij zamknięte dziury w masce (zmienia kontur tylko przy kieszeniach na brzegu)


@dataclass
class ApproximationParams:
    """Jak próbkować kontur i dopasowywać B-splajny."""
    roi_step_mm: float = 0.20           # krok próbkowania surowego łuku przełomu
    spline_input_step_mm: float = 0.25  # krok punktów wejściowych do B-splajnu
    contour_points: int = 599           # liczba punktów zapisanego konturu
    spline_points: int = 181            # liczba punktów zapisanego splajnu
    smoothing_sigmas: tuple = (0.04, 0.12, 0.35)  # [mm] oczekiwany szum konturu na poziom
    smoothing_labels: tuple = ("Szczegóły: 0,04 mm", "Roboczy: 0,12 mm", "Silne: 0,35 mm")


# -----------------
# KALIBRACJA PIKSELE <-> MM
# -----------------

def pixels_to_mm(points_px, image_shape, mm_per_pixel: float) -> np.ndarray:
    """(kolumna, wiersz) -> (x, y) [mm], środek obrazu w (0, 0), y w górę."""
    p = np.asarray(points_px, dtype=float)
    h, w = image_shape[:2]
    return np.c_[(p[:, 0] - w / 2) * mm_per_pixel, (h / 2 - p[:, 1]) * mm_per_pixel]


def mm_to_pixels(points_mm, image_shape, mm_per_pixel: float) -> np.ndarray:
    """Odwrotność pixels_to_mm."""
    p = np.asarray(points_mm, dtype=float)
    h, w = image_shape[:2]
    return np.c_[p[:, 0] / mm_per_pixel + w / 2, h / 2 - p[:, 1] / mm_per_pixel]


def load_gray(path) -> np.ndarray:
    """Wczytuje obraz jako tablicę uint8 w skali szarości (z obrotem wg EXIF ze zdjęć)."""
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("L"))


# -----------------
# SEGMENTACJA I KONTUR
# -----------------

def otsu(image) -> int:
    """Próg Otsu dla obrazu 8-bitowego."""
    counts = np.bincount(np.asarray(image, dtype=np.uint8).ravel(), minlength=256).astype(float)
    prob = counts / counts.sum()
    cumulative = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    variance = (mu[-1] * cumulative - mu) ** 2 / np.maximum(cumulative * (1 - cumulative), 1e-15)
    return int(np.argmax(variance))


def segment(gray, params: SegmentationParams = SegmentationParams()):
    """Maska największego obiektu na obrazie.

    Returns:
        (maska bool, próg Otsu)
    """
    image = np.asarray(gray)
    if params.blur_sigma_px > 0:
        image = np.uint8(np.clip(ndi.gaussian_filter(image.astype(float), params.blur_sigma_px), 0, 255))
    threshold = otsu(image)
    binary = image > threshold
    if params.foreground == "dark":
        binary = ~binary
    elif params.foreground == "auto":
        # Tło dotyka krawędzi obrazu. Jeśli brzeg jest głównie "obiektem", odwróć.
        border = np.r_[binary[0], binary[-1], binary[:, 0], binary[:, -1]]
        if border.mean() > 0.5:
            binary = ~binary
    elif params.foreground != "bright":
        raise ValueError(f"foreground musi być 'auto', 'bright' albo 'dark', jest {params.foreground!r}")
    labels, n = ndi.label(binary)
    if n == 0:
        raise ValueError("Nie znaleziono obiektu na obrazie (pusta maska po progowaniu).")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    binary = labels == np.argmax(sizes)
    if params.fill_holes:
        binary = ndi.binary_fill_holes(binary)
    return binary, threshold


def contour_from_mask(binary, mm_per_pixel: float) -> np.ndarray:
    """Zewnętrzny kontur maski (marching squares, dokładność subpikselowa) w mm, CCW."""
    # Ten sam algorytm, którego domyślnie używa matplotlib.contour (mpl2014).
    generator = contourpy.contour_generator(z=np.asarray(binary, dtype=float), name="mpl2014", corner_mask=True,
                                            line_type=contourpy.LineType.SeparateCode)
    lines, _codes = generator.lines(0.5)
    pixel_path = max(lines, key=len)
    if np.allclose(pixel_path[0], pixel_path[-1]):
        pixel_path = pixel_path[:-1]
    contour = pixels_to_mm(pixel_path, binary.shape, mm_per_pixel)
    return contour if area(contour) >= 0 else contour[::-1]


def extract_roi(contour, endpoints, via=None) -> np.ndarray:
    """Łuk konturu między dwoma klikniętymi końcami przełomu (kierunek CCW).

    Bez `via` wybierany jest krótszy z dwóch łuków. Z `via` - ten, który
    przechodzi bliżej punktu `via`.
    """
    contour = np.asarray(contour)
    i, j = [int(np.argmin(np.linalg.norm(contour - np.asarray(e), axis=1))) for e in endpoints]
    n = len(contour)
    forward = contour[(i + np.arange((j - i) % n + 1)) % n]
    other = contour[(j + np.arange((i - j) % n + 1)) % n]
    if via is not None:
        dist = [np.min(np.linalg.norm(arc - np.asarray(via), axis=1)) for arc in (forward, other)]
        return forward if dist[0] <= dist[1] else other
    return forward if arc_length(forward)[-1] < arc_length(other)[-1] else other


# -----------------
# B-SPLAJNY
# -----------------

def spline_curve(points, sigma: float, count: int = 181, input_step: float = 0.25):
    """Wygładzający B-splajn 3. stopnia (make_splprep, s = N * sigma^2).

    Returns:
        (punkty równomiernie po długości, ich długości łuku, krzywizna ze znakiem,
         słownik ze szczegółami splajnu)
    """
    p = sample_polyline(points, step=input_step)
    u = arc_length(p)
    u /= u[-1]
    spl, _ = make_splprep(p.T, u=u, k=3, s=len(p) * sigma * sigma)
    dense_u = np.linspace(0, 1, 2500)
    dense_s = arc_length(spl(dense_u).T)
    sampled_s = np.linspace(0, dense_s[-1], count)
    uq = np.interp(sampled_s, dense_s, dense_u)
    q = spl(uq).T
    d1, d2 = spl.derivative(1)(uq).T, spl.derivative(2)(uq).T
    speed = np.linalg.norm(d1, axis=1)
    kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(speed, 1e-9) ** 3
    details = {"controlPoints": spl.c.tolist(), "knots": spl.t.tolist(), "degree": int(spl.k),
               "residualRmse": float(np.sqrt(np.mean(np.sum((p - spl(u).T) ** 2, axis=1))))}
    return q, sampled_s, kappa, details


def approximate_arc(raw, params: ApproximationParams = ApproximationParams()) -> list[dict]:
    """Splajny łuku na wszystkich poziomach wygładzenia z `params`."""
    levels = []
    for sigma, label in zip(params.smoothing_sigmas, params.smoothing_labels):
        q, s, k, details = spline_curve(raw, sigma, params.spline_points, params.spline_input_step_mm)
        levels.append({"label": label, "sigma": sigma, "points": q.tolist(), "s": s.tolist(),
                       "curvature": k.tolist(), **details})
    return levels


# -----------------
# CAŁY FRAGMENT / CAŁY PLIK ADNOTACJI
# -----------------

def analyse_fragment(gray, mm_per_pixel: float, rois: list[dict], fragment_id: str,
                     seg: SegmentationParams = SegmentationParams(),
                     approx: ApproximationParams = ApproximationParams()) -> dict:
    """Analiza jednego obrazu. `rois`: [{"id", "endpoints" [mm], opcjonalnie "via" [mm]}].

    Zwracany słownik ma klucz "_mask" (tablica bool) - zapisuje go results.save_masks.
    """
    binary, threshold = segment(gray, seg)
    contour = contour_from_mask(binary, mm_per_pixel)
    item = {"id": fragment_id, "mmPerPixel": mm_per_pixel, "imageShape": list(np.shape(gray)[:2]),
            "contour": resample_closed(contour, approx.contour_points).tolist(),
            "threshold": threshold, "area": abs(area(contour)), "arcs": [], "_mask": binary}
    for roi in rois:
        raw = extract_roi(contour, np.asarray(roi["endpoints"]), roi.get("via"))
        raw = sample_polyline(raw, step=approx.roi_step_mm)
        item["arcs"].append({"id": roi["id"], "points": raw.tolist(), "length": float(arc_length(raw)[-1]),
                             "endpoints": np.asarray(roi["endpoints"]).tolist(),
                             "smoothLevels": approximate_arc(raw, approx)})
    return item


def load_annotations(path) -> dict:
    """Wczytuje plik adnotacji i sprawdza, czy ma wszystko, co potrzebne."""
    path = Path(path)
    annot = json.loads(path.read_text(encoding="utf-8"))
    annot.setdefault("case", path.parent.name)
    for f in annot.get("fragments", []):
        if "mmPerPixel" not in f and "mmPerPixel" not in annot:
            raise ValueError(f"Brak skali 'mmPerPixel' dla fragmentu {f.get('id')} w {path}")
        for k, roi in enumerate(f.get("rois", [])):
            roi.setdefault("id", f"{f['id']}{k + 1}")
            if len(roi.get("endpoints", [])) != 2:
                raise ValueError(f"Przełom {roi['id']} musi mieć dokładnie 2 końce (endpoints)")
    if len(annot.get("fragments", [])) < 2:
        raise ValueError(f"{path}: potrzeba co najmniej 2 fragmentów")
    return annot


def analyse_annotations(annotation_path, seg: SegmentationParams = SegmentationParams(),
                        approx: ApproximationParams = ApproximationParams()) -> dict:
    """Analizuje wszystkie fragmenty z pliku adnotacji.

    Analiza dostaje WYŁĄCZNIE obrazy, skalę i końce przełomów - nie zna sąsiedztwa
    ani położeń fragmentów.

    Returns:
        słownik przypadku: {"id", "title", "description", "fragments": [...]}
    """
    annotation_path = Path(annotation_path)
    annot = load_annotations(annotation_path)
    units = annot.get("endpointUnits", "px")
    fragments = []
    for source in annot["fragments"]:
        image_path = (annotation_path.parent / source["image"]).resolve()
        gray = load_gray(image_path)
        scale = float(source.get("mmPerPixel", annot.get("mmPerPixel")))
        rois = []
        for roi in source["rois"]:
            ends, via = np.asarray(roi["endpoints"], dtype=float), roi.get("via")
            if units == "px":
                ends = pixels_to_mm(ends, gray.shape, scale)
                via = pixels_to_mm([via], gray.shape, scale)[0] if via is not None else None
            rois.append({"id": roi["id"], "endpoints": ends, "via": via})
        item = analyse_fragment(gray, scale, rois, source["id"], seg, approx)
        item["image"] = str(image_path)
        fragments.append(item)
    return {"id": annot["case"], "title": annot.get("title", annot["case"]),
            "description": annot.get("description", ""), "fragments": fragments}

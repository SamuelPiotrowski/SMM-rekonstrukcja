"""
Etap 6: zapis wyników - JSON, CSV, maski i obrazy 2D (PNG).

Na jeden przypadek (folder `out_dir`) zapisuje:
    case.json         - wszystko: kontury, splajny, kandydaci, złożenie, miary
    candidates.csv    - jeden wiersz na parę przełomów
    assembly.csv      - podsumowanie złożenia
    <id>_mask.png     - maska każdego fragmentu
    input.png         - zestawienie obrazów wejściowych z klikniętymi końcami przełomów
    masks.png         - zestawienie masek
    assembly.png      - złożenie 2D (z konturem referencyjnym, jeśli jest prawda)
    reference.png     - model referencyjny (tylko dane z prawdą)

Wykresy używają obiektowego API matplotlib (Figure), więc nie zmieniają
backendu - narzędzie do klikania (annotate.py) dalej może otwierać okna.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageDraw

from approximation import mm_to_pixels
from geometry import transform

COLORS = ["#477eae", "#299784", "#ce8b3e", "#8b6ab3", "#c4566a", "#5b8f3a", "#3f6fb5", "#a7743a"]

NOTES = [
    "Dwa końce każdego przełomu są dostarczoną adnotacją. Ich wskazanie jest etapem ręcznym.",
    "Wyszukiwanie nie zna sąsiedztwa ani prawdziwych transformacji. Przeszukuje odwrócone okna o tej samej długości w mm.",
    "Pole kolizji jest estymacją siatkową, więc ma skończoną rozdzielczość.",
    "Wszystkie pary sąsiadów są hipotezami; wysoki wynik rankingu nie jest prawdopodobieństwem.",
    "Globalny montaż jest zachłannym wyborem niekonfliktujących połączeń i kompozycją transformacji w drzewie.",
]


def color(i: int) -> str:
    return COLORS[i % len(COLORS)]


# -----------------
# JSON
# -----------------

def rounded(data):
    """Kopia danych gotowa do JSON: numpy -> Python, liczby zaokrąglone do 6 miejsc."""
    if isinstance(data, np.ndarray):
        return rounded(data.tolist())
    if isinstance(data, (np.bool_, bool)):
        return bool(data)
    if isinstance(data, (np.floating, float)):
        if not np.isfinite(data):
            raise ValueError("Wartość nieskończona / NaN w wynikach")
        return round(float(data), 6)
    if isinstance(data, (np.integer, int)):
        return int(data)
    if isinstance(data, dict):
        return {k: rounded(v) for k, v in data.items() if not str(k).startswith("_")}
    if isinstance(data, (list, tuple)):
        return [rounded(x) for x in data]
    return data


def save_json(path, data) -> None:
    Path(path).write_text(json.dumps(rounded(data), ensure_ascii=False, indent=2), encoding="utf-8")


def load_case(path) -> dict:
    """Wczytuje case.json zapisany przez save_case (np. do ponownej wizualizacji 3D)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative_path(path, base) -> str:
    """Ścieżka względem `base`, jeśli się da (inny dysk na Windows -> bezwzględna)."""
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:
        return Path(path).as_posix()


def save_case(case: dict, out_dir) -> Path:
    path = Path(out_dir) / "case.json"
    save_json(path, case)
    return path


# -----------------
# MASKI I ZESTAWIENIA OBRAZÓW
# -----------------

def save_masks(case: dict, out_dir) -> None:
    """Zapisuje f["_mask"] każdego fragmentu jako <id>_mask.png i wpisuje ścieżkę do f["mask"]."""
    out_dir = Path(out_dir)
    for f in case["fragments"]:
        mask = f.pop("_mask", None)
        if mask is None:
            continue
        name = f"{f['id']}_mask.png"
        Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(out_dir / name)
        f["mask"] = name


def _scale_bar_mm(mm_per_pixel: float, zoom: float, tile: int) -> float:
    target = 0.15 * tile / zoom * mm_per_pixel
    return min((1, 2, 5, 10, 20, 50, 100, 200, 500), key=lambda v: abs(math.log(v / target)))


def contact_sheet(case: dict, out_dir, key: str = "image", name: str = "input.png", tile: int = 512) -> Path:
    """Zestawienie obrazów fragmentów (key="image") albo masek (key="mask") w siatce."""
    out_dir = Path(out_dir)
    fragments = case["fragments"]
    cols = 2 if len(fragments) <= 4 else math.ceil(math.sqrt(len(fragments)))
    rows = math.ceil(len(fragments) / cols)
    pad, top = 20, 35
    sheet = Image.new("RGB", (pad + cols * (tile + 28), top + rows * (tile + 38)), "#edf3f6")
    draw = ImageDraw.Draw(sheet)
    for i, f in enumerate(fragments):
        with Image.open(out_dir / f[key]) as im:
            im = im.convert("RGB")
            zoom = min(tile / im.width, tile / im.height)
            shape = (im.height, im.width)
            im = im.resize((max(1, round(im.width * zoom)), max(1, round(im.height * zoom))))
        x, y = pad + (i % cols) * (tile + 28), top + (i // cols) * (tile + 38)
        sheet.paste(im, (x, y))
        draw.text((x + 8, y + 5), f["id"], fill="#ffcd68", stroke_width=1)
        if key == "image":
            for arc in f["arcs"]:
                for p in mm_to_pixels(arc["endpoints"], shape, f["mmPerPixel"]) * zoom:
                    cx, cy = p + [x, y]
                    draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), outline="#f6b752", width=2)
                    draw.text((cx + 7, cy - 6), arc["id"], fill="#f6b752")
        bar_mm = _scale_bar_mm(f["mmPerPixel"], zoom, tile)
        bar_px = bar_mm / f["mmPerPixel"] * zoom
        by = y + im.height - 32
        draw.line((x + 20, by, x + 20 + bar_px, by), fill="#ffcd68", width=3)
        draw.text((x + 20, by + 6), f"{bar_mm:g} mm", fill="#ffcd68")
    sheet.save(out_dir / name)
    return out_dir / name


# -----------------
# WYKRESY 2D
# -----------------

def reference_image(truth: dict, out_dir, name: str = "reference.png") -> Path | None:
    """Model referencyjny (fragmenty na swoich miejscach). Tylko gdy prawda ma originalPolygon."""
    if not truth or not all("originalPolygon" in t for t in truth.get("fragments", [])):
        return None
    fig = Figure(figsize=(5, 8))
    ax = fig.subplots()
    for i, t in enumerate(truth["fragments"]):
        p = np.asarray(t["originalPolygon"])
        ax.fill(p[:, 0], p[:, 1], color=color(i), alpha=.75, ec="white", lw=1.2)
        ax.text(*p.mean(0), t["id"], ha="center", va="center", fontsize=18, color="white", weight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Model referencyjny — {len(truth['fragments'])} fragmenty")
    ax.grid(alpha=.13)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / name, dpi=150)
    return Path(out_dir) / name


def component_offsets(case: dict, gap: float = 20.0) -> dict:
    """Przesunięcie [mm] każdej grupy złożenia, żeby rozłączne grupy nie nachodziły (tylko prezentacja)."""
    contours = {f["id"]: np.asarray(f["contour"]) for f in case["fragments"]}
    offsets, x_cursor = {}, None
    for comp in case["assembly"]["components"]:
        pts = np.vstack([transform(contours[p["id"]], p["theta"], p["translation"]) for p in comp["poses"]])
        dx = 0.0 if x_cursor is None else x_cursor - pts[:, 0].min()
        offsets[comp["id"]] = np.array([dx, 0.0])
        x_cursor = pts[:, 0].max() + dx + gap
    return offsets


def final_poses(case: dict) -> dict:
    """Pozy wszystkich fragmentów po złożeniu, rozłączne grupy obok siebie: id -> (theta, t)."""
    offsets = component_offsets(case)
    poses = {}
    for comp in case["assembly"]["components"]:
        for p in comp["poses"]:
            poses[p["id"]] = (p["theta"], np.asarray(p["translation"]) + offsets[comp["id"]])
    return poses


def assembly_image(case: dict, out_dir, name: str = "assembly.png") -> Path:
    """Złożenie 2D: wszystkie fragmenty w wyznaczonych pozach."""
    fig = Figure(figsize=(6, 8))
    ax = fig.subplots()
    ref = case["assembly"].get("referenceContour")
    if ref:
        ref = np.asarray(ref)
        ax.plot(ref[:, 0], ref[:, 1], "--", c="#adbac5", label="Referencja", lw=1.5)
    poses = final_poses(case)
    for i, f in enumerate(case["fragments"]):
        p = transform(f["contour"], *poses[f["id"]])
        ax.fill(p[:, 0], p[:, 1], color=color(i), alpha=.8, ec="white", lw=.8, label=f["id"])
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    metrics = case["assembly"].get("metrics", {})
    if len(case["assembly"]["components"]) > 1:
        ax.set_title(f"{len(case['assembly']['components'])} osobne grupy — brak pewnego połączenia wszystkich")
    elif metrics.get("iou") is not None:
        ax.set_title(f"Wynik algorytmu • IoU = {metrics['iou']:.3f}")
    else:
        ax.set_title("Wynik algorytmu")
    ax.legend(loc="upper right")
    ax.grid(alpha=.12)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / name, dpi=150)
    return Path(out_dir) / name


# -----------------
# CSV
# -----------------

CANDIDATE_COLUMNS = ["case", "candidate", "a", "b", "correct", "selected", "angle_deg", "tx_mm", "ty_mm",
                     "rmse_mm", "paired_rmse_mm", "raw_rmse_mm", "p95_mm", "coverage_A_pct", "coverage_B_pct",
                     "contact_mm", "collision_mm2", "score", "angle_error_deg", "translation_error_mm"]
ASSEMBLY_COLUMNS = ["case", "IoU", "components", "correct_edges", "selected_edges", "fragments_placed",
                    "mean_angle_error_deg", "mean_translation_error_mm", "overlap_mm2"]


def _blank(v):
    return "" if v is None else v


def export_candidates_csv(cases: list[dict], path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8-sig") as h:
        w = csv.writer(h)
        w.writerow(CANDIDATE_COLUMNS)
        for case in cases:
            for c in case["candidates"]:
                m = c["metrics"]
                w.writerow([_blank(v) for v in [
                    case["id"], c["id"], c["a"], c["b"], c.get("correct"), c["accepted"],
                    math.degrees(c["theta"]), *c["translation"], m.get("rmse"), m.get("pairedRmse"),
                    m.get("rawRmse"), m.get("p95"), m.get("coverageA"), m.get("coverageB"), m.get("length"),
                    m.get("collisionArea"), m.get("score"), m.get("angleError"), m.get("translationError")]])


def export_assembly_csv(cases: list[dict], path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8-sig") as h:
        w = csv.writer(h)
        w.writerow(ASSEMBLY_COLUMNS)
        for c in cases:
            m = c["assembly"].get("metrics", {})
            w.writerow([_blank(v) for v in [
                c["id"], m.get("iou"), m.get("components"), m.get("correctEdges"), m.get("selectedEdges"),
                m.get("fragmentsPlaced"), m.get("angleError"), m.get("translationError"), m.get("collisionArea")]])


# -----------------
# WSZYSTKO NARAZ
# -----------------

def save_all(case: dict, out_dir, truth: dict | None = None) -> None:
    """Zapisuje wszystkie wyniki 2D jednego przypadku do `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_masks(case, out_dir)
    case["inputImage"] = contact_sheet(case, out_dir, "image", "input.png").name
    case["maskImage"] = contact_sheet(case, out_dir, "mask", "masks.png").name
    case["assemblyImage"] = assembly_image(case, out_dir).name
    ref = reference_image(truth, out_dir) if truth else None
    if ref:
        case["referenceImage"] = ref.name
    export_candidates_csv([case], out_dir / "candidates.csv")
    export_assembly_csv([case], out_dir / "assembly.csv")
    save_case(case, out_dir)


def save_study(cases: list[dict], out_dir, meta: dict) -> None:
    """Zbiorcze pliki kilku przypadków: study_data.json/.js (dla raportu HTML) i CSV."""
    out_dir = Path(out_dir)
    data = rounded({"meta": meta, "cases": cases})
    save_json(out_dir / "study_data.json", data)
    (out_dir / "study_data.js").write_text(
        "window.STUDY_DATA=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    export_candidates_csv(cases, out_dir / "candidate_results.csv")
    export_assembly_csv(cases, out_dir / "assembly_results.csv")

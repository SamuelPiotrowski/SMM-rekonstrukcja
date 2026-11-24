"""
Ręczna adnotacja własnych zdjęć: klikanie końców przełomów -> annotations.json.

Dla każdego zdjęcia otwiera się okno matplotlib:
    1. (tylko z ruler_mm) kliknij 2 punkty na linijce odległe o ruler_mm mm,
       Enter - skala mm/piksel jest liczona z tych punktów,
    2. klikaj PARAMI końce kolejnych przełomów (2 kliknięcia = 1 przełom),
       Enter - następne zdjęcie.
    Prawy przycisk / Backspace - cofnij ostatnie kliknięcie.

Zdjęcia: jeden fragment na zdjęcie, kontrastowe tło (jasna kość na ciemnym
albo odwrotnie), aparat prostopadle do stołu, ta sama odległość dla wszystkich
zdjęć (albo linijka na każdym zdjęciu).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from approximation import load_gray

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def find_images(folder) -> list[Path]:
    """Zdjęcia w folderze (bez masek i zestawień generowanych przez program)."""
    skip = ("_mask", "input", "masks", "assembly", "reference")
    return sorted(p for p in Path(folder).iterdir()
                  if p.suffix.lower() in IMAGE_EXTENSIONS and not p.stem.lower().startswith(skip)
                  and not p.stem.lower().endswith("_mask"))


def _click(image, title: str):
    import matplotlib.pyplot as plt  # okno GUI - import tylko przy klikaniu

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    points = plt.ginput(n=-1, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=None)
    plt.close(fig)
    return np.asarray(points, dtype=float).reshape(-1, 2)


def annotate_folder(folder, case: str | None = None, mm_per_pixel: float | None = None,
                    ruler_mm: float | None = None, output: str = "annotations.json") -> Path:
    """Tworzy `folder/output` z kliknięć użytkownika. Zwraca ścieżkę pliku."""
    folder = Path(folder)
    images = find_images(folder)
    if len(images) < 2:
        raise ValueError(f"W {folder} potrzeba co najmniej 2 zdjęć fragmentów ({', '.join(IMAGE_EXTENSIONS)})")
    if mm_per_pixel is None and ruler_mm is None:
        raise ValueError("Podaj skalę: mm_per_pixel albo ruler_mm (długość odcinka na linijce do kliknięcia)")
    annot = {"case": case or folder.name, "title": case or folder.name, "endpointUnits": "px", "fragments": []}
    if mm_per_pixel:
        annot["mmPerPixel"] = mm_per_pixel
    for path in images:
        gray = load_gray(path)
        fid = path.stem
        entry = {"id": fid, "image": path.name, "rois": []}
        if ruler_mm:
            pts = _click(gray, f"{path.name}: kliknij 2 punkty na linijce odległe o {ruler_mm} mm, potem Enter")
            if len(pts) != 2:
                raise ValueError(f"{path.name}: do skali potrzeba dokładnie 2 kliknięć, jest {len(pts)}")
            entry["mmPerPixel"] = float(ruler_mm / np.linalg.norm(pts[1] - pts[0]))
        pts = _click(gray, f"{path.name}: klikaj PARAMI końce przełomów (2 kliknięcia = 1 przełom). "
                           "Enter = dalej, prawy przycisk = cofnij")
        if len(pts) % 2:
            print(f"  {path.name}: nieparzysta liczba kliknięć - ostatnie pominięte")
            pts = pts[:-1]
        for k in range(len(pts) // 2):
            entry["rois"].append({"id": f"{fid}_{k + 1}", "endpoints": pts[2 * k:2 * k + 2].round(1).tolist()})
        print(f"  {fid}: {len(entry['rois'])} przełomy")
        annot["fragments"].append(entry)
    out = folder / output
    out.write_text(json.dumps(annot, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

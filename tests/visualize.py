"""
Test wizualizacji: pokazuje każdy fragment z out/ OSOBNO, po kolei.

Zamknięcie okna (Q / Esc) otwiera następny fragment. Funkcje są
w src/visualizing_3d.py, tu są tylko ich wywołania. Co i jak pokazać -
sekcja "view" w settings.json. Uruchomienie (z głównego folderu projektu):
    python tests/visualize.py

Przy "screenshot" w settings.json zamiast okien zapisuje osobny obraz
dla każdego fragmentu, np. out/podglad.png -> out/podglad_lowerjaw.png, ...

Sterowanie w oknie open3d: lewy przycisk - obrót, kółko - zoom,
Ctrl + lewy / środkowy przycisk - przesuwanie, "+"/"-" - rozmiar punktów,
"R" - reset widoku, "H" - pomoc w konsoli, "Q"/Esc - zamknięcie (następny fragment).
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# wczytuje ustawienia z config który to z kolei z settings.json
from config import (  # noqa: E402
    VIEW_AXES,
    VIEW_COLORS,
    VIEW_LAYOUT,
    VIEW_NORMALS,
    VIEW_ONLY,
    VIEW_SCREENSHOT,
    VIEW_SHOW,
)
from data_parsing import load_prepared_fragments  # noqa: E402
from visualizing_3d import build_scene, render_to_file, select_fragments, show_interactive  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")

fragments = load_prepared_fragments()
if not fragments:
    raise SystemExit("Brak przygotowanych fragmentów w out/ - uruchom najpierw: python scripts/prepare_data.py")

fragments = select_fragments(fragments, VIEW_ONLY)
print(f"Fragmenty do pokazania po kolei: {len(fragments)} (tryb: {VIEW_SHOW})")

for i, frag in enumerate(fragments, start=1):
    part = frag.meta.get("part") or frag.slug
    scene, legend = build_scene([frag], show=VIEW_SHOW, layout=VIEW_LAYOUT, colors=VIEW_COLORS,
                                normals=VIEW_NORMALS, axes=VIEW_AXES)
    print(f"\n({i}/{len(fragments)}) {legend[0].split('] ', 1)[1]}")  # bez "[1] " z build_scene

    if VIEW_SCREENSHOT:
        path = VIEW_SCREENSHOT.with_name(f"{VIEW_SCREENSHOT.stem}_{part}{VIEW_SCREENSHOT.suffix}")
        print("  Zapisano zrzut:", render_to_file(scene, path))
    else:
        show_interactive(scene, title=f"({i}/{len(fragments)}) {part} - {VIEW_SHOW}")

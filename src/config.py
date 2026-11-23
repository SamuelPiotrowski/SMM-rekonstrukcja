"""
Wczytuje ustawienia projektu z settings.json (w głównym folderze projektu).

Wszystkie wartości i ich opisy są w settings.json - ten plik tylko je wczytuje
i udostępnia jako stałe dla reszty kodu. settings.json może mieć komentarze //,
które są usuwane przed wczytaniem (zwykły JSON ich nie obsługuje).
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "settings.json"

# Tekst w cudzysłowie (zostaje bez zmian, np. "http://...") albo komentarz // do końca linii.
_STRING_OR_COMMENT = re.compile(r'("(?:\\.|[^"\\])*")|//[^\n]*')


def load_settings(path: Path = SETTINGS_FILE) -> dict:
    """Wczytuje plik JSON, w którym mogą być komentarze //."""
    text = path.read_text(encoding="utf-8")
    text = _STRING_OR_COMMENT.sub(lambda m: m.group(1) or "", text)
    return json.loads(text)


_s = load_settings()
_paths, _parsing, _vis, _view = _s["paths"], _s["data_parsing"], _s["visualizing"], _s["view"]

# -----------------
# PATHS
# -----------------

DATA_IN = ROOT / _paths["data_in"]
DATA_OUT = ROOT / _paths["data_out"]

# -----------------
# DATA PARSING
# -----------------

PREFERRED_FORMATS = tuple(_parsing["preferred_formats"])
MIN_COMPONENT_AREA_RATIO = float(_parsing["min_component_area_ratio"])
CENTER_FRAGMENTS = bool(_parsing["center_fragments"])
VOXEL_SIZE = float(_parsing["voxel_size"])
SAMPLING_DENSITY = _parsing["sampling_density"]
NORMAL_RADIUS_FACTOR = float(_parsing["normal_radius_factor"])
FPFH_RADIUS_FACTOR = float(_parsing["fpfh_radius_factor"])

# -----------------
# 3D VISUALIZING
# -----------------

VIS_WINDOW_SIZE = tuple(_vis["window_size"])
VIS_NORMAL_LENGTH_MM = float(_vis["normal_length_mm"])
VIS_MAX_NORMALS = int(_vis["max_normals"])
AXES_SIZE_MM = float(_vis["axes_size_mm"])
VIS_BACKGROUND = tuple(_vis["background"])
VIS_BONE_COLOR = tuple(_vis["bone_color"])
VIS_NORMAL_COLOR = tuple(_vis["normal_color"])
CAMERA_TILT = float(_vis["camera_tilt"])
CAMERA_ZOOM = float(_vis["camera_zoom"])

# -----------------
# VIEW - co pokazuje tests/visualize.py
# -----------------

VIEW_SHOW = _view["show"]
VIEW_LAYOUT = _view["layout"]
VIEW_COLORS = _view["colors"]
VIEW_ONLY = _view["only"]
VIEW_NORMALS = bool(_view["normals"])
VIEW_AXES = bool(_view["axes"])
VIS_GAP_MM = float(_view["gap"])
VIS_POINT_SIZE = float(_view["point_size"])
VIEW_SCREENSHOT = ROOT / _view["screenshot"] if _view["screenshot"] else None

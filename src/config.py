"""
Wczytuje ustawienia projektu z settings.json (w głównym folderze projektu).

Wszystkie wartości i ich opisy są w settings.json - ten plik tylko je wczytuje
i zamienia na obiekty parametrów poszczególnych modułów. settings.json może mieć
komentarze //, które są usuwane przed wczytaniem (zwykły JSON ich nie obsługuje).

Brakujący klucz w settings.json = wartość domyślna z danego modułu.
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


SETTINGS = load_settings()
_paths = SETTINGS.get("paths", {})

# -----------------
# PATHS
# -----------------

DATA_IN = ROOT / _paths.get("data_in", "bone_photos")
DATA_OUT = ROOT / _paths.get("data_out", "out")


# -----------------
# PARAMETRY MODUŁÓW
# -----------------

def _section(name: str, cls, skip=()):
    values = {k: v for k, v in SETTINGS.get(name, {}).items() if k not in skip}
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
    return cls(**values)


def segmentation_params():
    from approximation import SegmentationParams
    return _section("segmentation", SegmentationParams)


def approximation_params():
    from approximation import ApproximationParams
    return _section("approximation", ApproximationParams)


def match_params():
    from matching import MatchParams
    return _section("matching", MatchParams)


def view3d_params():
    from visualization_3d import View3DParams
    return _section("visualization_3d", View3DParams, skip=("open_browser",))


def synthetic_params():
    from synthetic import SyntheticParams
    return _section("synthetic", SyntheticParams, skip=("cases",))


OPEN_BROWSER = bool(SETTINGS.get("visualization_3d", {}).get("open_browser", True))
SYNTHETIC_CASES = tuple(SETTINGS.get("synthetic", {}).get("cases", ("clean", "noise", "loss")))

"""
Wczytuje ustawienia projektu z settings.json (w głównym folderze projektu).

Wszystkie wartości i ich opisy są w settings.json - ten plik tylko je wczytuje
i zamienia na obiekty parametrów poszczególnych modułów. settings.json może mieć
komentarze //, które są usuwane przed wczytaniem (zwykły JSON ich nie obsługuje).

Brakujący klucz w settings.json = wartość domyślna z danego modułu.
Literówka w nazwie klucza = czytelny błąd z listą poprawnych nazw.
"""

import json
import re
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "settings.json"

# Tekst w cudzysłowie (zostaje bez zmian, np. "http://...") albo komentarz // do końca linii.
_STRING_OR_COMMENT = re.compile(r'("(?:\\.|[^"\\])*")|//[^\n]*')


def load_settings(path: Path = SETTINGS_FILE) -> dict:
    """Wczytuje plik JSON, w którym mogą być komentarze //."""
    text = path.read_text(encoding="utf-8")
    text = _STRING_OR_COMMENT.sub(lambda m: m.group(1) or "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Błąd składni w {path.name}, linia {e.lineno}: {e.msg}\n"
                         "Sprawdź przecinki (bez przecinka po ostatnim elemencie), cudzysłowy \"...\" "
                         "i true/false/null małymi literami.") from None


SETTINGS = load_settings()
_paths = SETTINGS.get("paths", {})

# -----------------
# PATHS
# -----------------

DATA_IN = ROOT / _paths.get("data_in", "scanned_bones")      # surowe skany (z podfolderami)
DATA_OUT = ROOT / _paths.get("data_out", "out/3d")           # wyniki - generowane, można usunąć
CONTACTS_DIR = ROOT / _paths.get("contacts", "contacts")     # ręczne oznaczenia powierzchni styku


# -----------------
# PARAMETRY MODUŁÓW
# -----------------

def _section(name: str, cls, skip=()):
    """Tworzy obiekt parametrów `cls` z sekcji `name` settings.json."""
    values = {k: v for k, v in SETTINGS.get(name, {}).items() if k not in skip}
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise SystemExit(f"settings.json, sekcja \"{name}\": nieznane klucze {unknown}. "
                         f"Dostępne: {sorted(known)}")
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
    return cls(**values)


def scan_params():
    from data_parsing import ScanParams
    return _section("scans", ScanParams)


def merge_params():
    from merging import MergeParams
    return _section("merging", MergeParams)


def feature_params():
    from bones import FeatureParams
    return _section("features", FeatureParams)


def contact_params():
    from contacts import ContactParams
    return _section("contacts", ContactParams, skip=("port",))


def matching_params():
    from matching_surfaces import SurfaceMatchParams
    return _section("matching", SurfaceMatchParams)


def view3d_params():
    from visualization_3d import View3DParams
    return _section("visualization_3d", View3DParams, skip=("open_browser",))


OPEN_BROWSER = bool(SETTINGS.get("visualization_3d", {}).get("open_browser", True))
CONTACT_TOOL_PORT = int(SETTINGS.get("contacts", {}).get("port", 8765))

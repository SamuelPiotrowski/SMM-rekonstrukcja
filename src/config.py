"""
Wczytuje ustawienia projektu z settings.json (w głównym folderze projektu).

Wszystkie wartości i ich opisy są w settings.json - ten plik tylko je wczytuje
i zamienia na obiekty parametrów poszczególnych modułów. settings.json może mieć
komentarze //, które są usuwane przed wczytaniem (zwykły JSON ich nie obsługuje).

Brakujący klucz w settings.json = wartość domyślna z danego modułu.
Literówka w nazwie klucza = czytelny błąd z listą poprawnych nazw.

Co udostępnia:
    ROOT, SETTINGS_FILE             - główny folder projektu i ścieżka do settings.json,
    SETTINGS                        - cały settings.json jako słownik,
    DATA_IN, DATA_OUT, CONTACTS_DIR - foldery (sekcja "paths"),
    scan_params(), merge_params(), feature_params(), contact_params(),
    matching_params(), surface_feature_params(), view3d_params()
                                    - obiekty parametrów modułów (po jednym na sekcję),
    OPEN_BROWSER, CONTACT_TOOL_PORT, MATCH_TOOL_PORT
                                    - pojedyncze ustawienia skryptów.

Funkcje *_params() importują swój moduł dopiero przy wywołaniu - dzięki temu config
można zaimportować w każdym module bez cyklicznych importów.

    import config
    params = config.merge_params()      # MergeParams z sekcji "merging"
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
    """Wczytuje plik JSON, w którym mogą być komentarze //.

    Komentarze // (do końca linii) są usuwane przed parsowaniem. Tekst w cudzysłowie
    zostaje nietknięty, więc np. "http://..." się nie psuje. Komentarze /* ... */
    nie są obsługiwane.

    Args:
        path: plik do wczytania (domyślnie settings.json w głównym folderze projektu).

    Returns:
        Zawartość pliku, dla settings.json: {nazwa sekcji: {klucz: wartość}}.

    Raises:
        SystemExit: błąd składni JSON - komunikat podaje numer linii i typowe przyczyny.
    """
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
    """Tworzy obiekt parametrów `cls` z sekcji `name` settings.json.

    Klucze, których brakuje w sekcji, dostają wartości domyślne z dataclassy `cls`.
    Listy JSON są zamieniane na krotki (tak jak wartości domyślne w dataclassach).

    Args:
        name: nazwa sekcji w settings.json, np. "merging".
        cls:  dataclass z parametrami modułu, np. merging.MergeParams.
        skip: klucze sekcji, których nie ma w `cls`, bo są czytane osobno (np. "port").

    Returns:
        Obiekt `cls` z wartościami z settings.json.

    Raises:
        SystemExit: sekcja zawiera nieznany klucz (np. literówkę) - komunikat podaje poprawne nazwy.
    """
    values = {k: v for k, v in SETTINGS.get(name, {}).items() if k not in skip}
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise SystemExit(f"settings.json, sekcja \"{name}\": nieznane klucze {unknown}. "
                         f"Dostępne: {sorted(known)}")
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
    return cls(**values)


def scan_params():
    """Wyszukiwanie i czyszczenie skanów: sekcja "scans" settings.json.

    Returns:
        data_parsing.ScanParams (brakujące klucze = wartości domyślne).

    Raises:
        SystemExit: nieznany klucz w sekcji (patrz _section).
    """
    from data_parsing import ScanParams
    return _section("scans", ScanParams)


def merge_params():
    """Łączenie skanów tej samej kości: sekcja "merging" settings.json.

    Returns:
        merging.MergeParams (brakujące klucze = wartości domyślne).

    Raises:
        SystemExit: nieznany klucz w sekcji (patrz _section).
    """
    from merging import MergeParams
    return _section("merging", MergeParams)


def feature_params():
    """Chmura punktów i FPFH kości: sekcja "features" settings.json.

    Returns:
        bones.FeatureParams (brakujące klucze = wartości domyślne).

    Raises:
        SystemExit: nieznany klucz w sekcji (patrz _section).
    """
    from bones import FeatureParams
    return _section("features", FeatureParams)


def contact_params():
    """Powierzchnie styku (narzędzie + algorytm): sekcja "contacts" settings.json.

    Klucz "port" jest pomijany - to CONTACT_TOOL_PORT.

    Returns:
        contacts.ContactParams (brakujące klucze = wartości domyślne).

    Raises:
        SystemExit: nieznany klucz w sekcji (patrz _section).
    """
    from contacts import ContactParams
    return _section("contacts", ContactParams, skip=("port",))


def matching_params():
    """Dopasowanie powierzchni styku: sekcja "matching" settings.json.

    Klucz "port" jest pomijany - to MATCH_TOOL_PORT.

    Returns:
        matching_surfaces.SurfaceMatchParams (brakujące klucze = wartości domyślne).

    Raises:
        SystemExit: nieznany klucz w sekcji (patrz _section).
    """
    from matching_surfaces import SurfaceMatchParams
    return _section("matching", SurfaceMatchParams, skip=("port",))


def surface_feature_params():
    """Porównanie cech przełomu i kory: sekcja "surface_features" settings.json.

    Returns:
        surface_features.SurfaceFeatureParams (brakujące klucze = wartości domyślne).

    Raises:
        SystemExit: nieznany klucz w sekcji (patrz _section).
    """
    from surface_features import SurfaceFeatureParams
    return _section("surface_features", SurfaceFeatureParams)


def view3d_params():
    """Wygląd wizualizacji HTML: sekcja "visualization_3d" settings.json.

    Klucz "open_browser" jest pomijany - to OPEN_BROWSER.

    Returns:
        visualization_3d.View3DParams (brakujące klucze = wartości domyślne).

    Raises:
        SystemExit: nieznany klucz w sekcji (patrz _section).
    """
    from visualization_3d import View3DParams
    return _section("visualization_3d", View3DParams, skip=("open_browser",))


# Czy skrypty mają same otwierać wygenerowane pliki HTML w przeglądarce.
OPEN_BROWSER = bool(SETTINGS.get("visualization_3d", {}).get("open_browser", True))
# Port lokalnego serwera narzędzia do oznaczania (scripts/mark_contacts.py).
CONTACT_TOOL_PORT = int(SETTINGS.get("contacts", {}).get("port", 8765))
# Port lokalnego serwera podglądu dopasowania z suwakami (scripts/tune_matching.py).
MATCH_TOOL_PORT = int(SETTINGS.get("matching", {}).get("port", 8766))

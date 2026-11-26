"""
Wczytuje ustawienia projektu z settings.json (w głównym folderze projektu).

Wszystkie wartości i ich opisy są w settings.json - ten plik tylko je wczytuje
i zamienia na obiekty parametrów poszczególnych modułów. settings.json może mieć
komentarze //, które są usuwane przed wczytaniem (zwykły JSON ich nie obsługuje).

Brakujący klucz w settings.json = wartość domyślna z danego modułu.
Literówka w nazwie klucza = czytelny błąd z listą poprawnych nazw.

Zestawy danych: każdy zestaw to osobny podfolder w trzech miejscach (ta sama nazwa):
    scanned_bones/<zestaw>/   skany            -> DATA_IN
    out/3d/<zestaw>/          wyniki           -> DATA_OUT
    contacts/<zestaw>/        oznaczenia styku -> CONTACTS_DIR
Domyślny zestaw = "dataset" w settings.json; skrypty zmieniają go jednorazowo opcją
--dataset <nazwa> (add_dataset_argument + use_dataset), a w kodzie: config.use_dataset("nazwa").
Zestaw może mieć własne ustawienia: scanned_bones/<zestaw>/settings.json (te same sekcje) -
nadpisują pojedyncze klucze głównego settings.json tylko dla tego zestawu.

Co udostępnia:
    ROOT, SETTINGS_FILE             - główny folder projektu i ścieżka do settings.json,
    SETTINGS                        - cały settings.json jako słownik,
    SCANS_ROOT, OUT_ROOT, CONTACTS_ROOT
                                    - foldery główne (sekcja "paths"), w nich podfoldery zestawów,
    DATASET                         - nazwa bieżącego zestawu,
    DATA_IN, DATA_OUT, CONTACTS_DIR - foldery bieżącego zestawu (zmienia je use_dataset),
    use_dataset(), list_datasets(), dataset_dirs(), add_dataset_argument()
                                    - wybór i lista zestawów,
    ATLAS_DIR, ATLAS_INDEX, use_atlas(), add_atlas_argument()
                                    - baza modeli kompletnych kości (atlas anatomiczny, src/atlas.py):
                                      pliki modeli i zindeksowana baza,
    scan_params(), merge_params(), feature_params(), contact_params(),
    matching_params(), surface_feature_params(), view3d_params(), synthetic_params(),
    atlas_params()                  - obiekty parametrów modułów (po jednym na sekcję),
    DATASET_SETTINGS                - ustawienia bieżącego zestawu (jego settings.json, {} = brak),
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


# Stare nazwy kluczy z jednostką mm (np. "tolerance_mm", "area_mm2") - teraz jednostki
# arbitralne: "tolerance_au", "area_au2". Wartości się nie zmieniły, tylko nazwy.
_OLD_UNIT_KEY = re.compile(r"_mm(2?)$")


def new_unit_keys(data):
    """Zamienia stare nazwy kluczy *_mm / *_mm2 na *_au / *_au2 (w całym zagnieżdżonym JSON-ie).

    Dzięki temu pliki zapisane przed zmianą jednostek (settings.json, contacts/*.json,
    meta.json, merging.json, ground_truth.json, ...) wczytują się bez zmian. Gdy w słowniku
    są oba klucze (stary i nowy), wygrywa nowy.

    Args:
        data: wczytany JSON (słownik, lista albo pojedyncza wartość).

    Returns:
        To samo z przemianowanymi kluczami (nowe obiekty dla słowników i list).
    """
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            new = _OLD_UNIT_KEY.sub(r"_au\1", k) if isinstance(k, str) else k
            if new != k and new in data:
                continue
            out[new] = new_unit_keys(v)
        return out
    if isinstance(data, list):
        return [new_unit_keys(v) for v in data]
    return data


def read_json(path: Path):
    """Wczytuje plik JSON (UTF-8) i zamienia stare klucze *_mm na *_au (patrz new_unit_keys)."""
    return new_unit_keys(json.loads(Path(path).read_text(encoding="utf-8")))


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
        return new_unit_keys(json.loads(text))
    except json.JSONDecodeError as e:
        raise SystemExit(f"Błąd składni w {path.name}, linia {e.lineno}: {e.msg}\n"
                         "Sprawdź przecinki (bez przecinka po ostatnim elemencie), cudzysłowy \"...\" "
                         "i true/false/null małymi literami.") from None


SETTINGS = load_settings()
_paths = SETTINGS.get("paths", {})

# -----------------
# FOLDERY I ZESTAWY DANYCH
# -----------------

SCANS_ROOT = ROOT / _paths.get("data_in", "scanned_bones")    # skany, podfolder na każdy zestaw
OUT_ROOT = ROOT / _paths.get("data_out", "out/3d")            # wyniki, podfolder na każdy zestaw
CONTACTS_ROOT = ROOT / _paths.get("contacts", "contacts")     # oznaczenia styku, podfolder na każdy zestaw
PRESENTATIONS_ROOT = ROOT / _paths.get("presentations", "raporty/prezentacje")  # presentation.py, podfolder na zestaw
# Pliki skanów - po nich list_datasets rozpoznaje zestaw.
SCAN_EXTENSIONS = (".ply", ".obj", ".stl")

# Bieżący zestaw i jego foldery - ustawia use_dataset() (niżej, przy imporcie: zestaw z settings.json).
DATASET: str = ""
DATA_IN: Path = SCANS_ROOT        # surowe skany zestawu (z podfolderami)
DATA_OUT: Path = OUT_ROOT         # wyniki zestawu - generowane, można usunąć
CONTACTS_DIR: Path = CONTACTS_ROOT  # oznaczenia powierzchni styku zestawu (+ auto/)
# Ustawienia zestawu: DATA_IN/settings.json (opcjonalny, te same sekcje co główny settings.json) -
# nadpisują pojedyncze klucze głównego, np. {"merging": {"enabled": false}} tylko dla tego zestawu.
DATASET_SETTINGS_NAME = "settings.json"
DATASET_SETTINGS: dict = {}


def dataset_dirs(name: str) -> tuple[Path, Path, Path]:
    """Foldery zestawu `name`: (skany, wyniki, oznaczenia styku).

    Args:
        name: nazwa zestawu = nazwa podfolderu, np. "first_scans" (może zawierać "/",
              np. "syntetyczne/s1"); "" = foldery główne bez podfolderu (stary układ).

    Returns:
        (SCANS_ROOT/name, OUT_ROOT/name, CONTACTS_ROOT/name) - foldery nie muszą istnieć.
    """
    return SCANS_ROOT / name, OUT_ROOT / name, CONTACTS_ROOT / name


def _rel(path: Path) -> str:
    """Ścieżka względem głównego folderu projektu (do komunikatów); spoza niego - pełna."""
    try:
        return Path(path).relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _has_scans(folder: Path) -> bool:
    """Czy w `folder` (albo jego podfolderach) jest choć jeden plik skanu."""
    return any(p.suffix.lower() in SCAN_EXTENSIONS for p in folder.rglob("*") if p.is_file())


def list_datasets() -> list[str]:
    """Nazwy dostępnych zestawów: podfoldery SCANS_ROOT ze skanami albo OUT_ROOT z kośćmi.

    Zestaw bez skanów, ale z przygotowanymi kośćmi w out/ (np. skany są tylko u kogoś
    innego) też się liczy - da się go oglądać i oznaczać.

    Returns:
        Posortowane nazwy (bez folderów zaczynających się od "." i "_").
    """
    names = set()
    if SCANS_ROOT.is_dir():
        names |= {d.name for d in SCANS_ROOT.iterdir() if d.is_dir() and _has_scans(d)}
    if OUT_ROOT.is_dir():
        names |= {d.name for d in OUT_ROOT.iterdir() if (d / "bones").is_dir()}
    return sorted(n for n in names if not n.startswith((".", "_")))


def use_dataset(name: str | None = None, check: bool = True) -> str:
    """Ustawia bieżący zestaw danych: DATASET, DATA_IN, DATA_OUT, CONTACTS_DIR.

    Wszystkie funkcje biorą domyślne foldery z tych zmiennych w chwili wywołania, więc po
    use_dataset("x") każda kolejna funkcja (prepare_all, load_bones, load_contacts, ...)
    pracuje na zestawie "x". Skrypty wywołują to z wartością opcji --dataset.

    Args:
        name:  nazwa zestawu; None / "" = domyślny z settings.json ("dataset").
        check: sprawdź, czy zestaw istnieje (folder skanów albo przygotowane kości).

    Returns:
        Nazwa ustawionego zestawu.

    Raises:
        SystemExit: check=True i zestawu nie ma - komunikat podaje dostępne zestawy.
    """
    global DATASET, DATA_IN, DATA_OUT, CONTACTS_DIR, DATASET_SETTINGS
    name = (name or SETTINGS.get("dataset") or "").strip().strip("/\\")
    data_in, data_out, contacts = dataset_dirs(name)
    if check and not data_in.is_dir() and not (data_out / "bones").is_dir():
        available = list_datasets()
        hint = (f"Dostępne zestawy: {', '.join(available)}" if available else
                f"Brak zestawów - utwórz folder {SCANS_ROOT.name}/<nazwa zestawu>/ i wrzuć do niego skany")
        if (SCANS_ROOT / "first_test").is_dir():
            hint += (". Skany są jeszcze w starym układzie (bez zestawów) - przenieś je raz: "
                     "python scripts/migrate_first_scans.py")
        raise SystemExit(f"Nie ma zestawu \"{name}\" (brak folderu {_rel(data_in)} i wyników "
                         f"w {_rel(data_out)}). {hint}.\n"
                         "Domyślny zestaw: settings.json -> \"dataset\", jednorazowo: --dataset <nazwa>.")
    DATASET, DATA_IN, DATA_OUT, CONTACTS_DIR = name, data_in, data_out, contacts
    own = data_in / DATASET_SETTINGS_NAME
    DATASET_SETTINGS = load_settings(own) if name and own.is_file() else {}
    return name


def save_dataset_settings(section: str, values: dict) -> Path:
    """Zapisuje klucze sekcji w ustawieniach bieżącego zestawu (DATA_IN/settings.json), resztę zostawia.

    Plik zapisywany jest jako JSON bez komentarzy - gdy miał komentarze //, poprzednia wersja
    zostaje obok jako settings.json.bak. Po zapisie wywołaj use_dataset(DATASET), żeby je wczytać.

    Args:
        section: nazwa sekcji, np. "atlas".
        values:  {klucz: wartość} do wpisania (nadpisują te same klucze).

    Returns:
        Ścieżka zapisanego pliku.
    """
    path = DATA_IN / DATASET_SETTINGS_NAME
    data = load_settings(path) if path.is_file() else {}
    if path.is_file() and "//" in path.read_text(encoding="utf-8"):
        path.with_suffix(".json.bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    data.setdefault(section, {}).update(values)
    DATA_IN.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def add_dataset_argument(parser) -> None:
    """Dodaje do skryptu opcję -d / --dataset (zestaw danych zamiast domyślnego z settings.json).

    Po parser.parse_args() skrypt wywołuje config.use_dataset(args.dataset).

    Args:
        parser: argparse.ArgumentParser skryptu.
    """
    default = SETTINGS.get("dataset") or "-"
    parser.add_argument("-d", "--dataset", default=None, metavar="NAZWA",
                        help=f"zestaw danych (podfolder {_rel(SCANS_ROOT)}/, {_rel(OUT_ROOT)}/ i {_rel(CONTACTS_ROOT)}/); "
                             f"domyślnie z settings.json: {default}")


def dataset_flag() -> str:
    """" --dataset <nazwa>" do podpowiedzi w komunikatach - pusty tekst dla zestawu domyślnego."""
    return "" if DATASET == (SETTINGS.get("dataset") or "") else f" --dataset {DATASET}"


def no_bones_message() -> str:
    """Komunikat "brak przygotowanych kości" z nazwą bieżącego zestawu i poleceniem do uruchomienia."""
    return (f"Brak przygotowanych kości w zestawie \"{DATASET}\" ({_rel(DATA_OUT)}) - uruchom najpierw: "
            f"python scripts/prepare_data.py{dataset_flag()}")


use_dataset(check=False)


# -----------------
# ATLAS ANATOMICZNY (baza modeli kompletnych kości) - src/atlas.py
# -----------------

# Dwa foldery (wspólne dla wszystkich zestawów), zmienia je use_atlas():
#   ATLAS_DIR   - pliki modeli 3D kompletnych kości (.ply/.obj/.stl), potrzebne tylko do zindeksowania,
#   ATLAS_INDEX - zindeksowana baza (~1 MB na model) - TYLKO z niej korzysta wykrywanie; oryginały
#                 można potem usunąć albo trzymać gdzie indziej.
ATLAS_DIR: Path = ROOT / _paths.get("atlas", "atlas")
ATLAS_INDEX: Path = ROOT / _paths.get("atlas_index", "atlas_index")


def _project_path(path: str | Path) -> Path:
    """Ścieżka względna = względem głównego folderu projektu."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def use_atlas(source: str | Path | None = None, index: str | Path | None = None) -> tuple[Path, Path]:
    """Ustawia foldery atlasu (ATLAS_DIR, ATLAS_INDEX) - np. z opcji --atlas / --atlas-index skryptu.

    Args:
        source: folder z plikami modeli; None / "" = z settings.json ("paths" -> "atlas").
        index:  folder zindeksowanej bazy; None / "" = z settings.json ("paths" -> "atlas_index").

    Returns:
        (ATLAS_DIR, ATLAS_INDEX) - foldery nie muszą istnieć.
    """
    global ATLAS_DIR, ATLAS_INDEX
    ATLAS_DIR = _project_path(source or _paths.get("atlas", "atlas"))
    ATLAS_INDEX = _project_path(index or _paths.get("atlas_index", "atlas_index"))
    return ATLAS_DIR, ATLAS_INDEX


def add_atlas_argument(parser) -> None:
    """Dodaje do skryptu opcje --atlas i --atlas-index (inne foldery atlasu niż w settings.json).

    Po parser.parse_args() skrypt wywołuje config.use_atlas(args.atlas, args.atlas_index).

    Args:
        parser: argparse.ArgumentParser skryptu.
    """
    parser.add_argument("--atlas", default=None, metavar="FOLDER",
                        help=f"folder z plikami modeli kompletnych kości (domyślnie z settings.json: {_rel(ATLAS_DIR)})")
    parser.add_argument("--atlas-index", default=None, metavar="FOLDER",
                        help=f"folder zindeksowanej bazy modeli (domyślnie z settings.json: {_rel(ATLAS_INDEX)})")


# -----------------
# PARAMETRY MODUŁÓW
# -----------------

def _section(name: str, cls, skip=()):
    """Tworzy obiekt parametrów `cls` z sekcji `name` settings.json.

    Klucze, których brakuje w sekcji, dostają wartości domyślne z dataclassy `cls`.
    Klucze z ustawień bieżącego zestawu (DATASET_SETTINGS) nadpisują te z settings.json.
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
    merged = {**SETTINGS.get(name, {}), **DATASET_SETTINGS.get(name, {})}
    values = {k: v for k, v in merged.items() if k not in skip}
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        where = "settings.json" if not set(unknown) & set(DATASET_SETTINGS.get(name, {})) else \
            _rel(DATA_IN / DATASET_SETTINGS_NAME)
        raise SystemExit(f"{where}, sekcja \"{name}\": nieznane klucze {unknown}. "
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

    Ręczne grupy ("groups") mogą być wspólne albo osobne dla każdego zestawu:
    {"first_scans": {"kość": ["fragment nazwy skanu", ...]}, "inny_zestaw": {...}}.

    Returns:
        merging.MergeParams (brakujące klucze = wartości domyślne).

    Raises:
        SystemExit: nieznany klucz w sekcji (patrz _section).
    """
    from merging import MergeParams
    params = _section("merging", MergeParams)
    groups = params.groups
    # grupy osobno dla zestawów: {"nazwa zestawu": {"kość": [...]}, ...} -> grupy bieżącego zestawu
    if isinstance(groups, dict) and groups and all(isinstance(v, dict) for v in groups.values()):
        from dataclasses import replace
        params = replace(params, groups=groups.get(DATASET) or None)
    return params


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
    from dataclasses import replace
    from contacts import ContactParams
    params = _section("contacts", ContactParams, skip=("port",))
    # stara nazwa cechy (przed zmianą mm -> a.u.)
    return replace(params, feature_keys=tuple("edge_au" if k == "edge_mm" else k for k in params.feature_keys))


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


def synthetic_params():
    """Generator syntetycznych kości (scripts/make_synthetic.py): sekcja "synthetic" settings.json.

    Returns:
        synthetic_bones.SyntheticParams (brakujące klucze = wartości domyślne).

    Raises:
        SystemExit: nieznany klucz w sekcji (patrz _section).
    """
    from synthetic_bones import SyntheticParams
    return _section("synthetic", SyntheticParams)


def atlas_params():
    """Atlas anatomiczny (rozpoznawanie kości, algorytm "atlas"): sekcja "atlas" settings.json.

    Returns:
        atlas.AtlasParams (brakujące klucze = wartości domyślne).

    Raises:
        SystemExit: nieznany klucz w sekcji (patrz _section).
    """
    from atlas import AtlasParams
    return _section("atlas", AtlasParams)


def presentation_params():
    """Prezentacja (scripts/presentation.py): sekcja "presentation" settings.json.

    Returns:
        presentation.PresentationParams (brakujące klucze = wartości domyślne).

    Raises:
        SystemExit: nieznany klucz w sekcji (patrz _section).
    """
    from presentation import PresentationParams
    return _section("presentation", PresentationParams)


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

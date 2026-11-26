"""
Prezentacja wyników JEDNEJ metody - jeden przenośny plik HTML (scripts/presentation.py).

Łączy na jednej stronie to, co pokazują osobno:
    1. prepare_data.py / show_merging.py  - skany -> kości (łączenie skanów, macierz nakładania),
    2. show_bones.py                      - kości z powierzchniami styku tej metody
                                            (+ porównanie z ręcznymi oznaczeniami, jeśli są),
    3. tune_matching.py --html            - dopasowanie: najlepsze pary w 3D, macierz, proponowane złożenie.

SPÓJNOŚĆ - wszystko na stronie pochodzi z JEDNEJ metody, JEDNEGO zestawu i JEDNYCH ustawień:
    metoda = skąd powierzchnie styku: "manual" (ręczne oznaczenia) albo algorytm z contacts.DETECTORS
    ("roughness", "features", "atlas"). Ustawienia = obecny settings.json (+ settings.json zestawu),
    z "contacts" -> "method" i "matching" -> "source" ustawionymi na tę metodę.
    Tryb "manual-auto" (mieszanka ręcznych i algorytmu) celowo nie jest metodą prezentacji.

NIC NIE JEST PRZELICZANE. Strona czyta tylko gotowe wyniki:
    - kości z out/3d/<zestaw>/bones/ i merging.json (prepare_data.py),
    - ZAPIS METODY out/3d/<zestaw>/methods/<metoda>/ - robi go match_surfaces.py po każdym pełnym
      przebiegu (wszystkie kości, tryb "manual" albo "auto"): matching.json + powierzchnie styku
      DOKŁADNIE te, które dopasowywano (contacts/<id>.json) + odciski siatek kości (snapshot.json).
      Dzięki temu kilka metod może mieć wyniki naraz (np. atlas i features) - nie nadpisują się.
    Gdy czegoś brakuje albo zostało policzone z innymi ustawieniami niż obecne (inne kości, inne
    ustawienia "matching" / "contacts" / atlasu, zmienione ręczne oznaczenia), w tym miejscu strony
    jest ramka "PRZELICZ" z powodem i gotową komendą - starego wyniku nie pokazuje (żeby nie mieszać).

KROKI - które części ma strona (settings.json -> "presentation" -> "steps" albo --steps):
    "dane"         - 1. skany -> kości,
    "powierzchnie" - 2. kości z powierzchniami styku metody,
    "porownanie"   - 2b. algorytm vs ręczne oznaczenia (tylko metoda-algorytm),
    "dopasowanie"  - 3. dopasowanie końcowe (pary, macierz, złożenie).
    Bez "dopasowanie" zapis metody (match_surfaces.py) nie jest potrzebny: powierzchnie są z obecnych
    plików (ręczne / contacts/<zestaw>/auto/), a gdy ich brak - PRZELICZ z komendą detect_contacts.py
    (szybkie, bez dopasowania).

Wynik: raporty/prezentacje/<zestaw>/prezentacja_<metoda>.html (wszystkie kroki) albo
prezentacja_<metoda>_<kroki>.html (np. prezentacja_atlas_powierzchnie-porownanie.html) - plotly.js
jest zawsze wklejony w plik, więc działa bez internetu i bez serwera; można go po prostu komuś wysłać.

    from presentation import build, methods
    build("atlas")                                      # -> Path pliku HTML, status sekcji
    build("atlas", steps=["powierzchnie", "porownanie"])  # bez dopasowania końcowego
"""

from __future__ import annotations

import html
import json
import shutil
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path

import numpy as np

import config
import contacts as ct
import matching_surfaces as ms
from bones import Bone, load_bones
from config import read_json

MANUAL = "manual"
SNAPSHOT_FORMAT = "bones-smm-method/1"
TOOL_KEYS = ("brush_au", "wand_angle_deg")      # ustawienia narzędzia do oznaczania - nie wpływają na wynik
DISPLAY_KEYS = ("max_shown",)                    # ustawienia samego wyglądu - nie wpływają na wynik
PARAM_NAMES = {f.name for f in fields(ms.SurfaceMatchParams)}
STEPS = ("dane", "powierzchnie", "porownanie", "dopasowanie")     # kroki strony, w tej kolejności
STEP_ALIASES = {"porównanie": "porownanie", "data": "dane", "contacts": "powierzchnie",
                "compare": "porownanie", "matching": "dopasowanie"}


@dataclass(frozen=True)
class PresentationParams:
    """Sekcja "presentation" settings.json.

    Attributes:
        steps: kroki strony (podzbiór STEPS), np. ("powierzchnie", "porownanie") = bez dopasowania.
    """
    steps: tuple = STEPS


def parse_steps(values) -> tuple[str, ...]:
    """Kroki z listy tekstów (także "a,b" i polskie znaki) -> krotka w kolejności STEPS.

    Raises:
        ValueError: nieznany krok albo pusta lista.
    """
    out = set()
    for v in values:
        for name in str(v).replace(",", " ").split():
            name = STEP_ALIASES.get(name.lower(), name.lower())
            if name not in STEPS:
                raise ValueError(f"Nieznany krok {name!r}, dostępne: {', '.join(STEPS)}")
            out.add(name)
    if not out:
        raise ValueError(f"Pusta lista kroków, dostępne: {', '.join(STEPS)}")
    return tuple(s for s in STEPS if s in out)


def default_steps() -> tuple[str, ...]:
    """Kroki z settings.json (sekcja "presentation", klucz "steps").

    Raises:
        SystemExit: nieznany klucz albo krok.
    """
    try:
        return parse_steps(config.presentation_params().steps)
    except ValueError as e:
        raise SystemExit(f"settings.json, sekcja „presentation”: {e}") from None


# -----------------
# METODY I ICH USTAWIENIA
# -----------------

def methods() -> list[str]:
    """Wszystkie metody prezentacji: "manual" + algorytmy wykrywania (contacts.DETECTORS)."""
    return [MANUAL, *ct.DETECTORS]


def method_key(source: str, contact_method: str) -> str | None:
    """Metoda dla trybu dopasowania: "manual" -> "manual", "auto" -> nazwa algorytmu, "manual-auto" -> None."""
    if source == "manual":
        return MANUAL
    if source == "auto":
        return contact_method
    return None


def current_method() -> str | None:
    """Metoda z settings.json: "manual" przy matching.source = "manual", inaczej contacts.method
    (None przy "manual-auto" - mieszanka nie ma osobnego zapisu)."""
    return method_key(config.matching_params().source, config.contact_params().method)


def report_method(rep: dict) -> str | None:
    """Metoda, którą policzono raport dopasowania (matching.json); None = "manual-auto"."""
    return method_key(rep.get("params", {}).get("source", "manual"), rep.get("contact_params", {}).get("method"))


def method_label(method: str | None) -> str:
    """Nazwa metody do strony i konsoli."""
    if method is None:
        return "ręczne, a gdy brak - z algorytmu (manual-auto)"
    return "ręczne oznaczenia" if method == MANUAL else f"algorytm „{method}”"


def method_settings(method: str) -> tuple[ms.SurfaceMatchParams, ct.ContactParams]:
    """Obecne ustawienia (settings.json + zestawu) ustawione na metodę `method`.

    Args:
        method: "manual" albo nazwa algorytmu.

    Returns:
        (ustawienia dopasowania z source = "manual" / "auto", ustawienia wykrywania z method = metoda).

    Raises:
        ValueError: nieznana metoda.
    """
    if method not in methods():
        raise ValueError(f"Nieznana metoda {method!r}, dostępne: {', '.join(methods())}")
    p, cp = config.matching_params(), config.contact_params()
    if method == MANUAL:
        return replace(p, source="manual"), cp
    return replace(p, source="auto"), replace(cp, method=method)


def recompute_command(method: str) -> str:
    """Komenda, która liczy wszystko dla metody (powierzchnie styku + dopasowanie + zapis metody)."""
    return f"python scripts/match_surfaces.py --method {method}{config.dataset_flag()}"


def detect_command(method: str) -> str:
    """Komenda, która liczy same powierzchnie styku algorytmu (bez dopasowania)."""
    return f"python scripts/detect_contacts.py --method {method}{config.dataset_flag()}"


def prepare_command() -> str:
    """Komenda przygotowania kości zestawu."""
    return f"python scripts/prepare_data.py{config.dataset_flag()}"


def snapshot_dir(method: str, out_dir: Path | None = None) -> Path:
    """Folder zapisu metody: out/3d/<zestaw>/methods/<metoda>/."""
    return Path(out_dir or config.DATA_OUT) / "methods" / method


def last_matching_path(out_dir: Path | None = None) -> Path:
    """Ostatni przebieg match_surfaces.py (dowolna metoda, także wybrane kości): out/3d/<zestaw>/matching.json."""
    return Path(out_dir or config.DATA_OUT) / "matching.json"


def find_matching(method: str | None = None, last: bool = False,
                  out_dir: Path | None = None) -> tuple[Path, str | None, list[str]]:
    """Który wynik dopasowania pokazać / ocenić (tune_matching.py, compare_reference.py) - bez liczenia.

    Domyślnie zapis metody out/3d/<zestaw>/methods/<metoda>/matching.json (metoda z settings.json albo
    `method`), więc po przełączeniu metody nie trzeba nic przeliczać, o ile ta metoda była już policzona.
    Gdy zapisu metody brak, a ostatni przebieg (matching.json) jest z tej metody - ostatni przebieg.

    Args:
        method:  metoda (None = z settings.json).
        last:    zawsze ostatni przebieg out/3d/<zestaw>/matching.json (np. tylko wybrane kości, manual-auto).
        out_dir: folder wyników (None = DATA_OUT).

    Returns:
        (plik matching.json, metoda, którą go policzono (None = manual-auto), uwagi do wypisania).

    Raises:
        SystemExit: brak wyniku dla tej metody (komunikat podaje komendę, która go policzy).
    """
    last_path = last_matching_path(out_dir)
    last_rep = read_json(last_path) if last_path.exists() else None
    last_method = report_method(last_rep) if last_rep is not None else None
    notes = []
    if not last and method is None:
        method = current_method()
        if method is None:
            notes.append("settings.json: matching.source = \"manual-auto\" (mieszanka nie ma osobnego zapisu) "
                         "- pokazuję ostatni przebieg")
            last = True
    if last:
        if last_rep is None:
            raise SystemExit(f"Brak {config._rel(last_path)} - uruchom najpierw: "
                             f"python scripts/match_surfaces.py{config.dataset_flag()}")
        return last_path, last_method, notes
    snap = snapshot_dir(method, out_dir) / "matching.json"
    same_last = last_rep is not None and last_method == method
    if snap.exists():
        if same_last and last_path.stat().st_mtime > snap.stat().st_mtime + 1:
            notes.append("ostatni przebieg tej metody (matching.json, np. tylko wybrane kości) jest nowszy niż "
                         "jej pełny zapis - pokaż go: --last")
        return snap, method, notes
    if same_last:
        notes.append("brak pełnego zapisu metody (dopasowano tylko wybrane kości?) - pokazuję ostatni przebieg")
        return last_path, method, notes
    other = (f"; ostatni przebieg (matching.json) to: {method_label(last_method)} - pokaż go: --last"
             if last_rep is not None else "")
    raise SystemExit(f"Brak wyników dopasowania dla metody: {method_label(method)} - policz: "
                     f"{recompute_command(method)}{other}")


def presentations_dir() -> Path:
    """Folder prezentacji zestawu: raporty/prezentacje/<zestaw>/ (settings.json -> "paths" -> "presentations")."""
    return config.PRESENTATIONS_ROOT / config.DATASET


def page_path(method: str, out_dir: Path | None = None, steps=STEPS) -> Path:
    """Plik prezentacji metody: raporty/prezentacje/<zestaw>/prezentacja_<metoda>.html (wszystkie kroki)
    albo prezentacja_<metoda>_<krok>-<krok>.html (tylko wybrane kroki); `out_dir` = inny folder."""
    steps = tuple(s for s in STEPS if s in steps)
    suffix = "" if steps == STEPS else "_" + "-".join(steps)
    return Path(out_dir or presentations_dir()) / f"prezentacja_{method}{suffix}.html"


def _norm(obj):
    """Wartość jak po zapisie i odczycie JSON (krotki -> listy, NaN -> None) - do porównywania."""
    return json.loads(json.dumps(ms.json_safe(obj), ensure_ascii=False))


def _diff(then: dict, now: dict, skip=(), cls=None) -> list[str]:
    """Różnice dwóch słowników ustawień: ["klucz: było -> jest", ...] (bez kluczy `skip`).

    Klucz, którego w zapisanym wyniku nie ma (ustawienie dodane do programu później), liczy się jako
    wartość domyślna z dataclassy `cls` - wynik liczono wtedy tak, jakby miał wartość domyślną.
    """
    if cls is not None:
        then = {**_norm(asdict(cls())), **then}
    keys = sorted((set(then) | set(now)) - set(skip))
    return [f"{k}: {then.get(k)} → {now.get(k)}" for k in keys if then.get(k) != now.get(k)]


def atlas_key() -> str | None:
    """Odcisk atlasu (baza modeli + sekcje "atlas" i "surface_features") albo None, gdy nie da się go policzyć."""
    try:
        import atlas
        return atlas.data_key()
    except Exception:       # brak bazy / zła ścieżka - wtedy porównujemy po prostu None
        return None


def _uses_atlas(method: str, p: ms.SurfaceMatchParams) -> bool:
    """Czy wynik metody zależy od atlasu (algorytm "atlas" albo dopasowanie z use_atlas)."""
    return method == "atlas" or p.use_atlas


# -----------------
# ZAPIS METODY (robi go match_surfaces.py)
# -----------------

def save_snapshot(method: str, rep: dict, bones: list[Bone], p: ms.SurfaceMatchParams, cp: ct.ContactParams,
                  out_dir: Path | None = None, root: Path | None = None) -> Path:
    """Zapisuje wynik metody do out/3d/<zestaw>/methods/<metoda>/ (poprzedni zapis tej metody jest usuwany).

    Pliki: matching.json (jak out/3d/<zestaw>/matching.json), contacts/<id>.json - powierzchnie styku każdej
    kości dokładnie takie, jak przy dopasowaniu (format contacts.save_contacts), snapshot.json - zestaw,
    data, odciski siatek WSZYSTKICH kości i odcisk atlasu (gdy metoda z niego korzysta).

    Args:
        method:  metoda (method_key).
        rep:     raport dopasowania (matching_surfaces.report).
        bones:   WSZYSTKIE kości zestawu (zapis tylko po pełnym przebiegu).
        p, cp:   ustawienia, z którymi liczono.
        out_dir: folder wyników (None = DATA_OUT).
        root:    folder oznaczeń zamiast contacts/ (testy).

    Returns:
        Folder zapisu.
    """
    d = snapshot_dir(method, out_dir)
    if d.exists():
        shutil.rmtree(d)
    (d / "contacts").mkdir(parents=True)
    saved = []
    for b in bones:
        c, _ = ms.bone_contacts(b, p.source, cp, root)      # z cache - właśnie policzone przez dopasowanie
        if c is None:
            continue
        ct.save_contacts(b.mesh, c, d / "contacts" / f"{b.id}.json", ct.scan_frames(b))
        saved.append(b.id)
    meta = {"format": SNAPSHOT_FORMAT, "method": method, "dataset": config.DATASET,
            "created": datetime.now().isoformat(timespec="seconds"),
            "bones": {b.id: ct.mesh_fingerprint(b.mesh) for b in bones}, "contacts": saved,
            "atlas_key": atlas_key() if _uses_atlas(method, p) else None}
    (d / "matching.json").write_text(json.dumps(ms.json_safe(rep), indent=1, ensure_ascii=False, allow_nan=False),
                                     encoding="utf-8")
    (d / "snapshot.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return d


# -----------------
# CZY WYNIKI SĄ AKTUALNE
# -----------------

def check_data(bones: list[Bone], out_dir: Path | None = None) -> list[str]:
    """Czy kości (prepare_data.py) pasują do obecnych ustawień "scans", "merging" i "features".

    Args:
        bones:   kości z dysku.
        out_dir: folder wyników (None = DATA_OUT).

    Returns:
        Powody do przeliczenia (pusta lista = aktualne).
    """
    from visualization_3d import load_report

    if not bones:
        return [f"brak przygotowanych kości w zestawie „{config.DATASET}”"]
    report = load_report(Path(out_dir or config.DATA_OUT))
    if report is None:
        return ["brak merging.json (raportu prepare_data.py)"]
    out = []
    ids = {b["id"] for b in report.get("bones", [])}
    if ids != {b.id for b in bones}:
        out.append("kości na dysku nie zgadzają się z merging.json (przerwane albo stare prepare_data.py)")
    saved = report.get("settings")
    if saved is None:
        out.append("merging.json z poprzedniej wersji programu (bez zapisanych ustawień) - nie da się sprawdzić, "
                   "z jakimi ustawieniami powstały kości")
        return out
    now = {"scans": config.scan_params(), "merging": config.merge_params(), "features": config.feature_params()}
    for sec, values in now.items():
        diff = _diff(_norm(saved.get(sec, {})), _norm(asdict(values)), cls=type(values))
        if diff:
            out.append(f"settings.json, sekcja „{sec}”: {', '.join(diff)}")
    return out


def load_snapshot(method: str, bones: list[Bone], out_dir: Path | None = None,
                  root: Path | None = None) -> tuple[dict | None, list[str]]:
    """Wczytuje zapis metody i sprawdza, czy pasuje do obecnych kości, ustawień i oznaczeń.

    Args:
        method:  metoda.
        bones:   obecne kości z dysku (load_bones).
        out_dir: folder wyników (None = DATA_OUT).
        root:    folder oznaczeń zamiast contacts/ (testy).

    Returns:
        (zapis albo None, powody do przeliczenia). Zapis = {"meta", "report", "params": SurfaceMatchParams,
        "contacts": {id kości: Contacts}}; pokazywać go wolno TYLKO przy pustej liście powodów.
    """
    d = snapshot_dir(method, out_dir)
    if not (d / "matching.json").exists() or not (d / "snapshot.json").exists():
        return None, [f"brak wyników dopasowania dla metody: {method_label(method)}"]
    meta, rep = read_json(d / "snapshot.json"), read_json(d / "matching.json")
    if meta.get("format") != SNAPSHOT_FORMAT:
        return None, ["zapis metody w starym formacie"]
    p_now, cp_now = method_settings(method)
    reasons = []

    now_fp = {b.id: ct.mesh_fingerprint(b.mesh) for b in bones}
    then_fp = meta.get("bones", {})
    if now_fp != then_fp:
        added, gone = sorted(set(now_fp) - set(then_fp)), sorted(set(then_fp) - set(now_fp))
        changed = sorted(k for k in set(now_fp) & set(then_fp) if now_fp[k] != then_fp[k])
        what = [f"nowe: {', '.join(added)}" if added else "", f"usunięte: {', '.join(gone)}" if gone else "",
                f"zmienione siatki: {', '.join(changed)}" if changed else ""]
        reasons.append("kości zmieniły się od dopasowania (prepare_data.py) - " + "; ".join(w for w in what if w))

    diff = _diff(_norm(rep.get("params", {})), _norm(asdict(p_now)), skip=DISPLAY_KEYS, cls=ms.SurfaceMatchParams)
    if diff:
        reasons.append(f"settings.json, sekcja „matching”: {', '.join(diff)}")
    cdiff = []
    if method != MANUAL or p_now.use_atlas:
        skip = TOOL_KEYS + (("method", "exclude") if method == MANUAL else ())
        cdiff = _diff(_norm(rep.get("contact_params", {})), _norm(asdict(cp_now)), skip=skip, cls=ct.ContactParams)
        if cdiff:
            reasons.append(f"settings.json, sekcja „contacts”: {', '.join(cdiff)}")
    if _uses_atlas(method, p_now) and meta.get("atlas_key") != atlas_key():
        reasons.append("atlas zmienił się (baza modeli albo sekcje „atlas” / „surface_features”)")

    contacts, changed = {}, []
    stored = set(meta.get("contacts", []))
    for b in bones:
        if b.id not in stored:
            if method == MANUAL:
                now = ct.load_contacts(b.id, b.mesh, "manual", root=root, frames=ct.scan_frames(b))
                if now is not None and now.labels.any():
                    changed.append(b.label)
            continue
        c = ct.load_contacts(b.id, b.mesh, path=d / "contacts" / f"{b.id}.json")
        if c is None or c.remapped:
            continue                        # siatka inna - powód już jest wyżej (odciski kości)
        if method == MANUAL:
            now = ct.load_contacts(b.id, b.mesh, "manual", root=root, frames=ct.scan_frames(b))
            if now is None or not np.array_equal(now.labels, c.labels):
                changed.append(b.label)
        elif not cdiff and c.detector != _norm(ct.detector_info(b, method, cp_now)):
            changed.append(b.label)
        contacts[b.id] = c
    if changed:
        what = "ręczne oznaczenia zmieniły się" if method == MANUAL else "wynik algorytmu nieaktualny"
        reasons.append(f"{what} po dopasowaniu: {', '.join(changed)}")
    params = ms.SurfaceMatchParams(**{k: v for k, v in rep.get("params", {}).items() if k in PARAM_NAMES})
    return {"meta": meta, "report": rep, "params": params, "contacts": contacts}, reasons


def current_contacts(method: str, bones: list[Bone], root: Path | None = None) -> tuple[dict | None, list[str]]:
    """Powierzchnie styku metody z obecnych plików - gdy zapisu metody brak, ale same powierzchnie są aktualne.

    Nic nie liczy: algorytm tylko z cache contacts/<zestaw>/auto/, i tylko gdy policzony tą metodą
    z obecnymi ustawieniami na obecnej siatce.

    Returns:
        ({id kości: Contacts albo None} albo None, powody do przeliczenia).
    """
    _, cp = method_settings(method)
    out, stale = {}, []
    for b in bones:
        if method == MANUAL:
            c = ct.load_contacts(b.id, b.mesh, "manual", root=root, frames=ct.scan_frames(b))
        elif ct.is_excluded(b, cp):
            c = None
        else:
            c = ct.load_contacts(b.id, b.mesh, "auto", root=root, frames=ct.scan_frames(b))
            if c is None or c.remapped or c.detector != _norm(ct.detector_info(b, method, cp)):
                stale.append(b.label)
                continue
        out[b.id] = c if c is not None and c.labels.any() else None
    if stale:
        return None, [f"brak aktualnego wyniku algorytmu „{method}” (obecne ustawienia) dla: {', '.join(stale)}"]
    return out, []


def method_status(method: str, bones: list[Bone], data_ok: bool, matching: bool = True) -> tuple[bool, str]:
    """Krótki status metody do nagłówka: (aktualne?, opis). `matching` = False: tylko powierzchnie styku."""
    if not data_ok:
        return False, "najpierw prepare_data"
    if not matching:
        contacts, _ = current_contacts(method, bones)
        return (True, "aktualne") if contacts is not None else (False, "brak - detect_contacts")
    snap, reasons = load_snapshot(method, bones)
    if snap is None:
        return False, "brak - przelicz"
    if reasons:
        return False, "nieaktualne - przelicz"
    return True, f"aktualne ({snap['meta'].get('created', '?').replace('T', ' ')})"


# -----------------
# STRONA
# -----------------

def _e(text) -> str:
    """Tekst bezpieczny w HTML."""
    return html.escape(str(text))


def _steps_flag(steps) -> str:
    """" --steps a b" do podpowiedzi - pusty tekst, gdy kroki jak w settings.json."""
    if steps is None or tuple(steps) == default_steps():
        return ""
    return " --steps " + " ".join(steps)


def todo_box(reasons: list[str], command: str, method: str, steps=None) -> str:
    """Ramka "PRZELICZ" (HTML): powody, komenda do uruchomienia i jak potem odświeżyć stronę metody `method`."""
    items = "".join(f"<li>{_e(r)}</li>" for r in reasons)
    again = f"python scripts/presentation.py --method {method}{config.dataset_flag()}{_steps_flag(steps)}"
    return (f"<div class='todo'><b>PRZELICZ</b> - tej części nie pokazuję, bo nie pasowałaby do reszty strony "
            f"(inne ustawienia albo dane).<ul>{items}</ul>Uruchom: <code>{_e(command)}</code>, "
            f"potem ponownie: <code>{_e(again)}</code></div>")


def _kv_table(title: str, values: dict) -> str:
    """Tabelka „klucz: wartość” z nagłówkiem."""
    rows = "".join(f"<tr><td>{_e(k)}</td><td>{_e(json.dumps(v, ensure_ascii=False))}</td></tr>"
                   for k, v in values.items())
    return f"<div><table class='kv'><tr><th colspan='2'>{_e(title)}</th></tr>{rows}</table></div>"


def settings_html(method: str, p: ms.SurfaceMatchParams, cp: ct.ContactParams, matching: bool = True) -> str:
    """Wszystkie ustawienia, z którymi powstała strona (zwinięte w <details>; bez "matching", gdy bez dopasowania)."""
    parts = [_kv_table("scans", _norm(asdict(config.scan_params()))),
             _kv_table("merging", _norm(asdict(config.merge_params()))),
             _kv_table("features", _norm(asdict(config.feature_params())))]
    if method != MANUAL or (matching and p.use_atlas):
        c = {k: v for k, v in _norm(asdict(cp)).items() if k not in TOOL_KEYS}
        parts.append(_kv_table("contacts (wykrywanie)", c))
    if matching:
        parts.append(_kv_table("matching", {k: v for k, v in _norm(asdict(p)).items() if k not in DISPLAY_KEYS}))
    if method == "atlas" or (matching and p.use_atlas):
        parts.append(_kv_table("atlas", _norm(asdict(config.atlas_params()))))
    return ("<details><summary>Pokaż wszystkie ustawienia (settings.json"
            + (f" + scanned_bones/{_e(config.DATASET)}/settings.json" if config.DATASET_SETTINGS else "")
            + ")</summary><div class='grid'>" + "".join(parts) + "</div></details>")


def _status_html(steps: list[tuple[str, bool | None, str]]) -> str:
    """Lista kroków ze statusem ✓ / przelicz / – (None = pominięty albo nie dotyczy)."""
    def cell(ok, text):
        cls, mark = ("muted", "–") if ok is None else ("ok", "✓") if ok else ("bad", "✗")
        return f"<td class='{cls}'>{mark} {_e(text)}</td>"
    rows = "".join(f"<tr><td>{_e(name)}</td>{cell(ok, text)}</tr>" for name, ok, text in steps)
    return f"<table class='kv'>{rows}</table>"


def _surfaces_table(bones: list[Bone], contacts: dict) -> str:
    """Tabela: kość, liczba powierzchni styku i ich pola."""
    rows = []
    for nr, b in enumerate(bones, 1):
        c = contacts.get(b.id)
        stats = ct.surface_stats(b.mesh, c.labels) if c is not None else []
        areas = ", ".join(f"{s['area_au2']:.0f}" for s in stats) or "-"
        rows.append(f"<tr><td>{nr}.</td><td>{_e(b.label)}</td><td>{len(stats)}</td><td>{areas}</td></tr>")
    return ("<table class='kv'><tr><th></th><th>kość</th><th>powierzchni</th><th>pola [a.u.²]</th></tr>"
            + "".join(rows) + "</table>")


def _compare_table(rows: list[tuple[int, str, dict]]) -> str:
    """Tabela IoU / precyzja / czułość na kość + średnia."""
    body = "".join(f"<tr><td>{nr}.</td><td>{_e(label)}</td><td>{s['iou']:.2f}</td><td>{s['precision']:.2f}</td>"
                   f"<td>{s['recall']:.2f}</td></tr>" for nr, label, s in rows)
    mean = {k: sum(s[k] for _, _, s in rows) / len(rows) for k in ("iou", "precision", "recall")}
    body += (f"<tr><th></th><th>średnio ({len(rows)})</th><th>{mean['iou']:.2f}</th><th>{mean['precision']:.2f}</th>"
             f"<th>{mean['recall']:.2f}</th></tr>")
    return ("<table class='kv'><tr><th></th><th>kość</th><th>IoU</th><th>precyzja</th><th>czułość</th></tr>"
            + body + "</table>")


def _chosen_table(rep: dict) -> str:
    """Tabela proponowanego złożenia (pary "chosen" z matching.json)."""
    chosen = [r for r in rep.get("pairs", []) if r.get("chosen")]
    if not chosen:
        return "<p>Proponowane złożenie: brak par powyżej progu assembly_min_score.</p>"
    rows = "".join(f"<tr><td>{k}.</td><td><b>{r['score']:.0f}</b>{_e(r.get('score_unit', ''))}</td>"
                   f"<td>{_e(r['a_name'])}</td><td>{_e(r['b_name'])}</td><td>{r['contact_au2']:.0f}</td>"
                   f"<td>{r['collision_au2']:.0f}</td><td>{r['rmse_au']:.2f}</td></tr>"
                   for k, r in enumerate(chosen, 1))
    used = {key for r in chosen for key in (r["a"], r["b"])}
    free = [s["name"] for s in rep.get("surfaces", []) if s["key"] not in used]
    tail = f"<p>Przełomy bez pary ({len(free)}): {_e('; '.join(free))}</p>" if free else ""
    return ("<p><b>Proponowane złożenie</b> (każdy przełom i każda para kości najwyżej raz):</p>"
            "<table class='kv'><tr><th></th><th>wynik</th><th>A</th><th>B</th><th>styk [a.u.²]</th>"
            "<th>kolizja</th><th>RMSE [a.u.]</th></tr>" + rows + "</table>" + tail)


def _info_box(text: str) -> str:
    """Szara ramka z informacją (krok nie dotyczy tej metody / brak danych do porównania)."""
    return f"<div class='infobox'>{text}</div>"


def build(method: str, out: Path | None = None, open_browser: bool = False, log=print,
          steps=None) -> tuple[Path, dict]:
    """Buduje prezentację metody (opis: docstring modułu). Nic nie przelicza.

    Args:
        method:       "manual" albo nazwa algorytmu.
        out:          plik wynikowy (None = page_path(metoda, kroki)).
        open_browser: otwórz po zapisie.
        log:          komunikaty w konsoli (None = cisza).
        steps:        kroki strony (podzbiór STEPS; None = settings.json "presentation" -> "steps").

    Returns:
        (ścieżka pliku, {krok: True = pokazany, False = "przelicz"} - tylko wybrane kroki, które dotyczą
        tej metody; "porownanie" bez ręcznych oznaczeń albo dla "manual" nie jest liczone).

    Raises:
        ValueError: nieznana metoda albo krok.
    """
    from match_tool import BEST_PAIRS_NOTE, matching_description, matrix_note
    from visualization_3d import (bones_figure, compare_layers, load_report, match_matrix_figure, matching_figure,
                                  merging_figure, overlap_figure, write_page)

    log = log or (lambda *_: None)
    steps = default_steps() if steps is None else parse_steps(steps)
    want_match = "dopasowanie" in steps
    p_now, cp_now = method_settings(method)
    view = replace(config.view3d_params(), embed_plotly=True)      # przenośny plik: plotly.js w środku
    bones = load_bones()
    numbers = list(range(1, len(bones) + 1))
    data_reasons = check_data(bones)
    if not want_match:
        snap, snap_reasons = None, []
    elif data_reasons:
        snap, snap_reasons = None, ["najpierw przelicz kości (krok 1)"]
    else:
        snap, snap_reasons = load_snapshot(method, bones)
    snap_ok = snap is not None and not snap_reasons
    status, todo = {}, []          # todo: [(powody, komenda)] do konsoli
    sections = []

    def box(reasons, command):
        todo.append((reasons, command))
        return todo_box(reasons, command, method, steps)

    # 1. DANE
    if "dane" in steps:
        if data_reasons:
            data_section = [box(data_reasons, prepare_command())]
            status["dane"] = False
        else:
            report = load_report(config.DATA_OUT)
            merged = sum(b.n_scans > 1 for b in bones)
            noise = sum(b.meta.get("noise_removed_au2", 0) for b in bones)
            merge = report["settings"]["merging"]
            info = (f"<p>{len(report['scans'])} skanów → <b>{len(bones)} kości</b>; z kilku skanów: {merged}; "
                    f"usunięty szum: {noise:.0f} a.u.². Łączenie skanów: "
                    f"<b>{'włączone' if merge.get('enabled') else 'wyłączone'}</b>"
                    + (f" (ta sama kość, gdy nakładanie ≥ {merge.get('min_overlap')})" if merge.get("enabled") else "")
                    + ".</p>")
            data_section = [info, merging_figure(bones, view)]
            if report.get("pairs"):
                data_section.append(overlap_figure(report, view))
            status["dane"] = True
        sections.append(("1. Dane: skany → kości", "Skany każdej kości nałożone na siebie (każdy w innym kolorze), "
                         "czerwony = usunięty szum; macierz nakładania par skanów.", data_section))

    # 2. POWIERZCHNIE STYKU (potrzebne też do porównania)
    contacts, c_reasons, origin, c_command = None, [], "", prepare_command()
    if "powierzchnie" in steps or "porownanie" in steps:
        if data_reasons:
            c_reasons = ["najpierw przelicz kości (krok 1)"]
        elif snap_ok:
            contacts = {b.id: snap["contacts"].get(b.id) for b in bones}
            origin = "te same, które dopasowywano (zapis metody)"
        else:
            contacts, c_reasons = current_contacts(method, bones)
            origin = ("z obecnych plików (bez dopasowania)" if not want_match
                      else "z obecnych plików (dopasowanie jeszcze nie policzone - krok 3)")
            c_command = recompute_command(method) if want_match else detect_command(method)

    if "powierzchnie" in steps:
        if contacts is None:
            sections.append(("2. Kości i powierzchnie styku", "", box(c_reasons, c_command)))
            status["powierzchnie"] = False
        else:
            sections.append((
                "2. Kości i powierzchnie styku",
                f"Powierzchnie styku: {_e(method_label(method))} - {_e(origin)}. Lista „Pokaż” wybiera kość, "
                "najechanie myszą = szczegóły.",
                [bones_figure(bones, contacts, "scan", view, numbers), _surfaces_table(bones, contacts)]))
            status["powierzchnie"] = True

    # 2b. ALGORYTM VS RĘCZNE
    compare_text = "pominięte"
    if "porownanie" in steps:
        title = "2b. Algorytm vs ręczne oznaczenia"
        if method == MANUAL:
            compare_text = "nie dotyczy (metoda = ręczne)"
            sections.append((title, "", _info_box(
                "Porównanie dotyczy algorytmu - ta strona pokazuje same ręczne oznaczenia. Wybierz metodę-algorytm, "
                f"np. <code>python scripts/presentation.py --method atlas{_e(config.dataset_flag())}"
                f"{_e(_steps_flag(steps))}</code>.")))
        elif contacts is None:
            sections.append((title, "", box(c_reasons, c_command)))
            status["porownanie"] = False
            compare_text = "przelicz"
        else:
            manual = {b.id: ct.load_contacts(b.id, b.mesh, "manual", frames=ct.scan_frames(b)) for b in bones}
            manual = {k: (c if c is not None and c.labels.any() else None) for k, c in manual.items()}
            rows = []
            for nr, b in zip(numbers, bones):
                _, score = compare_layers(b, contacts.get(b.id), manual.get(b.id))
                if score is not None:
                    rows.append((nr, b.label, score))
            if rows:
                sections.append((
                    title,
                    "Zielony = zgodne, pomarańczowy = tylko algorytm (fałszywy alarm), niebieski = tylko ręczne "
                    "(przeoczone), szary = kość bez ręcznych oznaczeń. IoU = pokrycie, precyzja = ile z wykrytego "
                    "to styk, czułość = ile styku wykryto (1.00 = idealnie).",
                    [bones_figure(bones, contacts, "scan", view, numbers, reference=manual), _compare_table(rows)]))
                status["porownanie"] = True
                compare_text = f"aktualne ({len(rows)} kości z ręcznymi oznaczeniami)"
            else:
                compare_text = "brak ręcznych oznaczeń"
                sections.append((title, "", _info_box(
                    "Żadna kość nie ma ręcznych oznaczeń - nie ma z czym porównać. Oznacz: "
                    f"<code>python scripts/mark_contacts.py{_e(config.dataset_flag())}</code>.")))

    # 3. DOPASOWANIE
    if want_match:
        if snap_ok:
            rep, p = snap["report"], snap["params"]
            by_id = {b.id: b for b in bones}
            sections += [
                ("3. Dopasowanie: najlepsze pary", f"{_e(matching_description(p, rep))} {BEST_PAIRS_NOTE}",
                 [matching_figure(rep["pairs"], by_id, snap["contacts"], view, p.max_shown), _chosen_table(rep)]),
                ("3b. Dopasowanie: wszystkie pary", matrix_note(p), match_matrix_figure(rep, view))]
            status["dopasowanie"] = True
        else:
            sections.append(("3. Dopasowanie", "", box(snap_reasons, prepare_command() if data_reasons
                                                       else recompute_command(method))))
            status["dopasowanie"] = False

    # NAGŁÓWEK: ta metoda + inne metody tego zestawu
    others = []
    for m in methods():
        ok, text = method_status(m, bones, not data_reasons, want_match) if m != method else \
            (all(status.values()), "ta strona")
        link = f"<a href='{_e(page_path(m, steps=steps).name)}'>{_e(m)}</a>" if m != method else f"<b>{_e(m)}</b>"
        others.append(f"{link}: <span class='{'ok' if ok else 'bad'}'>{_e(text)}</span>")

    def row(key, name, ok_text):
        if key not in steps:
            return name, None, "pominięty (kroki strony)"
        ok = status.get(key)
        return name, ok, ok_text if ok else ("przelicz" if ok is False else compare_text)
    step_rows = [row("dane", "1. Dane (prepare_data.py)", "aktualne"),
                 row("powierzchnie", "2. Powierzchnie styku", "aktualne"),
                 row("porownanie", "2b. Algorytm vs ręczne", compare_text),
                 row("dopasowanie", "3. Dopasowanie (match_surfaces.py)",
                     f"aktualne, policzone {snap['meta'].get('created', '?').replace('T', ' ')}" if snap_ok else "")]
    skipped = [s for s in STEPS if s not in steps]
    intro = ("<p>Wszystko na tej stronie pochodzi z <b>jednej metody</b>, jednego zestawu i jednych ustawień - "
             "części policzone inaczej nie są pokazywane (ramka PRZELICZ).</p>"
             + _status_html(step_rows)
             + (f"<p>Pominięte kroki: {_e(', '.join(skipped))} (settings.json → „presentation” → „steps” "
                "albo <code>--steps</code>).</p>" if skipped else "")
             + "<p>Metody tego zestawu (plik prezentacji obok, w tym samym folderze): "
             + " · ".join(others) + "</p>" + settings_html(method, p_now, cp_now, want_match))
    lead = (f"Zestaw <b>{_e(config.DATASET)}</b> · metoda: <b>{_e(method_label(method))}</b> · "
            f"strona z {datetime.now():%Y-%m-%d %H:%M}")
    path = Path(out) if out else page_path(method, steps=steps)
    write_page(path, f"Prezentacja · {config.DATASET} · {method}", lead,
               [("Metoda i ustawienia", "", intro), *sections], view, open_browser=open_browser)
    for name, ok in status.items():
        log(f"  {name}: {'OK' if ok else 'PRZELICZ'}")
    if "porownanie" in steps and "porownanie" not in status:
        log(f"  porownanie: {compare_text}")
    shown = set()
    for reasons, _ in todo:
        for r in reasons:
            if r not in shown:
                log(f"    - {r}")
                shown.add(r)
    for command in dict.fromkeys(c for _, c in todo):
        log(f"  Przelicz: {command}")
    return path, status

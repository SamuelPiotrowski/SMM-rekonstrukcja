"""
Narzędzie do ręcznego oznaczania powierzchni styku - lokalna strona w przeglądarce.

Uruchamia mały serwer HTTP (tylko na tym komputerze, 127.0.0.1) i otwiera stronę
src/web/contact_tool.html. Strona pokazuje kość w 3D (three.js), a malowanie
zapisuje się przez serwer do contacts/<id_kości>.json (contacts.py).

    from contact_tool import serve
    serve()                       # Ctrl+C w konsoli kończy

API (używa go strona):
    GET  /api/bones                lista kości + stan oznaczeń
    GET  /api/bones/<id>           siatka (base64) + etykiety
    PUT  /api/bones/<id>/contacts  zapis etykiet {labels: base64 uint8, names: {k: nazwa}}
    POST /api/bones/<id>/detect    propozycja algorytmu (nic nie zapisuje)

three.js jest pobierany raz z CDN i zapisywany w DATA_OUT/.cache/web, potem działa bez internetu.
"""

from __future__ import annotations

import base64
import json
import threading
import urllib.request
import uuid
import webbrowser
from collections import OrderedDict
from functools import lru_cache
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit
from zipfile import BadZipFile, ZipFile

import numpy as np

import bone_materials
import contacts as ct
from bones import Bone, load_bones
from bundled_data import bundled_directory
from case_history import CaseHistory
from case_replays import ARCHIVE_NAME, MAX_BYTES, attach_scan_links, export_replays, read_replays
from detailed_matching import DetailedMatching
from fracture_library import catalog, load_case

WEB_DIR = Path(__file__).resolve().parent / "web"
OUR_SCAN_DIR = WEB_DIR.parents[1] / "out" / "wasze-skany"
OUR_SCAN_REPORT = OUR_SCAN_DIR / "assembly-manual.html"
CASE_HISTORY = WEB_DIR.parents[1] / "out" / "historia-skanow.sqlite3"
STARTER_REPLAYS = WEB_DIR.parents[1] / "dane-startowe" / ARCHIVE_NAME
FIBULA_DEMO = WEB_DIR.parents[1] / "out" / "fibula-demo"
MIXED_DEMO = WEB_DIR.parents[1] / "out" / "test-udowa-strzalkowa"
FIBULA_PAIR_DEMO = WEB_DIR.parents[1] / "out" / "test-dwie-strzalkowe"
FIBULA_REJECTION_DEMO = WEB_DIR.parents[1] / "out" / "test-obce-odlamki"
FIBULA_GAP_DEMO = WEB_DIR.parents[1] / "out" / "test-brakujacy-odlamek"
DEMO_PACKS = {FIBULA_DEMO: "pokazy-1-2", FIBULA_PAIR_DEMO: "pokazy-1-2",
              FIBULA_REJECTION_DEMO: "pokazy-3-4", FIBULA_GAP_DEMO: "pokazy-3-4"}
DEMO_RESULTS = {"demo-fibula": FIBULA_DEMO / "result.json",
                "demo-mixed": MIXED_DEMO / "demo-result.json",
                "demo-fibula-pair": FIBULA_PAIR_DEMO / "result.json",
                "demo-fibula-rejection": FIBULA_REJECTION_DEMO / "result.json",
                "demo-fibula-gap": FIBULA_GAP_DEMO / "result.json"}
DEMO_ARCHIVES = {"mixed": (MIXED_DEMO / "dwie-kosci-10-fragmentow.zip", 10),
                 "fibula-pair": (FIBULA_PAIR_DEMO / "dwie-strzalkowe-11-odlamkow.zip", 11),
                 "fibula-rejection": (FIBULA_REJECTION_DEMO / "strzalkowa-z-obcymi-odlamkami.zip", 7),
                 "fibula-gap": (FIBULA_GAP_DEMO / "strzalkowa-z-ubytkiem.zip", 4)}
SCAN_GALLERY = WEB_DIR.parents[1] / "out" / "wizualizacja-gab" / "gab-skany-3d.html"
STATIC = {"/": ("contact_tool.html", "text/html; charset=utf-8"),
          "/library": ("case_start.html", "text/html; charset=utf-8"),
          "/case-start": ("case_start.html", "text/html; charset=utf-8"),
          "/case-start.js": ("case_start.js", "text/javascript; charset=utf-8"),
          "/case-preview.webp": ("case-preview.webp", "image/webp"),
          "/contact_tool.js": ("contact_tool.js", "text/javascript; charset=utf-8"),
          "/contact_tool.css": ("contact_tool.css", "text/css; charset=utf-8"),
          "/assembly": ("assembly.html", "text/html; charset=utf-8"),
          "/assembly.js": ("assembly.js", "text/javascript; charset=utf-8")}
THREE_VERSION = "0.170.0"
VENDOR = {"/vendor/three.module.js": f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.module.js",
          "/vendor/OrbitControls.js":
              f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/examples/jsm/controls/OrbitControls.js"}


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def view_rotation(mesh) -> np.ndarray:
    """Obrót do widoku: najdłuższa oś kości poziomo (X), najcieńsza w stronę kamery (Z)."""
    v = np.asarray(mesh.vertices)
    c = v - v.mean(axis=0)
    _, _, vt = np.linalg.svd(c[:: max(1, len(c) // 20000)], full_matrices=False)
    r = vt.copy()
    if np.linalg.det(r) < 0:
        r[2] *= -1
    return r


class ToolState:
    """Kości wczytane do pamięci + ustawienia narzędzia."""

    def __init__(self, out_dir: Path | None, params: ct.ContactParams, bones=None, poses=None):
        self.params = params
        self.out_dir = out_dir
        self.lock = threading.Lock()
        self.match_lock = threading.Lock()
        self.matches = OrderedDict()
        self.reference_poses = poses
        self.bones: dict[str, Bone] = {b.id: b for b in (load_bones(out_dir) if bones is None else bones)}
        self.order = list(self.bones)

    def poses(self):
        if self.reference_poses is not None:
            return self.reference_poses
        if self.out_dir and (self.out_dir / "reference_poses.json").is_file():
            return json.loads((self.out_dir / "reference_poses.json").read_text(encoding="utf-8"))
        return {}

    def bone(self, bone_id: str) -> Bone:
        if bone_id not in self.bones:
            raise KeyError(bone_id)
        return self.bones[bone_id]

    def summary(self) -> dict:
        items = []
        for nr, bid in enumerate(self.order, 1):
            b = self.bones[bid]
            c = ct.load_contacts(bid, b.mesh)
            stats = ct.surface_stats(b.mesh, c.labels) if c else []
            items.append({
                "nr": nr, "id": bid, "label": b.label, "scans": b.n_scans,
                "area_mm2": (b.meta.get("stats", {}).get("area_mm2") or b.mesh.get_surface_area()),
                "extent_mm": b.meta.get("stats", {}).get("extent_mm"),
                "vertices": len(b.mesh.vertices),
                "surfaces": [{"id": s["id"], "name": c.name(s["id"]), "area_mm2": s["area_mm2"]} for s in stats],
                "updated": c.updated if c else None,
                "auto": ct.contacts_path(bid, "auto").exists(),
            })
        return {"bones": items, "palette": list(ct.PALETTE), "detectors": list(ct.DETECTORS),
                "params": {"brush_mm": self.params.brush_mm, "wand_angle_deg": self.params.wand_angle_deg}}

    def bone_payload(self, bone_id: str) -> dict:
        b = self.bone(bone_id)
        m = b.mesh
        colors = (np.asarray(m.vertex_colors) * 255).round().astype(np.uint8) if m.has_vertex_colors() \
            else np.full((len(m.vertices), 3), 220, dtype=np.uint8)
        c = ct.load_contacts(bone_id, m)
        labels = c.labels if c else np.zeros(len(m.vertices), dtype=np.uint8)
        return {
            "id": bone_id, "label": b.label,
            "positions": _b64(np.asarray(m.vertices, dtype=np.float32)),
            "normals": _b64(np.asarray(m.vertex_normals, dtype=np.float32)),
            "indices": _b64(np.asarray(m.triangles, dtype=np.uint32)),
            "colors": _b64(colors),
            "labels": _b64(labels.astype(np.uint8)),
            "names": {str(k): v for k, v in (c.names if c else {}).items()},
            "remapped": bool(c and c.remapped),
            "rotation": view_rotation(m).tolist(),
        }

    def save(self, bone_id: str, body: dict) -> dict:
        b = self.bone(bone_id)
        labels = np.frombuffer(base64.b64decode(body["labels"]), dtype=np.uint8).copy()
        names = {int(k): str(v)[:80] for k, v in body.get("names", {}).items()}
        c = ct.Contacts(bone_id=bone_id, labels=labels, names=names, source="manual")
        with self.lock:
            path = ct.save_contacts(b.mesh, c)
        stats = ct.surface_stats(b.mesh, labels)
        return {"saved": str(path) if path else None,
                "surfaces": [{"id": s["id"], "name": c.name(s["id"]), "area_mm2": s["area_mm2"],
                              "max_diameter_mm": s["max_diameter_mm"]} for s in stats]}

    def detect(self, bone_id: str, method: str | None) -> dict:
        b = self.bone(bone_id)
        method = method or next(iter(ct.DETECTORS))
        c = ct.detect_contacts(b, method, self.params)
        return {"labels": _b64(c.labels), "names": {str(k): v for k, v in c.names.items()}, "method": method}

    def match(self, body: dict) -> dict:
        from fracture_matching import evaluate_reference, match_surfaces

        if not self.match_lock.acquire(blocking=False):
            raise ValueError("Trwa już dopasowanie. Poczekaj na jego wynik.")
        try:
            selected = []
            with self.lock:
                for side in ("a", "b"):
                    choice = body.get(side)
                    if not isinstance(choice, dict) or not isinstance(choice.get("bone"), str):
                        raise ValueError("Wybierz dwa fragmenty i ich przełomy.")
                    bone = self.bone(choice["bone"])
                    surface = choice.get("surface")
                    contacts = ct.load_contacts(bone.id, bone.mesh)
                    if type(surface) is not int or not contacts or surface not in contacts.surface_ids:
                        raise ValueError(f"Zapisz wybraną powierzchnię przełomu dla {bone.label}.")
                    selected.extend((bone, contacts.labels, surface))
            result = match_surfaces(*selected)
            evaluate_reference(result, selected[0].id, selected[3].id, self.poses())
            key = uuid.uuid4().hex
            with self.lock:
                self.matches[key] = result
                if len(self.matches) > 8:
                    self.matches.popitem(last=False)
            return {"id": key}
        finally:
            self.match_lock.release()


def demo_file(path):
    if path.is_file() or path.parent not in DEMO_PACKS:
        return path
    bundle = bundled_directory(DEMO_PACKS[path.parent])
    return bundle / path.parent.name / path.name if bundle else path


def fetch_vendor(path: str, cache_dir: Path) -> bytes:
    """Read packaged Three.js, with the existing cache and CDN as a fallback."""
    packaged = WEB_DIR / "vendor" / Path(path).name
    if packaged.is_file():
        return packaged.read_bytes()
    target = cache_dir / Path(path).name
    if not target.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(VENDOR[path], timeout=30) as r:
            target.write_bytes(r.read())
    return target.read_bytes()


def make_handler(state: ToolState, cache_dir: Path, history_path: Path = CASE_HISTORY, *, detailed=None):
    history = CaseHistory(history_path)
    if STARTER_REPLAYS.is_file():
        history.import_cases(read_replays(STARTER_REPLAYS.read_bytes()))
    active_cases = {}
    solve_lock = threading.Lock()

    @lru_cache(maxsize=1)
    def team_inspection():
        from case_solver import team_result
        return team_result(OUR_SCAN_DIR)

    @lru_cache(maxsize=6)
    def team_source_scan(part_id, scan_id, full_detail):
        from case_solver import team_scan_preview
        return team_scan_preview(OUR_SCAN_DIR, part_id, scan_id, full_detail)

    def process_case(key):
        def progress(message):
            with state.lock:
                active_cases[key] = message

        try:
            from case_upload import load_scan
            from case_solver import make_result

            title, mixed, scans, anchor = history.inputs(key)
            meshes = [(f"{i} · {Path(name).stem[:55]}", load_scan(name, data))
                      for i, (name, data) in enumerate(scans, 1)]
            result = make_result(title, meshes, mixed=mixed, upload=True, progress=progress, target_anchor=anchor)
            attach_scan_links(result, key, scans)
            history.finish(key, result)
        except Exception as error:
            history.fail(key, f"Nie udało się złożyć przypadku. {error}")
        finally:
            with state.lock:
                active_cases.pop(key, None)
            solve_lock.release()

    def start_case(key):
        with state.lock:
            active_cases[key] = "Wczytuję siatki…"
        try:
            threading.Thread(target=process_case, args=(key,), daemon=True).start()
        except Exception:
            with state.lock:
                active_cases.pop(key, None)
            raise

    @lru_cache(maxsize=3)
    def open_case(key):
        if key == "our-scans":
            child = ToolState(OUR_SCAN_DIR, state.params)
        else:
            bones, poses = load_case(key)
            child = ToolState(None, state.params, bones=bones, poses=poses)
        child.lock, child.match_lock, child.matches = state.lock, state.match_lock, state.matches
        return child

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):     # bez logu każdego zapytania w konsoli
            pass

        def _send(self, status, body: bytes, ctype: str, download_name=None, *, headers=None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if download_name is not None:
                self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(download_name, safe=""))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status=HTTPStatus.OK):
            self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json")

        def _error(self, status, msg):
            self._json({"error": msg}, status)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

        def _route(self):
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            return parts

        def _case(self):
            return parse_qs(urlsplit(self.path).query).get("case", [""])[0]

        def _state(self):
            key = self._case()
            return open_case(key) if key else state

        def do_GET(self):
            path = self.path.split("?")[0]
            try:
                if path in STATIC:
                    name, ctype = STATIC[path]
                    return self._send(HTTPStatus.OK, (WEB_DIR / name).read_bytes(), ctype)
                if path == "/api/bone-materials":
                    return self._json(bone_materials.catalog())
                if path.startswith(bone_materials.PREFIX):
                    resource = bone_materials.material(path.removeprefix(bone_materials.PREFIX))
                    if resource is None:
                        return self._error(HTTPStatus.NOT_FOUND, "Materiał jest niedostępny.")
                    file, ctype, section = resource
                    headers = {"X-Content-Type-Options": "nosniff"}
                    if section == "exports":
                        headers["Content-Security-Policy"] = bone_materials.REPORT_CSP
                    download = file.name if section in bone_materials.GROUPS else None
                    return self._send(HTTPStatus.OK, file.read_bytes(), ctype, download, headers=headers)
                if path == "/detailed-matching":
                    try:
                        if detailed is None:
                            raise RuntimeError("Szczegółowe dopasowanie jest niedostępne.")
                        url = detailed.url()
                    except (OSError, RuntimeError) as error:
                        message = ("<!doctype html><html lang='pl'><meta charset='utf-8'>"
                                   "<title>Szczegółowe dopasowanie</title><h1>Szczegółowe dopasowanie</h1>"
                                   f"<p>{escape(str(error))}</p><p><a href='/detailed-matching'>Spróbuj ponownie</a></p>")
                        return self._send(HTTPStatus.SERVICE_UNAVAILABLE, message.encode(), "text/html; charset=utf-8")
                    self.send_response(HTTPStatus.FOUND)
                    self.send_header("Location", url)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if path == "/scan-gallery":
                    if not SCAN_GALLERY.is_file():
                        return self._error(HTTPStatus.NOT_FOUND, "Galeria skanów jest niedostępna.")
                    return self._send(HTTPStatus.OK, SCAN_GALLERY.read_bytes(), "text/html; charset=utf-8")
                if path == "/our-scans":
                    source = parse_qs(urlsplit(self.path).query).get("source", ["manual"])[0]
                    if source not in ("auto", "manual"):
                        return self._error(HTTPStatus.BAD_REQUEST, "Nieznany sposób dopasowania.")
                    report = OUR_SCAN_DIR / f"assembly-{source}.html"
                    if not report.is_file():
                        return self._error(HTTPStatus.NOT_FOUND, "Analiza własnych skanów nie została jeszcze zapisana.")
                    return self._send(HTTPStatus.OK, report.read_bytes(), "text/html; charset=utf-8")
                if path == "/plotly.js":
                    from plotly.offline import get_plotlyjs
                    return self._send(HTTPStatus.OK, get_plotlyjs().encode(), "text/javascript; charset=utf-8")
                if path in VENDOR:
                    try:
                        data = fetch_vendor(path, cache_dir)
                    except OSError as e:
                        return self._error(HTTPStatus.BAD_GATEWAY,
                                           f"Nie udało się pobrać three.js ({e}). Za pierwszym razem potrzebny jest internet.")
                    return self._send(HTTPStatus.OK, data, "text/javascript; charset=utf-8")
                parts = self._route()
                if parts == ["api", "library"]:
                    return self._json([entry for entry, _ in catalog().values()])
                if parts == ["api", "history"]:
                    with state.lock:
                        running = set(active_cases)
                    listing = history.list()
                    for case in listing["cases"]:
                        case["running"] = case["id"] in running and not case["done"]
                    return self._json(listing)
                if parts[:3] == ["api", "history", "replays"] and len(parts) in (3, 4):
                    key = parts[3] if len(parts) == 4 else None
                    try:
                        archive = export_replays(history, key)
                    except ValueError as error:
                        return self._error(HTTPStatus.BAD_REQUEST, str(error))
                    return self._send(HTTPStatus.OK, archive, "application/zip", download_name=ARCHIVE_NAME)
                if len(parts) == 4 and parts[:2] == ["api", "team-scans"]:
                    detail = parse_qs(urlsplit(self.path).query).get("detail", ["preview"])[0]
                    if detail not in ("preview", "full"):
                        return self._error(HTTPStatus.BAD_REQUEST, "Nieznany poziom szczegółowości.")
                    return self._json(team_source_scan(parts[2], parts[3], detail == "full"))
                if len(parts) == 5 and parts[0] == "api" and parts[1] in ("solve", "history") and parts[3] == "scan":
                    fragment = parts[4]
                    if not fragment.isascii() or not fragment.isdecimal() or len(fragment) > 18:
                        return self._error(HTTPStatus.NOT_FOUND, "Skan źródłowy jest niedostępny.")
                    scan = history.scan(parts[2], int(fragment))
                    if scan is None:
                        return self._error(HTTPStatus.NOT_FOUND, "Skan źródłowy jest niedostępny.")
                    if parts[1] == "history":
                        name, data = scan
                        return self._send(HTTPStatus.OK, data, "application/octet-stream", download_name=name)
                    from case_solver import scan_payload
                    from case_upload import load_scan

                    return self._json(scan_payload(load_scan(*scan)))
                if len(parts) == 5 and parts[:4] == ["api", "demo", "fibula", "scan"]:
                    if parts[4] not in ("1", "2", "3", "4"):
                        return self._error(HTTPStatus.NOT_FOUND, "Skan fragmentu jest niedostępny.")
                    source = demo_file(FIBULA_DEMO / f"fibula-fragment-{parts[4]}.ply")
                    if not source.is_file():
                        return self._error(HTTPStatus.NOT_FOUND, "Skan fragmentu jest niedostępny.")
                    if parse_qs(urlsplit(self.path).query).get("download") == ["1"]:
                        return self._send(HTTPStatus.OK, source.read_bytes(), "application/octet-stream",
                                          download_name=source.name)
                    from bones import read_mesh
                    from case_solver import scan_payload
                    return self._json(scan_payload(read_mesh(source)))
                if len(parts) == 5 and parts[:2] == ["api", "demo"] and parts[3] == "scan" and parts[2] in DEMO_ARCHIVES:
                    archive_path, count = DEMO_ARCHIVES[parts[2]]
                    fragment = parts[4]
                    if not fragment.isascii() or not fragment.isdecimal() or not 1 <= len(fragment) <= 2:
                        return self._error(HTTPStatus.NOT_FOUND, "Skan fragmentu jest niedostępny.")
                    number = int(fragment)
                    if not 1 <= number <= count:
                        return self._error(HTTPStatus.NOT_FOUND, "Skan fragmentu jest niedostępny.")
                    name = f"fragment_{number:02d}.ply"
                    try:
                        with ZipFile(demo_file(archive_path)) as archive:
                            data = archive.read(name)
                    except (FileNotFoundError, KeyError, BadZipFile):
                        return self._error(HTTPStatus.NOT_FOUND, "Skan fragmentu jest niedostępny.")
                    if parse_qs(urlsplit(self.path).query).get("download") == ["1"]:
                        return self._send(HTTPStatus.OK, data, "application/octet-stream", download_name=name)
                    from case_solver import scan_payload
                    from case_upload import load_scan
                    return self._json(scan_payload(load_scan(name, data)))
                if len(parts) == 3 and parts[:2] == ["api", "solve"]:
                    key = parts[2]
                    if key in DEMO_RESULTS:
                        report = demo_file(DEMO_RESULTS[key])
                        if not report.is_file():
                            return self._error(HTTPStatus.NOT_FOUND, "Zapisany pokaz jest niedostępny.")
                        return self._send(HTTPStatus.OK, b'{"status":"done","result":'
                                          + report.read_bytes() + b'}', "application/json")
                    if key == "team-mm":
                        return self._json({"status": "done", "result": team_inspection()})
                    if key in ("benchmark-auto", "benchmark-labelled", "benchmark-reference", "benchmark-mixed",
                               "benchmark-open3d"):
                        source = key.removeprefix("benchmark-")
                        report = WEB_DIR.parents[1] / "out" / "benchmark-3" / f"{source}.json"
                        if report.is_file():
                            return self._send(HTTPStatus.OK, b'{"status":"done","result":'
                                              + report.read_bytes() + b'}', "application/json")
                    with state.lock:
                        message = active_cases.get(key)
                    if message:
                        return self._json({"status": "running", "message": message})
                    saved = history.result(key)
                    if saved is None:
                        return self._error(HTTPStatus.NOT_FOUND, "Przypadek nie jest już dostępny.")
                    if saved["result"] is not None:
                        return self._send(HTTPStatus.OK, b'{"status":"done","result":'
                                          + saved["result"].encode() + b'}', "application/json")
                    message = saved["error"] or "Skany są zapisane. Uruchom analizę z historii skanów."
                    return self._json({"status": "error", "message": message})
                if parts == ["api", "reference"]:
                    from fracture_matching import display_mesh

                    view = self._state()
                    poses = view.poses()
                    if not poses:
                        return self._error(HTTPStatus.NOT_FOUND, "Ten przypadek nie ma animacji referencyjnej.")
                    return self._json({
                        "title": catalog()[self._case()][0]["label"] if self._case() else "Przykład testowy",
                        "parts": [{"label": b.label, **display_mesh(b.mesh), "matrix": poses[b.id]}
                                  for b in view.bones.values() if b.id in poses],
                    })
                if parts == ["api", "bones"]:
                    return self._json(self._state().summary())
                if len(parts) == 3 and parts[:2] == ["api", "match"]:
                    with state.lock:
                        result = state.matches.get(parts[2])
                    if result is None:
                        return self._error(HTTPStatus.NOT_FOUND, "Wynik wygasł. Uruchom dopasowanie ponownie.")
                    return self._json(result)
                if len(parts) == 3 and parts[:2] == ["api", "bones"]:
                    return self._json(self._state().bone_payload(parts[2]))
                return self._error(HTTPStatus.NOT_FOUND, "nie ma takiej strony")
            except KeyError as e:
                return self._error(HTTPStatus.NOT_FOUND, f"nieznana kość {e}")
            except Exception as e:      # błąd ma trafić na stronę, a nie zabić serwer
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}")

        def do_PUT(self):
            parts = self._route()
            try:
                if len(parts) == 3 and parts[:2] == ["api", "history"]:
                    if not history.move(parts[2], self._body().get("folder")):
                        return self._error(HTTPStatus.NOT_FOUND, "Nie znaleziono przypadku.")
                    return self._json({"saved": True})
                if len(parts) == 4 and parts[:2] == ["api", "bones"] and parts[3] == "contacts":
                    return self._json(self._state().save(parts[2], self._body()))
                return self._error(HTTPStatus.NOT_FOUND, "nie ma takiej strony")
            except KeyError as e:
                return self._error(HTTPStatus.NOT_FOUND, f"nieznana kość {e}")
            except (ValueError, TypeError) as e:
                return self._error(HTTPStatus.BAD_REQUEST, str(e))
            except Exception as e:
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}")

        def do_POST(self):
            parts = self._route()
            try:
                if parts == ["api", "history", "folders"]:
                    return self._json(history.create_folder(self._body().get("name")), HTTPStatus.CREATED)
                if parts == ["api", "history", "replays"]:
                    size = int(self.headers.get("Content-Length", 0))
                    if size <= 0 or size > MAX_BYTES:
                        return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Paczka może mieć do 200 MB.")
                    cases = read_replays(self.rfile.read(size))
                    imported = history.import_cases(cases)
                    return self._json({"imported": imported, "existing": len(cases) - imported}, HTTPStatus.CREATED)
                if len(parts) == 4 and parts[:2] == ["api", "history"] and parts[3] == "solve":
                    key = parts[2]
                    saved = history.result(key)
                    if saved is None:
                        return self._error(HTTPStatus.NOT_FOUND, "Nie znaleziono przypadku.")
                    if not saved["can_solve"]:
                        return self._error(HTTPStatus.BAD_REQUEST, "Do złożenia potrzebne są co najmniej dwa skany.")
                    if saved["result"] is not None:
                        return self._json({"id": key})
                    if not solve_lock.acquire(blocking=False):
                        return self._error(HTTPStatus.CONFLICT, "Analiza trwa. Poczekaj na jej zakończenie.")
                    try:
                        start_case(key)
                    except Exception:
                        solve_lock.release()
                        raise
                    return self._json({"id": key}, HTTPStatus.ACCEPTED)
                if parts in (["api", "solve"], ["api", "history"]):
                    from case_upload import MAX_UPLOAD, parse_upload

                    n = int(self.headers.get("Content-Length", 0))
                    if n <= 0 or n > MAX_UPLOAD:
                        return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Łączny rozmiar przesyłki może wynosić do 100 MB na przypadek.")
                    solve = parts[1] == "solve"
                    if solve and not solve_lock.acquire(blocking=False):
                        return self._error(HTTPStatus.CONFLICT, "Analiza trwa. Poczekaj na jej zakończenie.")
                    try:
                        data = self.rfile.read(n)
                        key = uuid.uuid4().hex
                        history.add(key, *parse_upload(self.headers.get("Content-Type", ""), data, solve=solve))
                        if solve:
                            start_case(key)
                    except Exception:
                        if solve:
                            solve_lock.release()
                        raise
                    return self._json({"id": key}, HTTPStatus.ACCEPTED if solve else HTTPStatus.CREATED)
                if parts == ["api", "match"]:
                    return self._json(self._state().match(self._body()))
                if len(parts) == 4 and parts[:2] == ["api", "bones"] and parts[3] == "detect":
                    return self._json(self._state().detect(parts[2], self._body().get("method")))
                return self._error(HTTPStatus.NOT_FOUND, "nie ma takiej strony")
            except KeyError as e:
                return self._error(HTTPStatus.NOT_FOUND, f"nieznana kość {e}")
            except (ValueError, TypeError) as e:
                return self._error(HTTPStatus.BAD_REQUEST, str(e))
            except Exception as e:
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}")

    return Handler


def serve(port: int | None = None, open_browser: bool = True, out_dir: Path | None = None,
          params: ct.ContactParams | None = None, start_library: bool = False) -> None:
    """Uruchamia narzędzie i czeka (Ctrl+C kończy)."""
    import config

    params = params or config.contact_params()
    port = config.CONTACT_TOOL_PORT if port is None else port
    state = ToolState(out_dir, params)
    if not state.bones and not start_library:
        raise SystemExit("Brak przygotowanych kości - uruchom najpierw: python scripts/prepare_data.py")
    cache_dir = config.DATA_OUT / ".cache" / "web"
    detailed = DetailedMatching()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state, cache_dir, detailed=detailed))
    except OSError:
        raise SystemExit(f"Port {port} jest zajęty - może narzędzie już działa w innym oknie? "
                         f"Inny port: python scripts/mark_contacts.py --port {port + 1}") from None
    url = f"http://127.0.0.1:{server.server_port}/"
    if start_library:
        url += "case-start"
    print(f"Narzędzie do oznaczania powierzchni styku: {url}")
    print(f"Kości: {len(state.bones)}. Oznaczenia zapisują się do: {config.CONTACTS_DIR}")
    print("Zamknij to okno albo naciśnij Ctrl+C, żeby zakończyć.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nZakończono.")
    finally:
        server.server_close()
        detailed.close()

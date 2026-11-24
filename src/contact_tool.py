"""
Narzędzie do ręcznego oznaczania powierzchni styku - lokalna strona w przeglądarce.

Uruchamia mały serwer HTTP (tylko na tym komputerze, 127.0.0.1) i otwiera stronę
src/web/contact_tool.html. Strona pokazuje kość w 3D (three.js), a malowanie
zapisuje się przez serwer do contacts/<id_kości>.json (contacts.py).

    from contact_tool import serve
    serve()                       # Ctrl+C w konsoli kończy

API (używa go strona):
    GET  /api/bones                lista kości + stan oznaczeń      (ToolState.summary)
    GET  /api/bones/<id>           siatka (base64) + etykiety       (ToolState.bone_payload)
    PUT  /api/bones/<id>/contacts  zapis etykiet {labels: base64 uint8, names: {k: nazwa}}
                                                                     (ToolState.save)
    POST /api/bones/<id>/detect    propozycja algorytmu (nic nie zapisuje)  (ToolState.detect)
Oprócz tego serwer podaje pliki strony (STATIC) i three.js (VENDOR). Błędy wracają
jako JSON {"error": "..."} i są pokazywane na stronie.

three.js jest pobierany raz z CDN i zapisywany w DATA_OUT/.cache/web, potem działa bez internetu.
"""

from __future__ import annotations

import base64
import json
import threading
import urllib.request
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import contacts as ct
from bones import Bone, load_bones

WEB_DIR = Path(__file__).resolve().parent / "web"
STATIC = {"/": ("contact_tool.html", "text/html; charset=utf-8"),
          "/contact_tool.js": ("contact_tool.js", "text/javascript; charset=utf-8"),
          "/contact_tool.css": ("contact_tool.css", "text/css; charset=utf-8")}
THREE_VERSION = "0.170.0"
VENDOR = {"/vendor/three.module.js": f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.module.js",
          "/vendor/OrbitControls.js":
              f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/examples/jsm/controls/OrbitControls.js"}


def _b64(a: np.ndarray) -> str:
    """Surowe bajty tablicy w base64 - strona czyta je jako Float32Array / Uint32Array / Uint8Array.

    Typ tablicy musi zgadzać się z tym, czego oczekuje contact_tool.js (float32, uint32, uint8).
    """
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def view_rotation(mesh) -> np.ndarray:
    """Obrót do widoku: najdłuższa oś kości poziomo (X), najcieńsza w stronę kamery (Z).

    Returns:
        Macierz 3x3 (wiersze = osie główne PCA wierzchołków, bez odbicia lustrzanego);
        strona mnoży przez nią pozycje wierzchołków.
    """
    v = np.asarray(mesh.vertices)
    c = v - v.mean(axis=0)
    _, _, vt = np.linalg.svd(c[:: max(1, len(c) // 20000)], full_matrices=False)
    r = vt.copy()
    if np.linalg.det(r) < 0:
        r[2] *= -1
    return r


class ToolState:
    """Stan serwera: kości wczytane raz przy starcie (bez chmur i FPFH) + ustawienia narzędzia.

    Metody odpowiadają zapytaniom API (opis w docstringu modułu). Serwer obsługuje
    zapytania w osobnych wątkach, dlatego zapis pliku jest chroniony blokadą `lock`.
    """

    def __init__(self, out_dir: Path | None, params: ct.ContactParams):
        """Wczytuje kości (reload).

        Args:
            out_dir: folder wyników z kośćmi (None = DATA_OUT z settings.json).
            params:  ustawienia narzędzia i algorytmu (sekcja "contacts").
        """
        self.params = params
        self.out_dir = out_dir
        self.lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        """Wczytuje kości z out_dir; kolejność jak w load_bones (= numery w skryptach)."""
        self.bones: dict[str, Bone] = {b.id: b for b in load_bones(self.out_dir)}
        self.order = list(self.bones)

    def bone(self, bone_id: str) -> Bone:
        """Kość o danym id. KeyError, gdy jej nie ma (serwer odpowiada wtedy 404)."""
        if bone_id not in self.bones:
            raise KeyError(bone_id)
        return self.bones[bone_id]

    def summary(self) -> dict:
        """GET /api/bones: lista kości do panelu bocznego + dane startowe strony.

        Returns:
            {"bones": [{"nr", "id", "label", "scans", "area_mm2", "extent_mm", "vertices",
                        "surfaces": [{"id", "name", "area_mm2"}] - ręczne oznaczenia,
                        "updated": data zapisu albo None, "auto": czy jest wynik algorytmu}],
             "palette": kolory powierzchni, "detectors": nazwy algorytmów,
             "params": {"brush_mm", "wand_angle_deg"} - startowe ustawienia pędzla i różdżki}.
        """
        items = []
        for nr, bid in enumerate(self.order, 1):
            b = self.bones[bid]
            c = ct.load_contacts(bid, b.mesh)
            stats = ct.surface_stats(b.mesh, c.labels) if c else []
            items.append({
                "nr": nr, "id": bid, "label": b.label, "scans": b.n_scans,
                "area_mm2": b.meta.get("stats", {}).get("area_mm2"),
                "extent_mm": b.meta.get("stats", {}).get("extent_mm"),
                "vertices": len(b.mesh.vertices),
                "surfaces": [{"id": s["id"], "name": c.name(s["id"]), "area_mm2": s["area_mm2"]} for s in stats],
                "updated": c.updated if c else None,
                "auto": ct.contacts_path(bid, "auto").exists(),
            })
        return {"bones": items, "palette": list(ct.PALETTE), "detectors": list(ct.DETECTORS),
                "params": {"brush_mm": self.params.brush_mm, "wand_angle_deg": self.params.wand_angle_deg}}

    def bone_payload(self, bone_id: str) -> dict:
        """GET /api/bones/<id>: siatka kości i jej ręczne etykiety dla three.js.

        Returns:
            {"id", "label",
             "positions": float32 (V, 3), "normals": float32 (V, 3), "indices": uint32 (T, 3),
             "colors": uint8 (V, 3) - kolory skanera albo szare, "labels": uint8 (V,)
             - wszystkie tablice jako base64 (_b64),
             "names": {"k": nazwa}, "remapped": etykiety przeniesione ze starej siatki,
             "rotation": 3x3 z view_rotation}.
        """
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
        """PUT /api/bones/<id>/contacts: zapisuje ręczne etykiety do contacts/<id>.json.

        Args:
            bone_id: id kości.
            body:    {"labels": base64 uint8 (V,), "names": {"k": nazwa}} - nazwy
                     przycinane do 80 znaków.

        Returns:
            {"saved": ścieżka pliku albo None (brak oznaczeń - plik usunięty),
             "surfaces": [{"id", "name", "area_mm2", "max_diameter_mm"}]}.
        """
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
        """POST /api/bones/<id>/detect: propozycja algorytmu - NIC nie zapisuje.

        Użytkownik poprawia propozycję na stronie i zapisuje ją sam (jako ręczne oznaczenia).

        Args:
            bone_id: id kości.
            method:  nazwa algorytmu z DETECTORS; None = params.method.

        Returns:
            {"labels": base64 uint8 (V,), "names": {"k": nazwa}, "method": użyty algorytm}.
        """
        b = self.bone(bone_id)
        method = method or self.params.method
        c = ct.detect_contacts(b, method, self.params)
        return {"labels": _b64(c.labels), "names": {str(k): v for k, v in c.names.items()}, "method": method}


def fetch_vendor(path: str, cache_dir: Path) -> bytes:
    """Plik three.js: z cache, a za pierwszym razem pobrany z CDN i zapisany w cache.

    Args:
        path:      ścieżka URL z VENDOR, np. "/vendor/three.module.js".
        cache_dir: folder cache (DATA_OUT/.cache/web).

    Returns:
        Zawartość pliku.

    Raises:
        OSError: nie udało się pobrać (np. brak internetu przy pierwszym uruchomieniu).
    """
    target = cache_dir / Path(path).name
    if not target.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(VENDOR[path], timeout=30) as r:
            target.write_bytes(r.read())
    return target.read_bytes()


def make_handler(state: ToolState, cache_dir: Path):
    """Tworzy klasę obsługi zapytań HTTP, która widzi `state` i `cache_dir`.

    HTTPServer sam tworzy obiekt obsługi dla każdego zapytania i nie pozwala przekazać
    mu własnych argumentów - dlatego klasa jest zdefiniowana wewnątrz funkcji.
    """

    class Handler(BaseHTTPRequestHandler):
        """Routing zapytań strony: pliki strony, three.js i API /api/bones/...

        Każdy wyjątek jest zamieniany na odpowiedź JSON {"error": ...} (KeyError -> 404,
        inne -> 500), żeby błąd było widać na stronie, a serwer działał dalej.
        """

        def log_message(self, fmt, *args):     # bez logu każdego zapytania w konsoli
            pass

        def _send(self, status, body: bytes, ctype: str):
            """Wysyła odpowiedź z treścią `body` typu `ctype` (bez cache przeglądarki)."""
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status=HTTPStatus.OK):
            """Wysyła `data` jako JSON (UTF-8, polskie znaki bez escape)."""
            self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json")

        def _error(self, status, msg):
            """Wysyła błąd {"error": msg} - strona pokazuje go w dymku."""
            self._json({"error": msg}, status)

        def _body(self) -> dict:
            """Treść zapytania jako słownik z JSON ({} dla pustej)."""
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

        def _route(self):
            """Ścieżka URL bez parametrów, podzielona na części: "/api/bones/x" -> ["api", "bones", "x"]."""
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            return parts

        def do_GET(self):
            """Pliki strony, three.js, lista kości i dane jednej kości."""
            path = self.path.split("?")[0]
            try:
                if path in STATIC:
                    name, ctype = STATIC[path]
                    return self._send(HTTPStatus.OK, (WEB_DIR / name).read_bytes(), ctype)
                if path in VENDOR:
                    try:
                        data = fetch_vendor(path, cache_dir)
                    except OSError as e:
                        return self._error(HTTPStatus.BAD_GATEWAY,
                                           f"Nie udało się pobrać three.js ({e}). Za pierwszym razem potrzebny jest internet.")
                    return self._send(HTTPStatus.OK, data, "text/javascript; charset=utf-8")
                parts = self._route()
                if parts == ["api", "bones"]:
                    return self._json(state.summary())
                if len(parts) == 3 and parts[:2] == ["api", "bones"]:
                    return self._json(state.bone_payload(parts[2]))
                return self._error(HTTPStatus.NOT_FOUND, "nie ma takiej strony")
            except KeyError as e:
                return self._error(HTTPStatus.NOT_FOUND, f"nieznana kość {e}")
            except Exception as e:      # błąd ma trafić na stronę, a nie zabić serwer
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}")

        def do_PUT(self):
            """Zapis oznaczeń kości: PUT /api/bones/<id>/contacts."""
            parts = self._route()
            try:
                if len(parts) == 4 and parts[:2] == ["api", "bones"] and parts[3] == "contacts":
                    return self._json(state.save(parts[2], self._body()))
                return self._error(HTTPStatus.NOT_FOUND, "nie ma takiej strony")
            except KeyError as e:
                return self._error(HTTPStatus.NOT_FOUND, f"nieznana kość {e}")
            except Exception as e:
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}")

        def do_POST(self):
            """Propozycja algorytmu: POST /api/bones/<id>/detect (body: {"method": ...} opcjonalnie)."""
            parts = self._route()
            try:
                if len(parts) == 4 and parts[:2] == ["api", "bones"] and parts[3] == "detect":
                    return self._json(state.detect(parts[2], self._body().get("method")))
                return self._error(HTTPStatus.NOT_FOUND, "nie ma takiej strony")
            except KeyError as e:
                return self._error(HTTPStatus.NOT_FOUND, f"nieznana kość {e}")
            except Exception as e:
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}")

    return Handler


def serve(port: int | None = None, open_browser: bool = True, out_dir: Path | None = None,
          params: ct.ContactParams | None = None) -> None:
    """Uruchamia narzędzie i czeka (Ctrl+C kończy).

    Serwer słucha tylko na 127.0.0.1 - z innych komputerów nie jest dostępny.

    Args:
        port:         port serwera (None = CONTACT_TOOL_PORT z settings.json).
        open_browser: otwórz stronę w przeglądarce po starcie.
        out_dir:      folder wyników z kośćmi (None = DATA_OUT).
        params:       ustawienia narzędzia i algorytmu (None = sekcja "contacts").

    Raises:
        SystemExit: brak przygotowanych kości albo port zajęty (komunikat mówi, co zrobić).
    """
    import config

    params = params or config.contact_params()
    port = port or config.CONTACT_TOOL_PORT
    state = ToolState(out_dir, params)
    if not state.bones:
        raise SystemExit("Brak przygotowanych kości - uruchom najpierw: python scripts/prepare_data.py")
    cache_dir = Path(out_dir or config.DATA_OUT) / ".cache" / "web"
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state, cache_dir))
    except OSError:
        raise SystemExit(f"Port {port} jest zajęty - może narzędzie już działa w innym oknie? "
                         f"Inny port: python scripts/mark_contacts.py --port {port + 1}") from None
    url = f"http://127.0.0.1:{port}/"
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

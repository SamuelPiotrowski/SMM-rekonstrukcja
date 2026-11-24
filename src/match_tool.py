"""
Podgląd dopasowania powierzchni styku z suwakami ustawień - lokalna strona w przeglądarce.

Uruchamia mały serwer HTTP (tylko na tym komputerze, 127.0.0.1) i otwiera stronę
src/web/match_tool.html. Strona pokazuje pary z out/3d/matching.json (wynik
scripts/match_surfaces.py) w 3D: kość A ustawiona na kości B. Suwaki zmieniają
ustawienia dopasowania, a po każdej zmianie serwer przelicza TYLKO pokazaną parę:
    - ustawienia szukania (próbkowanie, FPFH, RANSAC, ICP) -> pełne dopasowanie od nowa
      (matching_surfaces.match_pair, kilka sekund),
    - ustawienia oceny (styk, kolizja) -> tylko nowa ocena obecnego ustawienia
      (matching_surfaces.evaluate, ułamek sekundy).
Nic nie jest zapisywane - matching.json zostaje bez zmian. Strona pokazuje zmienione
ustawienia gotowe do wklejenia w settings.json.

    from match_tool import serve
    serve()                       # Ctrl+C w konsoli kończy

API (używa go strona):
    GET  /api/pairs               pary z matching.json + ustawienia         (MatchState.summary)
    GET  /api/geometry?key=<k>    siatka kości + nakładka powierzchni k     (MatchState.geometry)
                                  (k = "<id kości>#<numer powierzchni>")
    POST /api/match               przeliczenie pary                         (MatchState.match)
                                  {a, b, params, mode: "full" | "score", T (dla "score")}
Oprócz tego serwer podaje pliki strony (STATIC) i plotly.js (z pakietu plotly - działa bez internetu).
Błędy wracają jako JSON {"error": "..."} i są pokazywane na stronie.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from dataclasses import asdict, fields, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from scipy.spatial import cKDTree

import matching_surfaces as ms
from bones import Bone, load_bones
from contact_tool import _b64
from contacts import ContactParams, triangle_labels
from visualization_3d import View3DParams, _surface_overlay, decimate

WEB_DIR = Path(__file__).resolve().parent / "web"
STATIC = {"/": ("match_tool.html", "text/html; charset=utf-8"),
          "/match_tool.js": ("match_tool.js", "text/javascript; charset=utf-8"),
          "/match_tool.css": ("match_tool.css", "text/css; charset=utf-8")}
PARAM_NAMES = {f.name for f in fields(ms.SurfaceMatchParams)}


class MatchState:
    """Stan serwera: wyniki z matching.json, kości (z chmurami) i powierzchnie styku.

    Kości są wczytywane raz przy starcie (tylko te z matching.json). Powierzchnie są
    próbkowane przy pierwszym użyciu i pamiętane dla każdej wartości sample_mm.
    Obliczenia są chronione blokadą `lock` - open3d ma wspólne ziarno losowania,
    a serwer obsługuje zapytania w osobnych wątkach.
    """

    def __init__(self, out_dir: Path, contact_params: ContactParams, view: View3DParams, root: Path | None = None):
        """Wczytuje matching.json, kości i ich powierzchnie styku.

        Args:
            out_dir:        folder wyników (z matching.json i bones/).
            contact_params: algorytm wykrywania (gdy matching.json był liczony w trybie "auto").
            view:           wygląd (triangles_total - szczegółowość siatek na stronie).
            root:           folder oznaczeń zamiast contacts/ z settings.json (np. w testach).

        Raises:
            SystemExit: brak matching.json albo kości z niego (komunikat mówi, co zrobić).
        """
        path = Path(out_dir) / "matching.json"
        if not path.exists():
            raise SystemExit("Brak out/3d/matching.json - uruchom najpierw: python scripts/match_surfaces.py")
        self.report = json.loads(path.read_text(encoding="utf-8"))
        self.params = ms.SurfaceMatchParams(**{k: v for k, v in self.report["params"].items() if k in PARAM_NAMES})
        self.view = view
        self.lock = threading.Lock()
        needed = {s["bone"] for s in self.report["surfaces"]}
        self.bones: dict[str, Bone] = {b.id: b for b in load_bones(out_dir, features=True) if b.id in needed}
        missing = needed - set(self.bones)
        if missing:
            raise SystemExit(f"Kości z matching.json nie ma w {out_dir}/bones: {sorted(missing)} - "
                             "uruchom ponownie: python scripts/match_surfaces.py")
        self.contacts, self.tri_labels, self.trees = {}, {}, {}
        for bid, b in self.bones.items():
            c, _ = ms.bone_contacts(b, self.params.source, contact_params, root)
            if c is None:
                raise SystemExit(f"{b.label}: brak powierzchni styku (tryb {self.params.source!r}) - "
                                 "uruchom ponownie: python scripts/match_surfaces.py")
            self.contacts[bid] = c
        self._surfaces: dict[tuple[str, float], ms.Surface] = {}
        self._geometry: dict[str, dict] = {}

    def surface(self, key: str, sample_mm: float) -> ms.Surface:
        """Powierzchnia "<id kości>#<numer>" próbkowana co sample_mm (liczona raz dla każdej wartości).

        Raises:
            KeyError: nieznana kość albo powierzchnia (np. oznaczenia zmieniono po matching.json).
        """
        bid, sid = key.rsplit("#", 1)
        if bid not in self.bones:
            raise KeyError(bid)
        if (key, sample_mm) not in self._surfaces:
            b, c = self.bones[bid], self.contacts[bid]
            if bid not in self.tri_labels:
                self.tri_labels[bid] = triangle_labels(b.mesh, c.labels)
                self.trees[bid] = cKDTree(np.asarray(b.cloud.points))
            s = ms.make_surface(b, c, int(sid), sample_mm, self.tri_labels[bid], self.trees[bid])
            if s is None:
                raise KeyError(f"{key} (powierzchni nie ma już w oznaczeniach)")
            self._surfaces[(key, sample_mm)] = s
        return self._surfaces[(key, sample_mm)]

    def summary(self) -> dict:
        """GET /api/pairs: pary (od najlepszej), powierzchnie i ustawienia, z którymi je liczono.

        Returns:
            {"pairs": pary z matching.json, "surfaces": opisy powierzchni,
             "params": ustawienia z matching.json (startowe położenie suwaków)}.
        """
        return {"pairs": self.report["pairs"], "surfaces": self.report["surfaces"], "params": asdict(self.params)}

    def geometry(self, key: str) -> dict:
        """GET /api/geometry?key=...: uproszczona siatka kości i nakładka powierzchni styku.

        Wszystko w układzie kości (strona sama przenosi kość A macierzą T). Nakładka jest
        lekko odsunięta wzdłuż normalnych, żeby nie migotała na siatce.

        Returns:
            {"key", "label", "name", "positions": float32 (V, 3), "indices": uint32 (T, 3),
             "surf_positions", "surf_indices" - nakładka (puste, gdy po uproszczeniu nic nie zostało)}
            - tablice jako base64.
        """
        if key not in self._geometry:
            bid, sid = key.rsplit("#", 1)
            if bid not in self.bones:
                raise KeyError(bid)
            b = self.bones[bid]
            d = decimate(b.mesh, max(20000, self.view.triangles_total // 2))
            v = np.asarray(d.vertices)
            q, tri = _surface_overlay(b, self.contacts[bid].labels, int(sid), d,
                                      v + 0.04 * np.asarray(d.vertex_normals))
            if q is None:
                q, tri = np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
            self._geometry[key] = {
                "key": key, "label": b.label, "name": f"{b.label} · {self.contacts[bid].name(int(sid))}",
                "positions": _b64(v.astype(np.float32)), "indices": _b64(np.asarray(d.triangles, dtype=np.uint32)),
                "surf_positions": _b64(q.astype(np.float32)), "surf_indices": _b64(tri.astype(np.uint32)),
            }
        return self._geometry[key]

    def match(self, body: dict) -> dict:
        """POST /api/match: przelicza jedną parę z ustawieniami ze strony. Nic nie zapisuje.

        Args:
            body: {"a", "b": klucze powierzchni, "params": {nazwa: wartość} - nadpisują
                   ustawienia z matching.json, "mode": "full" (dopasowanie od nowa)
                   albo "score" (tylko ocena macierzy "T" 4x4)}.

        Returns:
            Wynik jak w matching.json (score, contact_mm2, collision_mm2, cover_a, cover_b,
            rmse_mm, T, seed) + "seconds" (czas liczenia) i "mode".

        Raises:
            ValueError: nieznane ustawienie albo tryb.
        """
        values = body.get("params", {})
        unknown = sorted(set(values) - PARAM_NAMES)
        if unknown:
            raise ValueError(f"nieznane ustawienia: {unknown}")
        types = {f.name: f.type for f in fields(ms.SurfaceMatchParams)}
        p = replace(self.params, **{k: (int(v) if types[k] in (int, "int") else float(v)) for k, v in values.items()})
        mode = body.get("mode", "full")
        with self.lock:
            start = time.perf_counter()
            a, b = self.surface(body["a"], p.sample_mm), self.surface(body["b"], p.sample_mm)
            if mode == "full":
                r = ms.match_pair(a, b, p)
            elif mode == "score":
                T = np.asarray(body["T"], dtype=float).reshape(4, 4)
                r = {**ms.evaluate(a, b, T, p), "T": T.tolist(), "seed": body.get("seed")}
            else:
                raise ValueError(f"mode = {mode!r}, dostępne: full, score")
            seconds = time.perf_counter() - start
        clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()}
        return {**clean, "seconds": round(seconds, 2), "mode": mode}


def make_handler(state: MatchState):
    """Tworzy klasę obsługi zapytań HTTP, która widzi `state` (jak w contact_tool.make_handler)."""
    plotly_js: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        """Routing: pliki strony, plotly.js i API /api/... Wyjątki -> JSON {"error": ...}."""

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

        def _guard(self, fn):
            """Wywołuje fn(); KeyError -> 404, inne wyjątki -> 500 (serwer działa dalej)."""
            try:
                return fn()
            except KeyError as e:
                return self._json({"error": f"nieznana kość albo powierzchnia {e}"}, HTTPStatus.NOT_FOUND)
            except Exception as e:      # błąd ma trafić na stronę, a nie zabić serwer
                return self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_GET(self):
            """Pliki strony, plotly.js, lista par i geometria powierzchni."""
            url = urlsplit(self.path)

            def handle():
                if url.path in STATIC:
                    name, ctype = STATIC[url.path]
                    return self._send(HTTPStatus.OK, (WEB_DIR / name).read_bytes(), ctype)
                if url.path == "/vendor/plotly.js":
                    if not plotly_js:
                        from plotly.offline import get_plotlyjs
                        plotly_js.append(get_plotlyjs().encode("utf-8"))
                    return self._send(HTTPStatus.OK, plotly_js[0], "text/javascript; charset=utf-8")
                if url.path == "/api/pairs":
                    return self._json(state.summary())
                if url.path == "/api/geometry":
                    return self._json(state.geometry(parse_qs(url.query)["key"][0]))
                return self._json({"error": "nie ma takiej strony"}, HTTPStatus.NOT_FOUND)

            self._guard(handle)

        def do_POST(self):
            """Przeliczenie pary: POST /api/match."""
            def handle():
                if urlsplit(self.path).path != "/api/match":
                    return self._json({"error": "nie ma takiej strony"}, HTTPStatus.NOT_FOUND)
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
                return self._json(state.match(body))

            self._guard(handle)

    return Handler


def serve(port: int | None = None, open_browser: bool = True, out_dir: Path | None = None) -> None:
    """Uruchamia podgląd i czeka (Ctrl+C kończy). Serwer słucha tylko na 127.0.0.1.

    Args:
        port:         port serwera (None = MATCH_TOOL_PORT z settings.json, "matching" -> "port").
        open_browser: otwórz stronę w przeglądarce po starcie.
        out_dir:      folder wyników z matching.json i kośćmi (None = DATA_OUT).

    Raises:
        SystemExit: brak matching.json / kości albo port zajęty (komunikat mówi, co zrobić).
    """
    import config

    port = port or config.MATCH_TOOL_PORT
    print("Wczytywanie kości i powierzchni styku...")
    state = MatchState(Path(out_dir or config.DATA_OUT), config.contact_params(), config.view3d_params())
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    except OSError:
        raise SystemExit(f"Port {port} jest zajęty - może podgląd już działa w innym oknie? "
                         f"Inny port: python scripts/tune_matching.py --port {port + 1}") from None
    url = f"http://127.0.0.1:{port}/"
    print(f"Podgląd dopasowania z suwakami: {url}")
    print(f"Pary: {len(state.report['pairs'])}, kości: {len(state.bones)}. Nic nie jest zapisywane.")
    print("Zamknij to okno albo naciśnij Ctrl+C, żeby zakończyć.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nZakończono.")
    finally:
        server.server_close()

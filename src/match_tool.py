"""
Podgląd wyników dopasowania (out/3d/<zestaw>/matching.json) - jak show_bones dla prepare_data.

scripts/match_surfaces.py liczy i zapisuje matching.json, a ten moduł go pokazuje:
    - write_html(): statyczna strona out/3d/<zestaw>/matching.html (najlepsze pary w 3D + macierz
      wyników wszystkich par) - jeden plik, można go komuś wysłać,
    - serve():      strona z suwakami ustawień (opis niżej).
Oba sprawdzają, czy matching.json jest aktualny (MatchState.warnings).

Strona z suwakami: uruchamia mały serwer HTTP (tylko na tym komputerze, 127.0.0.1) i otwiera stronę
src/web/match_tool.html. Strona pokazuje pary z out/3d/<zestaw>/matching.json (wynik
scripts/match_surfaces.py) w 3D: kość A ustawiona na kości B. Suwaki zmieniają
ustawienia dopasowania, a po każdej zmianie serwer przelicza TYLKO pokazaną parę:
    - ustawienia szukania (próbkowanie, FPFH, RANSAC, ICP) -> pełne dopasowanie od nowa
      (matching_surfaces.match_pair, kilka sekund),
    - ustawienia oceny (styk, kolizja) -> tylko nowa ocena obecnego ustawienia
      (matching_surfaces.evaluate, ułamek sekundy).
Nic nie jest zapisywane - matching.json zostaje bez zmian. Strona pokazuje zmienione
ustawienia gotowe do wklejenia w settings.json.

    from match_tool import serve, write_html
    write_html()                  # out/3d/<zestaw>/matching.html
    serve()                       # Ctrl+C w konsoli kończy

API (używa go strona):
    GET  /api/pairs               pary z matching.json + ustawienia + ostrzeżenia (MatchState.summary)
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
from visualization_3d import View3DParams, _surface_overlay, decimate, match_matrix_figure, matching_figure, write_page

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
        import config

        path = Path(out_dir) / "matching.json"
        run = f"python scripts/match_surfaces.py{config.dataset_flag()}"
        if not path.exists():
            raise SystemExit(f"Brak {path} - uruchom najpierw: {run}")
        self.report = json.loads(path.read_text(encoding="utf-8"))
        self.params = ms.SurfaceMatchParams(**{k: v for k, v in self.report["params"].items() if k in PARAM_NAMES})
        self.view = view
        self.lock = threading.Lock()
        needed = {s["bone"] for s in self.report["surfaces"]}
        self.bones: dict[str, Bone] = {b.id: b for b in load_bones(out_dir, features=True) if b.id in needed}
        missing = needed - set(self.bones)
        if missing:
            raise SystemExit(f"Kości z matching.json nie ma w {out_dir}/bones: {sorted(missing)} - "
                             f"uruchom ponownie: {run}")
        self.contacts, self.tri_labels, self.trees = {}, {}, {}
        for bid, b in self.bones.items():
            c, _ = ms.bone_contacts(b, self.params.source, contact_params, root)
            if c is None:
                raise SystemExit(f"{b.label}: brak powierzchni styku (tryb {self.params.source!r}) - "
                                 f"uruchom ponownie: {run}")
            self.contacts[bid] = c
        self._surfaces: dict[tuple[str, float], ms.Surface] = {}
        self._geometry: dict[str, dict] = {}
        self._settings: ms.SurfaceMatchParams | None = None   # ustawia _load() - do warnings()

    def surface(self, key: str, sample_mm: float) -> ms.Surface:
        """Powierzchnia styku próbkowana co `sample_mm` (liczona raz dla każdej wartości).

        Args:
            key:       klucz "<id kości>#<numer powierzchni>" (jak w matching.json).
            sample_mm: odstęp punktów powierzchni [mm] (ustawienie "sample_mm").

        Returns:
            Powierzchnia z matching_surfaces.make_surface (z pamięci, jeśli już liczona).

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
        return {"pairs": self.report["pairs"], "surfaces": self.report["surfaces"], "params": asdict(self.params),
                "warnings": self.warnings(self._settings)}

    def warnings(self, settings: ms.SurfaceMatchParams | None = None) -> list[str]:
        """Czy matching.json jest aktualny: ustawienia z settings.json i obecne oznaczenia.

        Args:
            settings: ustawienia z settings.json (config.matching_params()); None = nie porównuj.

        Returns:
            Opisy różnic (pusta lista = aktualny). Każdy kończy się podpowiedzią, co uruchomić.
        """
        out = []
        if settings is not None:
            now, then = asdict(settings), asdict(self.params)
            diff = [f"{k}: {then[k]} -> {now[k]}" for k in now if now[k] != then.get(k)]
            if diff:
                out.append("settings.json (sekcja \"matching\") różni się od ustawień, z którymi liczono "
                           f"matching.json ({', '.join(diff)}) - przelicz: python scripts/match_surfaces.py")
        gone = [s["key"] for s in self.report["surfaces"]
                if int(s["surface"]) not in set(np.unique(self.contacts[s["bone"]].labels).tolist())]
        if gone:
            out.append(f"oznaczenia zmieniły się po matching.json, nie ma już powierzchni: {', '.join(gone)} "
                       "- przelicz: python scripts/match_surfaces.py")
        return out

    def geometry(self, key: str) -> dict:
        """GET /api/geometry?key=...: uproszczona siatka kości i nakładka powierzchni styku.

        Wszystko w układzie kości (strona sama przenosi kość A macierzą T). Nakładka jest
        lekko odsunięta wzdłuż normalnych, żeby nie migotała na siatce. Wynik jest pamiętany.

        Args:
            key: klucz "<id kości>#<numer powierzchni>".

        Returns:
            {"key", "label", "name", "positions": float32 (V, 3), "indices": uint32 (T, 3),
             "surf_positions", "surf_indices" - nakładka (puste, gdy po uproszczeniu nic nie zostało)}
            - tablice jako base64.

        Raises:
            KeyError: nieznana kość.
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
            rmse_mm, T, seed) + "seconds" (czas liczenia) i "mode". Wartości nieskończone
            (np. rmse_mm bez dopasowania) są zamieniane na None, bo JSON ich nie obsługuje.

        Raises:
            ValueError: nieznane ustawienie albo tryb.
            KeyError:   nieznana kość albo powierzchnia, brak "a" / "b" (albo "T" w trybie "score").
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
    """Tworzy klasę obsługi zapytań HTTP, która widzi `state` (jak w contact_tool.make_handler).

    plotly.js (~4 MB) jest brany z zainstalowanego pakietu plotly przy pierwszym zapytaniu
    i pamiętany - strona działa bez internetu.

    Args:
        state: stan serwera, wspólny dla wszystkich zapytań.

    Returns:
        Klasa (podklasa BaseHTTPRequestHandler) do przekazania ThreadingHTTPServer.
    """
    plotly_js: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        """Routing: pliki strony, plotly.js i API /api/... Wyjątki -> JSON {"error": ...}."""

        def log_message(self, fmt, *args):
            """Wyłącza log każdego zapytania w konsoli (nadpisuje BaseHTTPRequestHandler)."""
            pass

        def _send(self, status, body: bytes, ctype: str):
            """Wysyła odpowiedź z treścią `body` typu `ctype` (bez cache przeglądarki).

            Args:
                status: kod HTTP (HTTPStatus).
                body:   treść odpowiedzi.
                ctype:  nagłówek Content-Type, np. "application/json".
            """
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status=HTTPStatus.OK):
            """Wysyła `data` jako JSON (UTF-8, polskie znaki bez escape).

            Args:
                data:   obiekt do zapisania w JSON (słownik, lista, ...).
                status: kod HTTP (domyślnie 200).
            """
            self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json")

        def _guard(self, fn):
            """Wywołuje fn() i zamienia wyjątki na odpowiedź JSON {"error": ...}.

            KeyError -> 404, inne wyjątki -> 500; serwer działa dalej, a błąd widać na stronie.

            Args:
                fn: funkcja bez argumentów, która sama wysyła odpowiedź.

            Returns:
                To, co zwróciła fn (albo wysłanie błędu) - wartość nie jest używana.
            """
            try:
                return fn()
            except KeyError as e:
                return self._json({"error": f"nieznana kość albo powierzchnia {e}"}, HTTPStatus.NOT_FOUND)
            except Exception as e:      # błąd ma trafić na stronę, a nie zabić serwer
                return self._json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_GET(self):
            """Pliki strony (STATIC), plotly.js, lista par i geometria powierzchni."""
            url = urlsplit(self.path)

            def handle():
                """Wysyła odpowiedź dla url.path (wywoływana przez _guard)."""
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
            """Przeliczenie pary: POST /api/match (treść: MatchState.match)."""
            def handle():
                """Czyta treść JSON, przelicza parę i wysyła wynik (wywoływana przez _guard)."""
                if urlsplit(self.path).path != "/api/match":
                    return self._json({"error": "nie ma takiej strony"}, HTTPStatus.NOT_FOUND)
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
                return self._json(state.match(body))

            self._guard(handle)

    return Handler


def _load(out_dir: Path | None) -> MatchState:
    """Wczytuje matching.json z kośćmi (ustawienia z settings.json) i wypisuje ostrzeżenia w konsoli.

    Args:
        out_dir: folder wyników z matching.json i kośćmi (None = DATA_OUT).

    Returns:
        Stan z ustawieniami z settings.json w `_settings` (do porównania w warnings()).
    """
    import config

    print("Wczytywanie kości i powierzchni styku...")
    state = MatchState(Path(out_dir or config.DATA_OUT), config.contact_params(), config.view3d_params())
    state._settings = config.matching_params()
    for w in state.warnings(state._settings):
        print(f"UWAGA: {w}")
    return state


def matching_page(state: MatchState, path: Path, open_browser: bool = False) -> Path:
    """Zapisuje statyczną stronę z wynikami `state`: najlepsze pary w 3D + macierz wyników.

    Args:
        state:        wczytany matching.json (MatchState).
        path:         plik wynikowy, np. out/3d/<zestaw>/matching.html.
        open_browser: otwórz stronę po zapisie.

    Returns:
        Ścieżka zapisanej strony.
    """
    p, rep = state.params, state.report
    algo = f" (algorytm: {rep['contact_params'].get('method')})" if p.source != "manual" else ""
    return write_page(
        path, "Dopasowanie powierzchni styku",
        f"Powierzchnie styku: {ms.SOURCE_NAMES.get(p.source, p.source)}{algo}. Każda dopasowana (RANSAC + ICP) "
        "do każdej powierzchni innej kości, twarzą w twarz. Wynik = pole styku (powierzchnie bliżej niż "
        f"{p.tolerance_mm:g} mm, normalne przeciwne) minus pole, na którym kości wchodzą w siebie.",
        [("Najlepsze pary", "Kość A (niebieska) ustawiona na kości B. Lista w lewym dolnym rogu wybiera parę. "
          "Suwaki ustawień z przeliczaniem jednej pary: python scripts/tune_matching.py",
          matching_figure(rep["pairs"], state.bones, state.contacts, state.view, p.max_shown)),
         ("Wszystkie pary", "Wynik każdej pary powierzchni [mm²]; puste pola = ta sama kość, "
          "„(algorytm)” = powierzchnia z algorytmu wykrywania.",
          match_matrix_figure(rep, state.view))],
        state.view, open_browser=open_browser)


def write_html(out_dir: Path | None = None, open_browser: bool = True) -> Path:
    """tune_matching.py --html: wczytuje matching.json i zapisuje out/3d/<zestaw>/matching.html (matching_page).

    Args:
        out_dir:      folder wyników z matching.json i kośćmi (None = DATA_OUT); strona trafia tam.
        open_browser: otwórz stronę po zapisie (o ile pozwala OPEN_BROWSER z settings.json).

    Returns:
        Ścieżka zapisanej strony.

    Raises:
        SystemExit: brak matching.json albo kości z niego (komunikat mówi, co zrobić).
    """
    import config

    state = _load(out_dir)
    page = matching_page(state, Path(out_dir or config.DATA_OUT) / "matching.html",
                         open_browser=config.OPEN_BROWSER and open_browser)
    print(f"Zapisano {page}")
    return page


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
    state = _load(out_dir)
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

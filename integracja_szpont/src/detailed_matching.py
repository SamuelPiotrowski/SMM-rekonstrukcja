"""Run the original matching server in a separate interpreter."""

import os
import subprocess
import sys
import threading
from pathlib import Path
from queue import Empty, Queue


MODULE = Path(__file__).resolve().parents[1] / "moduly" / "szczegolowe-dopasowanie"
READY = "OSTEOFORM_PORT="


class DetailedMatching:
    def __init__(self):
        self._lock = threading.Lock()
        self._process = None
        self._url = None

    def url(self):
        with self._lock:
            if self._url and self._process.poll() is None:
                return self._url
            self._stop()
            if not (MODULE / "src" / "match_tool.py").is_file():
                raise RuntimeError("Brakuje lokalnej kopii Szczegółowego dopasowania.")
            environment = os.environ.copy()
            environment.setdefault("OMP_NUM_THREADS", "4")
            process = subprocess.Popen(
                [sys.executable, "-u", "-B", str(Path(__file__).resolve())],
                cwd=MODULE, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._process = process
            ready = Queue()
            threading.Thread(target=self._read_startup, args=(process, ready), daemon=True).start()
            try:
                port, error = ready.get(timeout=45)
            except Empty:
                port, error = None, "Serwer Szczegółowego dopasowania nie zdążył się uruchomić."
            if port is None:
                self._stop()
                raise RuntimeError(error)
            self._url = f"http://127.0.0.1:{port}/"
            return self._url

    @staticmethod
    def _read_startup(process, ready):
        port = None
        last = ""
        with process.stdout as output:
            for line in output:
                value = line.strip().removeprefix(READY)
                if port is None and line.startswith(READY) and value.isdecimal():
                    port = int(value)
                    ready.put((port, ""))
                elif line.strip():
                    last = line.strip()
        if port is None:
            ready.put((None, last or "Serwer Szczegółowego dopasowania zakończył pracę."))

    def _stop(self):
        process, self._process = self._process, None
        self._url = None
        if process is None:
            return
        process.stdin.close()
        try:
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait()

    def close(self):
        with self._lock:
            self._stop()


def run_original_server():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, str(MODULE / "src"))
    import config
    from match_tool import LocalServer, _load, make_handler

    config.use_dataset()
    state = _load(None)
    with LocalServer(("127.0.0.1", 0), make_handler(state)) as server:
        def stop_after_parent_exits():
            sys.stdin.buffer.read(1)
            server.shutdown()

        threading.Thread(target=stop_after_parent_exits, daemon=True).start()
        threading.Thread(target=state.warm, daemon=True).start()
        print(f"{READY}{server.server_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    run_original_server()

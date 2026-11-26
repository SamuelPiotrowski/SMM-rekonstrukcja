"""Exercise the saved annotations, HTTP endpoint, ranking, and reference evaluation."""

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import config  # noqa: E402
import numpy as np  # noqa: E402
from contact_tool import ThreadingHTTPServer, ToolState, make_handler  # noqa: E402
from demo_fractures import example, write_fragment  # noqa: E402


def run_case(directory, loss):
    fragments, reference = example(loss)
    for fragment in fragments:
        write_fragment(directory, *fragment)
    (directory / "reference_poses.json").write_text(
        json.dumps({fragments[0][0]: np.eye(4).tolist(), fragments[1][0]: reference.tolist()}),
        encoding="utf-8",
    )
    state = ToolState(directory, config.contact_params())
    with ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state, directory / ".cache")) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            for path in ("/", "/assembly", "/assembly.js", "/plotly.js"):
                with urlopen(url + path) as response:
                    assert response.status == 200
            with urlopen(url + "/api/bones") as response:
                entries = json.load(response)["bones"]
            a, b = [{"bone": item["id"], "surface": item["surfaces"][0]["id"]} for item in entries]
            try:
                urlopen(Request(url + "/api/match", data=b"{}",
                                headers={"Content-Type": "application/json"}))
            except HTTPError as error:
                assert error.code == 400
            else:
                raise AssertionError("API did not reject the empty request.")
            request = Request(url + "/api/match", data=json.dumps({"a": a, "b": b}).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=150) as response:
                key = json.load(response)["id"]
            with urlopen(url + "/api/match/" + key) as response:
                result = json.load(response)
            best = result["candidates"][0]
            score = best["reference"]
            print(f"{'ubytek' if loss else 'czysty'}: błąd położenia {score['pose_rmse_mm']:.3f} mm, "
                  f"błąd obrotu {score['rotation_deg']:.3f}°, "
                  f"kontakt {best['contact_a']:.1%} / {best['contact_b']:.1%}", flush=True)
            assert best["eligible"] and score["pose_rmse_mm"] < 1.0 and score["rotation_deg"] < 2.0
        finally:
            server.shutdown()
            thread.join()


def test_fracture_matching():
    old_contacts = config.CONTACTS_DIR
    try:
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            config.CONTACTS_DIR = directory / "contacts"
            run_case(directory / "clean", loss=False)
            run_case(directory / "loss", loss=True)
    finally:
        config.CONTACTS_DIR = old_contacts


if __name__ == "__main__":
    test_fracture_matching()

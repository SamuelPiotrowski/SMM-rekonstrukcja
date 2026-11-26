"""Receive one local case. Files and computation are isolated per request."""

from email import policy
from email.parser import BytesParser
from pathlib import Path
from tempfile import TemporaryDirectory

from bones import read_mesh

MAX_UPLOAD = 100 * 1024 * 1024
FORMATS = {".ply", ".stl", ".obj"}


def parse_upload(content_type, data, *, solve=True):
    if not content_type.lower().startswith("multipart/form-data;"):
        raise ValueError("Wybierz skany w formularzu.")
    payload = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("ascii") + b"\r\n\r\n" + data)
    if not payload.is_multipart():
        raise ValueError("Nie udało się odczytać plików.")
    parts = list(payload.iter_parts())
    files = [part for part in parts if part.get_param("name", header="content-disposition") == "scans"]
    minimum = 2 if solve else 1
    if len(files) < minimum:
        raise ValueError("Do złożenia potrzebne są co najmniej dwa pliki." if solve else "Wybierz co najmniej jeden plik.")
    fields = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
              for part in parts if part not in files}
    mode = fields.get("mode", b"single")
    anchor = None
    if mode == b"target":
        choice = fields.get("anchor", b"")
        if len(choice) > len(str(len(files))) or not choice.isdigit() or not 1 <= int(choice) <= len(files):
            raise ValueError("Wskaż jeden z wgranych odłamków jako fragment odniesienia.")
        anchor = int(choice)
    title = fields.get("title", b"").decode("utf-8", errors="replace").strip()[:80]
    folder = fields.get("folder", b"").decode("ascii", errors="replace")
    scans = []
    for part in files:
        name = Path((part.get_filename() or "").replace("\\", "/")).name
        if Path(name).suffix.lower() not in FORMATS:
            raise ValueError(f"Dozwolone formaty to STL, OBJ i PLY. Sprawdź {name or 'wybrany plik'}.")
        scans.append((name, part.get_payload(decode=True) or b""))
    return title or "Własny przypadek", folder, mode in (b"mixed", b"target"), scans, anchor


def load_scan(name, data):
    suffix = Path(name).suffix.lower()
    with TemporaryDirectory(prefix="osteoform-case-") as scratch:
        target = Path(scratch) / f"fragment{suffix}"
        target.write_bytes(data)
        try:
            mesh = read_mesh(target)
        except (ValueError, OSError, IndexError) as error:
            raise ValueError(f"Nie udało się wczytać siatki {name}.") from error
    if len(mesh.triangles) < 20:
        raise ValueError(f"Plik {name} nie zawiera wystarczającej powierzchni 3D.")
    return mesh

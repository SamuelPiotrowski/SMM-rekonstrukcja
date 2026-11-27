"""Catalog of the local bone preview package."""

from pathlib import Path
from urllib.parse import quote, unquote

from bundled_data import bundled_directory


ROOT = Path(__file__).resolve().parents[1] / "out" / "materialy-kosci"
PREFIX = "/bone-materials/"
REPORTS = {
    "prezentacja_atlas.html": "Pełny podgląd atlasu",
    "prezentacja_atlas_powierzchnie-porownanie.html": "Powierzchnie styku i porównanie",
}
GROUPS = {
    "scanned_bones": ("Skany 3D", {".ply", ".obj", ".stl"}),
    "atlas": ("Modele atlasu", {".stl"}),
    "powierzchnie": ("Oznaczenia powierzchni styku", {".json"}),
    "atlas_index": ("Indeks atlasu", {".npz", ".json"}),
}
SUFFIXES = {
    **{folder: suffixes for folder, (_, suffixes) in GROUPS.items()},
    "exports": {".html"},
    "bone_photos": {".jpg", ".jpeg"},
    "miniatury": {".webp"},
}
MIME = {"exports": "text/html; charset=utf-8", "bone_photos": "image/jpeg", "miniatury": "image/webp"}
REPORT_CSP = ("sandbox allow-scripts allow-downloads; default-src 'none'; "
              "script-src 'unsafe-inline' 'unsafe-eval' blob:; style-src 'unsafe-inline'; "
              "img-src data: blob:; font-src data:; worker-src blob:")


def url(relative):
    return PREFIX + quote(relative.as_posix())


def report_path(name):
    local = ROOT / "exports" / name
    if local.is_file():
        return local
    bundle = bundled_directory("podglad-kosci")
    return bundle / name if bundle is not None else local


def catalog():
    reports = [{"label": label, "url": url(Path("exports") / name)}
               for name, label in REPORTS.items() if report_path(name).is_file()]
    photos = []
    for photo in sorted((ROOT / "bone_photos").glob("*")):
        if photo.is_file() and photo.suffix.lower() in SUFFIXES["bone_photos"]:
            preview = Path("miniatury") / f"{photo.stem}.webp"
            source = url(photo.relative_to(ROOT))
            photos.append({"name": photo.name, "url": source,
                           "preview": url(preview) if (ROOT / preview).is_file() else source})
    groups = []
    for section, (label, suffixes) in GROUPS.items():
        folder = ROOT / section
        files = [{"label": file.relative_to(folder).as_posix(), "url": url(file.relative_to(ROOT))}
                 for file in sorted(folder.rglob("*")) if file.is_file() and file.suffix.lower() in suffixes]
        if files:
            groups.append({"label": label, "files": files})
    return {"reports": reports, "photos": photos, "groups": groups}


def material(encoded_path):
    relative = unquote(encoded_path)
    pieces = relative.split("/")
    if (len(pieces) < 2 or "\\" in relative or "\x00" in relative or
            ":" in relative or any(part in ("", ".", "..") for part in pieces)):
        return None
    section = pieces[0]
    if section == "exports" and (len(pieces) != 2 or pieces[1] not in REPORTS):
        return None
    if section == "exports":
        report = report_path(pieces[1])
        return (report, MIME[section], section) if report.is_file() else None
    path = (ROOT / relative).resolve()
    if path.suffix.lower() not in SUFFIXES.get(section, ()) or not path.is_relative_to(ROOT.resolve()):
        return None
    if not path.is_file():
        return None
    mime = MIME.get(section, "application/octet-stream")
    return path, mime, section

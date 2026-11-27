"""Portable ZIPs of finished reconstructions and their source scans."""

import json
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile


MAX_BYTES = 200 * 1024 * 1024
ARCHIVE_NAME = "OSTEOFORM-zapisane-skladania.zip"
SCAN_FORMATS = {".obj", ".ply", ".stl"}


def attach_scan_links(result, key, scans):
    parts = result.get("parts", []) + [part for section in result.get("sections", []) for part in section["parts"]]
    if not parts:
        raise ValueError("Wynik nie zawiera odłamków.")
    for part in parts:
        fragment = part["id"]
        if not isinstance(fragment, str) or not re.fullmatch(r"[1-9][0-9]{0,8}", fragment):
            raise ValueError("Nieprawidłowy numer odłamka.")
        index = int(fragment) - 1
        if index >= len(scans):
            raise ValueError("Brakuje skanu odłamka.")
        part["uploaded_scan"] = {"url": f"/api/solve/{key}/scan/{fragment}",
                                 "download": f"/api/history/{key}/scan/{fragment}",
                                 "label": scans[index][0]}
    return result


def export_replays(history, key=None):
    cases = history.completed_cases(MAX_BYTES, key)
    if not cases:
        raise ValueError("Nie ma zakończonych złożeń do zapisania.")
    output = BytesIO()
    index = []
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=3) as archive:
        for case in cases:
            case_id = case["id"]
            scans = case.pop("scans")
            archive.writestr(f"{case_id}/result.json", case.pop("result"))
            for number, (_, data) in enumerate(scans, 1):
                archive.writestr(f"{case_id}/scans/{number}", data)
            index.append(dict(case, scans=[name for name, _ in scans]))
        archive.writestr("index.json", json.dumps({"version": 1, "cases": index}, ensure_ascii=False))
    return output.getvalue()


def read_replays(source):
    cases = []
    try:
        with ZipFile(BytesIO(source)) as archive:
            members = archive.infolist()
            names = {member.filename for member in members}
            if len(names) != len(members) or sum(member.file_size for member in members) > MAX_BYTES:
                raise ValueError("Nieprawidłowa lub zbyt duża paczka złożeń.")
            index = json.loads(archive.read("index.json"))
            if index["version"] != 1 or not isinstance(index["cases"], list) or not index["cases"]:
                raise ValueError("Nieobsługiwany format paczki złożeń.")
            used_ids = set()
            for entry in index["cases"]:
                key, title = entry["id"], entry["title"]
                if not re.fullmatch(r"[0-9a-f]{32}", key) or key in used_ids or not isinstance(title, str):
                    raise ValueError("Nieprawidłowy przypadek w paczce.")
                used_ids.add(key)
                created = datetime.fromisoformat(entry["created"]).isoformat()
                filenames = entry["scans"]
                anchor = entry["anchor"]
                if (not isinstance(filenames, list) or not filenames or type(entry["mixed"]) is not int
                        or entry["mixed"] not in (0, 1) or
                        (anchor is not None and (type(anchor) is not int or not 1 <= anchor <= len(filenames)))):
                    raise ValueError("Nieprawidłowe informacje o skanach.")
                scans = []
                for number, name in enumerate(filenames, 1):
                    if not isinstance(name, str) or re.search(r"[/\\:\x00-\x1f]", name) or Path(name).suffix.lower() not in SCAN_FORMATS:
                        raise ValueError("Nieprawidłowa nazwa skanu.")
                    scans.append((name, archive.read(f"{key}/scans/{number}")))
                result = json.loads(archive.read(f"{key}/result.json"))
                attach_scan_links(result, key, scans)
                cases.append(dict(id=key, title=title[:80], created=created, mixed=entry["mixed"], anchor=anchor,
                                  scans=scans, result=json.dumps(result, ensure_ascii=False, allow_nan=False)))
    except (BadZipFile, AttributeError, KeyError, OverflowError, RuntimeError, TypeError, NotImplementedError) as error:
        raise ValueError("Nie udało się odczytać paczki zapisanych złożeń.") from error
    return cases

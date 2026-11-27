"""Open the shipped preview data without replacing any local scans or results."""

from functools import lru_cache
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile


PROJECT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def bundled_directory(name):
    archive = PROJECT / "dane-startowe" / f"{name}.zip"
    if not archive.is_file():
        return None
    with archive.open("rb") as source:
        digest = file_digest(source, "sha256").hexdigest()
    cache = PROJECT / "out" / "dane-startowe"
    destination = cache / f"{name}-{digest[:16]}"
    if destination.is_dir():
        return destination
    cache.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache) as temporary, ZipFile(archive) as bundle:
        staging = Path(temporary) / "data"
        staging.mkdir()
        for member in bundle.infolist():
            pieces = member.filename.rstrip("/").split("/")
            if (any(piece in ("", ".", "..") for piece in pieces)
                    or "\\" in member.filename or ":" in member.filename):
                raise ValueError(f"Nieprawidłowa ścieżka w paczce startowej {name}.")
        bundle.extractall(staging)
        try:
            staging.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
    return destination

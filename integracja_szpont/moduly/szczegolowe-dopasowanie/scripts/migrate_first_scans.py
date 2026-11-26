"""
JEDNORAZOWO: przenosi pierwsze skany (sprzed zestawów danych) do zestawu "first_scans"
i nadaje kościom krótkie nazwy.

Wcześniej wszystkie skany leżały wprost w scanned_bones/ (foldery sesji Medita), wyniki
w out/3d/, oznaczenia w contacts/. Teraz każdy zestaw ma swój podfolder:
    scanned_bones/first_scans/   out/3d/first_scans/   contacts/first_scans/

Każdy skan dostaje własny folder nazwany jak kość (pliki w środku zostają bez zmian,
z oryginalną nazwą ze skanera). Nazwa kości = nazwa folderu, więc później wystarczy
zmienić nazwę folderu i uruchomić prepare_data.py - oznaczenia przejdą same.

    first_test/                     -> whole_scan/  (oba fragmenty strzałkowej + occlusion;
                                       kości: whole_scan-lowerjaw, whole_scan-upperjaw)
    modelowanie matematyczne_(5)/upperjaw, _(6)/lowerjaw, _(6)/upperjaw
                                    -> whole_upper_rescan, whole_lower_rescan_1, _2
                                       (ponowne skany tych samych fragmentów - łączenie
                                       skanów wykryło nakładanie 0.49-0.73)
    pozostałe 10 skanów             -> bone_01 ... bone_10

Użycie (z głównego folderu projektu; można uruchomić kilka razy - przeniesione pomija):
    python scripts/migrate_first_scans.py
Potem:
    python scripts/prepare_data.py
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # polskie znaki także przy przekierowaniu do pliku
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # ... i w komunikatach błędów (SystemExit)

import config  # noqa: E402
from data_parsing import slugify  # noqa: E402

DATASET = "first_scans"
SESSION = "23.11.2026-modelowanie matematyczne"
# (stary folder sesji, część skanu) -> nowy folder (= nazwa kości); część None = cały folder
MAPPING = [
    ("first_test", None, "whole_scan"),
    (SESSION, "lowerjaw", "bone_01"),
    (SESSION, "upperjaw", "bone_02"),
    (f"{SESSION}_(1)", "upperjaw", "bone_03"),
    (f"{SESSION}_(2)", "lowerjaw", "bone_04"),
    (f"{SESSION}_(2)", "upperjaw", "bone_05"),
    (f"{SESSION}_(3)", "lowerjaw", "bone_06"),
    (f"{SESSION}_(3)", "upperjaw", "bone_07"),
    (f"{SESSION}_(4)", "upperjaw", "bone_08"),
    (f"{SESSION}_(5)", "upperjaw", "whole_upper_rescan"),
    (f"{SESSION}_(6)", "lowerjaw", "whole_lower_rescan_1"),
    (f"{SESSION}_(6)", "upperjaw", "whole_lower_rescan_2"),
    (f"{SESSION}_(7)", "lowerjaw", "bone_09"),
    (f"{SESSION}_(7)", "upperjaw", "bone_10"),
]


def old_to_new_ids() -> dict[str, str]:
    """Stare id kości (sprzed zestawów) -> nowe id, np. "first_test-lowerjaw" -> "whole_scan-lowerjaw"."""
    ids = {}
    for folder, part, new in MAPPING:
        if part is None:
            for p in ("lowerjaw", "upperjaw"):
                ids[f"{slugify(folder)}-{p}"] = f"{new}-{p}"
        else:
            ids[f"{slugify(folder)}-{part}"] = new
    return ids


def move(src: Path, dst: Path) -> bool:
    """Przenosi plik / folder; istniejący cel = pominięcie (skrypt można uruchomić ponownie)."""
    if dst.exists():
        print(f"  pominięto (już jest): {dst.relative_to(config.ROOT)}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return True


def migrate_scans() -> int:
    """scanned_bones/<sesja>/... -> scanned_bones/first_scans/<kość>/...; zwraca liczbę przeniesionych plików."""
    root, target = config.SCANS_ROOT, config.SCANS_ROOT / DATASET
    moved = 0
    for folder, part, new in MAPPING:
        src_dir = root / folder
        if not src_dir.is_dir():
            continue
        for f in sorted(src_dir.iterdir()):
            if f.is_file() and (part is None or f.stem.endswith(f"-{part}")):
                moved += move(f, target / new / f.name)
    for folder in {m[0] for m in MAPPING}:            # puste foldery sesji
        d = root / folder
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    return moved


def migrate_out() -> int:
    """out/3d/* (wyniki sprzed zestawów) -> out/3d/first_scans/; .cache (biblioteki stron) zostaje wspólny."""
    root, target = config.OUT_ROOT, config.OUT_ROOT / DATASET
    if not (root / "bones").is_dir():
        return 0
    moved = 0
    for item in sorted(root.iterdir()):
        if item.name in (".cache", DATASET) or (item.is_dir() and (item / "bones").is_dir()):
            continue                                   # wspólny cache albo już zestaw
        moved += move(item, target / item.name)
    return moved


def migrate_contacts() -> int:
    """contacts/<stare id>.json (+ auto/) -> contacts/first_scans/<nowe id>.json; zwraca liczbę plików."""
    root, target = config.CONTACTS_ROOT, config.CONTACTS_ROOT / DATASET
    ids = old_to_new_ids()
    moved = 0
    for sub in ("", "auto"):
        for f in sorted((root / sub).glob("*.json")) if (root / sub).is_dir() else []:
            new_id = ids.get(f.stem)
            if new_id is None:
                print(f"  UWAGA: {f.relative_to(config.ROOT)} - nieznana kość, zostaje na miejscu")
                continue
            dst = target / sub / f"{new_id}.json"
            if dst.exists():
                print(f"  pominięto (już jest): {dst.relative_to(config.ROOT)}")
                continue
            text = f.read_text(encoding="utf-8")
            json.loads(text)                           # uszkodzony plik = błąd przed przeniesieniem
            dst.parent.mkdir(parents=True, exist_ok=True)
            # tekst bez ponownego formatowania: podmiana tylko pola "bone"
            text = text.replace(f'"bone": "{f.stem}"', f'"bone": "{new_id}"', 1)
            dst.write_text(text, encoding="utf-8")
            f.unlink()
            moved += 1
    auto = root / "auto"
    if auto.is_dir() and not any(auto.iterdir()):
        auto.rmdir()
    return moved


def main() -> None:
    """Przenosi skany, wyniki i oznaczenia do zestawu first_scans i wypisuje, co zrobiono."""
    print(f"Zestaw \"{DATASET}\":")
    n = migrate_scans()
    print(f"  skany:      {n} plików -> {config.SCANS_ROOT.name}/{DATASET}/")
    n = migrate_out()
    print(f"  wyniki:     {n} elementów -> {config.OUT_ROOT.relative_to(config.ROOT).as_posix()}/{DATASET}/")
    n = migrate_contacts()
    print(f"  oznaczenia: {n} plików -> {config.CONTACTS_ROOT.name}/{DATASET}/")
    print("Gotowe. Teraz: python scripts/prepare_data.py")


if __name__ == "__main__":
    main()

"""SMM 2026: pobierz startowe zbiory modeli 3D.
Użycie: python pobierz_dane.py
Opcjonalnie: python pobierz_dane.py --only brick
"""
from pathlib import Path
from urllib.request import Request, urlopen
import argparse
import hashlib
import zipfile
import shutil
import ssl
import certifi

ROOT = Path(__file__).resolve().parent
BASE_BRICK = 'https://www.geometrie.tuwien.ac.at/nbkdir/hofer/3dpuzzles/'
BONE = 'https://zenodo.org/records/6614647/files/Long%20Bone%20Dataset.zip?download=1'

def fetch(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print('Już pobrano:', target)
        return
    print('Pobieranie:', target.name)
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0 (SMM Research Dataset Downloader)'})
    temp = target.with_name(target.name + '.part')
    try:
        context = ssl.create_default_context(cafile=certifi.where())

        with urlopen(req, timeout=120, context=context) as response, temp.open('wb') as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        temp.replace(target)
        print('OK:', round(target.stat().st_size / 1024 / 1024, 2), 'MB')
    except Exception:
        temp.unlink(missing_ok=True)
        raise

def extract(zip_path, to):
    to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        # Bezpieczne rozpakowanie bez wychodzenia poza katalog wynikowy
        for m in z.infolist():
            path = (to / m.filename).resolve()
            if not path.is_relative_to(to.resolve()):
                raise ValueError('Niebezpieczna ścieżka w ZIP: ' + m.filename)
        z.extractall(to)
    print('Rozpakowano:', to)

def extract_brick(zip_path, to, expected_filename):
    """Bezpiecznie wyciąga dokładnie jeden oczekiwany plik PCD.

    Nie korzysta ze ścieżki ../ zapisanej wewnątrz archiwum.
    """
    to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        matches = [m for m in z.infolist()
                   if not m.is_dir()
                   and m.filename.replace('\\', '/').split('/')[-1] == expected_filename]
        if len(matches) != 1:
            raise ValueError(f'Oczekiwano jednego pliku {expected_filename}, znaleziono {len(matches)}')
        member = matches[0]
        if member.file_size > 250 * 1024 * 1024:
            raise ValueError('Plik PCD jest niespodziewanie duży')
        destination = to / expected_filename
        with z.open(member) as src, destination.open('wb') as dst:
            shutil.copyfileobj(src, dst)
    print('Bezpiecznie rozpakowano:', destination)

def convert_vienna_pcd_to_ply(source, target):
    # TU Wien PCD: pierwsza linia N, kolejne x y z nx ny nz.
    # To nie jest standardowe PCL PCD.
    rows = source.read_text(encoding='utf-8-sig').splitlines()
    n = int(rows[0].strip())
    coords = []
    for line in rows[1:]:
        if not line.strip():
            continue
        values = line.split()
        if len(values) < 6:
            raise ValueError('Nieoczekiwany wiersz w ' + str(source))
        coords.append(values[:6])
    if len(coords) != n:
        raise ValueError(f'Liczba punktów niezgodna: {len(coords)} != {n} w {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='ascii') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {n}\n')
        for name in ('x','y','z','nx','ny','nz'):
            f.write('property float ' + name + '\n')
        f.write('end_header\n')
        for c in coords:
            f.write(' '.join(c) + '\n')
    print('PLY:', target)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['brick','bones'], help='Pobierz tylko jeden zbiór')
    args = ap.parse_args()
    if args.only in (None, 'brick'):
        for i in range(1, 7):
            filename = f'brick_part{i:02d}.pcd.zip'
            archive = ROOT / '02_FRAGMENTY_ZLAMANE' / 'Vienna_Brick' / 'archiwa' / filename
            try:
                fetch(BASE_BRICK + filename, archive)
                raw = ROOT / '02_FRAGMENTY_ZLAMANE' / 'Vienna_Brick' / 'raw'
                extract_brick(archive, raw, f'brick_part{i:02d}.pcd')
                for path in raw.rglob(f'brick_part{i:02d}.pcd'):
                    convert_vienna_pcd_to_ply(path, ROOT / '02_FRAGMENTY_ZLAMANE' / 'Vienna_Brick' / 'PLY' / (path.stem + '.ply'))
            except Exception as ex:
                print('BŁĄD:', filename, type(ex).__name__, str(ex))
    if args.only == 'bones':
        archive = ROOT / '01_WZORCE_KOSCI' / 'Zenodo_Long_Bones' / 'Long_Bone_Dataset.zip'
        try:
            fetch(BONE, archive)
            extract(archive, ROOT / '01_WZORCE_KOSCI' / 'Zenodo_Long_Bones' / 'OBJ')
        except Exception as ex:
            print('BŁĄD: modele kości:', type(ex).__name__, str(ex))
    print('\nZakończono. Sprawdź komunikaty BŁĄD powyżej.')

if __name__ == '__main__':
    main()

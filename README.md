# PROJEKT KOSTNY

# PROBLEM

Mając fragmenty kości, należy je zeskanować skanerem świetlnym, który tworzy obrazy 3D. Następnie spróbować dopasować fragmenty do siebie tak, aby pasowały. Potem z dopasowanych fragmentów, można dalej analizować w jaki sposób doszło do uszkodzenia tych kości.

## Podproblemy:

1. Robienie zdjęć, skanowanie kości
2. Za pomocą bibliotek w pythonie przekonwertować na format, na którym można pracować
3. Oczyszczenie i przygotowanie danych
4. Dopasowywanie powierzchni ze zdjec 2d (niedziela)
5. Przełożenie na 3d

# Projekt Python

## Wymagania

- Windows (projekt testowany na Windows 11)
- [Python 3.13](https://www.python.org/downloads/) (przy instalacji zaznacz „Add Python to PATH”)
- [Git](https://git-scm.com/downloads)

> Linux coming soon

## Pobranie repozytorium

```powershell
git clone https://github.com/SamuelPiotrowski/SMM-rekonstrukcja.git
cd SMM-rekonstrukcja
```

Bez Gita: na GitHubie ([SamuelPiotrowski/SMM-rekonstrukcja](https://github.com/SamuelPiotrowski/SMM-rekonstrukcja)) **Code → Download ZIP** i rozpakuj.

## Przygotowywanie folderu

Należy samodzielnie utworzyć środowisko wirtualne (folder `.venv` nie jest w repo):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Jeśli PowerShell blokuje `activate`, uruchom raz:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Następnie zainstalować biblioteki z requirements:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Skany

Skanów **nie ma w repo** (są duże, a każdy nowy skan powiększałby historię gita na zawsze). Trzeba je samodzielnie wrzucić do folderu `scanned_bones_light/` w głównym folderze projektu (utworzyć go, jeśli nie istnieje). Wystarczą pliki `.ply`, bo kod i tak wybiera najpierw ten format.

Nazwy plików w formacie ze skanera: `<data>-<kość>-<część>.ply`, np. `23.11.2026-kośc strzałkowa 3-lowerjaw.ply`.

Inny folder można ustawić w `settings.json` (`paths.data_in`).

### Zdjęcia

Zdjęcia fragmentów (jeden fragment na zdjęcie) trzeba wrzucić do `bone_photos/` (nie ma ich w repo).
Zasady: kontrastowe tło (jasna kość na ciemnym albo odwrotnie), aparat prostopadle do stołu,
ta sama odległość dla wszystkich zdjęć albo linijka na każdym zdjęciu.

## Dopasowanie 2D (B-splajny)

### Struktura kodu (`src/`)

| plik | etap | co robi |
|---|---|---|
| `geometry.py` | – | wspólna geometria 2D: transformacje sztywne, pola, łamane, Kabsch/SVD |
| `synthetic.py` | 1 | syntetyczna kość, poszarpane przełomy (`seam`), obrazy testowe + `truth.json` |
| `approximation.py` | 2 | segmentacja (Otsu), kontur, wycięcie przełomów, B-splajny i krzywizna |
| `matching.py` | 3–5 | dopasowanie par przełomów, odrzucanie złych połączeń, złożenie, ocena względem prawdy |
| `results.py` | 6 | zapis wyników: `case.json`, CSV, maski, obrazy PNG |
| `visualization_3d.py` | – | interaktywna wizualizacja 3D (plik HTML, animacja składania) |
| `pipeline.py` | – | cały przebieg jednego przypadku (`run_case`) |
| `annotate.py` | – | klikanie końców przełomów na zdjęciach → `annotations.json` |
| `config.py` | – | wczytuje `settings.json` (wszystkie progi i parametry) |

### Dane syntetyczne (test metody)

```powershell
python scripts/run_synthetic.py
```

Wyniki w `out/synthetic/<clean|noise|loss>/`, m.in. `assembly_3d.html` (otwórz w przeglądarce).

### Własne zdjęcia

```powershell
# 1. kliknij końce przełomów (2 kliknięcia = 1 przełom, Enter = następne zdjęcie)
python scripts/annotate_photos.py --mm-per-pixel 0.1     # znana skala
python scripts/annotate_photos.py --ruler-mm 10          # albo skala z linijki na zdjęciu

# 2. analiza -> out/<nazwa>/ i wizualizacja 3D w przeglądarce
python scripts/run_own_data.py

# ponowna wizualizacja 3D bez liczenia od nowa
python scripts/show_3d.py out/<nazwa>/case.json
```

Format `annotations.json` jest opisany na początku `src/approximation.py`. Progi dopasowania są w `settings.json`
(sekcja `matching`). Dla małych kości zmniejsz `min_contact_mm`, bo musi być krótsze niż przełomy.

Test ogólności (inny rozmiar i skala zdjęć, ciemna kość na jasnym tle, JPG, bez prawdy):
`python tests/test_own_data.py`

# Szczegółowe notatki podproblemów

## 1. Skanowanie

### Surowe dane ze skanów, prosto z pliku
- Wierzchołki: współrzędne (x, y, z) każdego punktu na powierzchni.
- Ściany: trójkąty, czyli które wierzchołki są ze sobą połączone.
- Normalne: kierunek „na zewnątrz” w każdym punkcie. Jeśli ich nie ma, można je policzyć.
- Kolor na wierzchołek: tylko jeśli skaner go zapisał.

### Zdjęcia 2d z czterech stron





# PROJEKT KOSTNY

# PROBLEM

Mając fragmenty kości, należy je zeskanować skanerem świetlnym, który tworzy obrazy 3D. Następnie spróbować dopasować fragmenty do siebie tak, aby pasowały. Potem z dopasowanych fragmentów, można dalej analizować w jaki sposób doszło do uszkodzenia tych kości.

## Podproblemy:

1. Skanowanie kości, przede wszystkim ich powierzchni złamania. Rozwiązanie problemów z niedokładnym skanem. Skany mają formaty: (obj. ply, stl).
2. Ręczne dopasowywanie fragmentów w programie 3d. Zbaczenie czy skala się zgadza pomiędzy osobno zeskanowanymi fragmentami.
3. Za pomocą bibliotek w pythonie przekonwertować na format, na którym można pracować
4. Oczyszczenie i przygotowanie danych
5. Wstępna selekcja kości po wielkości powierzchni złamania, jej średnicy lub innych wspólnych cechach końcówek.
6. Dopasowywanie bardziej szczegółowe algorytmem dopasowywania punktów. Chyba to co robi Kasia z tymi klockami może się nadać.

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

## Uruchomienie

Skrypty uruchamiające są w `scripts/` (same wywołania funkcji), a funkcje w `src/`.
Uruchamiamy z głównego folderu projektu, przy aktywnym `.venv`.

1. Przygotowanie danych ze skanów (`scanned_bones_light/` → `out/`):

   ```powershell
   python scripts/prepare_data.py
   ```

   Co dokładnie robi `scripts/prepare_data.py` (funkcje z `src/data_parsing.py`):

   1. **Szuka skanów** w `scanned_bones_light/`. Skaner zapisuje ten sam skan jako `.ply`, `.obj` i `.stl`, więc pliki są grupowane po nazwie i z każdej grupy brany jest jeden, w kolejności **PLY → OBJ → STL** (PLY jest najmniejszy i ma kolory). Gdy nie ma żadnego skanu, program kończy się błędem `FileNotFoundError`.
   2. **Wczytuje siatkę** przez `trimesh` (open3d na Windowsie nie otwiera ścieżek z polskimi znakami) i przepisuje ją do open3d: wierzchołki w mm, trójkąty i kolory wierzchołków, jeśli są. Zapisuje statystyki surowej siatki: liczbę wierzchołków i trójkątów, pole, wymiary i to, czy siatka jest zamknięta.
   3. **Czyści siatkę z błędów skanowania**:
      - scala zduplikowane wierzchołki (STL zapisuje każdy trójkąt osobno),
      - usuwa zdegenerowane trójkąty (o zerowym polu) i zduplikowane trójkąty,
      - usuwa krawędzie nierozmaitościowe (wspólne dla więcej niż 2 trójkątów) i nieużywane wierzchołki,
      - usuwa małe oderwane kawałki (szum, odpryski), czyli części o polu mniejszym niż 5% największej (`MIN_COMPONENT_AREA_RATIO`),
      - liczy normalne wierzchołków i trójkątów.

      Siatka celowo **nie jest wygładzana ani upraszczana**, bo szczegóły powierzchni złamania są potrzebne do dopasowania. Jeśli po czyszczeniu zostaje kilka osobnych kawałków, w logu pojawia się ostrzeżenie.
   4. **Centruje fragment**: przesuwa go tak, żeby jego środek był w (0, 0, 0) (`CENTER_FRAGMENTS`). Wektor przesunięcia trafia do `meta.json`, więc da się wrócić do współrzędnych skanera.
   5. * **Tworzy chmurę punktów wraz z normalnymi**: losuje punkty równomiernie po powierzchni, a potem przerzedza je do ok. 1 punktu na woksel 0.5 mm (`VOXEL_SIZE`). Dzięki temu gęstość chmury nie zależy od gęstości oryginalnej siatki. Normalne punktów pochodzą z trójkątów, więc wszystkie są skierowane na zewnątrz kości.
   6. * **Liczy deskryptory FPFH**: dla każdego punktu chmury 33 liczby opisujące lokalny kształt powierzchni w promieniu 2.5 mm (5 × woksel). Nie zależą od położenia ani obrotu fragmentu, więc posłużą do zgrubnego dopasowania (RANSAC), a potem do dokładnego (ICP).
   7. **Zapisuje wyniki** do `out/<nazwa_ascii>/` (polskie znaki są usuwane z nazwy, np. `kośc strzałkowa` → `kosc_strzalkowa`):
      - `mesh_clean.ply`: oczyszczona siatka,
      - `cloud.ply`: chmura punktów z normalnymi,
      - `fpfh.npy`: deskryptory FPFH, tablica (N, 33),
      - `meta.json`: plik źródłowy, data, kość i część (z nazwy pliku), statystyki przed i po czyszczeniu, usunięte kawałki, przesunięcie, liczba punktów.
   8. **Tworzy `out/summary.csv`**: tabelę z jednym wierszem na fragment (separator `;`, kodowanie dla Excela).

   Ziarno losowania jest stałe, więc każde uruchomienie daje identyczne chmury punktów. Istniejące wyniki w `out/` są nadpisywane. Parametry (woksel, progi, promienie) można zmienić w `settings.json`, sekcja `data_parsing`.

    > **chmura i z niej deskryptory FPFH** : rozwiązanie dopasowywania zaproponowane przez AI,
    > jest to inna koncepcja niż nasza wymyślona ze średnicą, ale zostawiam w kodzie. 

   ---

    **Funkcje z `src/` można importować i używać osobno** (tak robią skrypty w `scripts/`)

    Na górze pliku wpisujemy:

    ```python
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # dla pliku w scripts/ albo tests/
    ```

    Dzięki temu zadziała nam poniższy import:

    ```python
    from data_parsing import [nazwa_funkcji]
    ```

## Ustawienia

Wszystkie ustawienia są w jednym pliku **`settings.json`** w głównym folderze projektu. Zmienia się je tylko tam, a `src/config.py` jedynie wczytuje ten plik. Przy każdej wartości w `settings.json` jest komentarz, co ona znaczy. Zmiany działają przy następnym uruchomieniu skryptu.

Przy edycji pamiętaj o składni JSON:
- `true` / `false` / `null` małymi literami (nie `True` / `None` jak w Pythonie),
- tekst w podwójnym cudzysłowie `"..."`, liczby z kropką (`0.5`, nie `0,5`),
- bez przecinka po ostatnim elemencie,
- komentarze `// ...` są dozwolone (zwykły JSON ich nie ma, ale `config.py` je usuwa przed wczytaniem, a VS Code traktuje ten plik jako JSON z komentarzami dzięki `.vscode/settings.json`).

### `paths`: foldery

| Klucz | Domyślnie | Znaczenie |
|---|---|---|
| `data_in` | `"scanned_bones_light"` | folder z surowymi skanami (względem głównego folderu projektu) |
| `data_out` | `"out"` | folder z wynikami przygotowania danych |

### `data_parsing`: przygotowanie danych (`scripts/prepare_data.py`)

Po zmianie tych wartości trzeba ponownie uruchomić `python scripts/prepare_data.py`, żeby wyniki w `out/` się przeliczyły.

| Klucz | Domyślnie | Znaczenie |
|---|---|---|
| `preferred_formats` | `["ply", "obj", "stl"]` | kolejność wyboru formatu, gdy ten sam skan jest w kilku plikach (PLY: najmniejszy, ma kolory; STL nie ma kolorów) |
| `min_component_area_ratio` | `0.05` | oderwane kawałki o polu mniejszym niż ten ułamek największego są usuwane jako szum (`0` = nic nie usuwaj) |
| `center_fragments` | `true` | przesuń każdy fragment tak, żeby jego środek był w (0, 0, 0); przesunięcie trafia do `meta.json` |
| `voxel_size` | `0.5` | rozmiar woksela [mm], czyli mniej więcej odstęp między punktami chmury; mniej = gęstsza chmura, wolniejsze liczenie |
| `sampling_density` | `4` | ile punktów losować na woksel² powierzchni przed przerzedzeniem |
| `normal_radius_factor` | `2.0` | promień liczenia normalnych = ta liczba × `voxel_size` |
| `fpfh_radius_factor` | `5.0` | promień deskryptorów FPFH = ta liczba × `voxel_size` (domyślnie 2.5 mm) |

### `visualizing`: wygląd okna 3D

| Klucz | Domyślnie | Znaczenie |
|---|---|---|
| `window_size` | `[1600, 900]` | rozmiar okna i zrzutu ekranu [px] (szerokość, wysokość) |
| `normal_length_mm` | `2.0` | długość rysowanych normalnych [mm] |
| `max_normals` | `3000` | maks. liczba rysowanych normalnych na fragment (wszystkie zasłoniłyby kość) |
| `axes_size_mm` | `10.0` | długość osi układu współrzędnych [mm] |
| `background` | `[1.0, 1.0, 1.0]` | kolor tła, RGB w zakresie 0–1 (tu biały) |
| `bone_color` | `[0.89, 0.85, 0.79]` | kolor kości, gdy skan nie ma własnych kolorów |
| `normal_color` | `[0.15, 0.15, 0.15]` | kolor odcinków normalnych |
| `camera_tilt` | `0.35` | pochylenie kamery startowej, żeby było widać głębię (`0` = patrzy prosto z góry na najcieńszą oś) |
| `camera_zoom` | `0.5` | zoom kamery startowej (mniej = bliżej; domyślne open3d to 0.7) |

### `view`: co pokazuje `tests/visualize.py`

| Klucz | Domyślnie | Znaczenie |
|---|---|---|
| `show` | `"mesh"` | `"mesh"` = oczyszczona siatka, `"cloud"` = chmura punktów, `"fpfh"` = chmura pokolorowana wg FPFH (podobny kolor = podobny lokalny kształt powierzchni, porównywalny między fragmentami) |
| `layout` | `"side"` | `"side"` = fragmenty obok siebie wzdłuż osi X, `"original"` = współrzędne skanera |
| `colors` | `"scan"` | `"scan"` = kolory skanera, `"fragment"` = każdy fragment w innym kolorze (legenda w konsoli); ignorowane przy `"fpfh"` |
| `only` | `null` | `null` = wszystkie fragmenty, albo lista fragmentów nazw, np. `["lowerjaw", "upperjaw"]` (wielkość liter bez znaczenia; literówka = błąd z listą dostępnych) |
| `normals` | `false` | dorysuj normalne chmury punktów |
| `axes` | `false` | pokaż osie w (0, 0, 0): X czerwona, Y zielona, Z niebieska |
| `gap` | `10.0` | odstęp między fragmentami w `layout: "side"` [mm], gdy kilka jest w jednym oknie (`tests/visualize.py` pokazuje po jednym, więc tam bez znaczenia) |
| `point_size` | `3.0` | rozmiar punktów chmury na ekranie [px] |
| `screenshot` | `null` | `null` = otwórz okno, albo ścieżka (względem głównego folderu), np. `"out/podglad.png"`: zapis obrazów bez okna, osobny dla każdego fragmentu (`out/podglad_lowerjaw.png`, ...) |

Przykład: chmura FPFH tylko dla dwóch fragmentów, zapisana do pliku:

```json
"view": {
  "show": "fpfh",
  "layout": "side",
  "colors": "scan",
  "only": ["lowerjaw", "upperjaw"],
  "normals": false,
  "axes": false,
  "gap": 10.0,
  "point_size": 3.0,
  "screenshot": "out/fpfh_szczeki.png"
}
```

## Testy

Folder `tests/` zawiera szybkie skrypty do ręcznego testowania funkcji z `src/`. Uruchamiamy je z głównego folderu projektu.

### Wizualizacja 3D: `tests/visualize.py`

Pokazuje **każdy fragment z `out/` osobno, po kolei**: zamknięcie okna (`Q` / `Esc`) otwiera następny. Wymaga wcześniej przygotowanych danych (`python scripts/prepare_data.py`).

```powershell
python tests/visualize.py
```

- Co i jak pokazać (siatka / chmura / FPFH, kolory, normalne, osie), ustawia się w `settings.json`, sekcja `view` (opis wyżej, w [Ustawienia](#ustawienia)).
- `only` ogranicza, które fragmenty zostaną pokazane, np. `["lowerjaw"]`.
- Przy ustawionym `screenshot` nie otwiera okien, tylko zapisuje osobny obraz dla każdego fragmentu, np. `out/podglad.png` → `out/podglad_lowerjaw.png`, `out/podglad_upperjaw.png`, ...
- Przed każdym oknem w konsoli wypisuje numer i opis fragmentu (trójkąty, pole, wymiary, liczba punktów chmury).
- Funkcje wizualizacji są w `src/visualizing_3d.py`.

Sterowanie w oknie: lewy przycisk - obrót, kółko - zoom, Ctrl + lewy / środkowy przycisk - przesuwanie, `+` / `-` - rozmiar punktów, `R` - reset widoku, `H` - pomoc w konsoli, `Q` / `Esc` - zamknięcie (następny fragment).


## Struktura

```
scanned_bones_light/   surowe skany (.obj, .ply, .stl) - nie ma w repo, patrz „Skany”
src/                   funkcje (config, data_parsing, visualizing_3d)
scripts/               skrypty uruchamiające (prepare_data)
settings.json          wszystkie ustawienia
tests/                 skrypty testowe (visualize)
out/                   wyniki przetwarzania (generowane, nie ma w repo)
```

# Szczegółowe notatki podproblemów

## 1. Skanowanie

### Surowe dane ze skanów, prosto z pliku
- Wierzchołki: współrzędne (x, y, z) każdego punktu na powierzchni.
- Ściany: trójkąty, czyli które wierzchołki są ze sobą połączone.
- Normalne: kierunek „na zewnątrz” w każdym punkcie. Jeśli ich nie ma, można je policzyć.
- Kolor na wierzchołek: tylko jeśli skaner go zapisał.


## 2. Ręczne dopasowywanie

### Ustalić czy skany zachowują skalę.

    Spróbować dopasować zeskanowane fragmenty ręcznie w programie.

## 5. Wstępna selekcja

### Jak wstępnie ustalić, które kości będą kandydatami na dopasowanie

    Można jako kryterium dopasowania wybrać największą długość dwóch punktów leżących na powierzchni 2d złamania (średnicę półosi wielkiej tak jakby).
    Jeśli dwa fragmenty mają ją podobną to jest szansa, że będą pasować. 

### Jak znaleźć front kości? Jak znaleźć powierzchnię złamania.

    Najpierw trzeba znaleźć oś kości i tak, żeby patrzeć prostopadle na przekrój poprzeczny.

    Pomysł: zeskanować dłuższy fragment poza końcówką i uznać linie wzdłuż kości jako równoległe i zrobić prostopadłą linie łączącą te dwie równoległe.

    Widokiem z boku uwzględniamy takie rzeczy jak:
    - kąt nachylenia do podstawy (rysujemy linie od najnizszego punktu do najwyzszego)
    - 

## Sposób dopasowania fragmentów kości za pomocą pełnych skanów

    Pomysły: mamy deskryptory FPFH
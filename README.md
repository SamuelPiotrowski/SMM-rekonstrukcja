# PROJEKT KOSTNY

# PROBLEM

Mając fragmenty kości, należy je zeskanować skanerem świetlnym, który tworzy obrazy 3D. Następnie spróbować dopasować fragmenty do siebie tak, aby pasowały. Potem z dopasowanych fragmentów, można dalej analizować w jaki sposób doszło do uszkodzenia tych kości.

## Podproblemy:

1. Skanowanie kości, przede wszystkim ich powierzchni złamania. Rozwiązanie problemów z niedokładnym skanem. Skany mają formaty: (obj. ply, stl).
2. Ręczne dopasowywanie fragmentów w programie 3d. Zbaczenie czy skala się zgadza pomiędzy osobno zeskanowanymi fragmentami.
3. Za pomocą bibliotek w pythonie przekonwertować na format, na którym można pracować
4. Oczyszczenie i przygotowanie danych (w tym łączenie wielu skanów tej samej kości)
5. Wstępna selekcja kości po wielkości powierzchni złamania, jej średnicy lub innych wspólnych cechach końcówek.
6. Dopasowywanie bardziej szczegółowe algorytmem dopasowywania punktów. Chyba to co robi Kasia z tymi klockami może się nadać.

# Projekt Python

## Wymagania

- Windows (projekt testowany na Windows 11)
- [Python 3.13](https://www.python.org/downloads/) (przy instalacji zaznacz „Add Python to PATH”)
- [Git](https://git-scm.com/downloads)
- przeglądarka (wizualizacje i narzędzie do oznaczania otwierają się w niej)

> Linux coming soon

## Pobranie repozytorium

```powershell
git clone https://github.com/SamuelPiotrowski/SMM-rekonstrukcja.git
cd SMM-rekonstrukcja
```

Bez Gita: na GitHubie ([SamuelPiotrowski/SMM-rekonstrukcja](https://github.com/SamuelPiotrowski/SMM-rekonstrukcja)) **Code → Download ZIP** i rozpakuj.

## Przygotowanie folderu

Środowisko wirtualne (folderu `.venv` nie ma w repo) i biblioteki:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jeśli PowerShell blokuje `activate`, uruchom raz: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Skany

Skanów **nie ma w repo** (są duże). Wrzuć je do folderu `scanned_bones/` w głównym folderze projektu - **tak, jak eksportuje je skaner**, razem z podfolderami sesji:

```
scanned_bones/
    23.11.2026-modelowanie matematyczne/
        23.11.2026-modelowanie matematyczne-lowerjaw.ply   (+ .obj, .stl - wystarczy jeden)
        23.11.2026-modelowanie matematyczne-upperjaw.ply
    23.11.2026-modelowanie matematyczne_(1)/
        23.11.2026-modelowanie matematyczne-upperjaw.ply   <- ta sama nazwa, ale inny skan
    first_test/
        ...
```

- skany są szukane we **wszystkich podfolderach**; ten sam plik w różnych folderach to różne skany,
- ten sam skan w kilku formatach jest wczytywany raz (kolejność: PLY → OBJ → STL),
- skany zgryzu Medita (`occlusion...`) są **pomijane** - to oba fragmenty złożone razem, więc skleiłyby dwie kości w jedną (zmiana: `scans.exclude` w `settings.json`),
- **nie trzeba** ręcznie wskazywać, które skany to ta sama kość - program sam to wykrywa.

Inny folder: `settings.json`, `paths.data_in`.

## Jak używać

Wszystko uruchamiamy z głównego folderu projektu, przy aktywnym `.venv`. Skrypty są w `scripts/` (same wywołania), funkcje w `src/`.

| krok | polecenie | co robi | wynik |
|---|---|---|---|
| 1 | `python scripts/prepare_data.py` | skany → kości: łączy skany tej samej kości, usuwa szum | `out/3d/bones/` |
| 2 | `python scripts/show_merging.py` | pokazuje, jak połączyły się skany | `out/3d/merging.html` |
| 3 | `python scripts/mark_contacts.py` | **ręczne oznaczanie powierzchni styku** w przeglądarce | `contacts/*.json` |
| 4 | `python scripts/show_bones.py` | wszystkie kości + oznaczone powierzchnie styku | `out/3d/bones.html` |
| 5 | `python scripts/match_surfaces.py` | **dopasowanie fragmentów po powierzchniach styku** (RANSAC + ICP) | `out/3d/matching.html` |
| 6 | `python scripts/tune_matching.py` | **podgląd dopasowania z suwakami** - zmiana ustawienia przelicza pokazaną parę | – (nic nie zapisuje) |
| – | `python scripts/detect_contacts.py` | algorytm wykrywania powierzchni styku + porównanie z ręcznymi | `contacts/auto/*.json` |
| – | `python tests/test_pipeline.py` | test całości na sztucznych skanach (~20 s) | – |

Wizualizacje (krok 2 i 4) to pojedyncze pliki HTML - otwierają się same, działają bez internetu i można je komuś wysłać.

### 1. Przygotowanie danych: `scripts/prepare_data.py`

1. **Szuka skanów** w `scanned_bones/` i podfolderach (patrz wyżej).
2. **Wczytuje i naprawia siatki**: scala zduplikowane wierzchołki, usuwa zdegenerowane i zduplikowane trójkąty oraz krawędzie nierozmaitościowe. Siatka **nie jest wygładzana** - szczegóły przełomu są potrzebne do dopasowania.
3. **Porównuje każdą parę skanów**: ustawia je na sobie (zgrubnie deskryptorami FPFH + RANSAC, dokładnie ICP) i mierzy **nakładanie** = jaka część mniejszego skanu leży na drugim (bliżej niż 0.15 mm, normalne w tę samą stronę). Na naszych skanach: różne fragmenty 0.05–0.20, ten sam fragment 0.49–0.73.
   Pierwsze uruchomienie trwa kilka minut (15 skanów = 105 par ≈ 2.5 min), wyniki par są w cache (`out/3d/cache/`), więc kolejne trwają ~20 s. Nowe skany liczą tylko nowe pary.
4. **Grupuje skany w kości**: nakładanie ≥ `min_overlap` (0.3) = ta sama kość. Działa też „przez pośrednika”: dwa skany różnych końców kości łączą się przez trzeci, który obejmuje oba.
5. **Usuwa szum**: oderwany kawałek skanu zostaje tylko, jeśli jest głównym kawałkiem, albo potwierdzają go inne skany tej kości, albo jest duży (≥ 5% głównego) i leży blisko (≤ 3 mm). Np. w `first_test/upperjaw` dwa kawałki leżące 120 mm od kości (połowicznie zeskanowany inny fragment) znikają.
6. **Skleja powierzchnię**: cały skan referencyjny (największy) + z pozostałych tylko to, czego w nim brakuje (oryginalna geometria i kolory, bez wygładzania). Alternatywa: `"fuse_method": "poisson"` - jedna gładka powierzchnia.
7. **Liczy chmurę punktów i FPFH** (do dopasowania fragmentów) i zapisuje wszystko do `out/3d/bones/<id kości>/`.

Konsola pokazuje, co się stało, np.:

```
  11. first_test / lowerjaw: 3 skan(y) [first_test / lowerjaw (ref.), modelowanie matematyczne_(6) / lowerjaw, ...] -> 3019 mm2
  12. first_test / upperjaw: 2 skan(y) [...] -> 2380 mm2, usunięty szum: 274 mm2
      first_test / upperjaw: usunięto kawałek 157 mm2 - SZUM: 120 mm od głównego, brak potwierdzenia
```

### 2. Sprawdzenie łączenia: `scripts/show_merging.py`

Strona z dwiema częściami:
- **3D**: skany każdej kości nałożone na siebie, każdy w innym kolorze, usunięty szum na czerwono. Drobne „paski” dwóch kolorów = skany dobrze leżą na sobie. Kość wybiera się z listy w lewym dolnym rogu.
- **Macierz nakładania** wszystkich par skanów; ramki = skany połączone w jedną kość.

Jeśli coś jest źle połączone (albo nie połączone), w `settings.json` zmień `merging.min_overlap` albo podaj grupy ręcznie:

```json
"groups": {"strzalkowa_A": ["first_test-lowerjaw", "matematyczne_6"]}
```

(nazwy to fragmenty nazw skanów wypisywanych przez `prepare_data.py`; skany spoza listy dalej są grupowane automatycznie).

### 3. Oznaczanie powierzchni styku: `scripts/mark_contacts.py`

Otwiera w przeglądarce narzędzie do malowania powierzchni styku (przełomów) na kościach. Lista kości jest po lewej (zielona kropka = oznaczona), narzędzia po prawej.

- **Różdżka** (`W`): kliknij przełom - zaznacza obszar aż do ostrej krawędzi. Za dużo? `Ctrl+Z` i zmniejsz tolerancję.
- **Pędzel** (`B`) / **Gumka** (`E`): poprawki lewym przyciskiem, rozmiar `[` `]`.
- Kilka przełomów na jednej kości = **Nowa powierzchnia** (`N`), każda ma swój kolor i nazwę (dwuklik zmienia nazwę).
- **Propozycja algorytmu**: wczytuje wynik algorytmu jako punkt wyjścia do poprawek.
- Obracanie **prawym** przyciskiem, przesuwanie środkowym, zoom kółkiem. `H` - pomoc.
- `Ctrl+S` zapisuje do `contacts/<id kości>.json`; przy zmianie kości zapis jest automatyczny.

Folder `contacts/` **jest w repozytorium** - to ręczna praca, warto ją commitować i dzielić się nią (poza podfolderem `contacts/auto/` z wynikami algorytmu, który jest generowany). Jeśli siatka kości się zmieni (np. po zmianie ustawień i ponownym `prepare_data.py`), oznaczenia są automatycznie przenoszone na nową siatkę po położeniu punktów.

Za pierwszym razem narzędzie potrzebuje internetu (pobiera bibliotekę three.js do `out/3d/.cache/`), potem działa offline. Zakończenie: `Ctrl+C` w konsoli.

### 4. Wizualizacja kości: `scripts/show_bones.py`

Wszystkie kości ułożone płasko „na stole”, ponumerowane, z oznaczonymi powierzchniami styku. Lista „Wszystkie kości” przybliża wybraną kość, przyciski: widok 3D / z góry / z boku, powierzchnie styku pokaż / ukryj, legenda ukrywa kości, najechanie myszą pokazuje szczegóły (skany, pole, wymiary, pola i średnice powierzchni styku).

```powershell
python scripts/show_bones.py                    # kolory skanera
python scripts/show_bones.py --colors fragment  # każda kość w innym kolorze
python scripts/show_bones.py --colors fpfh      # kolor wg lokalnego kształtu (FPFH), porównywalny między kośćmi
python scripts/show_bones.py --auto             # powierzchnie z algorytmu zamiast ręcznych
python scripts/show_bones.py 3 strzalkowa       # tylko kość nr 3 i kości z "strzalkowa" w nazwie (pełna rozdzielczość)
```

### Algorytm wykrywania powierzchni styku (na przyszłość): `scripts/detect_contacts.py`

Wszystko jest przygotowane pod algorytm, który zastąpi ręczne oznaczanie:
- algorytm to funkcja `(kość, parametry) -> etykiety wierzchołków` zarejestrowana dekoratorem `@detector("nazwa")` w `src/contacts.py`,
- wynik ma **ten sam format** co ręczne oznaczenia i leży obok nich: `contacts/auto/<id kości>.json` (pole `source` = `auto:<nazwa>`, pole `detector` = algorytm i jego ustawienia). Działa z wizualizacją (`show_bones.py --auto`), narzędziem („Propozycja algorytmu”) i dopasowaniem (`match_surfaces.py --source auto`),
- który algorytm jest używany: `settings.json`, sekcja `contacts`, klucz `method` (jednorazowo: `--method`),
- `detect_contacts.py` liczy zawsze od nowa; dopasowanie bierze zapisany wynik i liczy sam tylko wtedy, gdy go brak albo zmienił się algorytm, jego ustawienia lub siatka kości,
- `detect_contacts.py` od razu **porównuje wynik z ręcznymi oznaczeniami** (IoU, precyzja, czułość) - ręczne oznaczenia są „prawdą” do oceny algorytmu.

Teraz jest tylko prosty algorytm bazowy `roughness` (przełom = szorstka powierzchnia, kora = gładka): punkt wyjścia, nie rozwiązanie.

```python
# src/contacts.py
@detector("moja_metoda")
def moja_metoda(bone, params):
    labels = np.zeros(len(bone.mesh.vertices), dtype=np.uint8)
    ...                     # 0 = brak, 1, 2, ... = kolejne powierzchnie styku
    return labels
```

```powershell
python scripts/detect_contacts.py --method moja_metoda
```

### 5. Dopasowanie po powierzchniach styku: `scripts/match_surfaces.py`

Pierwsza wersja dopasowania fragmentów (podproblem 6). Bierze **tylko punkty powierzchni styku**, każdą powierzchnię osobno, i dopasowuje każdą do każdej powierzchni **innej** kości.

Skąd powierzchnie (`settings.json` → `matching` → `source`, jednorazowo `--source`):

| tryb | powierzchnie |
|---|---|
| `manual` (domyślny) | tylko ręczne (`contacts/<id>.json`); kości bez nich są pomijane |
| `auto` | tylko z algorytmu (`contacts/auto/<id>.json`; gdy brak albo nieaktualne - liczone i zapisywane) |
| `manual-auto` | ręczne, a dla kości bez ręcznych - z algorytmu |

Algorytm bazowy `roughness` jest słaby: na jego powierzchniach para kości strzałkowej w ogóle nie wychodzi, dlatego domyślnie tylko ręczne.

Jak działa:
1. Normalne powierzchni A są **odwracane**: dwa przełomy stykają się twarzą w twarz (wypukłość na A = wklęsłość na B), więc po odwróceniu A „wygląda jak” B i działa ta sama rejestracja co przy łączeniu skanów.
2. **RANSAC** na deskryptorach FPFH (zgrubnie) + **ICP** (dokładnie) → przekształcenie A → B. RANSAC w open3d jest losowy (wielowątkowy), dlatego zawsze przechodzi wszystkie 1 mln iteracji - przy 50 tys. dobre ustawienie wychodziło tylko w ~40% prób.
3. Ocena: **styk** = pole, na którym powierzchnie leżą bliżej niż `tolerance_mm` (0.5 mm) z przeciwnymi normalnymi; **kolizja** = pole, na którym całe kości wchodzą w siebie głębiej niż `collision_mm` (1 mm). **Wynik = styk − kolizja** [mm²].

Wszystkie ustawienia z opisami: `settings.json`, sekcja `matching`.

```powershell
python scripts/match_surfaces.py                      # wszystkie kości (~1-3 s na parę powierzchni)
python scripts/match_surfaces.py first_test 3         # tylko wybrane kości
python scripts/match_surfaces.py --source manual-auto # jednorazowo inny tryb
```

Strona `out/3d/matching.html`: najlepsze pary w 3D (kość A ustawiona na B, lista wybiera parę) + macierz wyników wszystkich par. Wszystkie pary z przekształceniami: `out/3d/matching.json`.

**Podgląd z suwakami: `python scripts/tune_matching.py`** (po `match_surfaces.py`). Lokalna strona w przeglądarce (jak narzędzie do oznaczania, działa bez internetu): lista par z `matching.json`, para w 3D i suwaki ustawień z sekcji `matching`. Po puszczeniu suwaka przeliczana jest **tylko pokazana para**:
- suwaki „Szukanie ustawienia” (odstęp punktów, FPFH, RANSAC, ICP) - dopasowanie od nowa (~2-3 s przy 1 mln iteracji × 2 próby),
- suwaki „Ocena” (styk, kolizja) - tylko nowa ocena obecnego ustawienia (ułamek sekundy).

Lista par pokazuje stary → nowy wynik, „Przelicz od nowa” powtarza losowy RANSAC, „Przywróć ustawienia” wraca do `matching.json`. Nic nie jest zapisywane - zmienione ustawienia strona wypisuje gotowe do wklejenia w `settings.json`; potem `match_surfaces.py` przelicza wszystkie pary. Port: `settings.json` → `matching` → `port` (8766).

Na naszych danych (24.11.2026, ręczne oznaczenia 4 kości): `modelowanie matematyczne / lowerjaw` ↔ `modelowanie matematyczne_(1) / upperjaw` 179 mm² i oba fragmenty kości strzałkowej (`first_test` lowerjaw pow. 2 ↔ upperjaw pow. 2) 118 mm², obie pary bez kolizji, RMSE ~0.17 mm, fragmenty układają się w jednej osi; najlepsza przypadkowa para ma 55. Im więcej kości ma ręczne oznaczenia, tym więcej par da się sprawdzić.

## Ustawienia

Wszystkie ustawienia są w **`settings.json`** (z komentarzem przy każdej wartości); `src/config.py` tylko je wczytuje. Brakujący klucz = wartość domyślna z kodu, literówka w nazwie = czytelny błąd z listą poprawnych nazw. Zmiany działają przy następnym uruchomieniu skryptu.

Składnia JSON: `true` / `false` / `null` małymi literami, tekst w `"..."`, liczby z kropką, bez przecinka po ostatnim elemencie. Komentarze `//` są dozwolone (VS Code traktuje plik jako JSON z komentarzami dzięki `.vscode/settings.json`).

| sekcja | najważniejsze klucze | do czego |
|---|---|---|
| `paths` | `data_in`, `data_out`, `contacts` | foldery: skany, wyniki (`out/3d`), powierzchnie styku |
| `scans` | `exclude`, `min_component_area_ratio`, `max_component_gap_mm` | które skany pominąć; kiedy oderwany kawałek to szum |
| `merging` | `min_overlap`, `tolerance_mm`, `min_support`, `fuse_method`, `groups` | kiedy skany to ta sama kość; jak skleić; ręczne grupy |
| `features` | `voxel_size`, `fpfh_radius_factor` | chmura punktów i FPFH do dopasowania |
| `contacts` | `method`, `rough_threshold`, `port`, `brush_mm`, ... | algorytm wykrywania i narzędzie do oznaczania |
| `matching` | `source`, `tolerance_mm`, `collision_mm`, `ransac_iterations`, ... | dopasowanie po powierzchniach styku |
| `visualization_3d` | `triangles_total`, `embed_plotly`, `open_browser` | wizualizacje HTML |

## Wyniki: `out/3d/` (generowane, można usunąć i policzyć od nowa)

```
out/3d/
    bones/<id kości>/
        mesh.ply            siatka kości (wyśrodkowana), z kolorami skanera
        source.npy          z którego skanu pochodzi każdy trójkąt
        cloud.npz           chmura punktów (points, normals, colors), co ~0.5 mm
        fpfh.npy            deskryptory FPFH chmury, (N, 33)
        meta.json           skany, ich przekształcenia, usunięty szum, statystyki
        scans/*.ply         każdy skan w układzie kości (*_noise.ply = usunięty szum)
    merging.json            nakładania wszystkich par skanów i przydział do kości
    matching.json           dopasowanie powierzchni styku: wszystkie pary, wyniki, przekształcenia
    summary.csv             tabela kości (separator ";", otwiera się w Excelu)
    bones.html, merging.html, matching.html
    cache/pairs.json        cache rejestracji par skanów
contacts/<id kości>.json         RĘCZNE oznaczenia powierzchni styku (w repozytorium!)
contacts/auto/<id kości>.json    wynik algorytmu wykrywania (generowany, nie ma w repo)
```

Kości w innym kodzie:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # dla pliku w scripts/ albo tests/

from bones import load_bones
from contacts import load_contacts, surface_stats

for bone in load_bones(features=True):              # bone.mesh, bone.cloud, bone.fpfh, bone.meta
    c = load_contacts(bone.id, bone.mesh)            # ręczne; None = nie oznaczona ("auto" = wynik algorytmu)
    if c:
        print(bone.label, surface_stats(bone.mesh, c.labels))   # pole, średnica, normalna, ...
```

## Struktura

```
scanned_bones/   surowe skany z podfolderami - nie ma w repo, patrz „Skany”
src/
    config.py            wczytuje settings.json
    data_parsing.py      wyszukiwanie skanów, wczytywanie, naprawa siatek
    merging.py           rejestracja par, grupowanie, usuwanie szumu, sklejanie
    bones.py             kość (siatka, chmura, FPFH, meta): zapis / odczyt
    pipeline.py          cały krok 1 (prepare_all)
    contacts.py          powierzchnie styku: format, statystyki, algorytmy, ocena
    matching_surfaces.py dopasowanie fragmentów po powierzchniach styku (RANSAC + ICP)
    contact_tool.py      serwer narzędzia do oznaczania
    web/                 strona narzędzia (HTML, CSS, JS - three.js)
    visualization_3d.py  wizualizacje plotly (HTML)
scripts/         skrypty uruchamiające (patrz „Jak używać”)
tests/           test_pipeline.py - test na sztucznych skanach
contacts/        ręczne oznaczenia powierzchni styku (+ auto/ - wyniki algorytmu, generowane)
settings.json    wszystkie ustawienia
out/             wyniki (generowane, nie ma w repo)
```

# Szczegółowe notatki podproblemów

## 1. Skanowanie

### Surowe dane ze skanów, prosto z pliku
- Wierzchołki: współrzędne (x, y, z) każdego punktu na powierzchni.
- Ściany: trójkąty, czyli które wierzchołki są ze sobą połączone.
- Normalne: kierunek „na zewnątrz” w każdym punkcie. Jeśli ich nie ma, można je policzyć.
- Kolor na wierzchołek: tylko jeśli skaner go zapisał.

### Co wiemy o naszych skanach (23.11.2026)
- Każdy skan to **otwarta, częściowa** powierzchnia fragmentu (skaner łapie tylko część), krawędzie trójkątów ~0.22 mm.
- 15 skanów (+ 1 occlusion) to **12 różnych kości**: `first_test/lowerjaw` = `modelowanie matematyczne_(6)` lowerjaw + upperjaw, `first_test/upperjaw` = `modelowanie matematyczne_(5)/upperjaw`. Pozostałe skany nie pokrywają się z niczym (nakładanie ≤ 0.20).
- Sloty Medita (upperjaw / lowerjaw) **nie oznaczają** konkretnej kości.
- `first_test/occlusionfirst` (skan zgryzu) pasuje do obu fragmentów kości strzałkowej naraz - to prawdopodobnie **oba fragmenty złożone razem**, czyli gotowa „prawda” do sprawdzania algorytmu dopasowania.

## 2. Ręczne dopasowywanie

### Ustalić czy skany zachowują skalę.

    Spróbować dopasować zeskanowane fragmenty ręcznie w programie.

## 5. Wstępna selekcja

### Jak wstępnie ustalić, które kości będą kandydatami na dopasowanie

    Można jako kryterium dopasowania wybrać największą długość dwóch punktów leżących na powierzchni 2d złamania (średnicę półosi wielkiej tak jakby).
    Jeśli dwa fragmenty mają ją podobną to jest szansa, że będą pasować.

    -> liczone już dla oznaczonych powierzchni: surface_stats() w src/contacts.py
       ("max_diameter_mm", "area_mm2", "extent_mm", "normal").

### Jak znaleźć front kości? Jak znaleźć powierzchnię złamania.

    Najpierw trzeba znaleźć oś kości i tak, żeby patrzeć prostopadle na przekrój poprzeczny.

    Pomysł: zeskanować dłuższy fragment poza końcówką i uznać linie wzdłuż kości jako równoległe i zrobić prostopadłą linie łączącą te dwie równoległe.

    Widokiem z boku uwzględniamy takie rzeczy jak:
    - kąt nachylenia do podstawy (rysujemy linie od najnizszego punktu do najwyzszego)
    -

## Sposób dopasowania fragmentów kości za pomocą pełnych skanów

    Pomysły: mamy deskryptory FPFH

# OSTEOFORM · start dla ekipy

Windows i Python 3.12 lub 3.13. Otwórz PowerShell i wykonaj poniższe polecenia.

```powershell
git clone -b osteoform-integracja --single-branch https://github.com/SamuelPiotrowski/SMM-rekonstrukcja.git
cd SMM-rekonstrukcja
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\integracja_szpont\requirements.txt
.\integracja_szpont\uruchom-demo.cmd
```

Przeglądarka otworzy się automatycznie. Zostaw konsolę otwartą. Kolejne uruchomienia zaczynaj od `uruchom-demo.cmd`.

**Nowy przypadek** pozwala wgrać własne pliki STL, OBJ lub PLY. Żeby obejrzeć gotowe testy, **Szczegółowe dopasowanie** i **Podgląd kości**, rozpakuj otrzymany plik `OSTEOFORM-dane.zip` do głównego folderu repozytorium. W archiwum są już właściwe ścieżki.

---

# PROJEKT KOSTNY

# PROBLEM

Mając fragmenty kości, należy je zeskanować skanerem świetlnym, który tworzy obrazy 3D. Następnie spróbować dopasować fragmenty do siebie tak, aby pasowały. Potem z dopasowanych fragmentów, można dalej analizować w jaki sposób doszło do uszkodzenia tych kości.

## Podproblemy:

1. Skanowanie kości, przede wszystkim ich powierzchni złamania. Rozwiązanie problemów z niedokładnym skanem. Skany mają formaty: (obj. ply, stl).
2. Przerobienie mesh na chmury puntków oraz punkty z deskryptorem fpfh
4. Oczyszczenie i przygotowanie danych (w tym łączenie wielu skanów tej samej kości, usunięcie dziwnych fragmentów daleko)
5. Wyznaczenie powierzchni styku na kościach
6. Wyznaczenie macierzy obrotu i translacji, które najlepiej dopasowują do siebie pary (obecnie RANSAC i ICP)
7. Policzenie wyniku dopasowania

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

### Skany i zestawy danych

Skanów **nie ma w repo** (są duże). Leżą w `scanned_bones/`, podzielone na **zestawy danych** - każdy zestaw to osobny folder:

```
scanned_bones/
    first_scans/                 <- zestaw (pierwsze skany z 23.11.2026)
        bone_01/                 <- kość = folder; nazwa folderu = nazwa kości
            23.11.2026-modelowanie matematyczne-lowerjaw.ply   (+ .obj, .stl - wystarczy jeden)
        bone_02/
            ...
        whole_scan/              <- kilka skanów w jednym folderze: kości whole_scan-lowerjaw, whole_scan-upperjaw
            23.11.2026-kośc strzałkowa 3-lowerjaw.ply
            23.11.2026-kośc strzałkowa 3-upperjaw.ply
            23.11.2026-kośc strzałkowa 3-occlusionfirst.ply   (pomijany - patrz niżej)
        whole_lower_rescan_1/ ...
    inny_zestaw/
        ...
```

Ten sam podział jest w wynikach i oznaczeniach: `out/3d/<zestaw>/` i `contacts/<zestaw>/`, więc zestawy się nie mieszają.

- **domyślny zestaw**: `settings.json` → `"dataset"` (teraz `"first_scans"`),
- **jednorazowo inny**: każdy skrypt przyjmuje `--dataset <nazwa>` (krócej `-d`), np. `python scripts/show_bones.py -d inny_zestaw`,
- **lista zestawów** (ile skanów, kości i ręcznych oznaczeń): `python scripts/datasets.py`,
- **nowy zestaw** = nowy folder w `scanned_bones/` ze skanami, potem `python scripts/prepare_data.py -d <nazwa>`,
- w kodzie: `import config; config.use_dataset("nazwa")` - potem wszystkie funkcje (`prepare_all`, `load_bones`, `load_contacts`, ...) działają na tym zestawie.

Nazwy kości:

- **jeden skan w folderze** → nazwa kości = nazwa folderu (`bone_01`),
- **kilka skanów w folderze** → folder + część z nazwy pliku Medita (`whole_scan-lowerjaw`),
- **plik wprost w folderze zestawu** → nazwa pliku (np. syntetyczne fragmenty `frag_01.stl`).

**Zmiana nazwy kości**: zmień nazwę folderu (albo pliku) i uruchom `python scripts/prepare_data.py`. Ręczne oznaczenia powierzchni styku same przejdą na nową nazwę (rozpoznawane po siatce kości albo po pliku skanu), konsola to wypisze. To samo przy zmianie nazwy całego zestawu (zmień też `"dataset"` w `settings.json`) - oznaczenia są wtedy kopiowane ze starego `contacts/<stara nazwa>/`. Dopasowanie (`match_surfaces.py`) trzeba po zmianie nazw policzyć od nowa.

Pozostałe zasady:

- skany są szukane we **wszystkich podfolderach** zestawu; ten sam plik w różnych folderach to różne skany,
- ten sam skan w kilku formatach jest wczytywany raz (kolejność: PLY → OBJ → STL) - zmieniając nazwę **pliku**, zmień ją we wszystkich formatach (dlatego wygodniej zmieniać nazwę folderu),
- skany zgryzu Medita (`occlusion...`) są **pomijane** - to oba fragmenty złożone razem, więc skleiłyby dwie kości w jedną (zmiana: `scans.exclude` w `settings.json`),
- **nie trzeba** ręcznie wskazywać, które skany to ta sama kość - program sam to wykrywa (gdy `merging.enabled` = `true`).

Masz jeszcze stary układ (foldery sesji Medita wprost w `scanned_bones/`, np. `first_test/`)? Przenieś go raz: `python scripts/migrate_first_scans.py` - przenosi skany, wyniki i oznaczenia do zestawu `first_scans` i nadaje kościom nowe nazwy (`first_test` → `whole_scan`, ponowne skany jego fragmentów → `whole_*_rescan*`, reszta → `bone_01` ... `bone_10`).

Inny folder główny: `settings.json`, `paths.data_in` (zestawy są jego podfolderami).

## Jak używać

Wszystko uruchamiamy z głównego folderu projektu, przy aktywnym `.venv`. Skrypty są w `scripts/` (same wywołania), funkcje w `src/`. Każdy skrypt działa na zestawie z `settings.json` (`"dataset"`), a z `--dataset <nazwa>` - na innym.

| krok | polecenie | co robi | wynik |
|---|---|---|---|
| 1 | `python scripts/prepare_data.py` | skany → kości: łączy skany tej samej kości, usuwa szum | `out/3d/<zestaw>/bones/` |
| 2 | `python scripts/show_merging.py` | pokazuje, jak połączyły się skany | `out/3d/<zestaw>/merging.html` |
| 3 | `python scripts/mark_contacts.py` | **ręczne oznaczanie powierzchni styku** w przeglądarce | `contacts/<zestaw>/*.json` |
| 4 | `python scripts/show_bones.py` | wszystkie kości + oznaczone powierzchnie styku | `out/3d/<zestaw>/bones.html` |
| 5 | `python scripts/match_surfaces.py` | **dopasowanie fragmentów po powierzchniach styku** (RANSAC + ICP) - tylko liczy i zapisuje | `out/3d/<zestaw>/methods/<metoda>/matching.json` |
| 6 | `python scripts/tune_matching.py` | **podgląd dopasowania** (jak krok 4 dla kroku 1): suwaki przeliczają pokazaną parę; `--html` = plik do wysłania | `out/3d/<zestaw>/matching.html` (z `--html`) |
| 7 | `python scripts/presentation.py` | **prezentacja jednej metody**: kroki 1, 4 i 6 w jednym przenośnym pliku, wszystko z jednej metody i jednych ustawień; czego brakuje - „przelicz” | `raporty/prezentacje/<zestaw>/prezentacja_<metoda>.html` |
| – | `python scripts/detect_contacts.py` | algorytm wykrywania powierzchni styku + porównanie z ręcznymi | `contacts/<zestaw>/auto/*.json` |
| – | `python scripts/build_atlas.py` | **baza kompletnych kości (atlas)**: indeksuje modele z `atlas/` | `atlas_index/` |
| – | `python scripts/identify_bones.py` | **z której kości z bazy pochodzi każdy fragment** (prawdopodobieństwa) + kora / przełom | `out/3d/<zestaw>/atlas.html`, `atlas.json`, `atlas.csv` |
| – | `python scripts/compare_reference.py` | **porównanie z kością wzorcową** (skan całej kości z zestawu): prawdziwe położenia fragmentów, prawdziwe pary przełomów, ocena `matching.json` | `out/3d/<zestaw>/reference.json` |
| – | `python scripts/datasets.py` | lista zestawów danych i ich stan | – |
| – | `python tests/test_pipeline.py` | test całości na sztucznych skanach (~20 s) | – |
| – | `python tests/test_atlas.py` | test atlasu na sztucznych kościach (~1 min) | – |

Wizualizacje (krok 2, 4 i 6 z `--html`) to pojedyncze pliki HTML - otwierają się same, działają bez internetu i można je komuś wysłać.

### 1. Przygotowanie danych: `scripts/prepare_data.py`

1. **Szuka skanów** w `scanned_bones/<zestaw>/` i podfolderach (patrz wyżej).
2. **Wczytuje i naprawia siatki**: scala zduplikowane wierzchołki, usuwa zdegenerowane i zduplikowane trójkąty oraz krawędzie nierozmaitościowe. Siatka **nie jest wygładzana** - szczegóły przełomu są potrzebne do dopasowania.
3. **Porównuje każdą parę skanów**: ustawia je na sobie (zgrubnie deskryptorami FPFH + RANSAC, dokładnie ICP) i mierzy **nakładanie** = jaka część mniejszego skanu leży na drugim (bliżej niż 0.15 a.u., normalne w tę samą stronę). Na naszych skanach: różne fragmenty 0.05–0.20, ten sam fragment 0.49–0.73.
   Pierwsze uruchomienie trwa kilka minut (15 skanów = 105 par ≈ 2.5 min), wyniki par są w cache (`out/3d/<zestaw>/cache/`), więc kolejne trwają ~20 s. Nowe skany liczą tylko nowe pary.
4. **Grupuje skany w kości**: nakładanie ≥ `min_overlap` (0.3) = ta sama kość. Działa też „przez pośrednika”: dwa skany różnych końców kości łączą się przez trzeci, który obejmuje oba.
5. **Usuwa szum**: oderwany kawałek skanu zostaje tylko, jeśli jest głównym kawałkiem, albo potwierdzają go inne skany tej kości, albo jest duży (≥ 5% głównego) i leży blisko (≤ 3 a.u.). Np. w `whole_scan/upperjaw` dwa kawałki leżące 120 a.u. od kości (połowicznie zeskanowany inny fragment) znikają.
6. **Skleja powierzchnię**: cały skan referencyjny (największy) + z pozostałych tylko to, czego w nim brakuje (oryginalna geometria i kolory, bez wygładzania). Alternatywa: `"fuse_method": "poisson"` - jedna gładka powierzchnia.
7. **Liczy chmurę punktów i FPFH** (do dopasowania fragmentów) i zapisuje wszystko do `out/3d/<zestaw>/bones/<id kości>/`.

Konsola pokazuje, co się stało, np.:

```
  13. whole_scan / lowerjaw: 3 skan(y) [whole_scan / lowerjaw (ref.), whole_lower_rescan_1, whole_lower_rescan_2] -> 3019 a.u.²
  14. whole_scan / upperjaw: 2 skan(y) [...] -> 2380 a.u.², usunięty szum: 274 a.u.²
      whole_scan / upperjaw: usunięto kawałek 157 a.u.² - SZUM: 120 a.u. od głównego, brak potwierdzenia
```

### 2. Sprawdzenie łączenia: `scripts/show_merging.py`

Strona z dwiema częściami:
- **3D**: skany każdej kości nałożone na siebie, każdy w innym kolorze, usunięty szum na czerwono. Drobne „paski” dwóch kolorów = skany dobrze leżą na sobie. Kość wybiera się z listy w lewym dolnym rogu.
- **Macierz nakładania** wszystkich par skanów; ramki = skany połączone w jedną kość.

Jeśli coś jest źle połączone (albo nie połączone), w `settings.json` zmień `merging.min_overlap` albo podaj grupy ręcznie:

```json
"groups": {"strzalkowa_A": ["whole_scan-lowerjaw", "whole_lower_rescan"]}
```

(nazwy to fragmenty nazw skanów wypisywanych przez `prepare_data.py`; skany spoza listy dalej są grupowane automatycznie).

### 3. Oznaczanie powierzchni styku: `scripts/mark_contacts.py`

Otwiera w przeglądarce narzędzie do malowania powierzchni styku (przełomów) na kościach. Lista kości jest po lewej (zielona kropka = oznaczona ręcznie, fioletowa = ustawienie automatyczne), narzędzia po prawej.

- **Ustawienie automatyczne**: kość bez ręcznych oznaczeń otwiera się od razu z wynikiem algorytmu (sekcja `"contacts"` w `settings.json`; z `contacts/<zestaw>/auto/`, a gdy brak albo nieaktualny - liczony). Strona pisze wtedy „ustawienie automatyczne”. Popraw je narzędziami poniżej i zapisz (`Ctrl+S`) - dopiero wtedy staje się ręcznym oznaczeniem w `contacts/<zestaw>/<id kości>.json`. Nieruszone i niezapisane zostaje automatyczne (nie trafia do ręcznych).
- **Różdżka** (`W`): kliknij przełom - zaznacza obszar aż do ostrej krawędzi. Za dużo? `Ctrl+Z` i zmniejsz tolerancję.
- **Pędzel** (`B`) / **Gumka** (`E`): poprawki lewym przyciskiem, rozmiar `[` `]`.
- Kilka przełomów na jednej kości = **Nowa powierzchnia** (`N`), każda ma swój kolor i nazwę (dwuklik zmienia nazwę).
- **Propozycja algorytmu**: zastępuje obecne oznaczenia wynikiem algorytmu (np. żeby zacząć od nowa).
- Obracanie **prawym** przyciskiem, przesuwanie środkowym, zoom kółkiem. `H` - pomoc.
- `Ctrl+S` zapisuje do `contacts/<zestaw>/<id kości>.json`; przy zmianie kości zmiany zapisują się same.

Folder `contacts/` **jest w repozytorium** - to ręczna praca, warto ją commitować i dzielić się nią (poza podfolderem `contacts/<zestaw>/auto/` z wynikami algorytmu, który jest generowany). Jeśli siatka kości się zmieni (np. po zmianie ustawień i ponownym `prepare_data.py`), oznaczenia są automatycznie przenoszone na nową siatkę po położeniu punktów.

Za pierwszym razem narzędzie potrzebuje internetu (pobiera bibliotekę three.js do `out/3d/.cache/`), potem działa offline. Zakończenie: `Ctrl+C` w konsoli.

### 4. Wizualizacja kości: `scripts/show_bones.py`

Wszystkie kości ułożone płasko „na stole”, ponumerowane, z oznaczonymi powierzchniami styku. Lista „Wszystkie kości” przybliża wybraną kość, przyciski: widok 3D / z góry / z boku, powierzchnie styku pokaż / ukryj, legenda ukrywa kości, najechanie myszą pokazuje szczegóły (skany, pole, wymiary, pola i średnice powierzchni styku).

**Linijka** (przycisk „📏 Linijka” na dole albo klawisz `L`): dwa kliknięcia w kość = odległość w a.u. (odcinek z podpisem na scenie i lista pomiarów). Obracanie myszą nie stawia punktu, `Esc` anuluje zaczęty odcinek, „Cofnij” / „Wyczyść” usuwa pomiary. Punkt przyciąga się do najbliższego wierzchołka pokazanej (uproszczonej) siatki - dla dokładniejszego pomiaru pokaż mniej kości (`show_bones.py 3`) albo zwiększ `triangles_total`. Linijka jest też w `merging.html`, `matching.html` i na stronie z suwakami (`tune_matching.py`).

```powershell
python scripts/show_bones.py                    # kolory skanera
python scripts/show_bones.py --colors fragment  # każda kość w innym kolorze
python scripts/show_bones.py --colors fpfh      # kolor wg lokalnego kształtu (FPFH), porównywalny między kośćmi
python scripts/show_bones.py --source auto      # powierzchnie z algorytmu zamiast ręcznych (krócej: --auto)
python scripts/show_bones.py --source manual-auto  # ręczne, a dla kości bez ręcznych - z algorytmu
python scripts/show_bones.py 3 strzalkowa       # tylko kość nr 3 i kości z "strzalkowa" w nazwie (pełna rozdzielczość)
python scripts/show_bones.py --compare          # porównanie: ręczne oznaczenia vs algorytm
```

**Porównanie ręczne vs algorytm** (`--compare`, wynik `out/3d/<zestaw>/bones_compare.html`): na każdej kości zielone = zgodne, pomarańczowe = tylko algorytm (fałszywy alarm), niebieskie = tylko ręczne (algorytm przeoczył), szare = wynik algorytmu na kości bez ręcznych oznaczeń. Najechanie na kość pokazuje jej IoU / precyzję / czułość (jak w `detect_contacts.py`), ramka na dole - średnią i objaśnienie miar:
- **IoU** - pokrycie wykrytego z ręcznym (część wspólna / suma obu),
- **precyzja** - ile z tego, co wykrył algorytm, to naprawdę styk (niska = fałszywe alarmy),
- **czułość** - ile prawdziwego styku algorytm wykrył (niska = przeoczenia).

Wszystkie od 0 do 1 (1.00 = idealnie), liczone na polu powierzchni, tylko dla kości z tego uruchomienia.

Które powierzchnie styku są pokazywane domyślnie, ustawia `settings.json`, sekcja `"visualization_3d"`, klucz `"source"`: `"manual"` / `"auto"` / `"manual-auto"` (tak samo jak `"source"` w sekcji `"matching"`). `--source` / `--auto` zmienia to jednorazowo. W trybach `auto` i `manual-auto` brakujący albo nieaktualny wynik algorytmu jest liczony i zapisywany w `contacts/<zestaw>/auto/`.

### Algorytm wykrywania powierzchni styku (na przyszłość): `scripts/detect_contacts.py`

Wszystko jest przygotowane pod algorytm, który zastąpi ręczne oznaczanie:
- algorytm to funkcja `(kość, parametry) -> etykiety wierzchołków` zarejestrowana dekoratorem `@detector("nazwa")` w `src/contacts.py`,
- wynik ma **ten sam format** co ręczne oznaczenia i leży obok nich: `contacts/<zestaw>/auto/<id kości>.json` (pole `source` = `auto:<nazwa>`, pole `detector` = algorytm i jego ustawienia). Działa z wizualizacją (`show_bones.py --auto` albo `"source"` w sekcji `"visualization_3d"`), narzędziem („Propozycja algorytmu”) i dopasowaniem (`match_surfaces.py --source auto`),
- który algorytm jest używany: `settings.json`, sekcja `contacts`, klucz `method` (jednorazowo: `--method`),
- `detect_contacts.py` liczy zawsze od nowa; dopasowanie bierze zapisany wynik i liczy sam tylko wtedy, gdy go brak albo zmienił się algorytm, jego ustawienia lub siatka kości,
- `detect_contacts.py` od razu **porównuje wynik z ręcznymi oznaczeniami** (IoU, precyzja, czułość) - ręczne oznaczenia są „prawdą” do oceny algorytmu.

Algorytmy:
- `roughness` - bazowy: przełom = szorstka powierzchnia, kora = gładka. Punkt wyjścia, nie rozwiązanie.
- `features` - **szorstkość + rozrzut normalnych + gęstość siatki naraz, bez żadnych oznaczeń**. Każda cecha jest porównywana z typową wartością na tej samej kości (mediana = kora, bo kory jest najwięcej): przełom jest bardziej szorstki, ma większy rozrzut normalnych i gęstszą siatkę. Wynik = średnia tych odchyleń, potem wygładzenie, próg i podział na obszary. Nie zależy od skali skanera. Na naszych 6 ręcznie oznaczonych kościach IoU 0.47 (`roughness` 0.43); opis cech: [raporty/przelom_vs_kora.md](raporty/przelom_vs_kora.md). Ustawienia: `feature_keys`, `feature_threshold`, `feature_smooth_au`, `feature_open_au` w sekcji `contacts`.
- `atlas` - **`features` + kontekst anatomiczny z bazy kompletnych kości**: rozpoznaje, z której kości pochodzi fragment, i na tej podstawie rozstrzyga, co jest korą (leży na powierzchni kości), a co przełomem (wnętrze kości, też kanał szpikowy). Dzięki temu naturalnie szorstka kora (nasady, przyczepy mięśni) nie jest brana za przełom, a gładki przełom nie jest przeoczony. Opis niżej: „Atlas anatomiczny”. Na naszych 6 ręcznie oznaczonych kościach, z bazą z jednej całej kości strzałkowej (`grzesiu/Test cała kość strzałkowa`): IoU 0.51 (`features` 0.47), na fragmentach kości strzałkowej do +0.10; na sztucznych kościach (`tests/test_atlas.py`) IoU 0.80 (`features` 0.36).

**Po każdym algorytmie: przybliżenie przełomu** (`contacts` → `surface_fit`, `split_ends`, `fit_tolerance_au`). Na trzonach kości długich algorytmy oznaczały jako przełom cały koniec fragmentu razem z rękawem kory (na `grzesiu` 2-12 przekrojów kości zamiast ~1-2, do 100% długości fragmentu), a czasem jedną powierzchnią oba końce. Dlatego:
1. **rozdzielenie końców**: na wydłużonym fragmencie (co najmniej `long_min_elongation` = 2.5 razy dłuższy niż szerszy) powierzchnia, która sięga obu połówek, dzielona jest na dwie - po jednej na każdy koniec;
2. **przybliżenie**: każdą powierzchnię przybliża płaszczyzna (`"plane"`) albo funkcja kwadratowa nad płaszczyzną (`"quadric"`), dopasowana odpornie (punkty daleko od niej mało ważą); na kości długiej liczą się głównie punkty, których normalna patrzy wzdłuż osi - twarz przełomu, nie kora z boku;
3. **odcięcie**: zostają punkty bliżej przybliżenia niż `fit_tolerance_au` (3 a.u.) - rękaw kory wzdłuż kości odpada.

Opis przybliżenia każdej powierzchni (normalna, kąt do osi, odchylenie od płaszczyzny) jest w pliku wyniku (`"info"` → `"fits"`). Odchylenie służy w dopasowaniu do odróżnienia **płaskiego, spiłowanego cięcia** (~0.1 a.u.) od nierównego przełomu, a samo przybliżenie - jako drugi, testowy tryb dopasowania (`"mode": "plane"`).

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

### Atlas anatomiczny: `scripts/build_atlas.py`, `scripts/identify_bones.py`, algorytm `atlas`

Baza modeli 3D **kompletnych** kości ludzkich daje algorytmowi kontekst: fragment to wycinek jakiejś kości, jego **kora** leży na zewnętrznej powierzchni tej kości, a **przełom** nie - odsłonięte wnętrze jest pod powierzchnią zdrowej kości. Kość długa ma w środku **kanał szpikowy**: na skanie przełomu to nie dziura, tylko wklęsły dół (skaner widzi ściany kanału) - on też jest wnętrzem kości, więc należy do przełomu.

**Baza.** Pliki modeli (`.ply` / `.obj` / `.stl`, jeden plik = jedna kość, nazwa pliku = nazwa kości, podfolder = typ kości) wrzuca się do `atlas/` (`settings.json` → `paths` → `atlas`, folder nie jest w repo). `python scripts/build_atlas.py` zamienia każdy model w mały wpis bazy w `atlas_index/` (**~1 MB na kość** zamiast dziesiątek MB - prawdziwy skan całej kości strzałkowej: 99 MB → 0.6 MB):
- tylko **zewnętrzna** powierzchnia (punkt, z którego żaden promień nie ucieka na zewnątrz, leży w środku - np. ściana kanału szpikowego w modelu z tomografii),
- jej punkty (skwantyzowane), własności do sita (długość, pole, histogramy krzywizn kawałków kości) i mała siatka podglądu.

Wykrywanie korzysta **tylko z bazy** - oryginalne pliki modeli można po zindeksowaniu usunąć albo trzymać gdzie indziej (`atlas_index/` jest mały, można go wrzucić do repo albo wysłać). Kolejne uruchomienia czytają gotowy indeks; liczone są tylko nowe i zmienione pliki (suma kontrolna), plik o zmienionej nazwie jest rozpoznawany po zawartości, model bez pliku zostaje w bazie (usunięcie: `--prune`). Nowe pliki w `atlas/` są doliczane same przy wykrywaniu. Deskryptory FPFH nie są zapisywane - liczą się w locie, tylko dla kości, które przejdą sito, i tylko w potrzebnej skali.

**Jak fragment jest porównywany z bazą** (`src/atlas.py`):
1. **Analiza lokalna** fragmentu: wynik `features` + **wgłębienia** (kanał szpikowy - z wnętrza wklęsłego dołu promienie trafiają w jego ściany; pas przy brzegu skanu jest pomijany, bo postrzępione brzegi też zasłaniają promienie).
2. **Sito własności - bez dopasowywania**: fragment nie może być dłuższy (ani mieć więcej kory) niż kość przy największej skali, a krzywizny jego kory (indeks kształtu - nie zależy od skali; krzywizna ogólna - skala tylko przesuwa histogram) muszą występować na jakimś kawałku tej kości. Kości, które nie pasują, są **wykluczane przed RANSAC**; przy okazji wychodzą prawdopodobne skale.
3. **Dopasowanie** pozostałych (najwyżej `max_models`): RANSAC na FPFH + ICP ze **skalą** (skan może mieć inną skalę i proporcje niż model), także dla **odbicia lustrzanego** (kość z drugiej strony ciała). Przełom i kanał szpikowy nie psują ustawienia (leżą w środku modelu, więc mają wagę ~0).
4. **Ocena**: jaka część kory fragmentu leży na powierzchni modelu, minus część fragmentu wystająca na zewnątrz → **rozkład prawdopodobieństwa** kości z bazy z możliwością **„żadna z bazy”** (gdy nic nie pasuje dość dobrze = niska pewność).
5. **Kora vs przełom** z najbardziej prawdopodobnych kości (ważone prawdopodobieństwem): na powierzchni modelu = kora, pod powierzchnią (wnętrze, kanał) = przełom, na zewnątrz = brak zdania (decyduje analiza lokalna). Połączone z analizą lokalną, dalej jak `features` (wygładzenie, próg, obszary). Przy niskiej pewności (`min_confidence`) - sama analiza lokalna, czyli praktycznie wynik `features`.

```powershell
python scripts/build_atlas.py                    # zindeksuj nowe modele z atlas/, pokaż bazę
python scripts/identify_bones.py                 # ranking kości dla każdego fragmentu + atlas.html
python scripts/detect_contacts.py --method atlas # powierzchnie styku z atlasu (contacts/<zestaw>/auto/, z rankingiem)
python scripts/identify_bones.py --atlas D:/modele --atlas-index D:/baza   # inne foldery bazy
```

Konsola, np.:
```
   4. bone_04: 1. Test cała kość strzałkowa-upperjaw (odbicie lustrzane) 99% (skala ×1.28, dopasowanie 0.90) · ... · żadna z bazy 0%
      sito: dopasowywane 1 z 1 modeli
```

`atlas.html`: mapa prawdopodobieństw (fragmenty × kości z bazy) i każdy fragment ustawiony na najbardziej prawdopodobnym modelu (szary, półprzezroczysty), z kolorami do wyboru: **decyzja** (pomarańczowe = szorstka kora, którą sama analiza lokalna wzięłaby za przełom; fioletowe = przełom, którego sama analiza lokalna by nie zaznaczyła), dowód z atlasu, analiza lokalna, razem. Ranking jest też w pliku wyniku algorytmu (`"info"` → `"atlas"`), w podpowiedzi `show_bones.py --auto` i w `atlas.json` / `atlas.csv` (do Excela).

**Wspólna skala zestawu.** Wszystkie fragmenty jednej osoby mają tę samą skalę względem modelu, ale dopasowane osobno potrafią wyjść w różnych skalach (na `grzesiu` ×0.68–1.06). Fragment w złej skali leży na modelu w złym miejscu. `identify_bones.py` wypisuje skalę zestawu dla każdego modelu (mediana ważona pewnością i długością fragmentu) i fragmenty z inną skalą. `python scripts/identify_bones.py --shared-scale` zapisuje ją w `scanned_bones/<zestaw>/settings.json` (`"atlas"` → `"model_scales"`) i dopasowuje wszystko od nowa w zakresie ±`shared_scale_tolerance` (10%). Potem tak samo działa algorytm `atlas` i dopasowanie fragmentów. Bardzo długie fragmenty (np. cała kość) mają rzadsze punkty i większą tolerancję RANSAC: kość innej osoby różni się kształtem tym bardziej, im dalej od siebie są punkty, więc przy stałej tolerancji 1.5 a.u. RANSAC nie znajdował ustawienia.

Najważniejsze ustawienia (`settings.json`, sekcja `atlas`): `reference_scale` (ile a.u. skanu to 1 jednostka modelu - model w mm, skan ~mm → 1), `scale_range` (różnice osobnicze), `mirror`, `max_unexplained` (siła sita), `tolerance_au` (jak blisko powierzchni modelu musi leżeć kora), `min_fit` / `temperature` (kiedy „żadna z bazy”), `atlas_weight`, `min_confidence`, `cavity_weight` (kanał szpikowy).

Generator syntetycznych kości ma kanał szpikowy: `settings.json` → `synthetic` → `marrow` (np. 0.5 = kanał o połowie promienia kości; na przełomie wklęsły dół głęboki na `marrow_depth_au`).

### 5. Dopasowanie po powierzchniach styku: `scripts/match_surfaces.py`

Pierwsza wersja dopasowania fragmentów (podproblem 6). Bierze **tylko punkty powierzchni styku**, każdą powierzchnię osobno, i dopasowuje każdą do każdej powierzchni **innej** kości.

Skąd powierzchnie (`settings.json` → `matching` → `source`, jednorazowo `--source`):

| tryb | powierzchnie |
|---|---|
| `manual` (domyślny) | tylko ręczne (`contacts/<zestaw>/<id>.json`); kości bez nich są pomijane |
| `auto` | tylko z algorytmu (`contacts/<zestaw>/auto/<id>.json`; gdy brak albo nieaktualne - liczone i zapisywane) |
| `manual-auto` | ręczne, a dla kości bez ręcznych - z algorytmu |

Algorytm bazowy `roughness` jest słaby: na jego powierzchniach para kości strzałkowej w ogóle nie wychodzi, dlatego domyślnie tylko ręczne.

Jak działa:
1. Normalne powierzchni A są **odwracane**: dwa przełomy stykają się twarzą w twarz (wypukłość na A = wklęsłość na B), więc po odwróceniu A „wygląda jak” B i działa ta sama rejestracja co przy łączeniu skanów.
2. **RANSAC** na deskryptorach FPFH (zgrubnie) + **ICP** (dokładnie) → przekształcenie A → B. RANSAC w open3d jest losowy (wielowątkowy), dlatego zawsze przechodzi wszystkie 1 mln iteracji - przy 50 tys. dobre ustawienie wychodziło tylko w ~40% prób. Większość złych opcji jest odrzucana od razu, bo są sprawdzane proste warunki. 
3. Ocena: **styk** = pole, na którym powierzchnie leżą bliżej niż `tolerance_au` (0.5 a.u.) z przeciwnymi normalnymi; **kolizja** = pole, na którym całe kości wchodzą w siebie głębiej niż `collision_au` (1 a.u.). **Wynik** (`score_mode`): domyślnie `"dice"` = (styk − kolizja) / średnie pole obu przełomów, w % - jaka część **obu** przełomów do siebie przylega. Część przełomu, która nic nie dotyka, obniża wynik, a wielkość fragmentów się skraca. Dawny wynik `"area"` = styk − kolizja [a.u.²] rósł z wielkością przełomów: duży przełom miał więcej pola do przypadkowego styku. Sprawdzone na gotowych wynikach: na `synthetic` (znana prawda) bez zmian, 15 najlepszych par = 15 prawdziwych; na `first_scans` dwie prawdziwe pary zostają na górze, a ich przewaga nad następną rośnie z ×1.2 (118 wobec 99 a.u.²) do ×1.6 (40% wobec 25%). Ustawienie przełomów (T) się nie zmienia, bo dla jednej pary obie oceny różnią się tylko stałym dzielnikiem; zmienia się kolejność par.

**Kości długie, tryb `"gap"`** (`long_mode`, włączony). Na `grzesiu_test` porównanie z całą kością pokazało, że nawet w prawdziwym złożeniu przełomy sąsiednich kawałków nie przylegają ściśle: są 2-3 a.u. od siebie (mediana), a w tolerancji 0.5 a.u. leży 1-3% punktów. Ścisły styk dawał więc prawdziwemu złożeniu wynik ujemny, a złym ustawieniom przypadkowy styk. W trybie `"gap"` para wydłużonych fragmentów ma tylko dwa stopnie swobody - obrót wokół wspólnej osi i przesunięcie wzdłuż niej (osie przy przełomach w jednej linii, A przedłuża B koniec w koniec) - a wynik to:
- **bliskość przełomów**: część punktów przełomu A bliżej przełomu B niż `gap_au` (2.5 a.u.) i nie głębiej w B niż `collision_au`, i odwrotnie,
- **ciągłość kory**: kora A tuż przy przełomie (1-6 a.u. od niego) ma leżeć na przedłużeniu kory B przy jej przełomie - blisko jej płaszczyzny stycznej (`continuity_au`), z normalnymi w tę samą stronę. To ustawia też obrót wokół osi (grzebienie trzonu),
- minus kolizja całych kości.

Wynik = 0.5 x bliskość + 0.5 x ciągłość - kolizja. Szukanie: siatka obrót co 15° x przesunięcie od -30 do +15 a.u. co 2 a.u. (ujemne = końce zachodzą na siebie, jak przy złamaniu spiralnym), dokładniej wokół 3 najlepszych, na koniec prostowanie. Na `grzesiu_test` z ręcznymi przełomami prawdziwa para test 1 - test 2 przeszła z miejsca poza ósemką na 1., a wszystkie 3 pary proponowanego złożenia są prawdziwe (wcześniej 0 z 2).

**Porównanie z kością wzorcową** (`python scripts/compare_reference.py`): gdy zestaw ma skan całej kości, z której pochodzą fragmenty (domyślnie kość z listy `exclude` sekcji `contacts`), każdy fragment jest na niej kładziony (RANSAC + ICP, bez skali; gdy pasuje słabo - przeszukanie całego trzonu). Skrypt wypisuje, jaka część kory fragmentu leży na wzorcu, prawdziwe pary przełomów i ocenę `matching.json`: czy para jest prawdziwa i jaki ma błąd położenia i obrotu. Uwaga: kawałek trzonu bez cech charakterystycznych można na wzorcu przesuwać wzdłuż osi bez straty dopasowania (na `grzesiu_test`: `test 2` o ±6 a.u.), więc jego „prawdziwe” położenie wzdłuż osi jest niepewne o kilka a.u.

**Kości długie: osie w jednej linii** (`axis_check`, włączone). Sam styk przełomów nie wystarcza. Kawałki trzonu często „pasują” lokalnie także pod kątem albo koniec do boku, a takie pary wygrywały: na `grzesiu` osie najlepszych par rozchodziły się o 23–88°. Gdy oba fragmenty są wydłużone (co najmniej `axis_min_elongation` = 2.5 razy dłuższe niż szersze, oś = kierunek, w którym fragment jest najdłuższy):
- **ocena**: kąt między osiami po złożeniu i rozsunięcie osi w miejscu styku (w promieniach kości; koniec do boku ≈ 2). Do `axis_max_deg` (20°) i `axis_max_offset` (1 promień) bez kary, przy dwukrotności styk już się nie liczy: wynik = styk × waga osi − kolizja;
- **starty wzdłuż osi**: oś A na osi B, A przedłuża B (koniec z przełomem do końca z przełomem), przełom na przełomie, obrót wokół osi co 30° (`axis_starts` = 12). Z każdego startu ICP; RANSAC liczy się dalej obok.

Na `grzesiu` wszystkie 10 najlepszych par ma teraz osie w zakresie 7–23° (wcześniej 23–88°), w tym samym czasie (~3 s na parę).

**Płaskie cięcia** (`flat_max_rms_au`): koniec spiłowany albo przecięty piłą jest prawie płaski, a przełom nierówny. Dwa płaskie cięcia zawsze do siebie pasują (płaszczyzna na płaszczyźnie), więc wygrywały z prawdziwymi przełomami. Każda powierzchnia jest przybliżana płaszczyzną. Gdy jej punkty odchylają się od niej średnio mniej niż `flat_max_rms_au`, powierzchnia nie jest dopasowywana. `match_surfaces.py` wypisuje odchylenie przy każdej powierzchni - z tego dobiera się próg.

**Tryb testowy `"plane"`** (`"mode"`): zamiast RANSAC + ICP na punktach dopasowywane są same przybliżenia płaszczyzną - płaszczyzna A na płaszczyźnie B (normalne przeciwne, środki razem), obrót wokół normalnej co 360° / `plane_steps`. Styk = wspólna część obrysów obu powierzchni rzutowanych na płaszczyznę. Ocena, osie, kolizja i premia jak w zwykłym trybie `"points"`. Szybki (bez RANSAC), do porównania z trybem `"points"`.

**Proponowane złożenie**: każdy przełom może pasować tylko do jednego partnera. Od najlepszej pary w dół brane są te, których przełomy (i para kości) nie zostały jeszcze użyte (`"chosen": true` w `matching.json`), i tylko z wynikiem co najmniej `assembly_min_score` (10%). Słabsze pary nie są wciskane na siłę: przełom zostaje bez pary, a konsola wypisuje takie przełomy osobno - tam może brakować kawałka pomiędzy (albo przełom jest źle wykryty). Konsola wypisuje je osobno, a strona oznacza „w proponowanym złożeniu”.

**Z atlasem anatomicznym** (`settings.json` → `matching` → `use_atlas`, jednorazowo `--use-atlas` / `--no-use-atlas`). Atlas mówi, z której kości z bazy pochodzi każdy fragment (z prawdopodobieństwami) i jak na niej leży. Korzysta z tego samego cache co algorytm `atlas` i `identify_bones.py`, więc zwykle nic nie liczy od nowa. Dopasowanie zawęża się na trzy sposoby:
- **pomijanie par**: gdy fragmenty na pewno pochodzą z **różnych** kości (`atlas_skip_different`, 95%), para nie jest liczona. „Ta sama kość” = ten sam model i ta sama strona ciała (odbicie lustrzane albo nie). Modele z jednego podfolderu bazy (typ kości) liczą się razem, bez strony. „Żadna z bazy” nie wyklucza pary;
- **start z atlasu**: gdy A i B leżą na tym samym modelu, ich położenie względem siebie wynika z ustawień na modelu (x_B = T_B · T_A⁻¹ · x_A, sam obrót i przesunięcie). Z tego startu ICP dociska przełomy. Gdy wyjdzie dobry styk (obie powierzchnie pokryte co najmniej w `start_enough_cover`, 35%), **RANSAC jest pomijany** i para liczy się w ułamku sekundy. Inaczej RANSAC liczy się jak zwykle i wygrywa lepsza próba;
- **premia**: styk liczy się do `atlas_boost` (25%) mocniej, gdy to ta sama kość: styk × (1 + 0.25 × P(ta sama kość)). Para bez styku nic nie zyskuje.

Konsola pokazuje najbardziej prawdopodobną kość każdego fragmentu, liczbę pominiętych par i ile par obeszło się bez RANSAC. W `matching.html` pominięte pary mają w macierzy znak ×.

Na `grzesiu` (8 fragmentów jednej strzałki, baza: `FIBULA_LEFT` + `HUMERUS_LEFT`) atlas w dopasowaniu niewiele daje. Żadna para nie jest pomijana, bo wszystkie fragmenty to strzałka, a strona ciała często wychodzi ~50/50. Start z atlasu nie trafia: na trzonie strzałki (prawie walec) atlas nie umie powiedzieć, w którym miejscu długości leży fragment ani który koniec jest górny. Starty są 15–245 a.u. od dobrego złożenia, często odwrócone o ~180°, także po ustaleniu wspólnej skali (`--shared-scale`, ×0.85 - krótkie fragmenty lądowały wtedy na granicy zakresu, a `GAB krótka 3` wypadła z bazy, dlatego dla `grzesiu` wspólna skala jest wyłączona). Atlas pewnie mówi tylko „to strzałka”. Złożenie trzonów załatwiają osie (wyżej). Pomijanie par przyda się przy fragmentach różnych kości, a start z atlasu - przy fragmentach z charakterystycznym kształtem (nasady, końce kości).

Wszystkie ustawienia z opisami: `settings.json`, sekcja `matching`.

```powershell
python scripts/match_surfaces.py                      # wszystkie kości (~1-3 s na parę powierzchni)
python scripts/match_surfaces.py whole_scan 3         # tylko wybrane kości
python scripts/match_surfaces.py --source manual-auto # jednorazowo inny tryb
python scripts/match_surfaces.py --no-use-atlas       # jednorazowo bez atlasu (porównanie)
```

Wynik: `out/3d/<zestaw>/methods/<metoda>/matching.json` - **osobno dla każdej metody** (`manual`, `roughness`, `features`, `atlas`; po pełnym przebiegu, wszystkie kości) - oraz ostatni przebieg (każdy, także wybrane kości i `manual-auto`) w `out/3d/<zestaw>/matching.json`. Wszystkie pary z wynikami, przekształceniami i użytymi ustawieniami. Skrypt **tylko liczy i zapisuje**; oglądanie to krok 6.

### 6. Podgląd dopasowania: `scripts/tune_matching.py`

Pokazuje zapisany wynik metody (tak jak `show_bones.py` pokazuje wynik `prepare_data.py`): domyślnie metody z `settings.json`, `--method` - innej. Po przełączeniu metody **nic nie trzeba przeliczać**, jeśli ta metoda była już policzona - powierzchnie styku są brane z zapisu metody (dokładnie te, które dopasowywano). Ostrzega w konsoli i na stronie, gdy wynik jest nieaktualny: inne ustawienia w `settings.json` niż te, z którymi liczono, zmienione kości, oznaczenia albo atlas - wtedy podaje komendę `match_surfaces.py --method <metoda>`. To samo wybieranie wyniku ma `compare_reference.py` (`--method`, `--last`).

```powershell
python scripts/tune_matching.py                  # strona z suwakami (lokalny serwer), metoda z settings.json
python scripts/tune_matching.py --method manual  # wynik innej metody
python scripts/tune_matching.py --last           # ostatni przebieg (out/3d/<zestaw>/matching.json)
python scripts/tune_matching.py --html           # statyczna strona out/3d/<zestaw>/matching_<metoda>.html
```

`--html`: najlepsze pary w 3D (kość A ustawiona na B, lista wybiera parę) + macierz wyników wszystkich par; jeden plik, można go wysłać.

**Suwaki** (domyślnie): lokalna strona w przeglądarce (jak narzędzie do oznaczania, działa bez internetu): lista par z `matching.json`, para w 3D i suwaki ustawień z sekcji `matching`. Po puszczeniu suwaka przeliczana jest **tylko pokazana para**:
- suwaki „Szukanie ustawienia” (odstęp punktów, FPFH, RANSAC, ICP) - dopasowanie od nowa (~2-3 s przy 1 mln iteracji × 2 próby),
- suwaki „Ocena” (styk, kolizja) - tylko nowa ocena obecnego ustawienia (ułamek sekundy).

W widoku 3D jest **linijka** (prawy dolny róg albo klawisz `L`, jak w `show_bones.py`) - np. do zmierzenia szczeliny między kośćmi; pomiary znikają po przeliczeniu albo zmianie pary.

Lista par pokazuje stary → nowy wynik, „Przelicz od nowa” powtarza losowy RANSAC, „Przywróć ustawienia” wraca do `matching.json`. Nic nie jest zapisywane - zmienione ustawienia strona wypisuje gotowe do wklejenia w `settings.json`; potem `match_surfaces.py` przelicza wszystkie pary. Port: `settings.json` → `matching` → `port` (8766).

Na naszych danych (24.11.2026, ręczne oznaczenia 4 kości): `bone_01` ↔ `bone_03` 179 a.u.² i oba fragmenty kości strzałkowej (`whole_scan` lowerjaw pow. 2 ↔ upperjaw pow. 2) 118 a.u.², obie pary bez kolizji, RMSE ~0.17 a.u., fragmenty układają się w jednej osi; najlepsza przypadkowa para ma 55. Im więcej kości ma ręczne oznaczenia, tym więcej par da się sprawdzić.

### 7. Prezentacja jednej metody: `scripts/presentation.py`

Jeden plik HTML do pokazania wyników: dane (łączenie skanów), kości z powierzchniami styku (+ porównanie z ręcznymi oznaczeniami) i dopasowanie (najlepsze pary, macierz, proponowane złożenie). **Wszystko z jednej metody, jednego zestawu i obecnych ustawień** - metoda to skąd powierzchnie styku: `manual` (ręczne) albo algorytm `roughness` / `features` / `atlas`.

```powershell
python scripts/match_surfaces.py --method atlas    # policz metodę (powierzchnie + dopasowanie, zapis metody)
python scripts/presentation.py --method atlas      # prezentacja -> raporty/prezentacje/<zestaw>/prezentacja_atlas.html
python scripts/presentation.py --all               # plik dla każdej metody (z linkami między nimi)
python scripts/presentation.py --method atlas --steps powierzchnie porownanie   # tylko powierzchnie + porównanie z ręcznymi
python scripts/presentation.py --all --no-matching # wszystko poza dopasowaniem końcowym
```

- **Kroki** (`settings.json` → `presentation` → `steps` albo jednorazowo `--steps`): `dane`, `powierzchnie`, `porownanie` (algorytm vs ręczne: IoU / precyzja / czułość), `dopasowanie`. Bez `dopasowanie` nie trzeba `match_surfaces.py` - wystarczą powierzchnie styku (ręczne albo `detect_contacts.py --method <algorytm>`), a plik dostaje w nazwie kroki, np. `prezentacja_atlas_powierzchnie-porownanie.html` (pełna prezentacja się nie nadpisuje).

- **Nic nie przelicza** - czyta gotowe wyniki. Część policzona z innymi ustawieniami niż obecne (inne kości, inne `matching` / `contacts` / atlas, zmienione ręczne oznaczenia) albo niepoliczona jest zastąpiona ramką **PRZELICZ** z powodem i komendą - stary wynik nie jest pokazywany, żeby nic się nie mieszało.
- Każda metoda ma **osobny zapis** `out/3d/<zestaw>/methods/<metoda>/` (robi go `match_surfaces.py` po pełnym przebiegu w trybie `manual` albo `auto`): `matching.json` + dokładnie te powierzchnie styku, które dopasowywano. Wyniki kilku metod nie nadpisują się. Tryb `manual-auto` (mieszanka) nie jest metodą prezentacji.
- Plik jest **przenośny**: plotly.js w środku, bez serwera i internetu - można go wysłać albo otworzyć na innym komputerze.

## Ustawienia

Wszystkie ustawienia są w **`settings.json`** (z komentarzem przy każdej wartości); `src/config.py` tylko je wczytuje. Brakujący klucz = wartość domyślna z kodu, literówka w nazwie = czytelny błąd z listą poprawnych nazw. Zmiany działają przy następnym uruchomieniu skryptu.

Składnia JSON: `true` / `false` / `null` małymi literami, tekst w `"..."`, liczby z kropką, bez przecinka po ostatnim elemencie. Komentarze `//` są dozwolone (VS Code traktuje plik jako JSON z komentarzami dzięki `.vscode/settings.json`).

| sekcja | najważniejsze klucze | do czego |
|---|---|---|
| `dataset` | (pojedyncza wartość) | domyślny zestaw danych (podfolder w każdym z folderów `paths`) |
| `paths` | `data_in`, `data_out`, `contacts`, `atlas`, `atlas_index` | foldery główne: skany, wyniki (`out/3d`), powierzchnie styku, pliki modeli i baza atlasu |
| `scans` | `exclude`, `min_component_area_ratio`, `max_component_gap_au` | które skany pominąć; kiedy oderwany kawałek to szum |
| `merging` | `min_overlap`, `tolerance_au`, `min_support`, `fuse_method`, `groups` | kiedy skany to ta sama kość; jak skleić; ręczne grupy |
| `features` | `voxel_size`, `fpfh_radius_factor` | chmura punktów i FPFH do dopasowania |
| `contacts` | `method`, `rough_threshold`, `port`, `brush_au`, ... | algorytm wykrywania i narzędzie do oznaczania |
| `atlas` | `reference_scale`, `scale_range`, `max_unexplained`, `tolerance_au`, `min_fit`, ... | atlas anatomiczny: rozpoznawanie kości i algorytm `atlas` |
| `matching` | `source`, `tolerance_au`, `collision_au`, `ransac_iterations`, `use_atlas`, `atlas_boost`, ... | dopasowanie po powierzchniach styku (z atlasem: pomijanie par, start, premia) |
| `visualization_3d` | `triangles_total`, `embed_plotly`, `open_browser` | wizualizacje HTML |

## Wyniki: `out/3d/<zestaw>/` (generowane, można usunąć i policzyć od nowa)

```
out/3d/<zestaw>/
    bones/<id kości>/
        mesh.ply            siatka kości (wyśrodkowana), z kolorami skanera
        source.npy          z którego skanu pochodzi każdy trójkąt
        cloud.npz           chmura punktów (points, normals, colors), co ~0.5 a.u.
        fpfh.npy            deskryptory FPFH chmury, (N, 33)
        meta.json           skany (z sumą kontrolną pliku), ich przekształcenia, usunięty szum, statystyki
        scans/*.ply         każdy skan w układzie kości (*_noise.ply = usunięty szum)
    merging.json            nakładania wszystkich par skanów i przydział do kości
    matching.json           ostatni przebieg dopasowania (dowolna metoda): wszystkie pary, wyniki, przekształcenia
    methods/<metoda>/       zapis metody (tune_matching, compare_reference, presentation): matching.json,
                            contacts/<id>.json, snapshot.json
    atlas.json, atlas.csv   rozpoznawanie kości z bazy (identify_bones.py): rankingi, skale, ustawienia na modelach
    cache/atlas/            cache dopasowań fragmentów do bazy
    summary.csv             tabela kości (separator ";", otwiera się w Excelu)
    bones.html, merging.html, matching.html, atlas.html
    cache/pairs.json        cache rejestracji par skanów
raporty/prezentacje/<zestaw>/prezentacja_<metoda>[_<kroki>].html   prezentacje (presentation.py, "paths" -> "presentations")
contacts/<zestaw>/<id kości>.json         RĘCZNE oznaczenia powierzchni styku (w repozytorium!)
contacts/<zestaw>/auto/<id kości>.json    wynik algorytmu wykrywania (generowany, nie ma w repo)
out/3d/.cache/                            biblioteki stron (three.js) - wspólne dla zestawów
```

Kości w innym kodzie:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # dla pliku w scripts/ albo tests/

import config
from bones import load_bones
from contacts import load_contacts, scan_frames, surface_stats

config.use_dataset("first_scans")                  # opcjonalnie: inny zestaw niż w settings.json
for bone in load_bones(features=True):              # bone.mesh, bone.cloud, bone.fpfh, bone.meta
    c = load_contacts(bone.id, bone.mesh, frames=scan_frames(bone))   # ręczne; None = nie oznaczona
    if c:
        print(bone.label, surface_stats(bone.mesh, c.labels))   # pole, średnica, normalna, ...
```

## Struktura

```
scanned_bones/   surowe skany, folder na każdy zestaw - nie ma w repo, patrz „Skany i zestawy danych”
src/
    config.py            wczytuje settings.json, wybór zestawu danych (use_dataset)
    data_parsing.py      wyszukiwanie skanów, wczytywanie, naprawa siatek
    merging.py           rejestracja par, grupowanie, usuwanie szumu, sklejanie
    bones.py             kość (siatka, chmura, FPFH, meta): zapis / odczyt
    pipeline.py          cały krok 1 (prepare_all)
    contacts.py          powierzchnie styku: format, statystyki, algorytmy, ocena
    atlas.py             atlas anatomiczny: baza kompletnych kości, sito, dopasowanie ze skalą, algorytm "atlas"
    matching_surfaces.py dopasowanie fragmentów po powierzchniach styku (RANSAC + ICP)
    contact_tool.py      serwer narzędzia do oznaczania
    web/                 strona narzędzia (HTML, CSS, JS - three.js)
    visualization_3d.py  wizualizacje plotly (HTML)
scripts/         skrypty uruchamiające (patrz „Jak używać”)
tests/           test_pipeline.py - test na sztucznych skanach, test_atlas.py - test atlasu
atlas/           pliki modeli kompletnych kości (nie ma w repo) - potrzebne tylko do zindeksowania
atlas_index/     zindeksowana baza atlasu (~1 MB na kość) - generowana przez build_atlas.py
contacts/        ręczne oznaczenia powierzchni styku, folder na każdy zestaw (+ auto/ - wyniki algorytmu, generowane)
settings.json    wszystkie ustawienia
out/             wyniki (generowane, nie ma w repo)
```

# Szczegółowe notatki podproblemów

## 1. Skanowanie

### Problem skanu

Skany, które wykonywaliśmy skanerem miały błędy. Nie dało się dobrze zeskanować gładkiej powierzchni kości. W podejściu 3D zakładamy, że 

### Surowe dane ze skanów, prosto z pliku
- Wierzchołki: współrzędne (x, y, z) każdego punktu na powierzchni.
- Ściany: trójkąty, czyli które wierzchołki są ze sobą połączone.
- Normalne: kierunek „na zewnątrz” w każdym punkcie. Jeśli ich nie ma, można je policzyć.
- Kolor na wierzchołek.

### Sposób zapisu skanów

Na razie skany są zapisane surowo i nie da się do końca wywnisokować, które to są kości. W przyszłości przy wielu skanach tego samego fragmentu trzeba podać w nazwie taki sam fragment (nazwa lub kod fragmentu). Przy podejściu ze skalowaniem jedynie końcówek fragmentów, (z powodu niedokładnego skanowania płaskich boków kości przez skaner) należy 

### Co wiemy o naszych skanach (23.11.2026)
- Każdy skan to **otwarta, częściowa** powierzchnia fragmentu (skaner łapie tylko część), krawędzie trójkątów ~0.22 a.u.
- 15 skanów (+ 1 occlusion) to **12 różnych kości**: `whole_scan-lowerjaw` = `whole_lower_rescan_1` + `whole_lower_rescan_2`, `whole_scan-upperjaw` = `whole_upper_rescan` (dawniej `first_test` i sesje `modelowanie matematyczne_(5)`, `_(6)`). Pozostałe skany nie pokrywają się z niczym (nakładanie ≤ 0.20).
- Sloty Medita (upperjaw / lowerjaw) **nie oznaczają** konkretnej kości.
- `whole_scan/...occlusionfirst` to **oba fragmenty złożone razem**, czyli gotowa kość do sprawdzania algorytmu dopasowania.

## 2. Przerobienie mesh na chmury puntków oraz punkty z deskryptorem fpfh

Tu 

## 6. Wyznaczenie macierzy obrotu i translacji, które najlepiej dopasowują do siebie pary (obecnie RANSAC i ICP)

Mamy kości i ich powierzchnie styku. Wszystkie powierzchnie styku łączymy w pary każda z każdą. Żeby sprawdzić, które pary pasuja najpierw musimy jak najlepiej je dopasować do siebie i mając jak najlepsze dopasowanie ocenić czy jest dobre.

Można wyznaczyć co najmniej dwa podejścia:
1. **roszerzoną wersję algorytmu niedzieli**, która aproksymuje punkty na krawędzi za pomocą krzywej i dopasowuje krzywe do siebie patrząc na ich korelację, żeby zawęzić kandydatów a potem dopasowanie iteracyjne
2. **Dopasowywanie za pomocą RANSAC i ICP** tę metodę wprowadziłam

** Przygotowanie danych**

Liczenie trzech rodzajów chmur dla punktów o różnej precyzji

| indeks | co                                | do czego              |
|:------:|-----------------------------------|-----------------------|
| `[0]`  | gęsta chmura (punkty co `sample_au`) | ostatni etap ICP      |
| `[1]`  | średnia (co 2 × `sample_au`)      | pierwsze etapy ICP    |
| `[2]`  | zgrubna (co `coarse_voxel_au`)    | RANSAC                |
| `[3]`  | FPFH zgrubnej                     | RANSAC                |

Dodatkowo dla powierzchni A robimy odwrócenie normalnych, aby zgadzały się z parowanym kawałkiem B. (FPFH użyte w algo zależy od normalnych).

**RANSAC**
Dla każdego punktu z A znajduje punkty B o najardziej podobnym FPFH. Jest to tylko jeden punkt przez cały algorytm. W wersji z mutual_filter=True prawdza czy dla punktu z B, także A jest najlepszym dopasowaniem, jesli nie porzuca parę.

W iteracji:
- Losuje trójkę par (3 punkty jednoznacznie wyznaczą obrót i przesunięcie w 3D)
- Wylicza przekształcenie T (obrót + przesunięcie)
- Odrzuca złe przekształcenia za pomocą szybkich do sprawdzenia warunków opierających się o punkty i T
- Gdy T nie odrzucono, to liczy dopasowanie czyli ile odległości punktów przekształconcyh T(a) leży dostatecznie blisko B

Jest to metoda niedokładna służąca do dopasowania wstępnego

**ICP**

ALgorytm, który znajduje lokalne minimum odległości punktów.

- Każdy punkt A przesuwa się T i szuka najbliższego punktu z B
- Szuka się małej poprawki do T, która minimalizuje dystans punktu `T(a)` do płaszczyzny stycznej w b.

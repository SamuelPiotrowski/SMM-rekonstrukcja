# Przełom vs kora: po czym odróżnić powierzchnię styku

*Raport z 24.11.2026 · dane: skany `scanned_bones/` (12 kości), 6 kości z ręcznie oznaczonymi powierzchniami styku · skrypt: `scripts/compare_surfaces.py`*

## W skrócie

- Przełom najlepiej odróżnia od kory **rozrzut normalnych** (AUC 0.86) i **szorstkość** (AUC 0.83). Działają na każdej kości, ale ich rozkłady częściowo się nakładają, więc sam próg na jednej cesze daje słabe obszary. Tak działa obecny algorytm `roughness`.
- **Siatka skanera jest na przełomie ok. 1.6 razy gęstsza** (AUC 0.19, czyli silnie, ale odwrotnie). To cecha skanera, a nie kości, ale jest bardzo wyraźna i nic nie kosztuje.
- **Kolor ze skanera (jasność, nasycenie) nie pomaga**: na różnych kościach wychodzi w przeciwne strony.
- **Wszystkie cechy razem** (regresja logistyczna sprawdzana na kości, której model nie widział) dają **AUC 0.90**, od 0.81 do 0.96 na pojedynczych kościach. To najlepszy kierunek dla nowego algorytmu wykrywania (`@detector`).

## Pytanie

Algorytm wykrywania powierzchni styku ma znaleźć przełom bez ręcznego oznaczania. Sprawdziliśmy, czym zaznaczone ręcznie przełomy różnią się od reszty kości (kory) i które cechy nadają się do tego, żeby je rozróżnić.

## Dane

6 kości z ręcznymi oznaczeniami (`contacts/*.json`). Porównywane są punkty przełomu z punktami kory tej samej kości.

| nr | kość | powierzchnie | przełom [a.u.²] | kora [a.u.²] |
|---|---|---|---|---|
| 1 | modelowanie matematyczne / lowerjaw | 1 | 284 | 536 |
| 2 | modelowanie matematyczne_(1) / upperjaw | 1 | 373 | 1011 |
| 3 | modelowanie matematyczne_(2) / lowerjaw | 1 | 128 | 595 |
| 4 | modelowanie matematyczne_(3) / upperjaw | 1 | 135 | 629 |
| 5 | first_test / lowerjaw (kość strzałkowa) | 2 | 350 | 2260 |
| 6 | first_test / upperjaw (kość strzałkowa) | 2 | 322 | 1207 |

Pominięte zostały dwa pasy:
- 1 a.u. przy otwartym brzegu skanu, bo brzeg jest postrzępiony, a to nie jest przełom;
- 0.5 a.u. po obu stronach granicy oznaczenia, bo ręczne oznaczenie nie jest idealnie dokładne, a cechy liczone w promieniu „widzą” obie strony granicy.

## Metoda

W każdym wierzchołku siatki kości liczymy cechy:

| cecha | co mierzy |
|---|---|
| szorstkość 0.5 / 1 / 2 a.u. | odchylenie powierzchni od dopasowanej płaszczyzny w danym promieniu, podzielone przez promień |
| rozrzut normalnych 0.5 / 1 / 2 a.u. | 1 − \|średnia normalnych w promieniu\|: 0 na płaskim, więcej na grudkach i krawędziach |
| wypukłość 2 a.u. | czy punkt wystaje ponad średnią okolicy (> 0), czy leży we wgłębieniu (< 0) |
| krawędź siatki | średnia długość krawędzi trójkątów przy wierzchołku [a.u.] |
| jasność, nasycenie koloru | kolor wierzchołka ze skanera |
| plamistość koloru | odchylenie standardowe jasności w promieniu 1 a.u. |
| normalna ∥ oś kości | \|cos\| kąta normalnej do długiej osi kości: 0 = ściana boczna, 1 = ściana poprzeczna |
| położenie na osi | odległość od środka kości wzdłuż osi: 0 = środek, 1 = koniec |

Cechy kształtu liczone są na równomiernej próbce powierzchni, więc nie zależą od gęstości siatki.

**Miara rozdzielności: AUC.** To prawdopodobieństwo, że losowy punkt przełomu ma większą wartość cechy niż losowy punkt kory (punkty ważone polem). Wartości czyta się tak:
- 0.5: cecha nic nie mówi;
- 1: przełom ma zawsze większą wartość;
- 0: przełom ma zawsze mniejszą wartość, czyli cecha też jest dobra, tylko działa odwrotnie.

**Połączenie cech.** Regresja logistyczna na wszystkich cechach, sprawdzana metodą „zostaw jedną kość”: model uczy się na 5 kościach i jest oceniany na szóstej. Dzięki temu wynik przybliża zachowanie na nowej, nieoznaczonej kości.

## Wyniki

### Ranking cech

AUC to średnia po 6 kościach. Mediana to mediana z median poszczególnych kości.

| cecha | AUC średnio | AUC min–max | mediana przełom | mediana kora | kierunek |
|---|---|---|---|---|---|
| **wszystkie cechy razem** | **0.90** | 0.81–0.96 | | | |
| rozrzut normalnych 2 a.u. | **0.86** | 0.79–0.92 | 0.29 | 0.028 | większy na przełomie |
| rozrzut normalnych 1 a.u. | 0.86 | 0.74–0.94 | 0.15 | 0.006 | większy |
| rozrzut normalnych 0.5 a.u. | 0.84 | 0.72–0.92 | 0.047 | 0.002 | większy |
| szorstkość 2 a.u. | 0.83 | 0.76–0.91 | 0.17 | 0.051 | większa |
| szorstkość 1 a.u. | 0.83 | 0.69–0.92 | 0.12 | 0.023 | większa |
| szorstkość 0.5 a.u. | 0.82 | 0.70–0.91 | 0.068 | 0.015 | większa |
| krawędź siatki | **0.19** | 0.09–0.31 | 0.20 a.u. | 0.34 a.u. | **mniejsza** na przełomie |
| wypukłość 2 a.u. | 0.71 | 0.61–0.82 | 0.22 | 0.066 | większa |
| plamistość koloru | 0.69 | 0.50–0.84 | 0.031 | 0.021 | większa (na kości 4 brak różnicy) |
| normalna ∥ oś kości | 0.69 | 0.54–0.89 | 0.29 | 0.11 | większa |
| położenie na osi | 0.64 | 0.44–0.88 | 0.44 | 0.33 | niespójnie |
| jasność koloru | 0.43 | 0.24–0.71 | 0.55 | 0.54 | niespójnie |
| nasycenie koloru | 0.46 | 0.31–0.62 | 0.28 | 0.30 | niespójnie |

„Niespójnie” znaczy, że na jednych kościach cecha jest większa na przełomie, a na innych mniejsza.

### AUC na każdej kości

| cecha | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| **wszystkie razem** | 0.87 | 0.87 | 0.93 | 0.81 | 0.96 | 0.94 |
| rozrzut normalnych 2 a.u. | 0.85 | 0.81 | 0.90 | 0.79 | 0.92 | 0.91 |
| rozrzut normalnych 1 a.u. | 0.81 | 0.83 | 0.92 | 0.74 | 0.92 | 0.94 |
| szorstkość 2 a.u. | 0.84 | 0.79 | 0.91 | 0.76 | 0.88 | 0.83 |
| szorstkość 1 a.u. | 0.80 | 0.82 | 0.92 | 0.69 | 0.91 | 0.85 |
| krawędź siatki | 0.27 | 0.18 | 0.09 | 0.31 | 0.13 | 0.16 |
| wypukłość 2 a.u. | 0.82 | 0.67 | 0.67 | 0.61 | 0.77 | 0.75 |
| plamistość koloru | 0.65 | 0.73 | 0.84 | 0.50 | 0.72 | 0.71 |
| normalna ∥ oś kości | 0.59 | 0.73 | 0.57 | 0.54 | 0.89 | 0.80 |
| położenie na osi | 0.49 | 0.66 | 0.53 | 0.44 | 0.88 | 0.81 |
| jasność koloru | 0.60 | 0.71 | 0.28 | 0.50 | 0.24 | 0.27 |
| nasycenie koloru | 0.31 | 0.33 | 0.62 | 0.40 | 0.59 | 0.55 |

Kości 5 i 6 mają po dwie powierzchnie. Każda z nich osobno daje wyniki bardzo podobne do całej kości (np. rozrzut normalnych 1 a.u. 0.91–0.96), więc oba końce zachowują się tak samo.

## Wnioski

1. **Kształt w małej skali to główny sygnał.** Przełom ma bardziej rozrzucone normalne i jest bardziej szorstki niż kora na wszystkich 6 kościach. Rozrzut normalnych jest trochę lepszy od szorstkości i najlepiej działa w promieniu 1–2 a.u. Najtrudniejsza jest kość 4 (AUC 0.70–0.79), której przełom jest gładszy.
2. **Gęstość siatki to mocna cecha „za darmo”.** Skaner (Medit) sam zagęszcza siatkę tam, gdzie widzi szczegóły, a więc na przełomie. Dwa zastrzeżenia:
   - to zależy od skanera i jego ustawień, więc na innych skanach może nie działać;
   - przy łączeniu kilku skanów jednej kości (`union`) krawędzie pochodzą z różnych skanów.
3. **Ułożenie względem osi pomaga tylko na długich fragmentach.** Na kościach strzałkowych (5, 6) przełomy są poprzeczne i na końcach kości (AUC 0.80–0.89). Na krótkich, krępych fragmentach (1–4) długa oś jest słabo określona i cecha prawie nic nie daje.
4. **Kolor nie jest wiarygodny.** Jasność i nasycenie wychodzą w różne strony na różnych kościach, prawdopodobnie przez oświetlenie i balans bieli w różnych sesjach skanowania. Plamistość koloru jest słaba, ale w miarę spójna.
5. **Łączenie cech wyraźnie pomaga:** AUC 0.90 wobec 0.86 dla najlepszej pojedynczej cechy, a najgorsza kość poprawia się z 0.79 do 0.81. Bez wygładzania obszarów zgodność punktów z ręcznym oznaczeniem (IoU) wynosi jednak tylko około 0.48. Pojedyncze punkty kory bywają szorstkie, a fragmenty przełomu gładkie.

## Co dalej (propozycja)

1. **Nowy algorytm w `src/contacts.py`:** wynik regresji logistycznej (rozrzut normalnych 1 i 2 a.u., szorstkość, krawędź siatki, wypukłość), potem wygładzenie obszarów (otwarcie / zamknięcie morfologiczne, jak w `roughness`) i odrzucenie małych obszarów. Ocena: `scripts/detect_contacts.py` (IoU względem ręcznych oznaczeń).
2. **Więcej ręcznych oznaczeń.** 6 kości to mało, a wyniki między kościami się różnią (np. kość 4). Każda nowa oznaczona kość poprawia i ocenę, i model.
3. **Ostrożnie z kolorem i gęstością siatki**, jeśli zmieni się skaner albo jego ustawienia: sprawdzić ponownie tym samym skryptem.
4. **Cechy granicy:** przełom jest zwykle otoczony ostrą krawędzią. Kąt między przełomem a korą na granicy nie był tu badany, a może pomóc w wyznaczaniu obszarów.

## Algorytm `features` (dodany po tej analizie)

Na podstawie tych wyników dodaliśmy algorytm wykrywania `features` (`src/contacts.py`). **Nie potrzebuje żadnych oznaczeń.** Działa tak:
1. Bierze trzy najlepsze cechy: rozrzut normalnych 2 a.u., szorstkość 1 a.u. i długość krawędzi siatki.
2. Każdą porównuje z **typową wartością na tej samej kości**. Kory jest najwięcej, więc mediana kości to poziom kory. Odchylenie od mediany jest liczone w jednostkach rozrzutu kości (1.4826 × MAD, czyli odporny odpowiednik odchylenia standardowego), więc wynik nie zależy od skali skanera.
3. Wynik to średnia odchyleń, przy czym krawędź siatki wchodzi ze znakiem minus, bo na przełomie siatka jest gęstsza.
4. Uśrednia wynik w promieniu 1 a.u., stosuje próg 0.15, otwarcie morfologiczne o 1 a.u. i odrzuca obszary mniejsze niż 15 a.u.².

Ręczne oznaczenia posłużyły tylko do oceny i wyboru progu. Wyniki na 6 oznaczonych kościach (IoU, precyzja i czułość liczone na całej kości):

| algorytm | AUC | IoU | precyzja | czułość | IoU na kościach 1–6 |
|---|---|---|---|---|---|
| `roughness` (bazowy) | | 0.43 | 0.63 | 0.61 | 0.32 0.43 0.48 0.32 0.51 0.54 |
| **`features`** | 0.86 | **0.47** | 0.58 | **0.76** | 0.49 0.47 0.53 0.44 0.40 0.49 |
| dla porównania: regresja uczona na oznaczeniach innych kości | 0.90 | 0.48 | | | |

AUC `features` (0.86) jest liczone z wyniku po uśrednieniu w promieniu 1 a.u., ale przed progiem i podziałem na obszary. AUC regresji (0.90) pochodzi z analizy wyżej, liczone bez wygładzania.

- **Bez oznaczeń prawie tak dobrze jak z oznaczeniami:** IoU 0.47 wobec 0.48 dla modelu uczonego.
- **Znajduje więcej przełomu niż `roughness`** (czułość 0.76 wobec 0.61), a najsłabsza kość ma IoU 0.40 zamiast 0.32.
- **Inne zestawy cech wypadały podobnie:** IoU 0.45–0.47, a sam próg mniej więcej taki sam (0.15).
- **Główna różnica to granica obszaru.** Większość nadmiarowo zaznaczonego pola leży do 1 a.u. od granicy ręcznego oznaczenia.
- **Do dopasowania fragmentów jeszcze się nie nadaje.** Wersja uczona, której obszary są podobne, na prawdziwych parach dawała wyniki bliskie 0 a.u.² albo kolizję; na ręcznych oznaczeniach te same pary mają 179 i 118 a.u.². Na razie to propozycja do poprawek w narzędziu („Propozycja algorytmu”).
- **Następny krok:** dociągnąć granicę obszaru do ostrej krawędzi przełom–kora, tak jak robi to różdżka w narzędziu.
- **Założenie:** większość skanu to kora. Na fragmencie, który jest prawie cały przełomem, algorytm zaznaczy za mało.

## Ograniczenia

- Ręczne oznaczenia traktujemy jako prawdę, ale same są niedokładne. Stąd pas 0.5 a.u. pominięty przy granicy.
- Niezaznaczona część kości to w większości kora, ale mogą się w niej trafić nieoznaczone fragmenty przełomu.
- Model łączący cechy był uczony na 5 kościach. Jego wagi (największe: rozrzut normalnych 2 a.u. +2.3, szorstkość 2 a.u. −1.1, krawędź siatki −0.65) są niepewne, bo podobne cechy dzielą się wagą.
- Wyniki dotyczą skanów z `scanned_bones/` z 23.11.2026. Po przejściu na nowe dane (`scanned_bones/3D-output`) kości mają nowe nazwy, więc stare ręczne oznaczenia do nich nie pasują. Analizę trzeba powtórzyć, gdy nowe kości zostaną oznaczone.

## Jak powtórzyć

```powershell
python scripts/prepare_data.py       # kości
python scripts/mark_contacts.py      # ręczne oznaczenia przełomów
python scripts/compare_surfaces.py   # ta analiza
```

Wynik:
- `out/3d/surface_features.html`: ranking cech, mapa AUC, rozkłady przełom / kora, wagi modelu oraz kość w 3D pokolorowana wybraną cechą;
- `out/3d/surface_features.csv`: AUC i percentyle każdej cechy dla każdej kości i powierzchni.

Ustawienia są w `settings.json`, w sekcji `surface_features`.

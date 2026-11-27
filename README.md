# OSTEOFORM · start 

Windows i Python 3.12 lub 3.13. Otwórz PowerShell i wykonaj poniższe polecenia.

```powershell
git clone -b osteoform-integracja --single-branch https://github.com/SamuelPiotrowski/SMM-rekonstrukcja.git
cd SMM-rekonstrukcja
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\integracja_szpont\requirements.txt
.\integracja_szpont\uruchom-demo.cmd
```

Przeglądarka otworzy się automatycznie. Zostaw konsolę otwartą. Kolejne uruchomienia zaczynaj od `uruchom-demo.cmd`.

**Nowy przypadek** pozwala wgrać pliki STL, OBJ lub PLY. Gotowe animacje w **Naszych skanach**, **Podgląd kości** i **Szczegółowe dopasowanie** działają od razu. Opcjonalna paczka `OSTEOFORM-dane.zip` dodaje pełną galerię, zdjęcia i dodatkowe pliki źródłowe.

Żeby przekazać własne wyniki z **Historii skanów**, wybierz **Pobierz złożenie** przy wyniku lub **Pobierz zapisane złożenia**. Wyślij ZIP osobie z ekipy. U siebie wybierze **Historia skanów → Wczytaj paczkę złożeń → Otwórz wynik**.

---

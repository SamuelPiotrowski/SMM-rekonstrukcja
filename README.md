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





# Szczegółowe notatki podproblemów

## 1. Skanowanie

### Surowe dane ze skanów, prosto z pliku
- Wierzchołki: współrzędne (x, y, z) każdego punktu na powierzchni.
- Ściany: trójkąty, czyli które wierzchołki są ze sobą połączone.
- Normalne: kierunek „na zewnątrz” w każdym punkcie. Jeśli ich nie ma, można je policzyć.
- Kolor na wierzchołek: tylko jeśli skaner go zapisał.

### Zdjęcia 2d z czterech stron





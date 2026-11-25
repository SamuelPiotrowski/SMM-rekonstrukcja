SMM — Vienna Brick

1. Rozpakuj ZIP i otwórz folder SMM_dane w VS Code.
2. Zainstaluj Python 3.12 oraz rozszerzenie Python (Microsoft).
3. Terminal > New Terminal (PowerShell).
4. Wpisz kolejno:
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install numpy scipy open3d matplotlib trimesh
   .\.venv\Scripts\python.exe pobierz_dane.py --only brick
   .\.venv\Scripts\python.exe test_fragmentow.py

Jeśli python nie działa, zainstaluj Python 3.12 z https://www.python.org/downloads/ i zaznacz Add Python to PATH, po czym zamknij i otwórz VS Code.
Pliki PLY: 02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY.
ZIP zawiera skrypty, nie skany. Pobieranie wymaga internetu.
Źródło: https://www.geometrie.tuwien.ac.at/ig/3dpuzzles.html

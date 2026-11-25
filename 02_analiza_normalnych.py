
import open3d as o3d
import numpy as np
from pathlib import Path

# Folder z fragmentami
folder = Path("02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY")

pliki = sorted(folder.glob("*.ply"))

fragmenty = []

kolory = [
    [0.95, 0.35, 0.35],
    [0.35, 0.65, 0.95],
    [0.35, 0.85, 0.55],
    [0.95, 0.75, 0.30],
    [0.70, 0.45, 0.95],
    [0.95, 0.55, 0.75]
]

for i, plik in enumerate(pliki):

    chmura = o3d.io.read_point_cloud(str(plik))

    # Pobranie normalnych do tablicy NumPy
    normalne = np.asarray(chmura.normals)

    # Dlugosc kazdego wektora normalnego
    dlugosci = np.linalg.norm(normalne, axis=1)

    # Sprawdzenie poprawnosci danych
    poprawne = np.isfinite(normalne).all(axis=1)

    liczba_zerowych = np.sum(dlugosci < 0.000001)
    liczba_niepoprawnych = np.sum(~poprawne)

    print("\nFragment:", plik.name)
    print("Liczba normalnych:", len(normalne))
    print("Najkrotsza normalna:", np.min(dlugosci))
    print("Najdluzsza normalna:", np.max(dlugosci))
    print("Srednia dlugosc:", np.mean(dlugosci))
    print("Normalne zerowe:", liczba_zerowych)
    print("Normalne niepoprawne:", liczba_niepoprawnych)

    # Zmniejszenie liczby punktow tylko do wizualizacji
    krok = max(1, len(chmura.points) // 4000)
    probka = chmura.uniform_down_sample(krok)

    probka.paint_uniform_color(kolory[i % len(kolory)])

    fragmenty.append(probka)

# Wyswietlenie fragmentow wraz z normalnymi
o3d.visualization.draw_geometries(
    fragmenty,
    window_name="Analiza normalnych",
    point_show_normal=True,
    width=1200,
    height=800
)

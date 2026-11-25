
import open3d as o3d
from pathlib import Path

# Folder z fragmentami
folder = Path("02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY")

# Pobranie wszystkich plikow PLY
pliki = sorted(folder.glob("*.ply"))

print("Liczba znalezionych fragmentow:", len(pliki))

fragmenty = []

kolory = [
    [0.95, 0.35, 0.35],
    [0.35, 0.65, 0.95],
    [0.35, 0.85, 0.55],
    [0.95, 0.75, 0.30],
    [0.70, 0.45, 0.95],
    [0.95, 0.55, 0.75]
]

# Wczytywanie i sprawdzanie fragmentow
for i, plik in enumerate(pliki):

    chmura = o3d.io.read_point_cloud(str(plik))

    print("\nFragment:", plik.name)
    print("Liczba punktow:", len(chmura.points))
    print("Czy posiada normalne:", chmura.has_normals())

    if chmura.is_empty():
        print("Blad: pusty fragment")
        continue

    # Wyznaczenie wymiarow fragmentu
    minimum = chmura.get_min_bound()
    maksimum = chmura.get_max_bound()

    wymiary = maksimum - minimum

    print("Wymiary X, Y, Z:", wymiary)

    # Nadanie koloru
    chmura.paint_uniform_color(kolory[i % len(kolory)])

    fragmenty.append(chmura)

# Wyswietlenie wszystkich fragmentow
if fragmenty:

    o3d.visualization.draw_geometries(
        fragmenty,
        window_name="Fragmenty 3D - kontrola danych",
        width=1200,
        height=800
    )

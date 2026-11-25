
import open3d as o3d
import numpy as np

# Wczytanie odszumionej chmury punktow
chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_odszumiony.ply"
)

print("Liczba punktow:", len(chmura.points))

# Kopia chmury do wyszukiwania plaszczyzn
pozostale_punkty = chmura

wyniki = []

# Szukamy dwoch najwiekszych plaszczyzn
for numer in range(2):

    plaszczyzna, indeksy = pozostale_punkty.segment_plane(
        distance_threshold=0.02,
        ransac_n=3,
        num_iterations=1000
    )

    if len(indeksy) == 0:
        print("Nie znaleziono kolejnej plaszczyzny.")
        break

    print("\nPLASZCZYZNA", numer + 1)
    print("Liczba punktow:", len(indeksy))
    print("Wspolczynniki plaszczyzny:", plaszczyzna)

    wyniki.append(
        pozostale_punkty.select_by_index(indeksy)
    )

    # Usuniecie znalezionej plaszczyzny z kopii
    pozostale_punkty = pozostale_punkty.select_by_index(
        indeksy,
        invert=True
    )

# Kolorowanie wynikow
pozostale_punkty.paint_uniform_color([0.7, 0.7, 0.7])

kolory = [
    [1.0, 0.1, 0.1],
    [0.1, 0.2, 1.0]
]

elementy = [pozostale_punkty]

for i in range(len(wyniki)):
    wyniki[i].paint_uniform_color(kolory[i])
    elementy.append(wyniki[i])

# Wizualizacja
o3d.visualization.draw_geometries(
    elementy,
    window_name="Wykrywanie plaskich powierzchni",
    width=1200,
    height=800
)

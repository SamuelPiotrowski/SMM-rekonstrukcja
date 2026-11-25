
import open3d as o3d
import numpy as np

# Wczytanie pelnej chmury
pelna = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_odszumiony.ply"
)

# Wczytanie roboczego boku
bok = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_bok_roboczy.ply"
)

punkty = np.asarray(pelna.points)
punkty_boku = np.asarray(bok.points)

# Drzewo najblizszych sasiadow
drzewo = o3d.geometry.KDTreeFlann(bok)

# Sprawdzenie, ktore punkty pelnej chmury
# naleza do obecnego boku
maska_boku = np.zeros(len(punkty), dtype=bool)

for i, punkt in enumerate(punkty):
    liczba, indeksy, odleglosci = drzewo.search_knn_vector_3d(
        punkt, 1
    )

    if liczba > 0 and odleglosci[0] < 1e-10:
        maska_boku[i] = True

# Kolorowanie
kolory = np.zeros((len(punkty), 3))

# Obecny bok - zielony
kolory[maska_boku] = [0.1, 0.8, 0.2]

# Wszystkie pozostale punkty - czerwone
kolory[~maska_boku] = [0.9, 0.1, 0.1]

pelna.colors = o3d.utility.Vector3dVector(kolory)

print("Pelna chmura:", len(punkty))
print("Obecny bok:", np.sum(maska_boku))
print("Pozostale punkty:", np.sum(~maska_boku))

o3d.visualization.draw_geometries(
    [pelna],
    window_name="Obecny bok i brakujace fragmenty",
    width=1200,
    height=800
)

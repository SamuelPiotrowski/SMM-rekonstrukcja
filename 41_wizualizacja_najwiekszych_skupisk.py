
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import colorsys

# 1. Wczytanie modelu
chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_krawedzie_probne.ply"
)

punkty = np.asarray(chmura.points)
kolory = np.asarray(chmura.colors)

# 2. Wybranie czerwonych punktow
maska = (
    (kolory[:, 0] > 0.9) &
    (kolory[:, 1] < 0.1) &
    (kolory[:, 2] < 0.1)
)

czerwone = punkty[maska]

# 3. Ponowne grupowanie tym samym parametrem
drzewo = cKDTree(czerwone)
odleglosci, _ = drzewo.query(czerwone, k=11)
eps = np.median(odleglosci[:, -1]) * 1.5

tymczasowa = o3d.geometry.PointCloud()
tymczasowa.points = o3d.utility.Vector3dVector(czerwone)

etykiety = np.array(
    tymczasowa.cluster_dbscan(
        eps=eps,
        min_points=8,
        print_progress=True
    )
)

# 4. Znalezienie 10 najwiekszych skupisk
liczba_grup = etykiety.max() + 1

rozmiary = []

for i in range(liczba_grup):
    rozmiary.append(np.sum(etykiety == i))

najwieksze = np.argsort(rozmiary)[::-1][:10]

print("\n10 NAJWIEKSZYCH SKUPISK")

for miejsce, grupa in enumerate(najwieksze):
    wybrane = czerwone[etykiety == grupa]

    srodek = np.mean(wybrane, axis=0)

    print(
        f"{miejsce + 1}. Skupisko {grupa}: "
        f"{len(wybrane)} punktow, "
        f"srodek: {np.round(srodek, 3)}"
    )

# 5. Przygotowanie kolorow
nowe_kolory = np.full(
    (len(punkty), 3),
    [0.65, 0.65, 0.65]
)

indeksy_czerwonych = np.where(maska)[0]

# Wszystkie pozostale skupiska - jasnozolte
for i, indeks in enumerate(indeksy_czerwonych):
    if etykiety[i] >= 0:
        nowe_kolory[indeks] = [0.9, 0.8, 0.3]
    else:
        nowe_kolory[indeks] = [1.0, 0.0, 0.0]

# 6. Kazde z 10 najwiekszych skupisk ma inny kolor
for miejsce, grupa in enumerate(najwieksze):
    odcien = miejsce / 10
    kolor = colorsys.hsv_to_rgb(odcien, 0.9, 1.0)

    indeksy = indeksy_czerwonych[etykiety == grupa]
    nowe_kolory[indeksy] = kolor

# 7. Wizualizacja i zapis
chmura.colors = o3d.utility.Vector3dVector(nowe_kolory)

o3d.io.write_point_cloud(
    "03_WYNIKI/brick_part01_10_najwiekszych_skupisk.ply",
    chmura
)

o3d.visualization.draw_geometries(
    [chmura],
    window_name="10 najwiekszych skupisk",
    width=1200,
    height=800
)

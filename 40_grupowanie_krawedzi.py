
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

# 1. Wczytanie wyniku poprzedniej analizy
sciezka = "03_WYNIKI/brick_part01_krawedzie_probne.ply"
chmura = o3d.io.read_point_cloud(sciezka)

punkty = np.asarray(chmura.points)
kolory = np.asarray(chmura.colors)

# 2. Wybranie czerwonych punktow
maska = (
    (kolory[:, 0] > 0.9) &
    (kolory[:, 1] < 0.1) &
    (kolory[:, 2] < 0.1)
)

czerwone_punkty = punkty[maska]

print("Wszystkie punkty:", len(punkty))
print("Czerwone punkty:", len(czerwone_punkty))

# 3. Automatyczne dobranie odleglosci grupowania
drzewo = cKDTree(czerwone_punkty)

odleglosci, indeksy = drzewo.query(
    czerwone_punkty,
    k=11
)

eps = np.median(odleglosci[:, -1]) * 1.5

print("\nPARAMETRY GRUPOWANIA")
print("Automatycznie dobrana odleglosc:", eps)

# 4. Grupowanie punktow
chmura_czerwona = o3d.geometry.PointCloud()
chmura_czerwona.points = o3d.utility.Vector3dVector(czerwone_punkty)

etykiety = np.array(
    chmura_czerwona.cluster_dbscan(
        eps=eps,
        min_points=8,
        print_progress=True
    )
)

# 5. Podsumowanie grup
liczba_grup = etykiety.max() + 1

print("\nWYNIKI GRUPOWANIA")
print("Liczba znalezionych skupisk:", liczba_grup)
print("Punkty nieprzypisane do skupisk:", np.sum(etykiety == -1))

rozmiary = []

for i in range(liczba_grup):
    liczba = np.sum(etykiety == i)
    rozmiary.append(liczba)

for i, liczba in enumerate(sorted(rozmiary, reverse=True)[:20]):
    print(f"Skupisko {i + 1}: {liczba} punktow")

# 6. Przygotowanie wizualizacji
kolory_wizualizacji = np.full(
    (len(punkty), 3),
    [0.65, 0.65, 0.65]
)

indeksy_czerwonych = np.where(maska)[0]

for i, indeks in enumerate(indeksy_czerwonych):
    if etykiety[i] == -1:
        kolory_wizualizacji[indeks] = [1.0, 0.0, 0.0]
    else:
        kolory_wizualizacji[indeks] = [1.0, 0.55, 0.0]

chmura.colors = o3d.utility.Vector3dVector(
    kolory_wizualizacji
)

# 7. Zapis i wyswietlenie
o3d.io.write_point_cloud(
    "03_WYNIKI/brick_part01_skupiska_krawedzi.ply",
    chmura
)

o3d.visualization.draw_geometries(
    [chmura],
    window_name="Skupiska nieregularnosci powierzchni",
    width=1200,
    height=800
)

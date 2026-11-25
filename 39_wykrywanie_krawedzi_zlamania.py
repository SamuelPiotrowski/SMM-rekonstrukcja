
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

# 1. Wczytanie roboczej powierzchni bocznej
sciezka = "03_WYNIKI/brick_part01_bok_roboczy.ply"
chmura = o3d.io.read_point_cloud(sciezka)

punkty = np.asarray(chmura.points)

print("Liczba punktow:", len(punkty))

# 2. Obliczenie normalnych
chmura.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
)

normalne = np.asarray(chmura.normals)

# 3. Znalezienie najblizszych sasiadow
drzewo = cKDTree(punkty)

odleglosci, indeksy = drzewo.query(punkty, k=16)

# 4. Obliczenie zmiennosci normalnych
wyniki = np.zeros(len(punkty))

for i in range(len(punkty)):
    sasiedzi = indeksy[i, 1:]
    normalna = normalne[i]
    normalne_sasiadow = normalne[sasiedzi]

    iloczyny = np.abs(normalne_sasiadow @ normalna)
    iloczyny = np.clip(iloczyny, 0, 1)

    katy = np.degrees(np.arccos(iloczyny))

    wyniki[i] = np.mean(katy)

print("\nANALIZA ZMIENNOSCI NORMALNYCH")
print("Minimum:", np.min(wyniki))
print("Maksimum:", np.max(wyniki))
print("Srednia:", np.mean(wyniki))
print("Mediana:", np.median(wyniki))

for p in [75, 85, 90, 95, 98]:
    print(f"Percentyl {p}:", np.percentile(wyniki, p))

# 5. Wybieramy punkty o najwiekszej zmiennosci
prog = np.percentile(wyniki, 90)
maska = wyniki >= prog

print("\nPunkty o najwyzszej zmiennosci:", np.sum(maska))
print("Zastosowany prog:", prog)

# 6. Kolorowanie
kolory = np.zeros((len(punkty), 3))

# Zwykla powierzchnia - szara
kolory[:] = [0.65, 0.65, 0.65]

# Najbardziej zmienne miejsca - czerwone
kolory[maska] = [1, 0, 0]

chmura.colors = o3d.utility.Vector3dVector(kolory)

# 7. Zapis wynikow
o3d.io.write_point_cloud(
    "03_WYNIKI/brick_part01_krawedzie_probne.ply",
    chmura
)

# 8. Wizualizacja
o3d.visualization.draw_geometries(
    [chmura],
    window_name="Proba wykrywania krawedzi zlamania",
    width=1200,
    height=800
)


import open3d as o3d
import numpy as np

# Wczytanie pozostalych punktow
chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_pozostale_punkty.ply"
)

print("Liczba punktow:", len(chmura.points))

# Obliczenie normalnych
chmura.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
)

normalne = np.asarray(chmura.normals)

# Analiza kierunkow normalnych
print("\nANALIZA NORMALNYCH")

print("Srednia X:", np.mean(normalne[:, 0]))
print("Srednia Y:", np.mean(normalne[:, 1]))
print("Srednia Z:", np.mean(normalne[:, 2]))

print("\nZakresy normalnych:")
print("X:", np.min(normalne[:, 0]), np.max(normalne[:, 0]))
print("Y:", np.min(normalne[:, 1]), np.max(normalne[:, 1]))
print("Z:", np.min(normalne[:, 2]), np.max(normalne[:, 2]))

# Kolorowanie wedlug kierunku normalnych
# Wartosc bezwzgledna usuwa problem przeciwnych zwrotow
kolory = np.abs(normalne)

# Wizualizacja
chmura.colors = o3d.utility.Vector3dVector(kolory)

o3d.visualization.draw_geometries(
    [chmura],
    window_name="Kierunki normalnych powierzchni bocznej",
    width=1200,
    height=800
)

sciezka = "03_WYNIKI/brick_part01_normalne_pozostalych.ply"

zapisano = o3d.io.write_point_cloud(sciezka, chmura)

print("Czy zapisano plik:", zapisano)
print("Sciezka:", sciezka)
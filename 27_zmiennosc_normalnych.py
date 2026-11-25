
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

# Wczytanie chmury z obliczonymi normalnymi
chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_normalne_pozostalych.ply"
)

punkty = np.asarray(chmura.points)
normalne = np.asarray(chmura.normals)

print("Liczba punktow:", len(punkty))

# Wyszukiwanie 16 najblizszych punktow
drzewo = cKDTree(punkty)
odleglosci, sasiedzi = drzewo.query(punkty, k=16, workers=-1)

# Pobranie normalnych sasiadow
normalne_sasiadow = normalne[sasiedzi]

# Porownanie kierunkow normalnych
iloczyny = np.sum(
    normalne[:, np.newaxis, :] * normalne_sasiadow,
    axis=2
)

# Wartosc bezwzgledna usuwa problem przeciwnych zwrotow normalnych
iloczyny = np.clip(np.abs(iloczyny), 0, 1)

# Obliczenie katow w stopniach
katy = np.degrees(np.arccos(iloczyny))

# Pomijamy pierwszy punkt, bo jest to sam badany punkt
srednie_katy = np.mean(katy[:, 1:], axis=1)

print("\nZMIENNOSC NORMALNYCH")
print("Minimum:", np.min(srednie_katy))
print("Maksimum:", np.max(srednie_katy))
print("Srednia:", np.mean(srednie_katy))
print("Mediana:", np.median(srednie_katy))

print("\nPercentyle:")
for p in [50, 75, 90, 95, 99]:
    print(p, ":", np.percentile(srednie_katy, p))

# Kolorowanie wedlug zmiennosci
progi = np.percentile(srednie_katy, [50, 75, 90])

kolory = np.zeros((len(punkty), 3))

kolory[srednie_katy <= progi[0]] = [0, 0, 1]
kolory[
    (srednie_katy > progi[0]) &
    (srednie_katy <= progi[1])
] = [0, 1, 0]

kolory[
    (srednie_katy > progi[1]) &
    (srednie_katy <= progi[2])
] = [1, 1, 0]

kolory[srednie_katy > progi[2]] = [1, 0, 0]

chmura.colors = o3d.utility.Vector3dVector(kolory)

o3d.visualization.draw_geometries(
    [chmura],
    window_name="Zmiennosc normalnych",
    width=1200,
    height=800
)

o3d.io.write_point_cloud(
    "03_WYNIKI/brick_part01_zmiennosc_normalnych.ply",
    chmura
)

print("\nZapisano wynik.")

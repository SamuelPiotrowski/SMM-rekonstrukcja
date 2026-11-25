
import open3d as o3d
import numpy as np

# Wczytanie pozostalych punktow
chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_pozostale_punkty.ply"
)

print("Liczba punktow:", len(chmura.points))

# Obliczenie lokalnych macierzy kowariancji
chmura.estimate_covariances(
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
)

kowariancje = np.asarray(chmura.covariances)

# Obliczenie wartosci wlasnych
wartosci = np.linalg.eigvalsh(kowariancje)

# Krzywizna jako stosunek najmniejszej wartosci wlasnej
# do sumy wszystkich wartosci wlasnych
suma = np.sum(wartosci, axis=1)

krzywizna = np.zeros(len(suma))

poprawne = suma > 0

krzywizna[poprawne] = (
    wartosci[poprawne, 0] / suma[poprawne]
)

print("\nANALIZA KRZYWIZNY")
print("Minimum:", np.min(krzywizna))
print("Maksimum:", np.max(krzywizna))
print("Srednia:", np.mean(krzywizna))
print("Mediana:", np.median(krzywizna))

print("\nPERCENTYLE")
for p in [25, 50, 75, 90, 95]:
    print(p, ":", np.percentile(krzywizna, p))

# Kolorowanie wedlug krzywizny
prog_niski = np.percentile(krzywizna, 50)
prog_sredni = np.percentile(krzywizna, 80)
prog_wysoki = np.percentile(krzywizna, 95)

kolory = np.zeros((len(krzywizna), 3))

# Niska krzywizna - niebieski
kolory[krzywizna <= prog_niski] = [0.1, 0.3, 1.0]

# Srednia krzywizna - zielony
maska = (
    (krzywizna > prog_niski) &
    (krzywizna <= prog_sredni)
)
kolory[maska] = [0.1, 0.8, 0.2]

# Podwyzszona krzywizna - zolty
maska = (
    (krzywizna > prog_sredni) &
    (krzywizna <= prog_wysoki)
)
kolory[maska] = [1.0, 0.8, 0.0]

# Najwyzsza krzywizna - czerwony
kolory[krzywizna > prog_wysoki] = [1.0, 0.1, 0.1]

chmura.colors = o3d.utility.Vector3dVector(kolory)

# Zapis wizualizacji
o3d.io.write_point_cloud(
    "03_WYNIKI/brick_part01_pozostale_krzywizna.ply",
    chmura
)

# Wizualizacja
o3d.visualization.draw_geometries(
    [chmura],
    window_name="Krzywizna pozostalych punktow",
    width=1200,
    height=800
)

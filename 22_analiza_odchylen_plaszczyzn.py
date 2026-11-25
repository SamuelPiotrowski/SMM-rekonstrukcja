
import open3d as o3d
import numpy as np

# Wczytanie chmury
chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_odszumiony.ply"
)

punkty = np.asarray(chmura.points)

# Wspolczynniki pierwszej plaszczyzny z pliku 20
plaszczyzna_1 = np.array([
    -0.00545321689,
    0.999984897,
    0.000683893682,
    1.15647083
])

# Wspolczynniki drugiej plaszczyzny z pliku 21
plaszczyzna_2 = np.array([
    -0.00321763,
    0.99998938,
    0.00330044,
    -1.85629718
])

# Progi odleglosci
progi = [0.02, 0.03, 0.05, 0.08]

# Analiza pierwszej plaszczyzny
print("\nPIERWSZA PLASZCZYZNA")

a, b, c, d = plaszczyzna_1

odleglosci_1 = np.abs(
    a * punkty[:, 0] +
    b * punkty[:, 1] +
    c * punkty[:, 2] + d
) / np.linalg.norm([a, b, c])

for prog in progi:
    liczba = np.sum(odleglosci_1 <= prog)
    print(
        "Prog:", prog,
        "| Liczba punktow:", liczba
    )

print("Odleglosci - percentyle:")
print(np.percentile(odleglosci_1, [10, 25, 50, 75, 90]))

# Analiza drugiej plaszczyzny
print("\nDRUGA PLASZCZYZNA")

maksimum_Y = np.max(punkty[:, 1])
minimum_Y = np.min(punkty[:, 1])

granica = maksimum_Y - (maksimum_Y - minimum_Y) * 0.2

indeksy_gora = np.where(punkty[:, 1] > granica)[0]
punkty_gora = punkty[indeksy_gora]

a, b, c, d = plaszczyzna_2

odleglosci_2 = np.abs(
    a * punkty_gora[:, 0] +
    b * punkty_gora[:, 1] +
    c * punkty_gora[:, 2] + d
) / np.linalg.norm([a, b, c])

for prog in progi:
    liczba = np.sum(odleglosci_2 <= prog)
    print(
        "Prog:", prog,
        "| Liczba punktow:", liczba
    )

print("Odleglosci - percentyle:")
print(np.percentile(odleglosci_2, [10, 25, 50, 75, 90]))

# Wizualizacja odchylen pierwszej plaszczyzny
kolory = np.full((len(punkty), 3), [0.7, 0.7, 0.7])

kolory[odleglosci_1 <= 0.02] = [1.0, 0.1, 0.1]
kolory[
    (odleglosci_1 > 0.02) & (odleglosci_1 <= 0.05)
] = [1.0, 0.6, 0.0]

chmura.colors = o3d.utility.Vector3dVector(kolory)

o3d.visualization.draw_geometries(
    [chmura],
    window_name="Odchylenia pierwszej plaszczyzny",
    width=1200,
    height=800
)

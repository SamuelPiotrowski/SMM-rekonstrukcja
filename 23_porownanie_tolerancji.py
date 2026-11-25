
import open3d as o3d
import numpy as np

chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_odszumiony.ply"
)

punkty = np.asarray(chmura.points)

# Wspolczynniki pierwszej plaszczyzny
plaszczyzna_1 = np.array([
    -0.00545321689,
    0.999984897,
    0.000683893682,
    1.15647083
])

# Wspolczynniki drugiej plaszczyzny
plaszczyzna_2 = np.array([
    -0.00321763,
    0.99998938,
    0.00330044,
    -1.85629718
])

# Odleglosci od pierwszej plaszczyzny
a, b, c, d = plaszczyzna_1

odleglosci_1 = np.abs(
    a * punkty[:, 0] +
    b * punkty[:, 1] +
    c * punkty[:, 2] + d
) / np.linalg.norm([a, b, c])

# Odleglosci od drugiej plaszczyzny
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

# Porownanie trzech tolerancji
for prog in [0.02, 0.03, 0.05]:

    print("\nTOLERANCJA:", prog)

    # Kolory pierwszego konca
    kolory_1 = np.full((len(punkty), 3), [0.7, 0.7, 0.7])
    kolory_1[odleglosci_1 <= prog] = [1, 0, 0]

    chmura_1 = o3d.geometry.PointCloud()
    chmura_1.points = o3d.utility.Vector3dVector(punkty)
    chmura_1.colors = o3d.utility.Vector3dVector(kolory_1)

    # Kolory drugiego konca
    kolory_2 = np.full((len(punkty_gora), 3), [0.7, 0.7, 0.7])
    kolory_2[odleglosci_2 <= prog] = [0, 0, 1]

    chmura_2 = o3d.geometry.PointCloud()
    chmura_2.points = o3d.utility.Vector3dVector(punkty_gora)
    chmura_2.colors = o3d.utility.Vector3dVector(kolory_2)

    print("Liczba punktow dolnego konca:",
          np.sum(odleglosci_1 <= prog))

    print("Liczba punktow gornego konca:",
          np.sum(odleglosci_2 <= prog))

    o3d.visualization.draw_geometries(
        [chmura_1],
        window_name=f"Dolny koniec - tolerancja {prog}",
        width=1200,
        height=800
    )

    o3d.visualization.draw_geometries(
        [chmura_2],
        window_name=f"Gorny koniec - tolerancja {prog}",
        width=1200,
        height=800
    )

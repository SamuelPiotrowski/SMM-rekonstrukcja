
import open3d as o3d
import numpy as np

# Wczytanie chmury
chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_odszumiony.ply"
)

punkty = np.asarray(chmura.points)

# Znalezienie gornego zakresu osi Y
maksimum_Y = np.max(punkty[:, 1])
minimum_Y = np.min(punkty[:, 1])

# Wybieramy gorna czesc klocka
granica = maksimum_Y - (maksimum_Y - minimum_Y) * 0.2

indeksy_gora = np.where(punkty[:, 1] > granica)[0]

gora = chmura.select_by_index(indeksy_gora)

print("Punkty w gornej czesci:", len(gora.points))
print("Granica Y:", granica)

# Szukanie plaszczyzny
plaszczyzna, indeksy = gora.segment_plane(
    distance_threshold=0.02,
    ransac_n=3,
    num_iterations=2000
)

koniec = gora.select_by_index(indeksy)
reszta = gora.select_by_index(indeksy, invert=True)

print("\nWYKRYTA PLASZCZYZNA")
print("Liczba punktow:", len(indeksy))
print("Wspolczynniki:", plaszczyzna)

# Kolory
koniec.paint_uniform_color([0.1, 0.3, 1.0])
reszta.paint_uniform_color([0.7, 0.7, 0.7])

o3d.visualization.draw_geometries(
    [koniec, reszta],
    window_name="Wykrywanie drugiego konca",
    width=1200,
    height=800
)


import open3d as o3d
import numpy as np

# Wczytanie pozostalych punktow
chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_pozostale_punkty.ply"
)

punkty = np.asarray(chmura.points)

y = punkty[:, 1]

# Zakres osi Y
y_min = np.min(y)
y_max = np.max(y)

dol = y_min + 0.05 * (y_max - y_min)
gora = y_max - 0.05 * (y_max - y_min)

granice = np.linspace(dol, gora, 13)

# Roboczy obszar boczny
maska_boku = (
    (y >= granice[1]) &
    (y < granice[11])
)

# Wyodrebnienie punktow
indeksy = np.where(maska_boku)[0]
bok = chmura.select_by_index(indeksy)

# Zapis
sciezka = "03_WYNIKI/brick_part01_bok_roboczy.ply"

o3d.io.write_point_cloud(sciezka, bok)

print("Liczba punktow boku:", len(bok.points))
print("Zapisano:", sciezka)

# Wizualizacja
bok.paint_uniform_color([0.1, 0.8, 0.2])

o3d.visualization.draw_geometries(
    [bok],
    window_name="Robocza powierzchnia boczna",
    width=1200,
    height=800
)


import open3d as o3d
import numpy as np

# Wczytanie pozostalych punktow
chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_pozostale_punkty.ply"
)

punkty = np.asarray(chmura.points)
y = punkty[:, 1]

y_min = np.min(y)
y_max = np.max(y)

dol = y_min + 0.05 * (y_max - y_min)
gora = y_max - 0.05 * (y_max - y_min)

granice = np.linspace(dol, gora, 13)

# Pierwszy i ostatni przekroj
maska_dol = (y >= granice[0]) & (y < granice[1])
maska_gora = (y >= granice[11]) & (y < granice[12])

# Srodkowa czesc boku
maska_srodek = (y >= granice[1]) & (y < granice[11])

print("DOLNY PIERŚCIEŃ:", np.sum(maska_dol))
print("GÓRNY PIERŚCIEŃ:", np.sum(maska_gora))
print("ŚRODEK BOKU:", np.sum(maska_srodek))

# Kolory
kolory = np.full((len(punkty), 3), [0.65, 0.65, 0.65])

kolory[maska_dol] = [1, 0.3, 0]
kolory[maska_gora] = [0, 0.3, 1]
kolory[maska_srodek] = [0.1, 0.8, 0.2]

chmura.colors = o3d.utility.Vector3dVector(kolory)

o3d.visualization.draw_geometries(
    [chmura],
    window_name="Pierscienie i srodkowa powierzchnia boku",
    width=1200,
    height=800
)


import open3d as o3d
import numpy as np

# Wczytanie pozostalych punktow
chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_pozostale_punkty.ply"
)

punkty = np.asarray(chmura.points)

# Zakres osi Y – taki sam jak w analizie obrysow
y_min = np.min(punkty[:, 1])
y_max = np.max(punkty[:, 1])

dol = y_min + 0.05 * (y_max - y_min)
gora = y_max - 0.05 * (y_max - y_min)

granice = np.linspace(dol, gora, 13)

# Wybieramy tylko ostatni przekroj
maska = (
    (punkty[:, 1] >= granice[11]) &
    (punkty[:, 1] < granice[12])
)

print("Punkty ostatniego przekroju:", np.sum(maska))

# Wszystkie punkty na szaro
kolory = np.full((len(punkty), 3), [0.65, 0.65, 0.65])

# Ostatni przekroj na pomaranczowo
kolory[maska] = [1, 0.3, 0]

chmura.colors = o3d.utility.Vector3dVector(kolory)

o3d.visualization.draw_geometries(
    [chmura],
    window_name="Sprawdzenie plamy w 3D",
    width=1200,
    height=800
)

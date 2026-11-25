
import open3d as o3d
import numpy as np

chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_pozostale_punkty.ply"
)

punkty = np.asarray(chmura.points)

y_min = np.min(punkty[:, 1])
y_max = np.max(punkty[:, 1])

dol = y_min + 0.05 * (y_max - y_min)
gora = y_max - 0.05 * (y_max - y_min)

granice = np.linspace(dol, gora, 13)

# Pierwszy przekroj
maska = (
    (punkty[:, 1] >= granice[0]) &
    (punkty[:, 1] < granice[1])
)

print("Punkty pierwszego przekroju:", np.sum(maska))

kolory = np.full((len(punkty), 3), [0.65, 0.65, 0.65])
kolory[maska] = [1, 0.3, 0]

chmura.colors = o3d.utility.Vector3dVector(kolory)

o3d.visualization.draw_geometries(
    [chmura],
    window_name="Sprawdzenie dolnej krawedzi",
    width=1200,
    height=800
)

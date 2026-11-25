
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

# Szerokosc waskiego pasa: 2% analizowanej wysokosci
szerokosc = 0.02 * (gora - dol)

# Wybor punktow przy dolnej i gornej granicy
dolna = punkty[:, 1] <= dol + szerokosc
gorna = punkty[:, 1] >= gora - szerokosc

print("Dolna granica:", np.sum(dolna), "punktow")
print("Gorna granica:", np.sum(gorna), "punktow")

kolory = np.full((len(punkty), 3), [0.65, 0.65, 0.65])

kolory[dolna] = [1, 0.3, 0]
kolory[gorna] = [0, 0.3, 1]

chmura.colors = o3d.utility.Vector3dVector(kolory)

o3d.visualization.draw_geometries(
    [chmura],
    window_name="Dokladne granice powierzchni bocznej",
    width=1200,
    height=800
)

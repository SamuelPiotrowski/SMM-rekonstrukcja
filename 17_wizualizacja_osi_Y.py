
import open3d as o3d
import numpy as np
from pathlib import Path

plik = Path("03_WYNIKI/brick_part01_odszumiony.ply")
chmura = o3d.io.read_point_cloud(str(plik))

punkty = np.asarray(chmura.points)

# Zakres osi Y
minimum = np.min(punkty[:, 1])
maksimum = np.max(punkty[:, 1])

# Podzial wysokosci na trzy czesci
zakres = (maksimum - minimum) / 3

dol = minimum + zakres
gora = maksimum - zakres

kolory = np.zeros((len(punkty), 3))

# Srodkowa czesc - szara
kolory[:] = [0.7, 0.7, 0.7]

# Dolna czesc - czerwona
kolory[punkty[:, 1] < dol] = [1, 0, 0]

# Gorna czesc - niebieska
kolory[punkty[:, 1] > gora] = [0, 0, 1]

chmura.colors = o3d.utility.Vector3dVector(kolory)

print("Dolna granica:", dol)
print("Gorna granica:", gora)

o3d.visualization.draw_geometries(
    [chmura],
    window_name="Podzial fragmentu wzdluz osi Y",
    width=1200,
    height=800
)

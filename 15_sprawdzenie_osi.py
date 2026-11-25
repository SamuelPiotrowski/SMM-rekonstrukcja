
import open3d as o3d
import numpy as np
from pathlib import Path

plik = Path("03_WYNIKI/brick_part01_odszumiony.ply")
chmura = o3d.io.read_point_cloud(str(plik))

punkty = np.asarray(chmura.points)

minimum = punkty.min(axis=0)
maksimum = punkty.max(axis=0)

print("ZAKRES WSPOLRZEDNYCH")
print("X:", minimum[0], "-", maksimum[0])
print("Y:", minimum[1], "-", maksimum[1])
print("Z:", minimum[2], "-", maksimum[2])

# Kolorowanie punktow
chmura.paint_uniform_color([0.7, 0.7, 0.7])

# Dodanie osi wspolrzednych
osie = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=20,
    origin=[0, 0, 0]
)

o3d.visualization.draw_geometries(
    [chmura, osie],
    window_name="Fragment kosci i osie",
    width=1200,
    height=800
)

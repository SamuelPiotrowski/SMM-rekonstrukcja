
import open3d as o3d
import numpy as np
from pathlib import Path

# Wczytanie odszumionej chmury
plik = Path("03_WYNIKI/brick_part01_odszumiony.ply")
chmura = o3d.io.read_point_cloud(str(plik))

punkty = np.asarray(chmura.points)
Y = punkty[:, 1]

# Podzial na 30 warstw
liczba_warstw = 30

granice = np.linspace(
    np.min(Y),
    np.max(Y),
    liczba_warstw + 1
)

# Kolory
kolory = np.zeros((len(punkty), 3))

# Wszystkie punkty na szaro
kolory[:] = [0.7, 0.7, 0.7]

# Pierwsza warstwa - czerwona
maska_dol = (
    (Y >= granice[0]) &
    (Y < granice[1])
)

kolory[maska_dol] = [1, 0, 0]

# Ostatnia warstwa - niebieska
maska_gora = (
    (Y >= granice[-2]) &
    (Y <= granice[-1])
)

kolory[maska_gora] = [0, 0, 1]

# Przypisanie kolorow
chmura.colors = o3d.utility.Vector3dVector(kolory)

print("Pierwsza warstwa:", np.sum(maska_dol))
print("Ostatnia warstwa:", np.sum(maska_gora))

# Wizualizacja
o3d.visualization.draw_geometries(
    [chmura],
    window_name="Analiza skrajnych warstw",
    width=1200,
    height=800
)

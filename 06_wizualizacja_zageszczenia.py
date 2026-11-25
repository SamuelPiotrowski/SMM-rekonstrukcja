
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie pierwszego fragmentu
chmura = o3d.io.read_point_cloud(
    "02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY/brick_part01.ply"
)

# Obliczenie odleglosci do najblizszego sasiada
odleglosci = np.asarray(
    chmura.compute_nearest_neighbor_distance()
)

# Wyznaczenie zakresu kolorowania
dolna_granica = np.percentile(odleglosci, 5)
gorna_granica = np.percentile(odleglosci, 95)

# Przeskalowanie odleglosci do zakresu od 0 do 1
wartosci = (odleglosci - dolna_granica) / (
    gorna_granica - dolna_granica
)

wartosci = np.clip(wartosci, 0, 1)

# Przypisanie kolorow
kolory = plt.get_cmap("jet")(wartosci)[:, :3]

# Pokolorowanie punktow
chmura.colors = o3d.utility.Vector3dVector(kolory)

# Zapis pokolorowanej chmury
o3d.io.write_point_cloud(
    "03_WYNIKI/brick_part01_zageszczenie.ply",
    chmura
)

# Wyswietlenie chmury
o3d.visualization.draw_geometries(
    [chmura],
    window_name="Zageszczenie punktow"
)

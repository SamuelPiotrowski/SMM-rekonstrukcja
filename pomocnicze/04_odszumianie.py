
import open3d as o3d
from pathlib import Path

# Wczytanie fragmentu
plik = Path("02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY/brick_part01.ply")

chmura = o3d.io.read_point_cloud(str(plik))

print("Liczba punktow przed odszumianiem:", len(chmura.points))

# Usuwanie punktow odstajacych
chmura_czysta, indeksy = chmura.remove_statistical_outlier(
    nb_neighbors=30,
    std_ratio=3.0
)

print("Liczba punktow po odszumianiu:", len(chmura_czysta.points))
print("Usuniete punkty:", len(chmura.points) - len(chmura_czysta.points))

# Obliczenie procentu usunietych punktow
procent = (
    (len(chmura.points) - len(chmura_czysta.points))
    / len(chmura.points) * 100
)

print("Procent usunietych punktow:", procent)

# Zachowanie oryginalnych danych
folder = Path("03_WYNIKI")
folder.mkdir(exist_ok=True)

o3d.io.write_point_cloud(
    str(folder / "brick_part01_odszumiony.ply"),
    chmura_czysta
)

print("Zapisano oczyszczony fragment.")

# Kolory do porownania

# Oddzielenie punktow usunietych
punkty_usuniete = chmura.select_by_index(
    indeksy,
    invert=True
)

# Kolorowanie punktow
chmura_czysta.paint_uniform_color([0.2, 0.8, 0.3])
punkty_usuniete.paint_uniform_color([1.0, 0.1, 0.1])

print("Liczba punktow usunietych:", len(punkty_usuniete.points))

# Wyswietlenie wyniku
o3d.visualization.draw_geometries(
    [chmura_czysta, punkty_usuniete],
    window_name="Wynik odszumiania",
    width=1200,
    height=800
)

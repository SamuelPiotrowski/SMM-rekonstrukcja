
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os

# Folder z oryginalnymi fragmentami
folder = "02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY"

# Lista plikow
pliki = [
    "brick_part01.ply",
    "brick_part02.ply",
    "brick_part03.ply",
    "brick_part04.ply",
    "brick_part05.ply",
    "brick_part06.ply"
]

# Przygotowanie wykresow
fig, wykresy = plt.subplots(2, 3, figsize=(15, 8))
wykresy = wykresy.flatten()

for i, nazwa in enumerate(pliki):

    sciezka = os.path.join(folder, nazwa)

    # Wczytanie chmury punktow
    chmura = o3d.io.read_point_cloud(sciezka)

    # Obliczenie odleglosci do najblizszego sasiada
    odleglosci = np.asarray(
        chmura.compute_nearest_neighbor_distance()
    )

    # Podstawowe statystyki
    if len(odleglosci) == 0:
        print("Brak punktow w chmurze!")
    continue
    print("\n", nazwa)
    print("Liczba punktow:", len(odleglosci))
    print("Minimalna odleglosc:", np.min(odleglosci))
    print("Maksymalna odleglosc:", np.max(odleglosci))
    print("Srednia odleglosc:", np.mean(odleglosci))
    print("Mediana:", np.median(odleglosci))
    print("95. percentyl:", np.percentile(odleglosci, 95))

    # Histogram odleglosci
    wykresy[i].hist(odleglosci, bins=50)
    wykresy[i].set_title(nazwa)
    wykresy[i].set_xlabel("Odleglosc do najblizszego sasiada")
    wykresy[i].set_ylabel("Liczba punktow")

plt.tight_layout()

# Zapis wykresu
os.makedirs("03_WYNIKI", exist_ok=True)
plt.savefig("03_WYNIKI/zageszczenie_punktow.png")

plt.show()


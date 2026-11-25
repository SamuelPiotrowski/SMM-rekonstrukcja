
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Wczytanie odszumionej chmury
plik = Path("03_WYNIKI/brick_part01_odszumiony.ply")
chmura = o3d.io.read_point_cloud(str(plik))

punkty = np.asarray(chmura.points)

# Pobranie wspolrzednych Y
wartosci_Y = punkty[:, 1]

print("MINIMUM Y:", np.min(wartosci_Y))
print("MAKSIMUM Y:", np.max(wartosci_Y))
print("SREDNIA Y:", np.mean(wartosci_Y))
print("MEDIANA Y:", np.median(wartosci_Y))

# Histogram wspolrzednych Y
plt.figure(figsize=(10, 5))

plt.hist(wartosci_Y, bins=100)

plt.title("Rozklad punktow wzdluz osi Y")
plt.xlabel("Wspolrzedna Y")
plt.ylabel("Liczba punktow")

plt.grid()
plt.show()


import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie pozostalych punktow
chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_pozostale_punkty.ply"
)

punkty = np.asarray(chmura.points)

print("Liczba punktow:", len(punkty))

# Wspolrzedne
x = punkty[:, 0]
y = punkty[:, 1]
z = punkty[:, 2]

# Zakres wysokosci
y_min = np.min(y)
y_max = np.max(y)

# Pomijamy skrajne 5% wysokosci,
# aby ograniczyc wplyw pozostalosci plaskich koncow
dol = y_min + 0.05 * (y_max - y_min)
gora = y_max - 0.05 * (y_max - y_min)

# Dzielimy wysokosc na 12 przekrojow
liczba_przekrojow = 12
granice = np.linspace(dol, gora, liczba_przekrojow + 1)

fig, osie = plt.subplots(3, 4, figsize=(16, 12))
osie = osie.flatten()

print("\nANALIZA OBRYSOW")

for i in range(liczba_przekrojow):

    # Punkty nalezace do danego przekroju
    maska = (
        (y >= granice[i]) &
        (y < granice[i + 1])
    )

    przekroj = punkty[maska]

    if len(przekroj) == 0:
        continue

    x_przekroj = przekroj[:, 0]
    z_przekroj = przekroj[:, 2]

    # Wymiary obrysu
    szerokosc = np.max(x_przekroj) - np.min(x_przekroj)
    glebokosc = np.max(z_przekroj) - np.min(z_przekroj)

    print(
        f"Przekroj {i + 1:2d} | "
        f"Punkty: {len(przekroj):6d} | "
        f"X: {szerokosc:.3f} | "
        f"Z: {glebokosc:.3f}"
    )

    # Rysowanie przekroju
    osie[i].scatter(
        x_przekroj,
        z_przekroj,
        s=1
    )

    osie[i].set_title(
        f"Przekroj {i + 1}\nY: {granice[i]:.2f} - {granice[i+1]:.2f}"
    )

    osie[i].set_xlabel("X")
    osie[i].set_ylabel("Z")
    osie[i].set_aspect("equal")
    osie[i].grid(True)

plt.tight_layout()
plt.show()


import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie pozostalych punktow
chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_pozostale_punkty.ply"
)

punkty = np.asarray(chmura.points)

x = punkty[:, 0]
y = punkty[:, 1]
z = punkty[:, 2]

print("Liczba punktow:", len(punkty))

# Rownanie dolnej plaszczyzny
dolna = np.array([
    -0.00545321689,
    0.999984897,
    0.000683882,
    1.15647083
])

# Rownanie gornej plaszczyzny
gorna = np.array([
    -0.00321763,
    0.99998938,
    0.00330044,
    -1.85629718
])

# Obliczenie odleglosci punktow od plaszczyzn
def oblicz_odleglosci(punkty, plaszczyzna):
    a, b, c, d = plaszczyzna

    licznik = np.abs(
        a * punkty[:, 0] +
        b * punkty[:, 1] +
        c * punkty[:, 2] + d
    )

    mianownik = np.sqrt(a**2 + b**2 + c**2)

    return licznik / mianownik


odleglosci_dol = oblicz_odleglosci(punkty, dolna)
odleglosci_gora = oblicz_odleglosci(punkty, gorna)

# Zakres wysokosci
y_min = np.min(y)
y_max = np.max(y)
zakres = y_max - y_min

# Analizujemy okolice dolnego i gornego konca
maska_dol = y <= y_min + 0.35 * zakres
maska_gora = y >= y_max - 0.35 * zakres

print("\nDOLNA OKOLICA")
print("Liczba punktow:", np.sum(maska_dol))
print("Odleglosc minimalna:", np.min(odleglosci_dol[maska_dol]))
print("Mediana:", np.median(odleglosci_dol[maska_dol]))
print("Percentyl 25:", np.percentile(odleglosci_dol[maska_dol], 25))
print("Percentyl 75:", np.percentile(odleglosci_dol[maska_dol], 75))

print("\nGORNA OKOLICA")
print("Liczba punktow:", np.sum(maska_gora))
print("Odleglosc minimalna:", np.min(odleglosci_gora[maska_gora]))
print("Mediana:", np.median(odleglosci_gora[maska_gora]))
print("Percentyl 25:", np.percentile(odleglosci_gora[maska_gora], 25))
print("Percentyl 75:", np.percentile(odleglosci_gora[maska_gora], 75))

# Wykresy odleglosci od plaszczyzn
fig, osie = plt.subplots(2, 1, figsize=(12, 9))

osie[0].scatter(
    y[maska_dol],
    odleglosci_dol[maska_dol],
    s=1,
    alpha=0.25
)
osie[0].axhline(0.05, linestyle="--")
osie[0].set_title("Odleglosc punktow od dolnej plaszczyzny")
osie[0].set_xlabel("Wspolrzedna Y")
osie[0].set_ylabel("Odleglosc")
osie[0].grid(True)

osie[1].scatter(
    y[maska_gora],
    odleglosci_gora[maska_gora],
    s=1,
    alpha=0.25
)
osie[1].axhline(0.05, linestyle="--")
osie[1].set_title("Odleglosc punktow od gornej plaszczyzny")
osie[1].set_xlabel("Wspolrzedna Y")
osie[1].set_ylabel("Odleglosc")
osie[1].grid(True)

plt.tight_layout()
plt.show()


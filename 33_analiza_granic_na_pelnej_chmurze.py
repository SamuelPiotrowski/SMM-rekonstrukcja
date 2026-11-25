
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie pelnej, odszumionej chmury
chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_odszumiony.ply"
)

punkty = np.asarray(chmura.points)

print("Liczba punktow:", len(punkty))

y = punkty[:, 1]

# Rownania plaszczyzn
dolna = np.array([
    -0.00545321689,
    0.999984897,
    0.000683882,
    1.15647083
])

gorna = np.array([
    -0.00321763,
    0.99998938,
    0.00330044,
    -1.85629718
])

def odleglosc(punkty, plaszczyzna):
    a, b, c, d = plaszczyzna

    return np.abs(
        a * punkty[:, 0] +
        b * punkty[:, 1] +
        c * punkty[:, 2] + d
    ) / np.sqrt(a**2 + b**2 + c**2)


odl_dol = odleglosc(punkty, dolna)
odl_gora = odleglosc(punkty, gorna)

# Analiza najblizszych okolic koncow
y_min = np.min(y)
y_max = np.max(y)
zakres = y_max - y_min

maska_dol = y <= y_min + 0.35 * zakres
maska_gora = y >= y_max - 0.35 * zakres

print("\nPELNA CHMURA - DOLNY KONIEC")
print("Punkty:", np.sum(maska_dol))
print("Mediana odleglosci:", np.median(odl_dol[maska_dol]))
print("Percentyl 25:", np.percentile(odl_dol[maska_dol], 25))
print("Percentyl 75:", np.percentile(odl_dol[maska_dol], 75))

print("\nPELNA CHMURA - GORNY KONIEC")
print("Punkty:", np.sum(maska_gora))
print("Mediana odleglosci:", np.median(odl_gora[maska_gora]))
print("Percentyl 25:", np.percentile(odl_gora[maska_gora], 25))
print("Percentyl 75:", np.percentile(odl_gora[maska_gora], 75))

# Wykresy
fig, osie = plt.subplots(2, 1, figsize=(12, 9))

osie[0].scatter(
    y[maska_dol], odl_dol[maska_dol],
    s=1, alpha=0.2
)
osie[0].axhline(0.05, linestyle="--")
osie[0].set_title("Pelna chmura - odleglosc od dolnej plaszczyzny")
osie[0].set_xlabel("Y")
osie[0].set_ylabel("Odleglosc")
osie[0].grid(True)

osie[1].scatter(
    y[maska_gora], odl_gora[maska_gora],
    s=1, alpha=0.2
)
osie[1].axhline(0.05, linestyle="--")
osie[1].set_title("Pelna chmura - odleglosc od gornej plaszczyzny")
osie[1].set_xlabel("Y")
osie[1].set_ylabel("Odleglosc")
osie[1].grid(True)

plt.tight_layout()
plt.show()

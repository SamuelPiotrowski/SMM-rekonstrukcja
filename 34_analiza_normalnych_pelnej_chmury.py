
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie pelnej chmury
chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_odszumiony.ply"
)

punkty = np.asarray(chmura.points)
y = punkty[:, 1]

print("Liczba punktow:", len(punkty))

# Obliczenie normalnych
chmura.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
)

normalne = np.asarray(chmura.normals)

# Kierunki normalnych wykrytych plaszczyzn
normalna_dol = np.array([
    -0.00545321689,
    0.999984897,
    0.000683882
])

normalna_gora = np.array([
    -0.00321763,
    0.99998938,
    0.00330044
])

# Normalizacja wektorow
normalna_dol = normalna_dol / np.linalg.norm(normalna_dol)
normalna_gora = normalna_gora / np.linalg.norm(normalna_gora)

# Porownanie normalnych
# Wartosc bezwzgledna uwzglednia niejednoznaczny zwrot normalnych
zgodnosc_dol = np.abs(normalne @ normalna_dol)
zgodnosc_gora = np.abs(normalne @ normalna_gora)

zgodnosc_dol = np.clip(zgodnosc_dol, 0, 1)
zgodnosc_gora = np.clip(zgodnosc_gora, 0, 1)

# Kat miedzy normalna punktu a normalna plaszczyzny
kat_dol = np.degrees(np.arccos(zgodnosc_dol))
kat_gora = np.degrees(np.arccos(zgodnosc_gora))

# Okolice koncow - 30% zakresu wysokosci
y_min = np.min(y)
y_max = np.max(y)
zakres = y_max - y_min

maska_dol = y <= y_min + 0.30 * zakres
maska_gora = y >= y_max - 0.30 * zakres

def pokaz_wyniki(nazwa, katy, maska):
    wartosci = katy[maska]

    print("\n", nazwa)
    print("Liczba punktow:", len(wartosci))
    print("Srednia:", np.mean(wartosci))
    print("Mediana:", np.median(wartosci))

    for p in [25, 50, 75, 90, 95]:
        print(
            f"Percentyl {p}:",
            np.percentile(wartosci, p)
        )

pokaz_wyniki("DOLNY KONIEC", kat_dol, maska_dol)
pokaz_wyniki("GORNY KONIEC", kat_gora, maska_gora)

# Wykresy katow w zaleznosci od wysokosci
fig, osie = plt.subplots(2, 1, figsize=(12, 9))

osie[0].scatter(
    y[maska_dol],
    kat_dol[maska_dol],
    s=1,
    alpha=0.2
)
osie[0].set_title("Dolna okolica - kat normalnych")
osie[0].set_xlabel("Wspolrzedna Y")
osie[0].set_ylabel("Kat w stopniach")
osie[0].grid(True)

osie[1].scatter(
    y[maska_gora],
    kat_gora[maska_gora],
    s=1,
    alpha=0.2
)
osie[1].set_title("Gorna okolica - kat normalnych")
osie[1].set_xlabel("Wspolrzedna Y")
osie[1].set_ylabel("Kat w stopniach")
osie[1].grid(True)

plt.tight_layout()
plt.show()

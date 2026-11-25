
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import os

# Wczytanie oryginalnego fragmentu
sciezka = "02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY/brick_part01.ply"
chmura = o3d.io.read_point_cloud(sciezka)

# Pobranie punktow i normalnych
punkty = np.asarray(chmura.points)
normalne = np.asarray(chmura.normals)

print("Liczba punktow:", len(punkty))

# Parametry wygladzania
liczba_sasiadow = 20
sigma_przestrzeni = 0.03
sigma_powierzchni = 0.02

# Przygotowanie drzewa sasiadow
drzewo = cKDTree(punkty)

# Tablica na nowe polozenia punktow
nowe_punkty = punkty.copy()

# Przetwarzanie punktow partiami
rozmiar_partii = 10000

for poczatek in range(0, len(punkty), rozmiar_partii):

    koniec = min(poczatek + rozmiar_partii, len(punkty))

    # Aktualna partia punktow
    aktualne = punkty[poczatek:koniec]
    aktualne_normalne = normalne[poczatek:koniec]

    # Znalezienie najblizszych sasiadow
    odleglosci, indeksy = drzewo.query(
        aktualne,
        k=liczba_sasiadow
    )

    sasiedzi = punkty[indeksy]

    # Roznice polozen punktow
    roznice = sasiedzi - aktualne[:, np.newaxis, :]

    # Odleglosci sasiadow od lokalnej plaszczyzny
    wysokosci = np.sum(
        roznice * aktualne_normalne[:, np.newaxis, :],
        axis=2
    )

    # Wagi zalezne od odleglosci przestrzennej
    wagi_przestrzeni = np.exp(
        -(odleglosci ** 2) / (2 * sigma_przestrzeni ** 2)
    )

    # Wagi zalezne od odleglosci od plaszczyzny
    wagi_powierzchni = np.exp(
        -(wysokosci ** 2) / (2 * sigma_powierzchni ** 2)
    )

    # Polaczenie wag
    wagi = wagi_przestrzeni * wagi_powierzchni

    # Obliczenie przesuniecia punktow
    przesuniecie = np.sum(
        wagi * wysokosci,
        axis=1
    ) / np.sum(wagi, axis=1)

    # Przesuniecie tylko w kierunku normalnej
    nowe_punkty[poczatek:koniec] = (
        aktualne
        + przesuniecie[:, np.newaxis] * aktualne_normalne
    )

    print("Przetworzono punktow:", koniec)

# Utworzenie wygladzonej chmury
chmura_wygladzona = o3d.geometry.PointCloud()
chmura_wygladzona.points = o3d.utility.Vector3dVector(nowe_punkty)
chmura_wygladzona.normals = chmura.normals

# Obliczenie wielkosci przesuniec
przesuniecia = np.linalg.norm(nowe_punkty - punkty, axis=1)

print("\nWyniki wygladzania:")
print("Srednie przesuniecie:", np.mean(przesuniecia))
print("Maksymalne przesuniecie:", np.max(przesuniecia))

# Zapis wyniku
os.makedirs("03_WYNIKI", exist_ok=True)

o3d.io.write_point_cloud(
    "03_WYNIKI/brick_part01_wygladzony.ply",
    chmura_wygladzona
)

print("\nZapisano wygladzona chmure.")

# Wyswietlenie oryginalu
oryginal = o3d.geometry.PointCloud(chmura)
oryginal.paint_uniform_color([0.2, 0.7, 0.9])

print("Wyswietlanie oryginalnej chmury...")
o3d.visualization.draw_geometries([oryginal])

# Wyswietlenie wyniku
chmura_wygladzona.paint_uniform_color([0.2, 0.8, 0.3])

print("Wyswietlanie wygladzonej chmury...")
o3d.visualization.draw_geometries([chmura_wygladzona])

# Wizualizacja przesuniec punktow

chmura_przesuniec = o3d.geometry.PointCloud()
chmura_przesuniec.points = o3d.utility.Vector3dVector(punkty)

# Kolorowanie wedlug wielkosci przesuniecia
kolory = np.zeros((len(punkty), 3))

maks = np.max(przesuniecia)

kolory[:, 0] = przesuniecia / maks
kolory[:, 1] = 0.2
kolory[:, 2] = 1 - przesuniecia / maks

chmura_przesuniec.colors = o3d.utility.Vector3dVector(kolory)

o3d.visualization.draw_geometries(
    [chmura_przesuniec],
    window_name="Wielkosc przesuniec punktow"
)
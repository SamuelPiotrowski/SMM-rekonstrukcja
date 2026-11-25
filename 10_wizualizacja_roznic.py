
import open3d as o3d
import numpy as np

# Wczytanie chmur punktow
oryginal = o3d.io.read_point_cloud(
    "02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY/brick_part01.ply"
)

wygladzona = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_wygladzony.ply"
)

print("Liczba punktow:", len(oryginal.points))


# Obliczenie odleglosci punktow oryginalnych
# od najblizszych punktow wygladzonej chmury
odleglosci = np.asarray(
    oryginal.compute_point_cloud_distance(wygladzona)
)

print("Najmniejsza odleglosc:", np.min(odleglosci))
print("Srednia odleglosc:", np.mean(odleglosci))
print("Najwieksza odleglosc:", np.max(odleglosci))


# Ustalenie zakresu kolorow
# Wykorzystujemy 95 percentyl, aby pojedyncze
# duze odleglosci nie zdominowaly wizualizacji
maksymalna_wartosc = np.percentile(odleglosci, 95)

if maksymalna_wartosc == 0:
    maksymalna_wartosc = 1


# Przeskalowanie odleglosci do zakresu 0-1
wartosci = odleglosci / maksymalna_wartosc
wartosci = np.clip(wartosci, 0, 1)


# Przygotowanie kolorow
kolory = np.zeros((len(wartosci), 3))

for i in range(len(wartosci)):

    if wartosci[i] < 0.33:
        # Niebieski
        kolory[i] = [0, 0, 1]

    elif wartosci[i] < 0.66:
        # Zielony
        kolory[i] = [0, 1, 0]

    elif wartosci[i] < 0.85:
        # Zolty
        kolory[i] = [1, 1, 0]

    else:
        # Czerwony
        kolory[i] = [1, 0, 0]


# Przypisanie kolorow do oryginalnej chmury
oryginal.colors = o3d.utility.Vector3dVector(kolory)


# Zapisanie pokolorowanej chmury
o3d.io.write_point_cloud(
    "03_WYNIKI/brick_part01_roznice.ply",
    oryginal
)


# Wyswietlenie modelu
o3d.visualization.draw_geometries(
    [oryginal],
    window_name="Roznice po wygladzaniu",
    width=1200,
    height=800
)

print("\nZapisano pokolorowana chmure:")
print("03_WYNIKI/brick_part01_roznice.ply")

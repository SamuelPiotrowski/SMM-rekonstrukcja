
import open3d as o3d
import numpy as np

# Wczytanie chmur punktow
oryginal = o3d.io.read_point_cloud(
    "02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY/brick_part01.ply"
)

wygladzona = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_wygladzony.ply"
)

print("Wczytano chmury punktow.")

# Obliczenie odleglosci miedzy chmurami
odleglosci = np.asarray(
    oryginal.compute_point_cloud_distance(wygladzona)
)

# Wyznaczenie progu 95 percentyla
prog = np.percentile(odleglosci, 95)

# Wybranie punktow o najwiekszych roznicach
indeksy = np.where(odleglosci >= prog)[0]

punkty = np.asarray(oryginal.points)

# Statystyki
print("\nANALIZA PUNKTOW O NAJWIEKSZYCH ROZNICACH")
print("Prog odleglosci:", prog)
print("Liczba wybranych punktow:", len(indeksy))
print("Procent wszystkich punktow:", len(indeksy) / len(punkty) * 100)
print("Srednia odleglosc:", np.mean(odleglosci[indeksy]))
print("Maksymalna odleglosc:", np.max(odleglosci))

# Przypisanie kolorow bezposrednio do oryginalnej chmury
kolory = np.zeros((len(punkty), 3))

# Wszystkie punkty na poczatku szare
kolory[:] = [0.65, 0.65, 0.65]

# Punkty o najwiekszych roznicach na czerwono
kolory[indeksy] = [1, 0, 0]

oryginal.colors = o3d.utility.Vector3dVector(kolory)

# Zapis pokolorowanej chmury
o3d.io.write_point_cloud(
    "03_WYNIKI/brick_part01_pokolorowany.ply",
    oryginal
)

# Wizualizacja
o3d.visualization.draw_geometries(
    [oryginal],
    window_name="Obszary najwiekszych roznic",
    width=1200,
    height=800
)

print("\nZapisano plik:")
print("03_WYNIKI/brick_part01_pokolorowany.ply")

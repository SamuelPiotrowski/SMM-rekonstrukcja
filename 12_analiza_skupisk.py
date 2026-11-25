
import open3d as o3d
import numpy as np

# 1. Wczytanie oryginalnej chmury
oryginal = o3d.io.read_point_cloud(
    "02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY/brick_part01.ply"
)

# 2. Wczytanie wygladzonej chmury
wygladzona = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_wygladzony.ply"
)

print("Wczytano chmury punktow.")

# 3. Obliczenie odleglosci miedzy chmurami
odleglosci = np.asarray(
    oryginal.compute_point_cloud_distance(wygladzona)
)

# 4. Wybranie 5% punktow o najwiekszych roznicach
prog = np.percentile(odleglosci, 95)
indeksy = np.where(odleglosci >= prog)[0]

punkty = np.asarray(oryginal.points)
punkty_roznic = punkty[indeksy]

print("\nWYBRANE PUNKTY")
print("Liczba punktow:", len(punkty_roznic))

# 5. Utworzenie chmury zawierajacej tylko wybrane punkty
chmura_roznic = o3d.geometry.PointCloud()
chmura_roznic.points = o3d.utility.Vector3dVector(
    punkty_roznic
)

# 6. Sprawdzenie odleglosci miedzy punktami
drzewo = o3d.geometry.KDTreeFlann(chmura_roznic)

odleglosci_sasiadow = []

for punkt in punkty_roznic:
    liczba, indeksy_sasiadow, odleglosci_kwadratowe = (
        drzewo.search_knn_vector_3d(punkt, 6)
    )

    if liczba == 6:
        odleglosci_sasiadow.append(
            np.sqrt(odleglosci_kwadratowe[5])
        )



# 7. Porownanie roznych parametrow DBSCAN

print("\nPOROWNANIE PARAMETROW DBSCAN")

wartosci_eps = [0.015, 0.02, 0.03, 0.04, 0.05, 0.06]

for eps in wartosci_eps:

    etykiety = np.array(
        chmura_roznic.cluster_dbscan(
            eps=eps,
            min_points=10,
            print_progress=False
        )
    )

    numery_skupisk = set(etykiety)
    numery_skupisk.discard(-1)

    liczba_szumu = np.sum(etykiety == -1)

    rozmiary = []

    for numer in numery_skupisk:
        rozmiary.append(np.sum(etykiety == numer))

    rozmiary.sort(reverse=True)

    print("\nEPS:", eps)
    print("Liczba skupisk:", len(numery_skupisk))
    print("Punkty poza skupiskami:", liczba_szumu)

    if len(rozmiary) > 0:
        print("Najwieksze skupiska:", rozmiary[:5])
    else:
        print("Nie znaleziono skupisk.")



# 8. Grupowanie punktow metoda DBSCAN
etykiety = np.array(
    chmura_roznic.cluster_dbscan(
        eps=eps,
        min_points=10,
        print_progress=True
    )
)

# 9. Zliczenie skupisk
numery_skupisk = set(etykiety)
numery_skupisk.discard(-1)

print("\nWYNIKI ANALIZY SKUPISK")
print("Liczba wykrytych skupisk:", len(numery_skupisk))

liczba_szumu = np.sum(etykiety == -1)

print("Liczba punktow poza skupiskami:", liczba_szumu)

for numer in sorted(numery_skupisk):
    liczba_punktow = np.sum(etykiety == numer)

    print(
        "Skupisko", numer,
        "- liczba punktow:", liczba_punktow
    )

# 10. Przygotowanie kolorow
kolory = np.zeros((len(punkty), 3))

# Caly klocek na szaro
kolory[:] = [0.75, 0.75, 0.75]

# Punkty poza skupiskami na czerwono
kolory[indeksy[etykiety == -1]] = [1, 0, 0]

# Kazde skupisko otrzymuje inny kolor
np.random.seed(42)

for numer in sorted(numery_skupisk):
    kolor = np.random.uniform(0.1, 0.95, 3)

    kolory[indeksy[etykiety == numer]] = kolor

# 11. Przypisanie kolorow do oryginalnej chmury
oryginal.colors = o3d.utility.Vector3dVector(kolory)

# 12. Zapis wyniku
o3d.io.write_point_cloud(
    "03_WYNIKI/brick_part01_skupiska.ply",
    oryginal
)

# 13. Wizualizacja
o3d.visualization.draw_geometries(
    [oryginal],
    window_name="Analiza skupisk punktow",
    width=1200,
    height=800
)

print("\nZapisano wynik:")
print("03_WYNIKI/brick_part01_skupiska.ply")

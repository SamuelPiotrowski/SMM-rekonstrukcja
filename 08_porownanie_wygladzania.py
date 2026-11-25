
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie oryginalnej i wygladzonej chmury
oryginal = o3d.io.read_point_cloud(
    "02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY/brick_part01.ply"
)

wygladzona = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_wygladzony.ply"
)

# Parametry analizy
rozmiar_voxela = 0.04
liczba_sasiadow = 30


# Funkcja obliczajaca zmiennosc powierzchni
def oblicz_zmiennosc(chmura):

    # Zmniejszenie liczby punktow
    chmura_mniejsza = chmura.voxel_down_sample(
        voxel_size=rozmiar_voxela
    )

    punkty = np.asarray(chmura_mniejsza.points)

    # Przygotowanie drzewa punktow
    drzewo = o3d.geometry.KDTreeFlann(chmura_mniejsza)

    wyniki = []

    # Analiza lokalnego otoczenia kazdego punktu
    for punkt in punkty:

        liczba, indeksy, odleglosci = drzewo.search_knn_vector_3d(
            punkt, liczba_sasiadow
        )

        sasiedzi = punkty[indeksy]

        # Obliczenie macierzy kowariancji
        macierz = np.cov(sasiedzi.T)

        # Wartosci wlasne PCA
        wartosci = np.linalg.eigvalsh(macierz)

        # Zmiennosc powierzchni
        suma = np.sum(wartosci)

        if suma > 0:
            zmiennosc = wartosci[0] / suma
        else:
            zmiennosc = 0

        wyniki.append(zmiennosc)

    return np.array(wyniki)


# Obliczenia dla obu chmur
print("Analiza oryginalnej chmury...")
wyniki_oryginal = oblicz_zmiennosc(oryginal)

print("Analiza wygladzonej chmury...")
wyniki_wygladzona = oblicz_zmiennosc(wygladzona)


# Wyswietlenie statystyk
print("\nORYGINALNA CHMURA")
print("Liczba punktow:", len(wyniki_oryginal))
print("Srednia:", np.mean(wyniki_oryginal))
print("Mediana:", np.median(wyniki_oryginal))

print("\nWYGLADZONA CHMURA")
print("Liczba punktow:", len(wyniki_wygladzona))
print("Srednia:", np.mean(wyniki_wygladzona))
print("Mediana:", np.median(wyniki_wygladzona))


# Porownanie na wykresie

# Porownanie na wykresie

# Wspolne przedzialy dla obu histogramow
wszystkie_wyniki = np.concatenate([
    wyniki_oryginal,
    wyniki_wygladzona
])

przedzialy = np.histogram_bin_edges(
    wszystkie_wyniki,
    bins=50
)

plt.figure(figsize=(10, 6))

plt.hist(
    wyniki_oryginal,
    bins=przedzialy,
    density=True,
    alpha=0.5,
    label="Oryginalna"
)

plt.hist(
    wyniki_wygladzona,
    bins=przedzialy,
    density=True,
    alpha=0.5,
    label="Wygladzona"
)

plt.xlabel("Zmiennosc powierzchni")
plt.ylabel("Gestosc")
plt.title("Porownanie powierzchni przed i po wygladzaniu")
plt.legend()

plt.tight_layout()
plt.savefig("03_WYNIKI/porownanie_wygladzania.png")
plt.show()

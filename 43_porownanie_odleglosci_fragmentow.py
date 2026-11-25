
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import os

folder = "02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY"

fragmenty = {}

# 1. Wczytanie wszystkich fragmentow
for numer in range(1, 7):

    nazwa = f"brick_part{numer:02d}.ply"
    sciezka = os.path.join(folder, nazwa)

    chmura = o3d.io.read_point_cloud(sciezka)

    # Zmniejszamy liczbe punktow do szybszego porownania
    chmura = chmura.voxel_down_sample(voxel_size=0.05)

    punkty = np.asarray(chmura.points)

    fragmenty[numer] = punkty

    print(
        f"Fragment {numer}: "
        f"{len(punkty)} punktow po zmniejszeniu"
    )

# 2. Funkcja obliczajaca odleglosc miedzy zakresami
def odleglosc_brylek(a_min, a_max, b_min, b_max):

    przerwy = np.maximum(
        0,
        np.maximum(a_min - b_max, b_min - a_max)
    )

    return np.linalg.norm(przerwy)


# 3. Porownywanie wszystkich par
wyniki = []

print("\n")
print("=" * 85)
print("POROWNANIE ODLEGLOSCI MIEDZY FRAGMENTAMI")
print("=" * 85)

for i in range(1, 7):

    for j in range(i + 1, 7):

        a = fragmenty[i]
        b = fragmenty[j]

        # Zakresy przestrzenne
        a_min = np.min(a, axis=0)
        a_max = np.max(a, axis=0)

        b_min = np.min(b, axis=0)
        b_max = np.max(b, axis=0)

        odleglosc_zakresow = odleglosc_brylek(
            a_min, a_max, b_min, b_max
        )

        # Najblizsze punkty
        drzewo_a = cKDTree(a)
        drzewo_b = cKDTree(b)

        odleglosci_ab, _ = drzewo_b.query(a, k=1)
        odleglosci_ba, _ = drzewo_a.query(b, k=1)

        odleglosci = np.concatenate([
            odleglosci_ab,
            odleglosci_ba
        ])

        minimum = np.min(odleglosci)
        p1 = np.percentile(odleglosci, 1)
        p5 = np.percentile(odleglosci, 5)
        p10 = np.percentile(odleglosci, 10)
        mediana = np.median(odleglosci)

        wyniki.append([
            i, j,
            odleglosc_zakresow,
            minimum, p1, p5, p10, mediana
        ])

# 4. Sortowanie wedlug odleglosci punktow
wyniki.sort(key=lambda x: x[4])

print(
    "\nPara | Zakresy | Minimum | P1 | P5 | P10 | Mediana"
)

for wynik in wyniki:

    i, j, zakresy, minimum, p1, p5, p10, mediana = wynik

    print(
        f"{i}-{j} | "
        f"{zakresy:.4f} | "
        f"{minimum:.4f} | "
        f"{p1:.4f} | "
        f"{p5:.4f} | "
        f"{p10:.4f} | "
        f"{mediana:.4f}"
    )

print("\nKONIEC POROWNANIA")

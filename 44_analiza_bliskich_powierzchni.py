
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import os

folder = "02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY"

fragmenty = {}

# 1. Wczytanie i zmniejszenie chmur
for numer in range(1, 7):

    sciezka = os.path.join(
        folder,
        f"brick_part{numer:02d}.ply"
    )

    chmura = o3d.io.read_point_cloud(sciezka)
    chmura = chmura.voxel_down_sample(voxel_size=0.05)

    fragmenty[numer] = np.asarray(chmura.points)

# 2. Progi odleglosci
progi = [0.05, 0.10, 0.20]

print("=" * 85)
print("ANALIZA BLISKOSCI POWIERZCHNI FRAGMENTOW")
print("=" * 85)

# 3. Porownanie wszystkich par
for i in range(1, 7):

    for j in range(i + 1, 7):

        a = fragmenty[i]
        b = fragmenty[j]

        drzewo_a = cKDTree(a)
        drzewo_b = cKDTree(b)

        # Odleglosci punktow A od B
        odleglosci_ab, _ = drzewo_b.query(a, k=1)

        # Odleglosci punktow B od A
        odleglosci_ba, _ = drzewo_a.query(b, k=1)

        print(f"\n{'-' * 65}")
        print(f"FRAGMENT {i} I FRAGMENT {j}")
        print(f"{'-' * 65}")

        for prog in progi:

            bliskie_ab = np.sum(odleglosci_ab < prog)
            bliskie_ba = np.sum(odleglosci_ba < prog)

            procent_ab = 100 * bliskie_ab / len(a)
            procent_ba = 100 * bliskie_ba / len(b)

            print(f"\nOdleglosc ponizej {prog:.2f}:")
            print(
                f"Punkty {i} blisko {j}: "
                f"{bliskie_ab} ({procent_ab:.2f}%)"
            )
            print(
                f"Punkty {j} blisko {i}: "
                f"{bliskie_ba} ({procent_ba:.2f}%)"
            )

print("\nKONIEC ANALIZY")


import open3d as o3d
import numpy as np
import os

folder = "02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY"

print("=" * 75)
print("ANALIZA WSZYSTKICH 6 FRAGMENTOW BRICK")
print("=" * 75)

for numer in range(1, 7):

    nazwa = f"brick_part{numer:02d}.ply"
    sciezka = os.path.join(folder, nazwa)

    chmura = o3d.io.read_point_cloud(sciezka)
    punkty = np.asarray(chmura.points)

    if len(punkty) == 0:
        print(f"\nNie udalo sie wczytac: {nazwa}")
        continue

    minimum = np.min(punkty, axis=0)
    maksimum = np.max(punkty, axis=0)
    zakres = maksimum - minimum
    srodek = np.mean(punkty, axis=0)

    print(f"\n{'-' * 55}")
    print(f"FRAGMENT {numer}: {nazwa}")
    print(f"{'-' * 55}")

    print("Liczba punktow:", len(punkty))

    print("X:", round(minimum[0], 4),
          "do", round(maksimum[0], 4),
          "| zakres:", round(zakres[0], 4))

    print("Y:", round(minimum[1], 4),
          "do", round(maksimum[1], 4),
          "| zakres:", round(zakres[1], 4))

    print("Z:", round(minimum[2], 4),
          "do", round(maksimum[2], 4),
          "| zakres:", round(zakres[2], 4))

    print("Srodek chmury:", np.round(srodek, 4))

print("\nKONIEC ANALIZY")

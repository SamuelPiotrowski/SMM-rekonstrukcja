
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pathlib import Path

folder = Path("02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY")
folder_wyniki = Path("03_WYNIKI")
folder_wyniki.mkdir(exist_ok=True)

pliki = sorted(folder.glob("*.ply"))

for plik in pliki:

    print("\nAnaliza:", plik.name)

    # Wczytanie fragmentu
    chmura = o3d.io.read_point_cloud(str(plik))

    print("Punkty przed zmniejszeniem:", len(chmura.points))

    # Zmniejszenie liczby punktow
    chmura = chmura.voxel_down_sample(voxel_size=0.04)

    punkty = np.asarray(chmura.points)

    print("Punkty po zmniejszeniu:", len(punkty))

    # Wyszukiwanie sasiadow
    drzewo = cKDTree(punkty)
    k = 30

    krzywizna = np.zeros(len(punkty))

    # Obliczanie krzywizny
    for poczatek in range(0, len(punkty), 5000):

        koniec = min(poczatek + 5000, len(punkty))

        sasiedzi = drzewo.query(
            punkty[poczatek:koniec],
            k=k
        )[1]

        lokalne_punkty = punkty[sasiedzi]

        srodek = np.mean(
            lokalne_punkty,
            axis=1,
            keepdims=True
        )

        przesuniecie = lokalne_punkty - srodek

        macierze = np.einsum(
            "nki,nkj->nij",
            przesuniecie,
            przesuniecie
        ) / k

        wartosci = np.linalg.eigvalsh(macierze)

        suma = np.sum(wartosci, axis=1)

        krzywizna[poczatek:koniec] = (
            wartosci[:, 0] / np.maximum(suma, 1e-12)
        )

    # Statystyki
    print("Minimum:", np.min(krzywizna))
    print("Maksimum:", np.max(krzywizna))
    print("Srednia:", np.mean(krzywizna))
    print("Mediana:", np.median(krzywizna))

    # Kolorowanie punktow
    dolna = np.percentile(krzywizna, 2)
    gorna = np.percentile(krzywizna, 98)

    wartosci_kolorow = np.clip(
        (krzywizna - dolna) / (gorna - dolna + 1e-12),
        0,
        1
    )

    kolory = plt.get_cmap("turbo")(wartosci_kolorow)[:, :3]

    chmura.colors = o3d.utility.Vector3dVector(kolory)

    # Zapis wyniku
    nazwa = plik.stem + "_krzywizna.ply"

    o3d.io.write_point_cloud(
        str(folder_wyniki / nazwa),
        chmura
    )

    print("Zapisano:", nazwa)

print("\nZakonczono analize wszystkich fragmentow.")

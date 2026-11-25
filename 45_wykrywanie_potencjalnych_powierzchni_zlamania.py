
import open3d as o3d
import numpy as np
import os


# ============================================================
# USTAWIENIA
# ============================================================

folder_wejsciowy = "02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY"

folder_wynikowy = "03_WYNIKI/45_WYKRYWANIE_POWIERZCHNI"

os.makedirs(folder_wynikowy, exist_ok=True)

# Zmniejszenie liczby punktow
voxel_size = 0.08

# Promien analizowanego sasiedztwa
promien = 0.20

# Ile najwiekszych wynikow uznajemy za kandydatow
procent_kandydatow = 15

# Parametry grupowania punktow
eps_grupowania = 0.16
min_punktow = 15


# ============================================================
# FUNKCJA: ZAMIANA WYNIKOW NA RANGI PROCENTOWE
# ============================================================

def zamien_na_rangi(wartosci):

    kolejnosc = np.argsort(np.argsort(wartosci))

    rangi = kolejnosc / (len(wartosci) - 1)

    return rangi


# ============================================================
# ANALIZA JEDNEGO FRAGMENTU
# ============================================================

def analizuj_fragment(numer):

    print("\n" + "=" * 70)
    print("ANALIZA FRAGMENTU", numer)
    print("=" * 70)

    sciezka = os.path.join(
        folder_wejsciowy,
        f"brick_part{numer:02d}.ply"
    )

    chmura = o3d.io.read_point_cloud(sciezka)

    if chmura.is_empty():
        print("BLAD: Nie udalo sie wczytac fragmentu.")
        return

    print("Punkty oryginalne:", len(chmura.points))

    # Zmniejszenie liczby punktow
    chmura = chmura.voxel_down_sample(voxel_size)

    punkty = np.asarray(chmura.points)

    liczba = len(punkty)

    print("Punkty po zmniejszeniu:", liczba)

    # --------------------------------------------------------
    # 1. OBLICZANIE NORMALNYCH
    # --------------------------------------------------------

    print("\nObliczanie normalnych...")

    chmura.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=promien,
            max_nn=40
        )
    )

    normalne = np.asarray(chmura.normals)

    drzewo = o3d.geometry.KDTreeFlann(chmura)

    # Tablice na wyniki
    krzywizna = np.zeros(liczba)
    chropowatosc = np.zeros(liczba)
    zmiennosc_normalnych = np.zeros(liczba)

    # --------------------------------------------------------
    # 2. ANALIZA LOKALNEJ GEOMETRII
    # --------------------------------------------------------

    print("Analiza lokalnej geometrii...")

    for i in range(liczba):

        _, indeksy, _ = drzewo.search_radius_vector_3d(
            punkty[i],
            promien
        )

        sasiedzi = punkty[indeksy]
        normalne_sasiadow = normalne[indeksy]

        if len(indeksy) < 8:
            continue

        # Srodek lokalnego sasiedztwa
        srodek = np.mean(sasiedzi, axis=0)

        # Macierz kowariancji
        przesuniecia = sasiedzi - srodek

        macierz = np.cov(
            przesuniecia.T
        )

        # Wartosci wlasne
        wartosci = np.linalg.eigvalsh(macierz)

        wartosci = np.maximum(wartosci, 0)

        # Najmniejsza wartosc odpowiada odchyleniu
        # od lokalnej plaszczyzny
        suma = np.sum(wartosci)

        if suma > 0:

            krzywizna[i] = wartosci[0] / suma

        # Chropowatosc: odchylenie od lokalnej plaszczyzny
        chropowatosc[i] = np.sqrt(wartosci[0])

        # Zmiennosc kierunku normalnych
        iloczyny = np.abs(
            normalne_sasiadow @ normalne[i]
        )

        iloczyny = np.clip(iloczyny, 0, 1)

        katy = np.degrees(
            np.arccos(iloczyny)
        )

        zmiennosc_normalnych[i] = np.mean(katy)

    # --------------------------------------------------------
    # 3. LACZENIE CECH GEOMETRYCZNYCH
    # --------------------------------------------------------

    print("\nLaczenie cech geometrycznych...")

    ranga_krzywizny = zamien_na_rangi(krzywizna)

    ranga_chropowatosci = zamien_na_rangi(chropowatosc)

    ranga_normalnych = zamien_na_rangi(
        zmiennosc_normalnych
    )

    # Kazda cecha ma taki sam udzial
    wynik = (
        ranga_krzywizny
        + ranga_chropowatosci
        + ranga_normalnych
    ) / 3

    # --------------------------------------------------------
    # 4. WYBOR KANDYDATOW
    # --------------------------------------------------------

    prog = np.percentile(
        wynik,
        100 - procent_kandydatow
    )

    indeksy_kandydatow = np.where(
        wynik >= prog
    )[0]

    kandydaci = chmura.select_by_index(
        indeksy_kandydatow
    )

    print("\nWYNIKI ANALIZY")

    print("Krzywizna:")
    print("  Srednia:", np.mean(krzywizna))

    print("Chropowatosc:")
    print("  Srednia:", np.mean(chropowatosc))

    print("Zmiennosc normalnych:")
    print("  Srednia:", np.mean(zmiennosc_normalnych))

    print("\nKandydaci na powierzchnie:")
    print("  Liczba punktow:", len(indeksy_kandydatow))
    print("  Procent:", round(
        100 * len(indeksy_kandydatow) / liczba, 2
    ))

    # --------------------------------------------------------
    # 5. KOLOROWA MAPA WYNIKU
    # --------------------------------------------------------

    kolory = np.zeros((liczba, 3))

    kolory[:, 0] = np.clip(2 * wynik, 0, 1)

    kolory[:, 1] = 1 - np.abs(2 * wynik - 1)

    kolory[:, 2] = np.clip(2 * (1 - wynik), 0, 1)

    chmura_wynik = o3d.geometry.PointCloud()

    chmura_wynik.points = o3d.utility.Vector3dVector(
        punkty
    )

    chmura_wynik.colors = o3d.utility.Vector3dVector(
        kolory
    )

    # --------------------------------------------------------
    # 6. GRUPOWANIE KANDYDATOW
    # --------------------------------------------------------

    print("\nGrupowanie kandydatow...")

    etykiety = np.array(
        kandydaci.cluster_dbscan(
            eps=eps_grupowania,
            min_points=min_punktow,
            print_progress=True
        )
    )

    liczba_grup = (
        len(set(etykiety)) - (1 if -1 in etykiety else 0)
    )

    liczba_szumu = np.sum(etykiety == -1)

    print("\nLiczba wykrytych skupisk:", liczba_grup)
    print("Punkty bez przypisania:", liczba_szumu)

    # Rozmiary skupisk
    rozmiary = []

    for etykieta in range(liczba_grup):

        liczba_punktow = np.sum(
            etykiety == etykieta
        )

        rozmiary.append(
            (etykieta, liczba_punktow)
        )

    rozmiary.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\n10 najwiekszych skupisk:")

    for etykieta, liczba_punktow in rozmiary[:10]:

        print(
            "Skupisko",
            etykieta,
            "-",
            liczba_punktow,
            "punktow"
        )

    # --------------------------------------------------------
    # 7. ZAPIS WYNIKOW
    # --------------------------------------------------------

    nazwa = f"fragment_{numer:02d}"

    o3d.io.write_point_cloud(
        os.path.join(
            folder_wynikowy,
            nazwa + "_mapa_cech.ply"
        ),
        chmura_wynik
    )

    kandydaci.paint_uniform_color(
        [1, 0, 0]
    )

    o3d.io.write_point_cloud(
        os.path.join(
            folder_wynikowy,
            nazwa + "_kandydaci.ply"
        ),
        kandydaci
    )

    # Zapis tylko punktow nalezacych do wykrytych skupisk
    indeksy_grup = np.where(etykiety >= 0)[0]

    skupiska = kandydaci.select_by_index(
        indeksy_grup
    )

    skupiska.paint_uniform_color(
        [1, 0, 0]
    )

    o3d.io.write_point_cloud(
        os.path.join(
            folder_wynikowy,
            nazwa + "_skupiska.ply"
        ),
        skupiska
    )

    print("\nZapisano wyniki fragmentu", numer)


# ============================================================
# ANALIZA WSZYSTKICH FRAGMENTOW
# ============================================================

for numer in range(1, 7):

    analizuj_fragment(numer)


print("\n" + "=" * 70)
print("KONIEC ANALIZY WSZYSTKICH FRAGMENTOW")
print("=" * 70)

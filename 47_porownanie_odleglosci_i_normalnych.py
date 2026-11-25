
import open3d as o3d
import numpy as np
import os

from scipy.spatial import cKDTree


# ============================================================
# USTAWIENIA
# ============================================================

folder_wejsciowy = "03_WYNIKI/45_WYKRYWANIE_POWIERZCHNI"

folder_wynikowy = "03_WYNIKI/47_ANALIZA_NORMALNYCH"

os.makedirs(folder_wynikowy, exist_ok=True)

eps = 0.16
min_punktow = 15
minimalny_rozmiar = 20

prog_odleglosci = 0.10

# Maksymalny kat miedzy normalna a kierunkiem
# laczacym najblizsze punkty
prog_kata = 30

prog_cos = np.cos(np.radians(prog_kata))


# ============================================================
# WCZYTYWANIE FRAGMENTOW I SKUPISK
# ============================================================

fragmenty = {}

print("=" * 80)
print("POROWNANIE ODLEGLOSCI I ORIENTACJI POWIERZCHNI")
print("=" * 80)

for numer in range(1, 7):

    sciezka = os.path.join(
        folder_wejsciowy,
        f"fragment_{numer:02d}_kandydaci.ply"
    )

    chmura = o3d.io.read_point_cloud(sciezka)

    if chmura.is_empty():
        print("Brak danych:", numer)
        continue

    # Normalne powinny byc zapisane w pliku z analizy 45.
    # Jesli ich nie ma, obliczamy je ponownie.

    if not chmura.has_normals():

        print(
            "Fragment", numer,
            "- brak zapisanych normalnych, obliczam ponownie."
        )

        chmura.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.20,
                max_nn=40
            )
        )

    punkty = np.asarray(chmura.points)
    normalne = np.asarray(chmura.normals)

    etykiety = np.array(
        chmura.cluster_dbscan(
            eps=eps,
            min_points=min_punktow,
            print_progress=False
        )
    )

    skupiska = []

    liczba_grup = (
        len(set(etykiety)) -
        (1 if -1 in etykiety else 0)
    )

    for etykieta in range(liczba_grup):

        indeksy = np.where(etykiety == etykieta)[0]

        if len(indeksy) < minimalny_rozmiar:
            continue

        skupiska.append({
            "numer": etykieta,
            "punkty": punkty[indeksy],
            "normalne": normalne[indeksy]
        })

    fragmenty[numer] = skupiska

    print(
        f"Fragment {numer}: "
        f"{len(skupiska)} skupisk"
    )


# ============================================================
# POROWNANIE DWÓCH SKUPISK
# ============================================================

def porownaj(a, b, normalne_a, normalne_b):

    drzewo_a = cKDTree(a)
    drzewo_b = cKDTree(b)

    # Najblizsze punkty A do B
    odleglosci_ab, indeksy_ab = drzewo_b.query(a, k=1)

    # Najblizsze punkty B do A
    odleglosci_ba, indeksy_ba = drzewo_a.query(b, k=1)

    # --------------------------------------------------------
    # KIERUNEK A -> B
    # --------------------------------------------------------

    punkty_b = b[indeksy_ab]

    wektory_ab = punkty_b - a

    dlugosci_ab = np.linalg.norm(
        wektory_ab,
        axis=1
    )

    dlugosci_ab = np.maximum(dlugosci_ab, 1e-12)

    kierunki_ab = wektory_ab / dlugosci_ab[:, None]

    norm_b = normalne_b[indeksy_ab]

    cos_a = np.abs(
        np.sum(normalne_a * kierunki_ab, axis=1)
    )

    cos_b = np.abs(
        np.sum(norm_b * kierunki_ab, axis=1)
    )

    cos_pary = np.abs(
        np.sum(normalne_a * norm_b, axis=1)
    )

    poprawne_ab = (
        (odleglosci_ab < prog_odleglosci)
        & (cos_a > prog_cos)
        & (cos_b > prog_cos)
        & (cos_pary > prog_cos)
    )

    # --------------------------------------------------------
    # KIERUNEK B -> A
    # --------------------------------------------------------

    punkty_a = a[indeksy_ba]

    wektory_ba = punkty_a - b

    dlugosci_ba = np.linalg.norm(
        wektory_ba,
        axis=1
    )

    dlugosci_ba = np.maximum(dlugosci_ba, 1e-12)

    kierunki_ba = wektory_ba / dlugosci_ba[:, None]

    norm_a = normalne_a[indeksy_ba]

    cos_b2 = np.abs(
        np.sum(normalne_b * kierunki_ba, axis=1)
    )

    cos_a2 = np.abs(
        np.sum(norm_a * kierunki_ba, axis=1)
    )

    cos_pary2 = np.abs(
        np.sum(normalne_b * norm_a, axis=1)
    )

    poprawne_ba = (
        (odleglosci_ba < prog_odleglosci)
        & (cos_b2 > prog_cos)
        & (cos_a2 > prog_cos)
        & (cos_pary2 > prog_cos)
    )

    procent_ab = 100 * np.mean(poprawne_ab)
    procent_ba = 100 * np.mean(poprawne_ba)

    # Zwykla bliskosc, bez sprawdzania normalnych
    bliskosc_ab = 100 * np.mean(
        odleglosci_ab < prog_odleglosci
    )

    bliskosc_ba = 100 * np.mean(
        odleglosci_ba < prog_odleglosci
    )

    ocena = min(procent_ab, procent_ba)

    mediana = np.median(
        np.concatenate([
            odleglosci_ab,
            odleglosci_ba
        ])
    )

    return {
        "bliskosc_ab": bliskosc_ab,
        "bliskosc_ba": bliskosc_ba,
        "zgodnosc_ab": procent_ab,
        "zgodnosc_ba": procent_ba,
        "ocena": ocena,
        "mediana": mediana
    }


# ============================================================
# POROWNANIE WSZYSTKICH SKUPISK
# ============================================================

wyniki = []

for numer_a in sorted(fragmenty):

    for numer_b in sorted(fragmenty):

        if numer_b <= numer_a:
            continue

        for skupisko_a in fragmenty[numer_a]:

            for skupisko_b in fragmenty[numer_b]:

                rezultat = porownaj(
                    skupisko_a["punkty"],
                    skupisko_b["punkty"],
                    skupisko_a["normalne"],
                    skupisko_b["normalne"]
                )

                wyniki.append({
                    "fragment_a": numer_a,
                    "fragment_b": numer_b,
                    "skupisko_a": skupisko_a["numer"],
                    "skupisko_b": skupisko_b["numer"],
                    "liczba_a": len(skupisko_a["punkty"]),
                    "liczba_b": len(skupisko_b["punkty"]),
                    "punkty_a": skupisko_a["punkty"],
                    "punkty_b": skupisko_b["punkty"],
                    **rezultat
                })


# ============================================================
# SORTOWANIE
# ============================================================

wyniki.sort(
    key=lambda x: x["ocena"],
    reverse=True
)


# ============================================================
# WYSWIETLENIE WYNIKOW
# ============================================================

print("\n" + "=" * 80)
print("NAJLEPSZE PARY PO UWZGLEDNIENIU NORMALNYCH")
print("=" * 80)

for miejsce, wynik in enumerate(wyniki[:30], start=1):

    print("\nMIEJSCE:", miejsce)

    print(
        "Fragmenty:",
        wynik["fragment_a"],
        "i",
        wynik["fragment_b"]
    )

    print(
        "Skupiska:",
        wynik["skupisko_a"],
        "i",
        wynik["skupisko_b"]
    )

    print(
        "Liczba punktow:",
        wynik["liczba_a"],
        "i",
        wynik["liczba_b"]
    )

    print(
        "Bliskosc A -> B:",
        round(wynik["bliskosc_ab"], 2), "%"
    )

    print(
        "Bliskosc B -> A:",
        round(wynik["bliskosc_ba"], 2), "%"
    )

    print(
        "Zgodnosc odleglosci i normalnych A -> B:",
        round(wynik["zgodnosc_ab"], 2), "%"
    )

    print(
        "Zgodnosc odleglosci i normalnych B -> A:",
        round(wynik["zgodnosc_ba"], 2), "%"
    )

    print(
        "Ocena symetryczna:",
        round(wynik["ocena"], 2), "%"
    )

    print(
        "Mediana odleglosci:",
        round(wynik["mediana"], 5)
    )


# ============================================================
# ZAPIS 10 NAJLEPSZYCH PAR
# ============================================================

print("\nZapisywanie najlepszych par...")

for miejsce, wynik in enumerate(wyniki[:10], start=1):

    chmura_a = o3d.geometry.PointCloud()
    chmura_b = o3d.geometry.PointCloud()

    chmura_a.points = o3d.utility.Vector3dVector(
        wynik["punkty_a"]
    )

    chmura_b.points = o3d.utility.Vector3dVector(
        wynik["punkty_b"]
    )

    chmura_a.paint_uniform_color([1, 0, 0])
    chmura_b.paint_uniform_color([0, 0.3, 1])

    polaczone = chmura_a + chmura_b

    nazwa = (
        f"para_{miejsce:02d}_"
        f"fragment_{wynik['fragment_a']}_"
        f"fragment_{wynik['fragment_b']}.ply"
    )

    o3d.io.write_point_cloud(
        os.path.join(folder_wynikowy, nazwa),
        polaczone
    )

print("\nZapisano najlepsze pary.")

print("\nKONIEC ANALIZY")

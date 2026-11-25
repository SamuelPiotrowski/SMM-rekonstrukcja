
import open3d as o3d
import numpy as np
import os

from scipy.spatial import cKDTree


# ============================================================
# USTAWIENIA
# ============================================================

folder_wejsciowy = "03_WYNIKI/45_WYKRYWANIE_POWIERZCHNI"

folder_wynikowy = "03_WYNIKI/46_POROWNANIE_POWIERZCHNI"

os.makedirs(folder_wynikowy, exist_ok=True)

# Parametry grupowania
eps = 0.16
min_punktow = 15

# Progi odleglosci
progi = [0.05, 0.10, 0.20]

# Pomijamy bardzo male skupiska
minimalny_rozmiar = 20


# ============================================================
# WCZYTANIE I GRUPOWANIE OBSZAROW
# ============================================================

fragmenty = {}

print("=" * 80)
print("POROWNYWANIE WYKRYTYCH OBSZAROW")
print("=" * 80)

for numer in range(1, 7):

    sciezka = os.path.join(
        folder_wejsciowy,
        f"fragment_{numer:02d}_kandydaci.ply"
    )

    chmura = o3d.io.read_point_cloud(sciezka)

    if chmura.is_empty():
        print("Brak danych dla fragmentu", numer)
        continue

    punkty = np.asarray(chmura.points)

    print("\nFragment", numer)
    print("Liczba kandydatow:", len(punkty))

    # Ponowne grupowanie punktow
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

        indeksy = np.where(
            etykiety == etykieta
        )[0]

        if len(indeksy) < minimalny_rozmiar:
            continue

        punkty_skupiska = punkty[indeksy]

        skupiska.append({
            "numer": etykieta,
            "punkty": punkty_skupiska
        })

    fragmenty[numer] = skupiska

    print("Wykryte skupiska:", liczba_grup)
    print("Skupiska do porownania:", len(skupiska))


# ============================================================
# POROWNYWANIE DWÓCH SKUPISK
# ============================================================

def porownaj_skupiska(a, b):

    drzewo_a = cKDTree(a)
    drzewo_b = cKDTree(b)

    odleglosci_ab, _ = drzewo_b.query(a, k=1)
    odleglosci_ba, _ = drzewo_a.query(b, k=1)

    wyniki = {}

    for prog in progi:

        bliskie_ab = np.sum(odleglosci_ab < prog)
        bliskie_ba = np.sum(odleglosci_ba < prog)

        procent_ab = 100 * bliskie_ab / len(a)
        procent_ba = 100 * bliskie_ba / len(b)

        wyniki[prog] = {
            "ab": procent_ab,
            "ba": procent_ba,
            "liczba_ab": bliskie_ab,
            "liczba_ba": bliskie_ba
        }

    mediana = np.median(
        np.concatenate([odleglosci_ab, odleglosci_ba])
    )

    return wyniki, mediana


# ============================================================
# POROWNANIE WSZYSTKICH SKUPISK
# ============================================================

wszystkie_wyniki = []

for numer_a in sorted(fragmenty):

    for numer_b in sorted(fragmenty):

        if numer_b <= numer_a:
            continue

        for skupisko_a in fragmenty[numer_a]:

            for skupisko_b in fragmenty[numer_b]:

                a = skupisko_a["punkty"]
                b = skupisko_b["punkty"]

                wyniki, mediana = porownaj_skupiska(a, b)

                wynik_005 = wyniki[0.05]
                wynik_010 = wyniki[0.10]
                wynik_020 = wyniki[0.20]

                # Wspolny wskaznik:
                # bierzemy slabszy z dwoch kierunkow.
                # Dzieki temu nie wystarczy, ze tylko jedna
                # mala czesc skupiska znajduje sie blisko drugiego.

                ocena = min(
                    wynik_010["ab"],
                    wynik_010["ba"]
                )

                wszystkie_wyniki.append({
                    "fragment_a": numer_a,
                    "fragment_b": numer_b,
                    "skupisko_a": skupisko_a["numer"],
                    "skupisko_b": skupisko_b["numer"],
                    "liczba_a": len(a),
                    "liczba_b": len(b),
                    "wyniki": wyniki,
                    "mediana": mediana,
                    "ocena": ocena,
                    "punkty_a": a,
                    "punkty_b": b
                })


# ============================================================
# SORTOWANIE WYNIKOW
# ============================================================

wszystkie_wyniki.sort(
    key=lambda x: (
        x["ocena"],
        -x["mediana"]
    ),
    reverse=True
)


# ============================================================
# WYSWIETLENIE NAJLEPSZYCH PAR
# ============================================================

print("\n" + "=" * 80)
print("NAJBARDZIEJ OBIECUJACE PARY SKUPISK")
print("=" * 80)

for i, wynik in enumerate(wszystkie_wyniki[:30], start=1):

    print("\nMIEJSCE:", i)

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

    for prog in progi:

        dane = wynik["wyniki"][prog]

        print(
            f"Odleglosc < {prog:.2f}: "
            f"{dane['ab']:.2f}% i {dane['ba']:.2f}%"
        )

    print(
        "Mediana odleglosci:",
        round(wynik["mediana"], 5)
    )

    print(
        "Ocena symetryczna przy 0.10:",
        round(wynik["ocena"], 2), "%"
    )


# ============================================================
# ZAPIS 10 NAJLEPSZYCH PAR DO PLIKOW PLY
# ============================================================

print("\nZapisywanie najlepszych par...")

for i, wynik in enumerate(wszystkie_wyniki[:10], start=1):

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
        f"para_{i:02d}_"
        f"fragment_{wynik['fragment_a']}_"
        f"fragment_{wynik['fragment_b']}.ply"
    )

    sciezka = os.path.join(
        folder_wynikowy,
        nazwa
    )

    o3d.io.write_point_cloud(
        sciezka,
        polaczone
    )

print("\nZapisano 10 najlepszych par.")

print("\nKONIEC ANALIZY")

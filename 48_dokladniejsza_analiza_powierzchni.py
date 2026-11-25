
import open3d as o3d
import numpy as np
import os

from scipy.spatial import cKDTree


# ============================================================
# USTAWIENIA
# ============================================================

folder_wejsciowy = "03_WYNIKI/45_WYKRYWANIE_POWIERZCHNI"

folder_wynikowy = "03_WYNIKI/48_DOKLADNIEJSZA_ANALIZA"

os.makedirs(folder_wynikowy, exist_ok=True)

eps = 0.16
min_punktow = 15
minimalny_rozmiar = 20

prog_odleglosci = 0.10
prog_kata = 30

prog_cos = np.cos(np.radians(prog_kata))


# ============================================================
# WCZYTYWANIE FRAGMENTOW
# ============================================================

fragmenty = {}

print("=" * 85)
print("DOKLADNIEJSZA ANALIZA BLISKOSCI I NORMALNYCH")
print("=" * 85)

for numer in range(1, 7):

    sciezka = os.path.join(
        folder_wejsciowy,
        f"fragment_{numer:02d}_kandydaci.ply"
    )

    chmura = o3d.io.read_point_cloud(sciezka)

    if chmura.is_empty():
        print("Brak danych:", numer)
        continue

    if not chmura.has_normals():

        print(
            "Fragment", numer,
            "- obliczam brakujace normalne."
        )

        chmura.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.20,
                max_nn=40
            )
        )

    punkty = np.asarray(chmura.points)
    normalne = np.asarray(chmura.normals)

    # Normalizacja normalnych
    dlugosci = np.linalg.norm(normalne, axis=1)

    poprawne = dlugosci > 1e-12

    normalne[poprawne] = (
        normalne[poprawne] /
        dlugosci[poprawne, None]
    )

    etykiety = np.array(
        chmura.cluster_dbscan(
            eps=eps,
            min_points=min_punktow,
            print_progress=False
        )
    )

    skupiska = []

    etykiety_grup = sorted(
        set(etykiety) - {-1}
    )

    for etykieta in etykiety_grup:

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
# ANALIZA W JEDNYM KIERUNKU
# ============================================================

def analizuj_kierunek(
    punkty_a,
    punkty_b,
    normalne_a,
    normalne_b
):

    drzewo_b = cKDTree(punkty_b)

    odleglosci, indeksy = drzewo_b.query(
        punkty_a,
        k=1
    )

    bliskie = odleglosci < prog_odleglosci

    procent_bliskich = 100 * np.mean(bliskie)

    # Analizujemy normalne tylko dla bliskich punktow.
    # Pomijamy punkty prawie identyczne, ponieważ
    # kierunek laczacy takie punkty jest niestabilny.

    poprawne_odleglosci = (
        bliskie &
        (odleglosci > 1e-8)
    )

    if np.sum(poprawne_odleglosci) == 0:

        return {
            "bliskosc": procent_bliskich,
            "rownoleglosc": 0,
            "zgodnosc_a": 0,
            "zgodnosc_b": 0,
            "zgodnosc_wszystko": 0,
            "mediana_bliskich": (
                np.median(odleglosci[bliskie])
                if np.any(bliskie)
                else np.nan
            ),
            "mediana_wszystkich": np.median(odleglosci)
        }

    indeksy_bliskie = indeksy[poprawne_odleglosci]

    wektory = (
        punkty_b[indeksy_bliskie] -
        punkty_a[poprawne_odleglosci]
    )

    dlugosci = np.linalg.norm(wektory, axis=1)

    kierunki = wektory / dlugosci[:, None]

    norm_a = normalne_a[poprawne_odleglosci]
    norm_b = normalne_b[indeksy_bliskie]

    # Zgodnosc normalnej z kierunkiem laczacym punkty

    cos_a = np.abs(
        np.sum(norm_a * kierunki, axis=1)
    )

    cos_b = np.abs(
        np.sum(norm_b * kierunki, axis=1)
    )

    # Rownoleglosc normalnych

    cos_pary = np.abs(
        np.sum(norm_a * norm_b, axis=1)
    )

    zgodna_a = cos_a > prog_cos
    zgodna_b = cos_b > prog_cos
    rownolegle = cos_pary > prog_cos

    wszystko = (
        zgodna_a &
        zgodna_b &
        rownolegle
    )

    liczba = len(poprawne_odleglosci[poprawne_odleglosci])

    # Procenty liczymy wsrod bliskich punktow,
    # dla ktorych kierunek laczacy jest okreslony.

    return {
        "bliskosc": procent_bliskich,

        "rownoleglosc": (
            100 * np.mean(rownolegle)
        ),

        "zgodnosc_a": (
            100 * np.mean(zgodna_a)
        ),

        "zgodnosc_b": (
            100 * np.mean(zgodna_b)
        ),

        "zgodnosc_wszystko": (
            100 * np.mean(wszystko)
        ),

        "mediana_bliskich": np.median(
            odleglosci[bliskie]
        ),

        "mediana_wszystkich": np.median(odleglosci)
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

                punkty_a = skupisko_a["punkty"]
                punkty_b = skupisko_b["punkty"]

                normalne_a = skupisko_a["normalne"]
                normalne_b = skupisko_b["normalne"]

                wynik_ab = analizuj_kierunek(
                    punkty_a,
                    punkty_b,
                    normalne_a,
                    normalne_b
                )

                wynik_ba = analizuj_kierunek(
                    punkty_b,
                    punkty_a,
                    normalne_b,
                    normalne_a
                )

                bliskosc_symetryczna = min(
                    wynik_ab["bliskosc"],
                    wynik_ba["bliskosc"]
                )

                zgodnosc_symetryczna = min(
                    wynik_ab["zgodnosc_wszystko"],
                    wynik_ba["zgodnosc_wszystko"]
                )

                wyniki.append({

                    "fragment_a": numer_a,
                    "fragment_b": numer_b,

                    "skupisko_a": skupisko_a["numer"],
                    "skupisko_b": skupisko_b["numer"],

                    "liczba_a": len(punkty_a),
                    "liczba_b": len(punkty_b),

                    "punkty_a": punkty_a,
                    "punkty_b": punkty_b,

                    "ab": wynik_ab,
                    "ba": wynik_ba,

                    "bliskosc_sym": bliskosc_symetryczna,
                    "zgodnosc_sym": zgodnosc_symetryczna
                })


# ============================================================
# SORTOWANIE
# ============================================================

wyniki.sort(
    key=lambda x: (
        x["bliskosc_sym"],
        x["zgodnosc_sym"]
    ),
    reverse=True
)


# ============================================================
# WYSWIETLANIE
# ============================================================

print("\n")
print("=" * 85)
print("NAJBLIZSZE PARY SKUPISK")
print("=" * 85)

for miejsce, wynik in enumerate(wyniki[:30], start=1):

    print("\n" + "-" * 70)

    print("MIEJSCE:", miejsce)

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
        round(wynik["ab"]["bliskosc"], 2), "%"
    )

    print(
        "Bliskosc B -> A:",
        round(wynik["ba"]["bliskosc"], 2), "%"
    )

    print(
        "Bliskosc symetryczna:",
        round(wynik["bliskosc_sym"], 2), "%"
    )

    print(
        "Rownoleglosc normalnych A -> B:",
        round(wynik["ab"]["rownoleglosc"], 2), "%"
    )

    print(
        "Rownoleglosc normalnych B -> A:",
        round(wynik["ba"]["rownoleglosc"], 2), "%"
    )

    print(
        "Zgodnosc wszystkich warunkow A -> B:",
        round(wynik["ab"]["zgodnosc_wszystko"], 2), "%"
    )

    print(
        "Zgodnosc wszystkich warunkow B -> A:",
        round(wynik["ba"]["zgodnosc_wszystko"], 2), "%"
    )

    print(
        "Zgodnosc symetryczna:",
        round(wynik["zgodnosc_sym"], 2), "%"
    )

    print(
        "Mediana bliskich odleglosci A -> B:",
        round(wynik["ab"]["mediana_bliskich"], 5)
        if not np.isnan(wynik["ab"]["mediana_bliskich"])
        else "brak"
    )

    print(
        "Mediana bliskich odleglosci B -> A:",
        round(wynik["ba"]["mediana_bliskich"], 5)
        if not np.isnan(wynik["ba"]["mediana_bliskich"])
        else "brak"
    )


# ============================================================
# ZAPIS NAJLEPSZYCH PAR
# ============================================================

print("\nZapisywanie 10 najlepszych par...")

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
        f"{wynik['fragment_b']}.ply"
    )

    o3d.io.write_point_cloud(
        os.path.join(folder_wynikowy, nazwa),
        polaczone
    )

print("\nZapisano najlepsze pary.")
print("Wyniki znajduja sie w:", folder_wynikowy)

print("\nKONIEC ANALIZY")

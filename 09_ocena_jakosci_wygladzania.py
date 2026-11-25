
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Wczytanie chmur punktow
oryginal = o3d.io.read_point_cloud(
    "02_FRAGMENTY_ZLAMANE/Vienna_Brick/PLY/brick_part01.ply"
)

wygladzona = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_wygladzony.ply"
)

# Sprawdzenie liczby punktow
print("Liczba punktow oryginalnych:", len(oryginal.points))
print("Liczba punktow wygladzonych:", len(wygladzona.points))


# Odleglosci: oryginalna -> wygladzona
odleglosci_oryginal = np.asarray(
    oryginal.compute_point_cloud_distance(wygladzona)
)

# Odleglosci: wygladzona -> oryginalna
odleglosci_wygladzona = np.asarray(
    wygladzona.compute_point_cloud_distance(oryginal)
)


# Funkcja obliczajaca statystyki
def pokaz_statystyki(nazwa, odleglosci):

    print("\n" + nazwa)
    print("Srednia odleglosc:", np.mean(odleglosci))
    print("Mediana:", np.median(odleglosci))
    print("95 percentyl:", np.percentile(odleglosci, 95))
    print("Maksymalna odleglosc:", np.max(odleglosci))


# Wyswietlenie wynikow
pokaz_statystyki(
    "ORYGINALNA -> WYGLADZONA",
    odleglosci_oryginal
)

pokaz_statystyki(
    "WYGLADZONA -> ORYGINALNA",
    odleglosci_wygladzona
)


# Wykres porownawczy
plt.figure(figsize=(10, 6))

plt.hist(
    odleglosci_oryginal,
    bins=50,
    alpha=0.5,
    label="Oryginalna -> wygladzona"
)

plt.hist(
    odleglosci_wygladzona,
    bins=50,
    alpha=0.5,
    label="Wygladzona -> oryginalna"
)

plt.xlabel("Odleglosc miedzy punktami")
plt.ylabel("Liczba punktow")
plt.title("Odchylenie geometrii po wygladzaniu")
plt.legend()

plt.tight_layout()
plt.savefig("03_WYNIKI/ocena_jakosci_wygladzania.png")
plt.show()


import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Wczytanie odszumionej chmury
plik = Path("03_WYNIKI/brick_part01_odszumiony.ply")
chmura = o3d.io.read_point_cloud(str(plik))

punkty = np.asarray(chmura.points)

# Pobranie wspolrzednych
X = punkty[:, 0]
Y = punkty[:, 1]
Z = punkty[:, 2]

# Podzial na 30 warstw wzdluz osi Y
liczba_warstw = 30

granice = np.linspace(
    np.min(Y),
    np.max(Y),
    liczba_warstw + 1
)

# Listy na wyniki
liczba_punktow = []
szerokosci = []
glebokosci = []
pola = []
srodki_warstw = []

for i in range(liczba_warstw):

    # Wybor punktow z aktualnej warstwy
    if i == liczba_warstw - 1:
        maska = (Y >= granice[i]) & (Y <= granice[i + 1])
    else:
        maska = (Y >= granice[i]) & (Y < granice[i + 1])

    warstwa = punkty[maska]

    srodki_warstw.append(
        (granice[i] + granice[i + 1]) / 2
    )

    liczba_punktow.append(len(warstwa))

    if len(warstwa) > 0:

        # Rozmiar warstwy w osiach X i Z
        szerokosc = np.max(warstwa[:, 0]) - np.min(warstwa[:, 0])
        glebokosc = np.max(warstwa[:, 2]) - np.min(warstwa[:, 2])

        szerokosci.append(szerokosc)
        glebokosci.append(glebokosc)

        pola.append(szerokosc * glebokosc)

    else:
        szerokosci.append(0)
        glebokosci.append(0)
        pola.append(0)

# Wyswietlenie wynikow
print("\nANALIZA PRZEKROJOW")
print("-" * 75)

print("Warstwa | Liczba punktow | Szerokosc | Glebokosc | Pole")

for i in range(liczba_warstw):
    print(
        f"{i+1:7} | "
        f"{liczba_punktow[i]:14} | "
        f"{szerokosci[i]:9.3f} | "
        f"{glebokosci[i]:9.3f} | "
        f"{pola[i]:8.3f}"
    )

# Wizualizacja zmian
fig, wykresy = plt.subplots(3, 1, figsize=(11, 12))

wykresy[0].plot(srodki_warstw, liczba_punktow, marker="o")
wykresy[0].set_title("Liczba punktow w warstwach")
wykresy[0].set_ylabel("Liczba punktow")
wykresy[0].grid()

wykresy[1].plot(srodki_warstw, szerokosci, marker="o", label="X")
wykresy[1].plot(srodki_warstw, glebokosci, marker="o", label="Z")
wykresy[1].set_title("Zmiana wymiarow przekrojow")
wykresy[1].set_ylabel("Rozmiar")
wykresy[1].legend()
wykresy[1].grid()

wykresy[2].plot(srodki_warstw, pola, marker="o")
wykresy[2].set_title("Pole prostokata obejmujacego przekroj")
wykresy[2].set_xlabel("Wspolrzedna Y")
wykresy[2].set_ylabel("Pole")
wykresy[2].grid()

plt.tight_layout()
plt.show()

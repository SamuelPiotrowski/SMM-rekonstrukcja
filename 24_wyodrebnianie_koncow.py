
import open3d as o3d
import numpy as np
from pathlib import Path

# Wczytanie odszumionej chmury
chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_odszumiony.ply"
)

punkty = np.asarray(chmura.points)

# Tolerancja
prog = 0.05

# Wspolczynniki dolnej plaszczyzny
plaszczyzna_1 = np.array([
    -0.00545321689,
    0.999984897,
    0.000683893682,
    1.15647083
])

# Wspolczynniki gornej plaszczyzny
plaszczyzna_2 = np.array([
    -0.00321763,
    0.99998938,
    0.00330044,
    -1.85629718
])

# Odleglosci od dolnej plaszczyzny
a, b, c, d = plaszczyzna_1

odleglosci_1 = np.abs(
    a * punkty[:, 0] +
    b * punkty[:, 1] +
    c * punkty[:, 2] + d
) / np.linalg.norm([a, b, c])

# Wyznaczenie gornej czesci
maksimum_Y = np.max(punkty[:, 1])
minimum_Y = np.min(punkty[:, 1])

granica = maksimum_Y - (
    maksimum_Y - minimum_Y
) * 0.2

indeksy_gora = np.where(punkty[:, 1] > granica)[0]
punkty_gora = punkty[indeksy_gora]

# Odleglosci od gornej plaszczyzny
a, b, c, d = plaszczyzna_2

odleglosci_2 = np.abs(
    a * punkty_gora[:, 0] +
    b * punkty_gora[:, 1] +
    c * punkty_gora[:, 2] + d
) / np.linalg.norm([a, b, c])

# Wybor punktow obu koncow
maska_dol = odleglosci_1 <= prog
maska_gora_lokalna = odleglosci_2 <= prog

# Zamiana lokalnych indeksow na indeksy calej chmury
maska_gora = np.zeros(len(punkty), dtype=bool)
maska_gora[indeksy_gora[maska_gora_lokalna]] = True

# Zabezpieczenie przed ewentualnym nakladaniem sie masek
maska_gora = maska_gora & ~maska_dol

# Pozostale punkty
maska_reszta = ~(maska_dol | maska_gora)

# Utworzenie trzech chmur
dol = chmura.select_by_index(np.where(maska_dol)[0])
gora = chmura.select_by_index(np.where(maska_gora)[0])
reszta = chmura.select_by_index(np.where(maska_reszta)[0])

print("DOLNY KONIEC:", len(dol.points))
print("GORNY KONIEC:", len(gora.points))
print("POZOSTALE PUNKTY:", len(reszta.points))

# Folder na wyniki
folder = Path("03_WYNIKI")
folder.mkdir(exist_ok=True)

# Zapis plikow
o3d.io.write_point_cloud(
    str(folder / "brick_part01_dolny_koniec.ply"),
    dol
)

o3d.io.write_point_cloud(
    str(folder / "brick_part01_gorny_koniec.ply"),
    gora
)

o3d.io.write_point_cloud(
    str(folder / "brick_part01_pozostale_punkty.ply"),
    reszta
)

print("\nZapisano trzy pliki PLY.")

# Kolorowanie do wspolnej wizualizacji
dol.paint_uniform_color([1, 0.1, 0.1])
gora.paint_uniform_color([0.1, 0.2, 1])
reszta.paint_uniform_color([0.7, 0.7, 0.7])

o3d.visualization.draw_geometries(
    [dol, gora, reszta],
    window_name="Podzial fragmentu kosci",
    width=1200,
    height=800
)

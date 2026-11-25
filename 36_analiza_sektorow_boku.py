
import open3d as o3d
import numpy as np

chmura = o3d.io.read_point_cloud(
    "03_WYNIKI/brick_part01_pozostale_punkty.ply"
)

punkty = np.asarray(chmura.points)

x = punkty[:, 0]
y = punkty[:, 1]
z = punkty[:, 2]

# Granice wysokosci
y_min = np.min(y)
y_max = np.max(y)

dol = y_min + 0.05 * (y_max - y_min)
gora = y_max - 0.05 * (y_max - y_min)

granice = np.linspace(dol, gora, 13)

# Srodkowy obszar boku
maska = (y >= granice[1]) & (y < granice[11])

# Srodek klocka w plaszczyznie XZ
srodek_x = np.median(x)
srodek_z = np.median(z)

# Kat polozenia punktow wokol osi Y
katy = np.degrees(
    np.arctan2(z - srodek_z, x - srodek_x)
)

# Podzial na 12 sektorow
granice_katow = np.linspace(-180, 180, 13)

print("\nANALIZA SEKTOROW BOKU")
print("-" * 75)
print("Sektor | Liczba punktow | Min Y | Max Y")

for i in range(12):

    maska_sektora = (
        maska &
        (katy >= granice_katow[i]) &
        (katy < granice_katow[i + 1])
    )

    punkty_sektora = punkty[maska_sektora]

    if len(punkty_sektora) == 0:
        print(i + 1, "| BRAK PUNKTOW")
        continue

    print(
        f"{i + 1:6} | "
        f"{len(punkty_sektora):14} | "
        f"{np.min(punkty_sektora[:, 1]):.4f} | "
        f"{np.max(punkty_sektora[:, 1]):.4f}"
    )

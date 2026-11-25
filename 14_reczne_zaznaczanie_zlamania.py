
import open3d as o3d
from pathlib import Path

# Wczytanie odszumionego fragmentu
plik = Path("03_WYNIKI/brick_part01_odszumiony.ply")

chmura = o3d.io.read_point_cloud(str(plik))
# Ustawienie jednolitego koloru punktow
chmura.paint_uniform_color([0.7, 0.7, 0.7])

print("Liczba punktow:", len(chmura.points))

# Reczne zaznaczanie punktow powierzchni zlamania.\.venv\Scripts\python.exe 14_reczne_zaznaczanie_zlamania.py
print("\nZAZNACZANIE POWIERZCHNI ZLAMANIA")
print("Shift + lewy przycisk myszy - zaznaczenie punktu")
print("Q - zakonczenie zaznaczania")

wizualizacja = o3d.visualization.VisualizerWithEditing()

wizualizacja.create_window(
    window_name="Zaznaczanie powierzchni zlamania",
    width=1200,
    height=800
)

wizualizacja.add_geometry(chmura)
wizualizacja.run()

# Pobranie indeksow zaznaczonych punktow
indeksy = wizualizacja.get_picked_points()

wizualizacja.destroy_window()

print("\nLiczba zaznaczonych punktow:", len(indeksy))

# Sprawdzenie, czy zaznaczono jakiekolwiek punkty
if len(indeksy) == 0:
    print("Nie zaznaczono zadnych punktow.")
else:

    # Wyodrebnienie zaznaczonych punktow
    powierzchnia = chmura.select_by_index(indeksy)

    # Kolorowanie zaznaczonej powierzchni
    powierzchnia.paint_uniform_color([1.0, 0.0, 0.0])

    # Przygotowanie folderu na wyniki
    folder = Path("03_WYNIKI")
    folder.mkdir(exist_ok=True)

    # Zapis powierzchni do osobnego pliku PLY
    wynik = folder / "brick_part01_powierzchnia_zlamania.ply"

    o3d.io.write_point_cloud(
        str(wynik),
        powierzchnia
    )

    print("Zapisano powierzchnie zlamania:", wynik)

    # Wizualizacja wyniku
    chmura_wizualizacja = chmura.select_by_index(
        indeksy,
        invert=True
    )

    chmura_wizualizacja.paint_uniform_color([0.7, 0.7, 0.7])

    o3d.visualization.draw_geometries(
        [chmura_wizualizacja, powierzchnia],
        window_name="Wyodrebniona powierzchnia zlamania",
        width=1200,
        height=800
    )

"""
Dopasowywanie fragmentów kości po POWIERZCHNIACH STYKU (podproblem 6 z README.md).

Bierze tylko punkty powierzchni styku (przełomów), każdą powierzchnię osobno (kość może
mieć kilka), i dopasowuje każdą do każdej powierzchni INNEJ kości.

SKĄD POWIERZCHNIE (SurfaceMatchParams.source, settings.json "matching" -> "source"):
    "manual"      - tylko ręczne oznaczenia contacts/<zestaw>/<id>.json; kości bez nich są pomijane,
    "auto"        - tylko wynik algorytmu (settings.json "contacts" -> "method" i jego ustawienia):
                    z contacts/<zestaw>/auto/<id>.json, a gdy pliku brak albo jest nieaktualny (inny
                    algorytm, inne ustawienia, zmieniona siatka) - liczony i zapisywany
                    (contacts.auto_contacts),
    "manual-auto" - ręczne, a dla kości bez ręcznych - z algorytmu jak w "auto".
    Powierzchnie mniejsze niż `min_area_au2` są pomijane.

DLA KAŻDEJ PARY POWIERZCHNI (A z jednej kości, B z innej):
1. PRÓBKOWANIE: równomierne punkty co `sample_au` z normalnymi (na zewnątrz kości).
2. ODWRÓCENIE NORMALNYCH A. Dwa przełomy stykają się "twarzą w twarz": wypukłość na A
   to wklęsłość na B, a ich normalne są przeciwne. Po odwróceniu normalnych A wygląda
   dla deskryptorów jak B, więc działa ta sama rejestracja co przy łączeniu skanów.
3. ZGRUBNE USTAWIENIE - RANSAC na deskryptorach FPFH (punkty co `coarse_voxel_au`,
   promień `fpfh_radius_au`): losuje trójki podobnych punktów, odrzuca ustawienia
   z normalnymi różnymi o więcej niż `ransac_normal_angle_deg`, zostawia to, z którym
   zgadza się najwięcej punktów. Zawsze wszystkie `ransac_iterations` (open3d liczy
   wielowątkowo, więc wynik jest losowy - mało iteracji = często złe ustawienie).
4. DOKŁADNE USTAWIENIE - ICP point-to-plane: pary punktów szukane do `icp_start_au`,
   potem do połowy tego, na koniec do 2 x `sample_au`.
   Kroki 3-4 są powtarzane `ransac_seeds` razy, wygrywa ustawienie z najlepszym wynikiem.
5. OCENA ustawienia T (A -> B):
   - styk [a.u.²]:    punkty A bliżej niż `tolerance_au` od B, z normalnymi przeciwnymi
                    z dokładnością do `normal_angle_deg` (i odwrotnie, B na A);
                    pole styku = mniejsze z pól obu stron (część pokryta x pole powierzchni),
   - RMSE [a.u.]:     średnia kwadratowa odległość punktów A w styku od B,
   - kolizja [a.u.²]: pole CAŁEJ kości A, które leży pod powierzchnią kości B głębiej niż
                    `collision_au` (sprawdzane do `collision_search_au` od B), i odwrotnie -
                    prawdziwe dopasowanie przełomów nie może wbijać jednej kości w drugą,
   - WYNIK (score_mode):
       "dice" (domyślny) = 100 x (styk - kolizja) / średnie pole obu przełomów [%] - jaka część obu
                przełomów do siebie przylega; nieprzylegająca część przełomu obniża wynik, a wielkość
                fragmentów się skraca (bez tego duże przełomy wygrywały samym polem),
       "area" = styk - kolizja [a.u.²] (dawny wynik, rośnie z wielkością przełomów).
     Pary są sortowane od najlepszego wyniku.

PŁASKIE CIĘCIA (flat_max_rms_au): przełom jest nierówny, a koniec spiłowany albo przecięty piłą jest
prawie płaski. Każda powierzchnia jest przybliżana płaszczyzną (Surface.fit, contacts.fit_surface);
powierzchnia, której punkty odchylają się od płaszczyzny średnio mniej niż flat_max_rms_au, nie jest
dopasowywana - dwa płaskie cięcia pasują do siebie zawsze (płaszczyzna na płaszczyźnie), więc
wygrywałyby z prawdziwymi przełomami.

TRYB "plane" (mode, do testów): zamiast RANSAC + ICP na punktach dopasowywane są same przybliżenia:
płaszczyzna A na płaszczyźnie B (normalne przeciwne, środki razem), obrót wokół normalnej co
360° / plane_steps. Styk = wspólna część obrysów obu powierzchni rzutowanych na płaszczyznę B (siatka
co 2 x sample_au); ocena, osie, kolizja i premia jak w trybie "points".

OŚ JAKO WARUNEK SZUKANIA (axis_constrain, match_long_constrained): zwykły RANSAC i ICP szukają bez osi, a oś
tylko ocenia gotowe ustawienie - gdy ICP z każdego startu zjedzie pod kątem, ocena wybiera najmniej złe. Dla pary
kości długich KAŻDY start (siatka ustawień na wprost: obrót co constrain_twist_deg x przesunięcie co
constrain_step_au, oceniana szybko - bliskość przełomów i kolizja; płaszczyzny; RANSAC tylko przy
constrain_ransac) jest prostowany i dociskany ICP, który może tylko obracać wokół wspólnej osi i przesuwać
(axis_locked_icp); potem mały zwykły ICP (pary punktów do 2 x sample_au), przyjęty tylko, gdy kąt osi nie
przekroczy axis_max_deg. Wygrywa najlepszy wynik.

OŚ PRZY PRZEŁOMIE i PROSTOWANIE (axis_lock): oś do warunku "osie w jednej linii" jest liczona z części
fragmentu przy przełomie (LOCAL_AXIS_FRAC długości, co najmniej LOCAL_AXIS_MIN_AU) - głowa kości i wygięcie
strzałki przekrzywiają oś całego fragmentu. Najlepsze ustawienie pary kości długich jest prostowane: oś A
obrócona na oś B wokół miejsca styku, potem ICP point-to-plane, w którym wolno tylko obracać wokół wspólnej
osi i przesuwać (axis_locked_icp) - wygrywa, jeśli ma lepszy wynik.

STARTY PŁASZCZYZNA NA PŁASZCZYŹNIE (plane_starts, jak w OSTEOFORM): środek przybliżenia A na środku B,
normalne przeciwne, obrót wokół normalnej co 360° / plane_starts; plane_starts_keep najlepszych wg styku
z luźną tolerancją idzie do ICP (obok startów wzdłuż osi, z atlasu i RANSAC).

KOŚCI DŁUGIE, TRYB "gap" (long_mode): przełomy dwóch kawałków trzonu rzadko przylegają ściśle (szczelina,
rozkruszone brzegi), więc ścisły styk (tolerance_au) nie odróżnia prawdziwej pary od przypadkowej. Dla pary
wydłużonych fragmentów ustawienie ma tylko dwa stopnie swobody: obrót wokół wspólnej osi i przesunięcie wzdłuż
niej (osie przy przełomach w jednej linii, A przedłuża B koniec w koniec). Ocena (long_terms):
    - BLISKOŚĆ przełomów: część punktów przełomu A bliżej przełomu B niż gap_au (i nie głębiej w B niż
      collision_au) i odwrotnie - jak styk, ale ze szczeliną: min(pokrycie A x pole A, pokrycie B x pole B) / średnie pole,
    - CIĄGŁOŚĆ KORY: kora A tuż przy przełomie (do rim_au od niego) ma leżeć na przedłużeniu kory B przy jej
      przełomie: punkt A bliżej płaszczyzny stycznej najbliższego punktu kory B niż continuity_au, normalne
      zgodne (do 35°) - wyrównuje też obrót wokół osi (grzebienie trzonu); w obie strony,
    - KOLIZJA całych kości jak zwykle.
    WYNIK = (0.5 x bliskość + 0.5 x ciągłość) x waga osi - kolizja (jednostki jak score_mode).
Szukanie (match_long_pair): siatka obrót co gap_twist_deg x przesunięcie w gap_shift_au co gap_step_au, potem
dokładniej (1/3 kroku) wokół 3 najlepszych, na koniec prostowanie (axis_locked_icp) - wygrywa lepszy wynik.

JAKOŚĆ (quality_factor, jak w OSTEOFORM): styk w wyniku x 1 / (1 + RMSE / tolerance_au) x exp(-kąt / 45°),
kąt = mediana odchylenia normalnych w styku od idealnie przeciwnych - ciasne dopasowanie wygrywa z luźnym.

PROPONOWANE ZŁOŻENIE (assemble): para wchodzi, gdy wynik >= assembly_min_score, pokrycie każdej strony
>= assembly_min_cover i kolizja <= assembly_max_collision x styk; każdy przełom najwyżej raz. Przy
assembly_beam > 0 (jak w OSTEOFORM) składane są CAŁE GRUPY: wiązka assembly_beam najlepszych częściowych
złożeń, dołączana grupa przenoszona sztywno, a połączenie odrzucane, gdy którekolwiek dwa fragmenty
łączonych grup wchodzą w siebie (część punktów głębiej niż collision_au) więcej niż assembly_max_penetration.
Wygrywa złożenie o największej sumie wyników. Słabsze pary nie są wciskane na siłę: przełom bez pary to
miejsce, gdzie może brakować kawałka pomiędzy (albo to nie przełom).

ATLAS ANATOMICZNY (use_atlas, settings.json "matching"; src/atlas.py, cache out/3d/<zestaw>/cache/atlas/):
    Dla każdej kości z powierzchniami atlas mówi, z której kompletnej kości z bazy pochodzi fragment
    (prawdopodobieństwa, też "żadna z bazy") i jak na niej leży (T: model -> fragment, ze skalą).
    "Ta sama kość" = ten sam model i ta sama strona (odbicie lustrzane albo nie); modele w jednym
    podfolderze bazy (typ kości) liczą się razem, bez strony - mogą być z różnych stron ciała.
    Z tego (atlas_relation): P(ta sama kość) = suma p_A x p_B po kościach, P(różne) = reszta znanych;
    "żadna z bazy" nie liczy się do żadnej, więc niepewny fragment nie wyklucza pary.
    a) POMIJANIE PAR: P(różne kości) >= atlas_skip_different -> para nie jest dopasowywana
       (przełomy różnych kości nie mogą do siebie pasować).
    b) START Z ATLASU (atlas_starts): gdy A i B leżą na tym samym modelu, ich położenie względem
       siebie wynika z ustawień na modelu: x_B = T_B T_A⁻¹ x_A (bez różnicy skal - sam obrót i
       przesunięcie zgodne w środku powierzchni A). Z tego startu ICP (pierwszy promień
       start_icp_au, dalej jak w kroku 4). Gdy najlepszy start pokrył obie powierzchnie co
       najmniej w start_enough_cover, RANSAC (krok 3) jest pomijany - inaczej liczony jak zwykle.
    c) PREMIA: styk w wyniku x (1 + atlas_boost x P(ta sama kość)); para bez styku nic nie zyskuje.

OSIE W JEDNEJ LINII (axis_check, settings.json "matching"): kawałki trzonu kości długiej łączą się
koniec do końca, wzdłuż jednej osi - nie pod kątem i nie koniec do boku. Dla każdej kości: oś główna
(PCA wierzchołków), wydłużenie (odchylenie wzdłuż osi / w poprzek), promień (mediana odległości od
osi) i miejsce powierzchni styku na osi (-1 / +1 = końce, 0 = środek boku). Gdy OBA fragmenty są
wydłużone (>= axis_min_elongation):
    a) OCENA (axis_fit): kąt między osiami po ustawieniu T i rozsunięcie osi w miejscu styku (odległość
       najbliższych punktów obu osi od styku, w promieniach kości). Waga = 1 do axis_max_deg /
       axis_max_offset, potem liniowo do 0 przy dwukrotności; styk w wyniku x waga.
    b) STARTY WZDŁUŻ OSI (axis_starts_for): oś A na osi B, koniec A z przełomem na końcu B z przełomem
       (A przedłuża B), przełom A na przełomie B, obrót wokół osi co 360° / axis_starts; z każdego
       startu ICP (pierwszy promień start_icp_au). Powierzchnia w środku boku (|miejsce| < 0.3) -
       obie strony.
    Gdy najlepszy start (z osi albo z atlasu) pokrył obie powierzchnie co najmniej w start_enough_cover
    przy wadze osi >= 0.5, RANSAC jest pomijany.

Użycie: scripts/match_surfaces.py liczy i zapisuje out/3d/<zestaw>/matching.json,
scripts/tune_matching.py (src/match_tool.py) go pokazuje: suwaki ustawień z przeliczaniem
jednej pary albo (--html) statyczna strona out/3d/<zestaw>/matching.html.
W kodzie:
    surfaces, info = extract_surfaces(load_bones(features=True), params, contact_params)
    results = match_all(surfaces, params)     # [{"a", "b", "score", "contact_au2", "T", ...}, ...]
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from bones import Bone
from contacts import (ContactParams, Contacts, auto_contacts, bone_axis, fit_surface, is_excluded, load_contacts,
                      scan_frames, triangle_labels)
from data_parsing import compute_fpfh, sample_surface, submesh, to_cloud, voxel_keep
from merging import covered, transform_points

log = logging.getLogger(__name__)
reg = o3d.pipelines.registration

SOURCES = ("manual", "auto", "manual-auto")
SCORE_MODES = ("dice", "area")             # wynik: udział styku w obu przełomach [%] albo pole [a.u.²]
MODES = ("points", "plane")                # dopasowanie: punkty przełomów (RANSAC + ICP) albo przybliżenia płaszczyzną
LOCAL_AXIS_FRAC = 0.4                      # oś przy przełomie: z części fragmentu +- tyle długości od przełomu ...
LOCAL_AXIS_MIN_AU = 15.0                   # ... ale co najmniej tyle a.u.
RIM_MIN_AU = 1.0                           # "gap": kora przy przełomie od tylu a.u. od niego (krawędź ma przekrzywione normalne)
RIM_NEIGHBOURS = 8                         # "gap": ciągłość kory - sprawdzanych najbliższych punktów kory drugiej strony
# Opisy trybów "source" do komunikatów i stron (match_surfaces.py, match_tool).
SOURCE_NAMES = {"manual": "tylko ręczne", "auto": "tylko z algorytmu",
                "manual-auto": "ręczne, a gdy brak - z algorytmu"}


@dataclass(frozen=True)
class SurfaceMatchParams:
    """Ustawienia dopasowania powierzchni styku (settings.json, sekcja "matching" - tam pełne opisy).

    Jak każde ustawienie wpływa na algorytm: kroki 1-5 w docstringu modułu.
    """

    source: str = "manual"                 # "manual" / "auto" / "manual-auto"
    score_mode: str = "dice"               # wynik: "dice" = udział styku w obu przełomach [%], "area" = pole [a.u.²]
    mode: str = "points"                   # "points" = RANSAC + ICP na punktach; "plane" = przybliżenia płaszczyzną (test)
    plane_steps: int = 36                  # tryb "plane": obroty wokół normalnej co 360° / tyle
    flat_max_rms_au: float = 0.0           # płaskie cięcie (odchylenie od płaszczyzny < tyle) - pomijane; 0 = bez [a.u.]
    assembly_min_score: float = 0.0        # proponowane złożenie: tylko pary z wynikiem >= tyle
    assembly_min_cover: float = 0.0        # ... i pokryciem każdej strony >= tyle
    assembly_max_collision: float = 0.0    # ... i kolizją <= tyle x styk; 0 = bez
    assembly_beam: int = 0                 # składanie całych grup: tyle częściowych złożeń (wiązka); 0 = zachłannie
    assembly_max_penetration: float = 0.06  # ... odrzuć, gdy dwa fragmenty grupy wchodzą w siebie bardziej (część punktów)
    plane_starts: int = 0                  # "points": starty płaszczyzna na płaszczyźnie, obroty co 360° / tyle; 0 = bez
    plane_starts_keep: int = 3             # ... tyle najlepszych idzie do ICP
    axis_lock: bool = False                # kości długie: wyprostuj najlepsze ustawienie (osie w linii) i dociśnij ICP
    axis_constrain: bool = False           # kości długie: oś jako WARUNEK szukania (każdy start prostowany, ICP wokół osi)
    constrain_grid_keep: int = 6           # ... z siatki ustawień na wprost tyle najlepszych idzie do ICP
    constrain_twist_deg: float = 15.0      # ... siatka: obrót wokół osi co tyle [stopnie]
    constrain_step_au: float = 3.0         # ... siatka: przesunięcie wzdłuż osi co tyle [a.u.]
    constrain_ransac: bool = False         # ... dodatkowo starty z RANSAC (szuka bez osi, wolne)
    quality_factor: bool = False           # styk x 1 / (1 + RMSE / tolerance_au) x exp(-kąt normalnych / 45°)
    max_area_ratio: float = 0.0            # pomiń pary, których pola przełomów różnią się więcej niż tyle razy; 0 = bez
    long_mode: str = "contact"             # kości długie: "contact" = ścisły styk, "gap" = bliskość + ciągłość kory
    gap_au: float = 2.5                    # "gap": przełomy bliżej niż to = blisko [a.u.]
    rim_au: float = 6.0                    # "gap": kora przy przełomie = do tylu a.u. od niego
    continuity_au: float = 0.8             # "gap": kora A na przedłużeniu kory B = bliżej jej płaszczyzny niż to [a.u.]
    gap_twist_deg: float = 15.0            # "gap": krok obrotu wokół osi w szukaniu [stopnie]
    gap_shift_au: tuple = (-30.0, 15.0)    # "gap": przesunięcie wzdłuż osi (ujemne = końce zachodzą na siebie) [a.u.]
    gap_step_au: float = 2.0               # "gap": krok przesunięcia [a.u.]
    min_area_au2: float = 20.0             # mniejsze powierzchnie styku są pomijane [a.u.²]
    sample_au: float = 0.3                 # odstęp punktów powierzchni [a.u.]
    coarse_voxel_au: float = 1.0           # odstęp punktów zgrubnego ustawienia (RANSAC) [a.u.]
    fpfh_radius_au: float = 4.0            # promień deskryptora FPFH [a.u.]
    ransac_iterations: int = 1000000       # zawsze wszystkie (bez wczesnego końca)
    ransac_seeds: int = 2                  # niezależnych prób RANSAC + ICP na parę
    ransac_normal_angle_deg: float = 35.0  # RANSAC: maks. różnica normalnych par punktów [stopnie]
    icp_start_au: float = 2.0              # ICP: pierwszy promień szukania par punktów [a.u.]
    tolerance_au: float = 0.5              # styk: punkty bliżej niż to [a.u.] ...
    normal_angle_deg: float = 45.0         # ... z normalnymi przeciwnymi z tą dokładnością [stopnie]
    collision_au: float = 1.0              # kolizja: głębiej niż to pod powierzchnią drugiej kości [a.u.]
    collision_search_au: float = 4.0       # ... sprawdzane do tej odległości od drugiej kości [a.u.]
    max_shown: int = 15                    # ile najlepszych par pokazać w 3D
    # atlas anatomiczny (docstring modułu, "ATLAS ANATOMICZNY")
    use_atlas: bool = False                # kontekst z atlasu: pomijanie par, start z atlasu, premia
    atlas_skip_different: float = 0.95     # pomiń parę, gdy P(różne kości) >= tyle; > 1 = nie pomijaj
    atlas_starts: int = 2                  # najwięcej startów z atlasu na parę (wspólne modele); 0 = bez
    atlas_boost: float = 0.25              # styk liczy się do tylu razy mocniej przy P(ta sama kość) = 1; 0 = bez
    # kości długie: osie w jednej linii (docstring modułu, "OSIE W JEDNEJ LINII")
    axis_check: bool = False               # ocena i starty wzdłuż osi dla wydłużonych fragmentów
    axis_min_elongation: float = 2.5       # fragment wydłużony = długość / szerokość (odchylenia PCA) >= tyle
    axis_max_deg: float = 20.0             # osie rozchodzą się do tylu stopni bez kary (2x tyle = styk się nie liczy)
    axis_max_offset: float = 1.0           # osie rozsunięte w miejscu styku do tylu promieni kości bez kary (2x = 0)
    axis_starts: int = 12                  # startów wzdłuż osi (obroty wokół osi co 360°/tyle); 0 = bez
    # starty z atlasu i wzdłuż osi
    start_icp_au: float = 6.0              # ICP z takiego startu: pierwszy promień szukania par [a.u.]
    start_enough_cover: float = 0.35       # start pokrył obie powierzchnie w tylu (i osie się zgadzają) -> bez RANSAC


# -----------------
# POWIERZCHNIE
# -----------------

@dataclass
class Surface:
    """Jedna powierzchnia styku jednej kości, w układzie tej kości (jak out/3d/<zestaw>/bones/<id>/mesh.ply).

    Tworzona przez extract_surfaces. Punkty są równomierną próbką samej powierzchni
    styku (co sample_au), a bone_tree - punktami całej kości (do liczenia kolizji).
    """

    bone: Bone
    contacts: Contacts
    sid: int                       # numer powierzchni w etykietach kości
    points: np.ndarray             # (N, 3)
    normals: np.ndarray            # (N, 3), na zewnątrz kości
    area_au2: float                # pole powierzchni (z siatki, nie z punktów)
    bone_tree: cKDTree | None = None   # punkty CAŁEJ kości (do kolizji), wspólne dla powierzchni kości
    atlas: dict | None = None          # z której kości z bazy pochodzi fragment (atlas_context); None = bez atlasu
    _cache: dict = field(default_factory=dict, repr=False)   # chmury i KD-drzewo (clouds, tree)

    @property
    def key(self) -> str:
        """Unikalny klucz "<id kości>#<numer powierzchni>" - tak zapisany w matching.json."""
        return f"{self.bone.id}#{self.sid}"

    @property
    def name(self) -> str:
        """Czytelna nazwa "<nazwa kości> · <nazwa powierzchni>"."""
        return f"{self.bone.label} · {self.contacts.name(self.sid) if self.contacts else f'Powierzchnia {self.sid}'}"

    @property
    def is_manual(self) -> bool:
        """True = powierzchnia oznaczona ręcznie, False = z algorytmu."""
        return self.contacts.source == "manual"

    def clouds(self, p: SurfaceMatchParams, flip: bool):
        """Chmury punktów powierzchni do rejestracji - liczone raz dla każdej wartości `flip`.

        Pamięć podręczna pamięta ustawienia, z którymi chmury policzono (sample_au,
        coarse_voxel_au, fpfh_radius_au) - po ich zmianie chmury są liczone od nowa.

        Args:
            p:    ustawienia (sample_au, coarse_voxel_au, fpfh_radius_au).
            flip: True = normalne odwrócone - tak przygotowuje się stronę "A" pary.

        Returns:
            (gęsta - wszystkie punkty, średnia - co 2 x sample_au, zgrubna - co coarse_voxel_au,
             FPFH zgrubnej w promieniu fpfh_radius_au); chmury z normalnymi.
        """
        key = ("clouds", flip, p.sample_au, p.coarse_voxel_au, p.fpfh_radius_au)
        if key not in self._cache:
            nrm = -self.normals if flip else self.normals
            mid = voxel_keep(self.points, 2 * p.sample_au)
            coarse_idx = voxel_keep(self.points, p.coarse_voxel_au)
            coarse = to_cloud(self.points[coarse_idx], nrm[coarse_idx])
            self._cache[key] = (to_cloud(self.points, nrm), to_cloud(self.points[mid], nrm[mid]), coarse,
                                compute_fpfh(coarse, p.fpfh_radius_au))
        return self._cache[key]

    def tree(self) -> cKDTree:
        """KD-drzewo punktów powierzchni (szukanie najbliższych sąsiadów) - liczone raz.

        Returns:
            cKDTree z `points` (te same indeksy co points i normals).
        """
        if "tree" not in self._cache:
            self._cache["tree"] = cKDTree(self.points)
        return self._cache["tree"]

    def axis(self) -> dict:
        """Oś główna kości i miejsce tej powierzchni na niej (docstring modułu, "OSIE W JEDNEJ LINII") - liczone raz.

        Returns:
            {"c": środek kości, "d": kierunek osi (jednostkowy), "elong": wydłużenie (odchylenie
             wzdłuż osi / największe w poprzek), "r": promień kości (mediana odległości wierzchołków
             od osi) [a.u.], "sc": środek powierzchni, "pos": miejsce powierzchni na osi (-1 / +1 =
             końce, 0 = środek), "cl", "dl": punkt i kierunek osi PRZY PRZEŁOMIE (PCA wierzchołków do
             LOCAL_AXIS_FRAC długości od środka powierzchni; zwrot jak "d")} - wszystko w układzie kości.
        """
        if "axis" not in self._cache:
            ax = bone_axis(self.bone.mesh)
            mid, half = (ax["hi"] + ax["lo"]) / 2, max((ax["hi"] - ax["lo"]) / 2, 1e-9)
            sc = self.points.mean(axis=0)
            v = np.asarray(self.bone.mesh.vertices)
            t = (v - ax["c"]) @ ax["d"]
            near = np.abs(t - (sc - ax["c"]) @ ax["d"]) <= max(LOCAL_AXIS_FRAC * 2 * half, LOCAL_AXIS_MIN_AU)
            cl, dl = ax["c"], ax["d"]
            if near.sum() >= 50 and np.ptp(t[near]) >= 2 * ax["r"]:      # dość długi kawałek trzonu
                cl = v[near].mean(axis=0)
                dl = np.linalg.eigh(np.cov((v[near] - cl).T))[1][:, -1]
                dl = dl if dl @ ax["d"] >= 0 else -dl
            self._cache["axis"] = {**ax, "sc": sc, "cl": cl, "dl": dl,
                                   "pos": float(((self.points - ax["c"]) @ ax["d"]).mean() - mid) / half}
        return self._cache["axis"]

    def rim(self, p: SurfaceMatchParams) -> tuple[np.ndarray, np.ndarray, cKDTree]:
        """Kora przy przełomie: wierzchołki kości poza powierzchniami styku, od RIM_MIN_AU do rim_au od tej powierzchni.

        Rzadsza próbka (co sample_au x 2). Do ciągłości kory w trybie "gap" - liczona raz.

        Returns:
            (punkty (N, 3), normalne (N, 3), KD-drzewo punktów) w układzie kości; puste, gdy brak.
        """
        key = ("rim", p.rim_au, p.sample_au)
        if key not in self._cache:
            v = np.asarray(self.bone.mesh.vertices)
            n = np.asarray(self.bone.mesh.vertex_normals)
            cortex = np.flatnonzero(self.contacts.labels == 0) if self.contacts is not None else np.arange(len(v))
            d = self.tree().query(v[cortex], distance_upper_bound=p.rim_au)[0]
            near = cortex[np.isfinite(d) & (d > RIM_MIN_AU)]         # bez samej krawędzi przełomu
            keep = near[voxel_keep(v[near], 2 * p.sample_au)] if len(near) else near
            pts = v[keep] if len(keep) else np.zeros((0, 3))
            self._cache[key] = (pts, n[keep] if len(keep) else np.zeros((0, 3)),
                                cKDTree(pts) if len(pts) else None)
        return self._cache[key]

    def fit(self, p: SurfaceMatchParams) -> dict:
        """Przybliżenie powierzchni płaszczyzną (contacts.fit_surface) - liczone raz.

        Na wydłużonym fragmencie (>= axis_min_elongation) punkty ważone (n . oś)² - twarz przełomu
        na końcu kości długiej patrzy wzdłuż osi. Używane przez PŁASKIE CIĘCIA i tryb "plane".

        Args:
            p: ustawienia (axis_min_elongation).

        Returns:
            Wynik fit_surface ("center", "normal", "u", "v", "rms_au", "axis_deg", ...).
        """
        key = ("fit", p.axis_min_elongation)
        if key not in self._cache:
            ax = self.axis()
            self._cache[key] = fit_surface(self.points, self.normals, None,
                                           ax["d"] if ax["elong"] >= p.axis_min_elongation else None, "plane")
        return self._cache[key]


def bone_contacts(bone: Bone, source: str, contact_params: ContactParams = ContactParams(),
                  root: Path | None = None) -> tuple[Contacts | None, str]:
    """Powierzchnie styku kości wg trybu `source` (opis trybów: docstring modułu).

    W trybach "auto" i "manual-auto" może uruchomić algorytm wykrywania i zapisać jego
    wynik w contacts/<zestaw>/auto/ (contacts.auto_contacts) - za pierwszym razem trwa to dłużej.

    Args:
        bone:           kość.
        source:         "manual" / "auto" / "manual-auto".
        contact_params: algorytm wykrywania i jego ustawienia (dla "auto").
        root:           folder oznaczeń zamiast contacts/ z settings.json (np. w testach).

    Returns:
        (etykiety albo None, skąd: "ręczne" / "algorytm (cache)" / "algorytm (policzone)" / "brak ...").

    Raises:
        ValueError: nieznany tryb `source`.
    """
    if source not in SOURCES:
        raise ValueError(f"matching.source = {source!r}, dostępne: {', '.join(SOURCES)}")
    if source in ("manual", "manual-auto"):
        c = load_contacts(bone.id, bone.mesh, "manual", root=root, frames=scan_frames(bone))
        if c is not None and c.labels.any():
            return c, "ręczne"
        if source == "manual":
            return None, "brak ręcznych oznaczeń"
    if is_excluded(bone, contact_params):
        return None, "pominięta (contacts -> exclude)"
    c, computed = auto_contacts(bone, contact_params, root)
    if not c.labels.any():
        return None, "brak (algorytm nic nie znalazł)"
    return c, "algorytm (policzone)" if computed else "algorytm (cache)"


def make_surface(bone: Bone, contacts: Contacts, sid: int, sample_au: float,
                 tri_labels: np.ndarray | None = None, bone_tree: cKDTree | None = None) -> Surface | None:
    """Jedna powierzchnia styku kości: punkty co `sample_au` z normalnymi i pole.

    Args:
        bone:       kość wczytana z chmurą punktów (load_bones(features=True)).
        contacts:   etykiety powierzchni tej kości.
        sid:        numer powierzchni.
        sample_au:  odstęp punktów [a.u.].
        tri_labels: etykiety trójkątów (triangle_labels) - podane, żeby nie liczyć ich dla każdej powierzchni.
        bone_tree:  KD-drzewo punktów całej kości (do kolizji); None = zbudowane tutaj.

    Returns:
        Powierzchnia albo None, gdy etykieta `sid` nie ma żadnego trójkąta.
    """
    tl = triangle_labels(bone.mesh, contacts.labels) if tri_labels is None else tri_labels
    m = submesh(bone.mesh, tl == sid)
    if not len(m.triangles):
        return None
    if bone_tree is None:
        bone_tree = cKDTree(np.asarray(bone.cloud.points))
    s = sample_surface(m, sample_au)
    return Surface(bone=bone, contacts=contacts, sid=sid, points=s.points, normals=s.normals,
                   area_au2=float(m.get_surface_area()), bone_tree=bone_tree)


def extract_surfaces(bones: list[Bone], p: SurfaceMatchParams, contact_params: ContactParams = ContactParams(),
                     root: Path | None = None, log=None) -> tuple[list[Surface], dict[str, str]]:
    """Powierzchnie styku wszystkich kości (każda osobno), bez mniejszych niż p.min_area_au2.

    Kości muszą być wczytane z chmurą punktów (load_bones(features=True)) - do kolizji.
    Przy p.use_atlas każda kość z powierzchniami dostaje kontekst z atlasu (Surface.atlas,
    atlas_context) - z cache, a gdy go brak, liczony (jak algorytm "atlas", za pierwszym razem dłużej).

    Args:
        bones:          kości do dopasowania.
        p:              ustawienia (source, min_area_au2, sample_au, use_atlas).
        contact_params: algorytm wykrywania (tryby "auto" i "manual-auto"); też analiza lokalna atlasu.
        root:           folder oznaczeń zamiast contacts/ z settings.json.
        log:            komunikaty, gdy atlas liczy dopasowanie fragmentu od nowa (None = cisza).

    Returns:
        (powierzchnie, {id kości: skąd wzięte powierzchnie - jak w bone_contacts; gdy
         wszystkie odpadły jako za małe, z dopiskiem ", powierzchnie < ... a.u.²"; przy
         use_atlas z dopiskiem ", atlas: <najbardziej prawdopodobna kość> <p>"}).

    Raises:
        ValueError: kość bez chmury punktów albo nieznany p.source.
    """
    out, info = [], {}
    atlas_off = ""                 # powód, gdy atlasu nie da się użyć (pusta baza) - wtedy bez atlasu
    for b in bones:
        if b.cloud is None:
            raise ValueError(f"{b.id}: brak chmury punktów - wczytaj kości przez load_bones(features=True)")
        c, info[b.id] = bone_contacts(b, p.source, contact_params, root)
        if c is None:
            continue
        tl = triangle_labels(b.mesh, c.labels)
        tree, flat = None, []
        for k in c.surface_ids:
            if tree is None:
                tree = cKDTree(np.asarray(b.cloud.points))
            s = make_surface(b, c, k, p.sample_au, tl, tree)
            if s is None or s.area_au2 < p.min_area_au2:
                continue
            if p.flat_max_rms_au > 0 and s.fit(p)["rms_au"] < p.flat_max_rms_au:
                flat.append(f"{c.name(k)} (odchylenie {s.fit(p)['rms_au']:.2f} a.u.)")
                continue
            out.append(s)
        if flat:
            info[b.id] += f", płaskie cięcia pominięte: {', '.join(flat)}"
        mine = [s for s in out if s.bone.id == b.id]
        if not mine:
            info[b.id] += f", powierzchnie < {p.min_area_au2:g} a.u.²"
        elif p.use_atlas and not atlas_off:
            try:
                ctx = atlas_context(b, contact_params, log)
            except ValueError:     # pusta baza modeli (atlas.load_atlas)
                atlas_off = "baza modeli pusta - python scripts/build_atlas.py"
                ctx = None
            for s in mine:
                s.atlas = ctx
            info[b.id] += f", atlas: {atlas_off or atlas_top(ctx)}"
    return out, info


# -----------------
# ATLAS: z której kości pochodzi fragment (src/atlas.py)
# -----------------

ATLAS_MIN_JOINT = 0.02     # start z atlasu tylko z modelu, na którym p_A x p_B >= tyle


def atlas_context(bone: Bone, contact_params: ContactParams, log=None) -> dict:
    """Ranking kości z bazy dla fragmentu, z ustawieniami na modelach (atlas.cached_match).

    Ten sam cache co algorytm "atlas" (out/3d/<zestaw>/cache/atlas/<id>.npz) - gdy wykrywanie
    liczono algorytmem "atlas" z tymi samymi ustawieniami, nic nie jest liczone od nowa.

    Args:
        bone:           kość (fragment).
        contact_params: analiza lokalna (feature_keys, feature_threshold) - jak przy wykrywaniu.
        log:            komunikaty przy liczeniu od nowa (brak cache); None = cisza.

    Returns:
        {"p_unknown": p "żadna z bazy", "candidates": [{"model", "group", "mirror", "label", "p",
         "rel_scale", "T": 4x4 model -> fragment (lista)}] od najbardziej prawdopodobnego}.

    Raises:
        ValueError: pusta baza modeli.
    """
    import atlas
    import config

    ap, sfp = config.atlas_params(), config.surface_feature_params()
    started: list = []

    def note(text: str) -> None:
        """Pierwszy komunikat atlasu dla tej kości poprzedzony jej nazwą (tylko przy liczeniu od nowa)."""
        if not started:
            started.append(True)
            log(f"  atlas: {bone.label} - liczę dopasowanie do bazy (raz, potem z cache)...")
        log(text)

    m = atlas.cached_match(bone, atlas.load_atlas(ap), ap, contact_params, sfp, log=note if log else None)
    keys = ("model", "group", "mirror", "label", "p", "rel_scale", "T")
    return {"p_unknown": round(float(m.p_unknown), 4),
            "candidates": [{k: c[k] for k in keys} for c in m.candidates if c["p"] >= 0.001]}


def atlas_top(ctx: dict | None) -> str:
    """Najbardziej prawdopodobna kość z kontekstu atlasu, np. "FIBULA_LEFT 82%" (do konsoli)."""
    if ctx is None:
        return "brak"
    import atlas

    best = ctx["candidates"][0] if ctx["candidates"] else None
    if best is None or best["p"] < ctx["p_unknown"]:
        return f"{atlas.UNKNOWN} {ctx['p_unknown']:.0%}"
    return f"{best['label']} {best['p']:.0%}"


def atlas_kinds(ctx: dict) -> dict[str, float]:
    """P(kość) fragmentu: model + strona (odbicie albo nie); modele z podfolderu bazy = jeden typ, bez strony.

    Args:
        ctx: kontekst z atlas_context.

    Returns:
        {klucz kości: prawdopodobieństwo}; suma = 1 - p "żadna z bazy".
    """
    acc: dict[str, float] = {}
    for c in ctx["candidates"]:
        key = c["group"] or f"{c['model']}|{int(bool(c['mirror']))}"
        acc[key] = acc.get(key, 0.0) + c["p"]
    return acc


def atlas_relation(a: Surface, b: Surface) -> tuple[float, float]:
    """Czy fragmenty a i b pochodzą z tej samej kości - z rankingów atlasu obu fragmentów.

    Args:
        a, b: powierzchnie (Surface.atlas z atlas_context; None = bez wiedzy).

    Returns:
        (P(ta sama kość), P(różne kości)); reszta do 1 = niewiadomo (któryś może być "żadna
        z bazy"). Bez kontekstu któregoś fragmentu: (0, 0).
    """
    if a.atlas is None or b.atlas is None:
        return 0.0, 0.0
    ka, kb = atlas_kinds(a.atlas), atlas_kinds(b.atlas)
    same = sum(q * kb.get(k, 0.0) for k, q in ka.items())
    return same, max(sum(ka.values()) * sum(kb.values()) - same, 0.0)


def atlas_starts(a: Surface, b: Surface, p: SurfaceMatchParams) -> list[tuple[str, float, np.ndarray]]:
    """Ustawienia startowe A -> B z atlasu: oba fragmenty leżą na tym samym modelu (i stronie).

    x_A = T_A x_modelu, x_B = T_B x_modelu  =>  x_B = T_B T_A⁻¹ x_A. Skale obu fragmentów na
    modelu mogą się różnić (osobno dopasowane), a skan jest w jednych jednostkach - zostaje
    sam obrót (najbliższy macierzy T_B T_A⁻¹) i przesunięcie zgodne w środku powierzchni A.

    Args:
        a, b: powierzchnie z kontekstem atlasu (Surface.atlas).
        p:    ustawienia (use_atlas, atlas_starts).

    Returns:
        [(nazwa modelu, p_A x p_B, macierz 4x4 A -> B)] od najbardziej prawdopodobnego, najwyżej
        p.atlas_starts, tylko modele z p_A x p_B >= ATLAS_MIN_JOINT; [] = bez startów z atlasu.
    """
    if not p.use_atlas or p.atlas_starts <= 0 or a.atlas is None or b.atlas is None:
        return []
    on_b = {(c["model"], bool(c["mirror"])): c for c in b.atlas["candidates"]}
    center = a.points.mean(axis=0)
    out = []
    for ca in a.atlas["candidates"]:
        cb = on_b.get((ca["model"], bool(ca["mirror"])))
        joint = ca["p"] * cb["p"] if cb is not None else 0.0
        if joint < ATLAS_MIN_JOINT:
            continue
        M = np.asarray(cb["T"], dtype=float) @ np.linalg.inv(np.asarray(ca["T"], dtype=float))
        U, _, Vt = np.linalg.svd(M[:3, :3])
        R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, transform_points(M, center[None])[0] - R @ center
        out.append((ca["label"], joint, T))
    out.sort(key=lambda s: -s[1])
    return out[:p.atlas_starts]


# -----------------
# DOPASOWANIE PARY
# -----------------

def contact_score(a: Surface, b: Surface, T: np.ndarray, p: SurfaceMatchParams) -> dict:
    """Jak dobrze powierzchnia a, przeniesiona przekształceniem T, przylega do b (krok 5 modułu).

    Punkt jest "w styku", gdy druga powierzchnia jest bliżej niż p.tolerance_au, a normalne
    są przeciwne z dokładnością do p.normal_angle_deg. Sprawdzane w obie strony.

    Args:
        a, b: powierzchnie (każda w układzie swojej kości).
        T:    przekształcenie 4x4 z układu kości a do układu kości b.
        p:    ustawienia (tolerance_au, normal_angle_deg).

    Returns:
        {"cover_a", "cover_b": ułamek punktów a / b w styku,
         "contact_au2": min(cover_a x pole a, cover_b x pole b),
         "rmse_au": RMSE odległości punktów a w styku od b, "angle_deg": mediana odchylenia normalnych
         w styku od idealnie przeciwnych (oba nan, gdy styku brak)}.
    """
    min_dot = float(np.cos(np.deg2rad(p.normal_angle_deg)))
    pa = transform_points(T, a.points)
    na = -(a.normals @ T[:3, :3].T)                  # odwrócone: w styku mają być zgodne z normalnymi b
    ok_a = covered(pa, na, b.tree(), b.normals, p.tolerance_au, min_dot)
    Ti = np.linalg.inv(T)
    pb = transform_points(Ti, b.points)
    nb = -(b.normals @ Ti[:3, :3].T)
    ok_b = covered(pb, nb, a.tree(), a.normals, p.tolerance_au, min_dot)
    d, idx = b.tree().query(pa[ok_a]) if ok_a.any() else (np.zeros(0), np.zeros(0, int))
    cos = np.einsum("ij,ij->i", na[ok_a], b.normals[idx]) if ok_a.any() else np.zeros(0)
    fa, fb = float(ok_a.mean()), float(ok_b.mean())
    return {"cover_a": fa, "cover_b": fb, "contact_au2": min(fa * a.area_au2, fb * b.area_au2),
            "rmse_au": float(np.sqrt(np.mean(d**2))) if len(d) else float("nan"),
            "angle_deg": float(np.degrees(np.median(np.arccos(np.clip(cos, -1, 1))))) if len(cos) else float("nan")}


def plane_contact(a: Surface, b: Surface, T: np.ndarray, p: SurfaceMatchParams) -> dict:
    """Styk w trybie "plane": wspólna część obrysów obu powierzchni na płaszczyźnie B (docstring modułu).

    Punkty A (po T) i B są rzutowane na płaszczyznę przybliżającą B i zaznaczane na siatce co
    2 x sample_au; tylko punkty A bliżej płaszczyzny B niż fit_tolerance (3 x odchylenie B, co
    najmniej 1 a.u.) i z normalną przeciwną (do normal_angle_deg).

    Args:
        a, b: powierzchnie.
        T:    przekształcenie 4x4 z układu kości a do układu kości b.
        p:    ustawienia (sample_au, normal_angle_deg, axis_min_elongation).

    Returns:
        {"cover_a", "cover_b": część komórek obrysu A / B we wspólnej części, "contact_au2": pole wspólnej
         części (mniejsze z pól obu stron, jak w contact_score), "rmse_au": odległość punktów A we wspólnej
         części od płaszczyzny B (nan, gdy jej brak)}.
    """
    fb = b.fit(p)
    cell = 2 * p.sample_au
    pa = transform_points(T, a.points)
    na = a.normals @ T[:3, :3].T
    da = (pa - fb["center"]) @ fb["normal"]
    near = (np.abs(da) <= max(3 * fb["rms_au"], 1.0)) & (-(na @ fb["normal"]) >= np.cos(np.radians(p.normal_angle_deg)))

    def cells(x: np.ndarray) -> set:
        """Komórki siatki (u, v) na płaszczyźnie B zajęte przez punkty x."""
        rel = x - fb["center"]
        return set(map(tuple, np.floor(np.column_stack([rel @ fb["u"], rel @ fb["v"]]) / cell).astype(int)))

    ca_all, cb = cells(pa), cells(b.points)
    common = cells(pa[near]) & cb
    fa, fbv = len(common) / max(len(ca_all), 1), len(common) / max(len(cb), 1)
    return {"cover_a": fa, "cover_b": fbv, "contact_au2": min(fa * a.area_au2, fbv * b.area_au2),
            "rmse_au": float(np.sqrt(np.mean(da[near] ** 2))) if near.any() else float("nan")}


def collision_area(a: Surface, b: Surface, T: np.ndarray, p: SurfaceMatchParams) -> float:
    """Pole kości a (po przekształceniu T), które wchodzi w kość b głębiej niż collision_au, i odwrotnie [a.u.²].

    Punkt jest "w środku" kości, jeśli najbliższy jej punkt jest bliżej niż collision_search_au,
    a punkt leży za jej powierzchnią (po przeciwnej stronie niż normalna) o więcej niż
    collision_au. Skany są otwarte, więc to przybliżenie - wystarcza do odrzucenia
    ustawień "kość w kości". Pole = liczba takich punktów chmury x (pole kości / liczba
    punktów chmury).

    Args:
        a, b: powierzchnie - liczone są ich CAŁE kości (bone.cloud, bone_tree).
        T:    przekształcenie 4x4 z układu kości a do układu kości b.
        p:    ustawienia (collision_au, collision_search_au).

    Returns:
        Suma pól z obu kierunków [a.u.²].
    """
    def one_way(src: Surface, dst: Surface, M: np.ndarray) -> float:
        """Pole kości src (przeniesionej M) wbite w kość dst [a.u.²].

        Args:
            src: powierzchnia, której kość jest przenoszona.
            dst: powierzchnia kości, w którą src może się wbijać.
            M:   przekształcenie 4x4 z układu kości src do układu kości dst.

        Returns:
            Pole punktów chmury src leżących głębiej niż p.collision_au pod powierzchnią dst.
        """
        ps = transform_points(M, np.asarray(src.bone.cloud.points))
        pd, nd = np.asarray(dst.bone.cloud.points), np.asarray(dst.bone.cloud.normals)
        d, idx = dst.bone_tree.query(ps, distance_upper_bound=p.collision_search_au)
        ok = np.isfinite(d)
        depth = np.einsum("ij,ij->i", pd[idx[ok]] - ps[ok], nd[idx[ok]])
        per_point = src.bone.meta.get("stats", {}).get("area_au2", 0) / max(len(ps), 1)
        return float((depth > p.collision_au).sum() * per_point)

    return one_way(a, b, T) + one_way(b, a, np.linalg.inv(T))


# -----------------
# OSIE W JEDNEJ LINII (kości długie)
# -----------------

def _ramp(x: float, limit: float) -> float:
    """1 do `limit`, potem liniowo do 0 przy 2 x `limit`."""
    return float(np.clip(2.0 - x / max(limit, 1e-9), 0.0, 1.0))


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Obrót o `angle` radianów wokół osi `axis` (wzór Rodriguesa)."""
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def _rotation_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Najmniejszy obrót przenoszący kierunek u na kierunek v (oba jednostkowe)."""
    axis = np.cross(u, v)
    sin, cos = float(np.linalg.norm(axis)), float(np.clip(u @ v, -1, 1))
    if sin < 1e-9:
        if cos > 0:
            return np.eye(3)
        other = np.eye(3)[np.argmin(np.abs(u))]           # przeciwne: pół obrotu wokół osi prostopadłej
        return _rotation(np.cross(u, other), math.pi)
    return _rotation(axis, math.atan2(sin, cos))


def long_pair(a: Surface, b: Surface, p: SurfaceMatchParams) -> bool:
    """Czy dla pary sprawdzać osie: axis_check i oba fragmenty wydłużone (>= axis_min_elongation)."""
    return p.axis_check and min(a.axis()["elong"], b.axis()["elong"]) >= p.axis_min_elongation


def axis_fit(a: Surface, b: Surface, T: np.ndarray, p: SurfaceMatchParams) -> dict | None:
    """Czy po ustawieniu T fragmenty leżą wzdłuż jednej osi (docstring modułu, "OSIE W JEDNEJ LINII", a).

    Args:
        a, b: powierzchnie.
        T:    przekształcenie 4x4 z układu kości a do układu kości b.
        p:    ustawienia (axis_check, axis_min_elongation, axis_max_deg, axis_max_offset).

    Returns:
        {"axis_deg": kąt między osiami [stopnie], "axis_offset": rozsunięcie osi w miejscu styku
         [promienie kości], "axis_weight": waga styku 0-1} albo None, gdy osi się nie sprawdza.
    """
    if not long_pair(a, b, p):
        return None
    A, B = a.axis(), b.axis()
    R, t = T[:3, :3], T[:3, 3]
    da, ca = R @ A["dl"], R @ A["cl"] + t                 # osie przy przełomach
    angle = math.degrees(math.acos(min(1.0, abs(float(da @ B["dl"])))))
    x = 0.5 * (R @ A["sc"] + t + B["sc"])                  # miejsce styku
    qa = ca + ((x - ca) @ da) * da                         # najbliższe punkty obu osi
    qb = B["cl"] + ((x - B["cl"]) @ B["dl"]) * B["dl"]
    offset = float(np.linalg.norm(qa - qb)) / (0.5 * (A["r"] + B["r"]))
    return {"axis_deg": round(angle, 2), "axis_offset": round(offset, 3),
            "axis_weight": round(_ramp(angle, p.axis_max_deg) * _ramp(offset, p.axis_max_offset), 4)}


def axis_starts_for(a: Surface, b: Surface, p: SurfaceMatchParams) -> list[tuple[str, np.ndarray]]:
    """Ustawienia startowe A -> B wzdłuż osi: A przedłuża B, przełom na przełomie, obroty wokół osi.

    Koniec A z przełomem (kierunek pos_A x d_A od środka A) ma patrzeć PRZECIWNIE do końca B z
    przełomem: wtedy ciało A leży za końcem B, w przedłużeniu jego osi. Środek przełomu A trafia
    na środek przełomu B, a obrót wokół osi jest sprawdzany co 360° / axis_starts. Powierzchnia
    w środku boku (|pos| < 0.3) - obie strony osi.

    Args:
        a, b: powierzchnie.
        p:    ustawienia (axis_check, axis_min_elongation, axis_starts).

    Returns:
        [(opis, macierz 4x4 A -> B)]; [] = para bez sprawdzania osi albo axis_starts = 0.
    """
    if p.axis_starts <= 0 or not long_pair(a, b, p):
        return []
    A, B = a.axis(), b.axis()

    def sides(pos: float) -> list[float]:
        """Strona osi, po której leży powierzchnia (obie, gdy jest w środku boku)."""
        return [1.0 if pos >= 0 else -1.0] if abs(pos) >= 0.3 else [1.0, -1.0]

    out = []
    for sa in sides(A["pos"]):
        for sb in sides(B["pos"]):
            v = -sb * B["dl"]
            R0 = _rotation_between(sa * A["dl"], v)
            for k in range(p.axis_starts):
                R = _rotation(v, 2 * math.pi * k / p.axis_starts) @ R0
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = R, B["sc"] - R @ A["sc"]
                out.append((f"oś {360 * k / p.axis_starts:.0f}°", T))
    return out


def axis_locked_icp(a: Surface, b: Surface, T: np.ndarray, p: SurfaceMatchParams) -> np.ndarray:
    """Prostowanie "na wprost": oś A na osi B (osie przy przełomach), potem ICP z obrotem tylko wokół tej osi.

    1. Obrót wokół miejsca styku, który kładzie oś A (po T) na oś B, i przesunięcie osi A na oś B w poprzek.
    2. ICP point-to-plane (punkty A co 2 x sample_au -> powierzchnia B) z 4 parametrami: obrót wokół osi B
       przez miejsce styku i przesunięcie; pary punktów do icp_start_au, potem połowa, na koniec 2 x sample_au.

    Args:
        a, b: powierzchnie kości długich.
        T:    ustawienie startowe A -> B (najlepsze z innych prób).
        p:    ustawienia (icp_start_au, sample_au).

    Returns:
        Wyprostowane i dociśnięte przekształcenie 4x4 A -> B (osie przy przełomach równoległe).
    """
    A, B = a.axis(), b.axis()
    T = np.asarray(T, dtype=float).copy()
    R = T[:3, :3]
    da, db = R @ A["dl"], B["dl"]
    d = db if da @ db >= 0 else -db
    x = 0.5 * (R @ A["sc"] + T[:3, 3] + B["sc"])
    F = np.eye(4)
    F[:3, :3] = _rotation_between(da, d)
    F[:3, 3] = x - F[:3, :3] @ x
    T = F @ T
    ca = T[:3, :3] @ A["cl"] + T[:3, 3]
    qa, qb = ca + ((x - ca) @ d) * d, B["cl"] + ((x - B["cl"]) @ d) * d
    shift = qb - qa
    T[:3, 3] += shift - (shift @ d) * d                    # oś A na osi B (tylko w poprzek)
    src = a.points[voxel_keep(a.points, 2 * p.sample_au)]
    for dist in (p.icp_start_au, p.icp_start_au / 2, 2 * p.sample_au):
        for _ in range(15):
            P = transform_points(T, src)
            dd, idx = b.tree().query(P, distance_upper_bound=dist)
            ok = np.isfinite(dd)
            if ok.sum() < 10:
                break
            q, nq, Pk = b.points[idx[ok]], b.normals[idx[ok]], P[ok]
            r = np.einsum("ij,ij->i", Pk - q, nq)
            J = np.column_stack([np.einsum("ij,ij->i", np.cross(d, Pk - x), nq), nq])
            theta = np.linalg.lstsq(J, -r, rcond=None)[0]
            M = np.eye(4)
            M[:3, :3] = _rotation(d, float(theta[0]))
            M[:3, 3] = x - M[:3, :3] @ x + theta[1:]
            T = M @ T
            x = x + theta[1:]
            if abs(theta[0]) < 1e-5 and np.abs(theta[1:]).max() < 1e-4:
                break
    return T


def plane_starts_for(a: Surface, b: Surface, p: SurfaceMatchParams) -> list[tuple[str, np.ndarray]]:
    """Starty płaszczyzna na płaszczyźnie (jak w OSTEOFORM): plane_starts_keep najlepszych z plane_starts obrotów.

    Środek przybliżenia A na środku przybliżenia B, normalne przeciwne, obrót wokół normalnej B co
    360° / plane_starts. Kolejność wg styku z luźną tolerancją (4 x tolerance_au, kąt 60°) - przed ICP
    nierówne przełomy jeszcze do siebie nie przylegają.

    Args:
        a, b: powierzchnie.
        p:    ustawienia (plane_starts, plane_starts_keep).

    Returns:
        [(opis, macierz 4x4 A -> B)], najlepsze pierwsze; [] przy plane_starts = 0.
    """
    if p.plane_starts <= 0:
        return []
    fa, fb = a.fit(p), b.fit(p)
    R0 = _rotation_between(fa["normal"], -fb["normal"])
    loose = replace(p, tolerance_au=4 * p.tolerance_au, normal_angle_deg=60.0)
    cand = []
    for k in range(p.plane_starts):
        T = np.eye(4)
        T[:3, :3] = _rotation(fb["normal"], 2 * math.pi * k / p.plane_starts) @ R0
        T[:3, 3] = fb["center"] - T[:3, :3] @ fa["center"]
        cand.append((contact_score(a, b, T, loose)["contact_au2"], f"płaszczyzna {360 * k / p.plane_starts:.0f}°", T))
    cand.sort(key=lambda c: -c[0])
    return [(label, T) for _, label, T in cand[:max(p.plane_starts_keep, 1)]]


LONG_MODES = ("contact", "gap")


def long_terms(a: Surface, b: Surface, T: np.ndarray, p: SurfaceMatchParams) -> dict:
    """Tryb "gap": bliskość przełomów i ciągłość kory przy ustawieniu T (docstring modułu, "KOŚCI DŁUGIE").

    Args:
        a, b: powierzchnie wydłużonych fragmentów.
        T:    przekształcenie 4x4 z układu kości a do układu kości b.
        p:    ustawienia (gap_au, rim_au, continuity_au).

    Returns:
        {"proximity": 0-1 (min(pokrycie A x pole A, pokrycie B x pole B) / średnie pole), "near_a", "near_b":
         pokrycie obu stron, "continuity": 0-1 (część kory przy przełomach na przedłużeniu kory drugiej strony,
         średnia z obu kierunków)}.
    """
    Ti = np.linalg.inv(T)

    def near(src: Surface, dst: Surface, M: np.ndarray) -> float:
        """Część przełomu src (po M) bliżej przełomu dst niż gap_au i nie głębiej w dst niż collision_au."""
        q = transform_points(M, src.points)
        d, idx = dst.tree().query(q, distance_upper_bound=p.gap_au)
        ok = np.isfinite(d)
        if ok.any():
            i = idx[ok]
            ok[ok] = np.einsum("ij,ij->i", q[ok] - dst.points[i], dst.normals[i]) > -p.collision_au
        return float(ok.mean())

    fa, fb = near(a, b, T), near(b, a, Ti)
    mean_area = max(0.5 * (a.area_au2 + b.area_au2), 1e-9)
    prox = min(fa * a.area_au2, fb * b.area_au2) / mean_area

    def cont(src: Surface, dst: Surface, M: np.ndarray) -> float:
        """Część kory przy przełomie src (po M) leżącej na przedłużeniu kory przy przełomie dst."""
        ps, ns, _ = src.rim(p)
        pd, nd, td = dst.rim(p)
        if td is None or not len(ps):
            return 0.0
        q = transform_points(M, ps)
        nq = ns @ M[:3, :3].T
        d, idx = td.query(q, k=RIM_NEIGHBOURS, distance_upper_bound=3 * p.rim_au)
        ok = np.isfinite(d)
        idx = np.where(ok, idx, 0)
        plane = np.abs(np.einsum("nkj,nkj->nk", q[:, None, :] - pd[idx], nd[idx]))
        agree = np.einsum("nj,nkj->nk", nq, nd[idx]) > math.cos(math.radians(35))
        return float((ok & (plane < p.continuity_au) & agree).any(axis=1).mean())

    return {"proximity": round(prox, 4), "near_a": round(fa, 4), "near_b": round(fb, 4),
            "continuity": round(0.5 * (cont(a, b, T) + cont(b, a, Ti)), 4)}


def evaluate(a: Surface, b: Surface, T: np.ndarray, p: SurfaceMatchParams) -> dict:
    """Pełna ocena ustawienia T (krok 5 modułu): contact_score + collision_area + wynik.

    Bez szukania ustawienia - szybkie, więc nadaje się do sprawdzenia, jak wynik
    zmienia się z ustawieniami oceny (tolerance_au, normal_angle_deg, collision_*).

    Args:
        a, b: powierzchnie (każda w układzie swojej kości).
        T:    przekształcenie 4x4 z układu kości a do układu kości b.
        p:    ustawienia oceny (score_mode, tolerance_au, normal_angle_deg, collision_au,
              collision_search_au, przy axis_check osie, przy use_atlas też atlas_boost).

    Returns:
        {"cover_a", "cover_b", "contact_au2", "rmse_au", "collision_au2", "dice" = 2 x styk / (pole A +
         pole B), "score" (WYNIK, docstring modułu, krok 5), "score_unit": "%" albo "a.u.²"}; dla pary
        wydłużonych fragmentów (axis_fit) dodatkowo "axis_deg", "axis_offset", "axis_weight"; przy
        use_atlas "atlas_same", "atlas_diff" (atlas_relation) i "atlas_bonus" = część wyniku z premii.

    Raises:
        ValueError: nieznany p.score_mode.
    """
    if p.score_mode not in SCORE_MODES:
        raise ValueError(f"matching.score_mode = {p.score_mode!r}, dostępne: {', '.join(SCORE_MODES)}")
    if p.mode not in MODES:
        raise ValueError(f"matching.mode = {p.mode!r}, dostępne: {', '.join(MODES)}")
    if p.long_mode not in LONG_MODES:
        raise ValueError(f"matching.long_mode = {p.long_mode!r}, dostępne: {', '.join(LONG_MODES)}")
    sc = plane_contact(a, b, T, p) if p.mode == "plane" else contact_score(a, b, T, p)
    sc["collision_au2"] = collision_area(a, b, T, p)
    ax = axis_fit(a, b, T, p)
    if ax is not None:
        sc.update(ax)
    mean_area = max(0.5 * (a.area_au2 + b.area_au2), 1e-9)
    sc["dice"] = round(sc["contact_au2"] / mean_area, 4)
    if p.score_mode == "dice":         # udział w obu przełomach [%] - nie zależy od wielkości fragmentów
        gain, loss, sc["score_unit"] = 100 * sc["contact_au2"] / mean_area, 100 * sc["collision_au2"] / mean_area, "%"
    else:                              # pole [a.u.²]
        gain, loss, sc["score_unit"] = sc["contact_au2"], sc["collision_au2"], "a.u.²"
    if p.long_mode == "gap" and p.mode == "points" and ax is not None:
        sc.update(long_terms(a, b, T, p))
        rel = 0.5 * sc["proximity"] + 0.5 * sc["continuity"]
        gain = 100 * rel if p.score_mode == "dice" else rel * mean_area
    gain *= ax["axis_weight"] if ax is not None else 1.0
    if p.quality_factor and p.mode == "points" and p.long_mode != "gap" and np.isfinite(sc["rmse_au"]):
        sc["quality"] = round(math.exp(-sc.get("angle_deg", 0.0) / 45.0) / (1 + sc["rmse_au"] / p.tolerance_au), 4)
        gain *= sc["quality"]
    sc["score"] = gain - loss
    if p.use_atlas:
        same, diff = atlas_relation(a, b)
        sc.update(atlas_same=round(same, 4), atlas_diff=round(diff, 4), atlas_bonus=gain * p.atlas_boost * same)
        sc["score"] += sc["atlas_bonus"]
    return sc


def refine_icp(src_clouds, tgt_clouds, T: np.ndarray, p: SurfaceMatchParams,
               first_au: float | None = None) -> np.ndarray:
    """ICP point-to-plane: icp_start_au i połowa tego na rzadszych chmurach, potem 2 x sample_au na gęstych.

    Args:
        src_clouds: chmury strony A z Surface.clouds(flip=True) (używane [0] i [1]).
        tgt_clouds: chmury strony B z Surface.clouds(flip=False).
        T:          przekształcenie startowe A -> B (z RANSAC albo z atlasu).
        p:          ustawienia (icp_start_au, sample_au).
        first_au:   dodatkowy pierwszy krok z tym promieniem (start z atlasu jest mniej dokładny
                    niż z RANSAC); None = bez.

    Returns:
        Poprawione przekształcenie 4x4 A -> B.
    """
    crit = reg.ICPConvergenceCriteria(max_iteration=50)
    steps = ((p.icp_start_au, 1), (p.icp_start_au / 2, 1), (2 * p.sample_au, 0))
    if first_au and first_au > p.icp_start_au:
        steps = ((first_au, 1),) + steps
    for dist, k in steps:
        T = reg.registration_icp(src_clouds[k], tgt_clouds[k], dist, T,
                                 reg.TransformationEstimationPointToPlane(), crit).transformation
    return T


def match_pair_plane(a: Surface, b: Surface, p: SurfaceMatchParams) -> dict:
    """Tryb "plane": płaszczyzna A na płaszczyźnie B, normalne przeciwne, obrót wokół normalnej co 360° / plane_steps.

    Args:
        a, b: powierzchnie dwóch różnych kości.
        p:    ustawienia (plane_steps i ocena).

    Returns:
        Jak match_pair ("start" = "płaszczyzna <obrót>°", "ransac" = False).
    """
    fa, fb = a.fit(p), b.fit(p)
    R0 = _rotation_between(fa["normal"], -fb["normal"])
    best = None
    steps = max(p.plane_steps, 1)
    for k in range(steps):
        R = _rotation(fb["normal"], 2 * math.pi * k / steps) @ R0
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, fb["center"] - R @ fa["center"]
        sc = evaluate(a, b, T, p)
        if best is None or sc["score"] > best["score"]:
            best = {**sc, "T": T.tolist(), "seed": k, "start": f"płaszczyzna {360 * k / steps:.0f}°", "ransac": False}
    return best


GAP_SAMPLE = 800             # "gap", szybka ocena siatki: tyle punktów chmury kości do kolizji


def _bone_sample(s: Surface) -> tuple[np.ndarray, float]:
    """Próbka chmury całej kości (najwyżej GAP_SAMPLE punktów) i pole przypadające na punkt próbki - liczone raz."""
    if "bone_sample" not in s._cache:
        pts = np.asarray(s.bone.cloud.points)
        step = max(1, len(pts) // GAP_SAMPLE)
        sample = pts[::step]
        s._cache["bone_sample"] = (sample, s.bone.meta.get("stats", {}).get("area_au2", 0) / max(len(sample), 1))
    return s._cache["bone_sample"]


def collision_fast(a: Surface, b: Surface, T: np.ndarray, p: SurfaceMatchParams) -> float:
    """Przybliżona kolizja (jak collision_area, ale z próbki GAP_SAMPLE punktów każdej kości) [a.u.²]."""
    def one(src: Surface, dst: Surface, M: np.ndarray) -> float:
        """Pole kości src (po M) wbite w kość dst głębiej niż collision_au."""
        sample, per_point = _bone_sample(src)
        ps = transform_points(M, sample)
        d, idx = dst.bone_tree.query(ps, distance_upper_bound=p.collision_search_au)
        ok = np.isfinite(d)
        if not ok.any():
            return 0.0
        pd, nd = np.asarray(dst.bone.cloud.points), np.asarray(dst.bone.cloud.normals)
        depth = np.einsum("ij,ij->i", pd[idx[ok]] - ps[ok], nd[idx[ok]])
        return float((depth > p.collision_au).sum() * per_point)

    return one(a, b, T) + one(b, a, np.linalg.inv(T))


def long_fast_score(a: Surface, b: Surface, T: np.ndarray, p: SurfaceMatchParams) -> float:
    """Szybka ocena ustawienia w trybie "gap" do siatki: 0.5 x bliskość + 0.5 x ciągłość - kolizja / średnie pole."""
    lt = long_terms(a, b, T, p)
    mean_area = max(0.5 * (a.area_au2 + b.area_au2), 1e-9)
    return 0.5 * lt["proximity"] + 0.5 * lt["continuity"] - collision_fast(a, b, T, p) / mean_area


def long_pose(a: Surface, b: Surface, sa: float, sb: float, twist: float, shift: float) -> np.ndarray:
    """Ustawienie A -> B w trybie "gap": osie przy przełomach w linii, A przedłuża B, obrót i przesunięcie.

    Koniec A z przełomem (kierunek sa x oś A) patrzy przeciwnie do końca B z przełomem (sb x oś B); punkt osi A
    naprzeciw środka przełomu A trafia na punkt osi B naprzeciw środka przełomu B przesunięty o `shift` wzdłuż
    kierunku końca B (dodatnie = szczelina, ujemne = końce zachodzą na siebie); obrót `twist` [rad] wokół osi B.
    """
    A, B = a.axis(), b.axis()
    uB = sb * B["dl"]
    R = _rotation(uB, twist) @ _rotation_between(sa * A["dl"], -uB)
    aA = A["cl"] + ((A["sc"] - A["cl"]) @ A["dl"]) * A["dl"]
    aB = B["cl"] + ((B["sc"] - B["cl"]) @ B["dl"]) * B["dl"]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, aB + shift * uB - R @ aA
    return T


def match_long_pair(a: Surface, b: Surface, p: SurfaceMatchParams) -> dict:
    """Tryb "gap" dla pary kości długich: siatka (obrót, przesunięcie), dokładniej wokół najlepszych, prostowanie.

    Returns:
        Jak match_pair ("start" = "oś <obrót>° / <przesunięcie> a.u.", "ransac" = False).
    """
    A, B = a.axis(), b.axis()

    def sides(pos: float) -> list[float]:
        """Strona osi z przełomem (obie, gdy przełom jest w środku boku)."""
        return [1.0 if pos >= 0 else -1.0] if abs(pos) >= 0.3 else [1.0, -1.0]

    lo, hi = p.gap_shift_au
    cand = []
    for sa in sides(A["pos"]):
        for sb in sides(B["pos"]):
            for tw in np.arange(0.0, 360.0, p.gap_twist_deg):
                for sh in np.arange(lo, hi + 1e-9, p.gap_step_au):
                    T = long_pose(a, b, sa, sb, math.radians(tw), sh)
                    cand.append((long_fast_score(a, b, T, p), sa, sb, tw, sh))
    cand.sort(key=lambda c: -c[0])
    fine = []
    for k, (_, sa, sb, tw0, sh0) in enumerate(cand[:3]):
        for tw in tw0 + np.arange(-1, 1.01, 1 / 3) * p.gap_twist_deg:
            for sh in sh0 + np.arange(-1, 1.01, 1 / 3) * p.gap_step_au:
                T = long_pose(a, b, sa, sb, math.radians(tw), sh)
                fine.append((long_fast_score(a, b, T, p), k, tw, sh, T))
    _, k, tw, sh, T = max(fine, key=lambda c: c[0])
    best = {**evaluate(a, b, T, p), "T": T.tolist(), "seed": k, "ransac": False,
            "start": f"oś {tw % 360:.0f}° / {sh:+.1f} a.u."}
    if p.axis_lock:
        T = axis_locked_icp(a, b, np.asarray(best["T"]), p)
        sc = evaluate(a, b, T, p)
        if sc["score"] > best["score"]:
            best = {**sc, "T": T.tolist(), "seed": best["seed"], "ransac": False,
                    "start": best["start"] + " + na wprost"}
    return best


def match_long_constrained(a: Surface, b: Surface, p: SurfaceMatchParams) -> dict:
    """Szukanie z osią jako warunkiem (docstring modułu, "OŚ JAKO WARUNEK SZUKANIA").

    Returns:
        Jak match_pair ("start" = rodzaj startu + "na wprost" albo "+ mały kąt").
    """
    A, B = a.axis(), b.axis()
    ca, cb = a.clouds(p, flip=True), b.clouds(p, flip=False)

    def sides(pos: float) -> list[float]:
        """Strona osi z przełomem (obie, gdy przełom jest w środku boku)."""
        return [1.0 if pos >= 0 else -1.0] if abs(pos) >= 0.3 else [1.0, -1.0]

    grid = []
    lo, hi = p.gap_shift_au
    sub_a = a.points[voxel_keep(a.points, 1.0)]          # rzadka próbka przełomów - szybka ocena siatki
    sub_b = b.points[voxel_keep(b.points, 1.0)]
    tree_a, tree_b = cKDTree(sub_a), cKDTree(sub_b)
    mean_area = max(0.5 * (a.area_au2 + b.area_au2), 1e-9)
    for sa in sides(A["pos"]):
        for sb in sides(B["pos"]):
            for tw in np.arange(0.0, 360.0, p.constrain_twist_deg):
                for sh in np.arange(lo, hi + 1e-9, p.constrain_step_au):
                    T = long_pose(a, b, sa, sb, math.radians(tw), sh)
                    fa = np.isfinite(tree_b.query(transform_points(T, sub_a), distance_upper_bound=p.gap_au)[0]).mean()
                    fb = np.isfinite(tree_a.query(transform_points(np.linalg.inv(T), sub_b),
                                                  distance_upper_bound=p.gap_au)[0]).mean()
                    quick = (min(fa * a.area_au2, fb * b.area_au2) - collision_fast(a, b, T, p)) / mean_area
                    grid.append((quick, f"na wprost {tw:.0f}° / {sh:+.0f} a.u.", T))
    grid.sort(key=lambda g: -g[0])
    starts = [(label, T) for _, label, T in grid[:p.constrain_grid_keep]] + plane_starts_for(a, b, p)
    dist = 1.5 * p.coarse_voxel_au
    for seed in range(p.ransac_seeds if p.constrain_ransac else 0):
        o3d.utility.random.seed(seed)
        r = reg.registration_ransac_based_on_feature_matching(
            ca[2], cb[2], ca[3], cb[3], True, dist, reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(dist),
             reg.CorrespondenceCheckerBasedOnNormal(np.deg2rad(p.ransac_normal_angle_deg))],
            reg.RANSACConvergenceCriteria(p.ransac_iterations, 1.0))
        starts.append(("RANSAC", refine_icp(ca, cb, r.transformation, p)))
    best = None
    for n, (label, T0) in enumerate(starts):
        T1 = axis_locked_icp(a, b, T0, p)
        T2 = reg.registration_icp(ca[0], cb[0], 2 * p.sample_au, T1, reg.TransformationEstimationPointToPlane(),
                                  reg.ICPConvergenceCriteria(max_iteration=50)).transformation
        tries = [(T1, f"{label}, oś na wprost")]
        ax2 = axis_fit(a, b, T2, p)
        if ax2 is not None and ax2["axis_deg"] <= p.axis_max_deg:
            tries.append((T2, f"{label}, + mały kąt"))
        for T, how in tries:
            sc = evaluate(a, b, T, p)
            if best is None or sc["score"] > best["score"]:
                best = {**sc, "T": T.tolist(), "seed": n, "start": how, "ransac": p.constrain_ransac}
    return best


def match_pair(a: Surface, b: Surface, p: SurfaceMatchParams) -> dict:
    """Szuka T kładącego powierzchnię a na b (twarzą w twarz). Zwraca najlepszą próbę.

    Próby: najpierw starty z atlasu (atlas_starts) i wzdłuż osi (axis_starts_for) -> refine_icp
    z pierwszym promieniem start_icp_au, potem p.ransac_seeds prób RANSAC na FPFH (ziarno = numer
    próby RANSAC) -> refine_icp. RANSAC jest pomijany, gdy najlepszy z tych startów pokrył obie
    powierzchnie co najmniej w p.start_enough_cover przy wadze osi >= 0.5. Każda próba oceniana
    przez evaluate; wygrywa największy wynik.

    Args:
        a, b: powierzchnie dwóch różnych kości.
        p:    ustawienia dopasowania.

    Returns:
        {"score", "contact_au2", "collision_au2", "cover_a", "cover_b", "rmse_au",
         "T": macierz 4x4 (lista) z układu kości a do układu kości b, "seed": numer wygranej
         próby (od 0, starty z atlasu i z osi pierwsze), "start": "RANSAC", "atlas: <model>"
         albo "oś <obrót>°", "ransac": czy liczono RANSAC} (+ pola osi i atlasu z evaluate).
    """
    if p.mode == "plane":
        return match_pair_plane(a, b, p)
    if p.long_mode == "gap" and long_pair(a, b, p):
        return match_long_pair(a, b, p)
    if p.axis_constrain and long_pair(a, b, p):
        return match_long_constrained(a, b, p)
    ca, cb = a.clouds(p, flip=True), b.clouds(p, flip=False)
    best, n = None, 0

    def keep(T: np.ndarray, start: str) -> None:
        """Ocenia próbę n i zapamiętuje ją, jeśli jest najlepsza."""
        nonlocal best, n
        sc = evaluate(a, b, T, p)
        if best is None or sc["score"] > best["score"]:
            best = {**sc, "T": T.tolist(), "seed": n, "start": start}
        n += 1

    for label, _, T0 in atlas_starts(a, b, p):
        keep(refine_icp(ca, cb, T0, p, first_au=p.start_icp_au), f"atlas: {label}")
    for label, T0 in axis_starts_for(a, b, p) + plane_starts_for(a, b, p):
        keep(refine_icp(ca, cb, T0, p, first_au=p.start_icp_au), label)
    ransac = (best is None or min(best["cover_a"], best["cover_b"]) < p.start_enough_cover
              or best.get("axis_weight", 1.0) < 0.5)
    if ransac:
        dist = 1.5 * p.coarse_voxel_au             # maks. odległość punktów uznanych za pasujące w RANSAC
        for seed in range(p.ransac_seeds):
            o3d.utility.random.seed(seed)
            coarse = reg.registration_ransac_based_on_feature_matching(
                ca[2], cb[2], ca[3], cb[3], True, dist,
                reg.TransformationEstimationPointToPoint(False), 3,
                [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                 reg.CorrespondenceCheckerBasedOnDistance(dist),
                 reg.CorrespondenceCheckerBasedOnNormal(np.deg2rad(p.ransac_normal_angle_deg))],
                reg.RANSACConvergenceCriteria(p.ransac_iterations, 1.0))    # 1.0 = bez wczesnego końca
            keep(refine_icp(ca, cb, coarse.transformation, p), "RANSAC")
    if best is not None and p.axis_lock and long_pair(a, b, p):
        keep(axis_locked_icp(a, b, np.asarray(best["T"]), p), f"{best['start']} + na wprost")
    if best is not None:
        best["ransac"] = ransac
    return best


def split_pairs(surfaces: list[Surface], p: SurfaceMatchParams) -> tuple[list[tuple[int, int]], list[dict]]:
    """Pary powierzchni z różnych kości: do dopasowania i pominięte (atlas: na pewno różne kości, pola różne).

    Args:
        surfaces: powierzchnie z extract_surfaces.
        p:        ustawienia (use_atlas, atlas_skip_different).

    Returns:
        ([(i, j) indeksy w `surfaces`] do dopasowania,
         [{"a", "b", "a_name", "b_name", "atlas_same", "atlas_diff"}] pominięte).
    """
    pairs, skipped = [], []
    for i in range(len(surfaces)):
        for j in range(i + 1, len(surfaces)):
            a, b = surfaces[i], surfaces[j]
            if a.bone.id == b.bone.id:
                continue
            same, diff = atlas_relation(a, b) if p.use_atlas else (0.0, 0.0)
            ratio = max(a.area_au2, b.area_au2) / max(min(a.area_au2, b.area_au2), 1e-9)
            why = (f"atlas: różne kości ({diff:.0%})" if diff > 0 and diff >= p.atlas_skip_different else
                   f"pola przełomów różnią się {ratio:.1f}x" if p.max_area_ratio > 0 and ratio > p.max_area_ratio else "")
            if why:
                skipped.append({"a": a.key, "b": b.key, "a_name": a.name, "b_name": b.name, "why": why,
                                "atlas_same": round(same, 4), "atlas_diff": round(diff, 4)})
            else:
                pairs.append((i, j))
    return pairs, skipped


def acceptable(r: dict, p: SurfaceMatchParams) -> bool:
    """Czy para może wejść do złożenia: wynik > 0 i >= assembly_min_score, pokrycie każdej strony >=
    assembly_min_cover, kolizja <= assembly_max_collision x styk (0 = bez tego warunku). W trybie "gap"
    pokrycie i styk liczone z bliskości przełomów (near_a / near_b, proximity)."""
    gap = "proximity" in r                 # tryb "gap": pokrycie i styk = bliskość przełomów (ze szczeliną)
    cover = (r.get("near_a", 1.0), r.get("near_b", 1.0)) if gap else (r.get("cover_a", 1.0), r.get("cover_b", 1.0))
    contact = (r["proximity"] * 0.5 * (r.get("a_area_au2", 0.0) + r.get("b_area_au2", 0.0)) if gap
               else r.get("contact_au2", 0.0))
    return (r["score"] > 0 and r["score"] >= p.assembly_min_score and min(cover) >= p.assembly_min_cover
            and (p.assembly_max_collision <= 0 or r.get("collision_au2", 0.0) <= p.assembly_max_collision * max(contact, 1e-9)))


def _bone_of(key: str) -> str:
    """Id kości z klucza powierzchni "<id kości>#<numer>"."""
    return key.rsplit("#", 1)[0]


def assemble(results: list[dict], p: SurfaceMatchParams | None = None,
             surfaces: list[Surface] | None = None) -> dict:
    """Proponowane złożenie (docstring modułu, "PROPONOWANE ZŁOŻENIE"). Ustawia w każdym wyniku "chosen".

    Bez wiązki (assembly_beam = 0 albo brak `surfaces`): zachłannie od najlepszej pary - każdy przełom
    i każda para kości najwyżej raz. Z wiązką: całe grupy fragmentów z kontrolą kolizji grup (_assemble_beam).

    Args:
        results:  wyniki match_all (od najlepszego), zmieniane w miejscu ("chosen").
        p:        ustawienia (assembly_*); None = domyślne.
        surfaces: powierzchnie z extract_surfaces (kości z chmurami - do kolizji grup).

    Returns:
        {"chosen": wybrane pary od najlepszej, "groups": {id kości: numer grupy}, "poses": {id kości:
         4x4 (lista) - położenie kości w układzie jej grupy}} ("groups" i "poses" tylko z wiązką).
    """
    p = p or SurfaceMatchParams()
    for r in results:
        r["chosen"] = False
    cands = [r for r in results if acceptable(r, p)]
    if p.assembly_beam > 0 and surfaces:
        out = _assemble_beam(cands, surfaces, p)
    else:
        used, bone_pairs, chosen = set(), set(), []
        for r in cands:
            pair = frozenset((_bone_of(r["a"]), _bone_of(r["b"])))
            if r["a"] not in used and r["b"] not in used and pair not in bone_pairs:
                used |= {r["a"], r["b"]}
                bone_pairs.add(pair)
                chosen.append(r)
        out = {"chosen": chosen, "groups": {}, "poses": {}}
    for r in out["chosen"]:
        r["chosen"] = True
    return out


def _assemble_beam(cands: list[dict], surfaces: list[Surface], p: SurfaceMatchParams) -> dict:
    """Składanie całych grup (jak w OSTEOFORM): wiązka p.assembly_beam częściowych złożeń.

    Stan = położenie każdej kości w układzie jej grupy, użyte przełomy i suma wyników. Krok: każda
    para łącząca dwie RÓŻNE grupy wolnymi przełomami przenosi sztywno całą grupę kości A:
    pose[j] = pose[B] @ T @ pose[A]⁻¹ @ pose[j]. Połączenie odpada, gdy którekolwiek dwa fragmenty
    obu grup wchodzą w siebie (część próbki chmury kości głębiej niż collision_au) więcej niż
    assembly_max_penetration. Wynik = stan o największej sumie wyników.

    Returns:
        Jak assemble.
    """
    by_bone: dict[str, Surface] = {}
    for s in surfaces:
        by_bone.setdefault(s.bone.id, s)
    bones = sorted(by_bone)
    sample = {b: np.asarray(by_bone[b].bone.cloud.points)[::max(1, len(by_bone[b].bone.cloud.points) // 1500)]
              for b in bones}

    def penetration(j: str, k: str, Tj: np.ndarray, Tk: np.ndarray) -> float:
        """Część próbki kości j (w położeniu Tj) leżąca głębiej niż collision_au pod powierzchnią kości k."""
        ps = transform_points(np.linalg.inv(Tk) @ Tj, sample[j])
        dst = by_bone[k]
        d, idx = dst.bone_tree.query(ps, distance_upper_bound=p.collision_search_au)
        ok = np.isfinite(d)
        if not ok.any():
            return 0.0
        pd, nd = np.asarray(dst.bone.cloud.points), np.asarray(dst.bone.cloud.normals)
        depth = np.einsum("ij,ij->i", pd[idx[ok]] - ps[ok], nd[idx[ok]])
        return float((depth > p.collision_au).sum() / len(ps))

    start = {"pose": {b: np.eye(4) for b in bones}, "group": {b: i for i, b in enumerate(bones)},
             "used": frozenset(), "edges": (), "score": 0.0}
    beam, best, seen = [start], start, set()
    for _ in range(len(bones) - 1):
        new = []
        for st in beam:
            for i, r in enumerate(cands):
                if r["a"] in st["used"] or r["b"] in st["used"]:
                    continue
                ba, bb = _bone_of(r["a"]), _bone_of(r["b"])
                ga, gb = st["group"][ba], st["group"][bb]
                key = frozenset(st["edges"] + (i,))
                if ga == gb or key in seen:
                    continue
                seen.add(key)
                delta = st["pose"][bb] @ np.asarray(r["T"]) @ np.linalg.inv(st["pose"][ba])
                pose = dict(st["pose"])
                movers = [j for j in bones if st["group"][j] == ga]
                stays = [k for k in bones if st["group"][k] == gb]
                for j in movers:
                    pose[j] = delta @ pose[j]
                worst = max(max(penetration(j, k, pose[j], pose[k]), penetration(k, j, pose[k], pose[j]))
                            for j in movers for k in stays)
                if worst > p.assembly_max_penetration:
                    continue
                group = {j: (gb if g == ga else g) for j, g in st["group"].items()}
                new.append({"pose": pose, "group": group, "used": st["used"] | {r["a"], r["b"]},
                            "edges": st["edges"] + (i,), "score": st["score"] + r["score"]})
        if not new:
            break
        new.sort(key=lambda x: -x["score"])
        beam = new[:p.assembly_beam]
        if beam[0]["score"] > best["score"]:
            best = beam[0]
    chosen = sorted((cands[i] for i in best["edges"]), key=lambda r: -r["score"])
    return {"chosen": chosen, "groups": dict(best["group"]),
            "poses": {b: np.round(best["pose"][b], 6).tolist() for b in bones}}


def match_all(surfaces: list[Surface], p: SurfaceMatchParams, progress=print) -> list[dict]:
    """Dopasowuje każdą parę powierzchni z różnych kości. Wynik posortowany od najlepszej pary.

    Liczba par rośnie kwadratowo z liczbą powierzchni (~N²/2), każda para trwa ~1-3 s.
    Przy use_atlas pary na pewno różnych kości są pomijane (split_pairs).

    Args:
        surfaces: powierzchnie z extract_surfaces.
        p:        ustawienia dopasowania.
        progress: funkcja do wypisywania postępu (co 10 par); lambda _: None = cisza.

    Returns:
        Wyniki match_pair uzupełnione o opis pary: "a", "b" (Surface.key), "a_name",
        "b_name", "a_area_au2", "b_area_au2", "a_manual", "b_manual"; od największego "score".
        Złożenie z tych par: assemble(results, p, surfaces).
    """
    pairs, skipped = split_pairs(surfaces, p)
    note = f" (pominięte {len(skipped)} - atlas: różne kości)" if skipped else ""
    progress(f"Powierzchnie: {len(surfaces)}, pary do sprawdzenia: {len(pairs)}{note}")
    results = []
    start = time.perf_counter()
    for n, (i, j) in enumerate(pairs, 1):
        a, b = surfaces[i], surfaces[j]
        r = match_pair(a, b, p)
        r.update(a=a.key, b=b.key, a_name=a.name, b_name=b.name, a_area_au2=a.area_au2, b_area_au2=b.area_au2,
                 a_manual=a.is_manual, b_manual=b.is_manual)
        results.append(r)
        if n % 10 == 0 or n == len(pairs):
            progress(f"  {n}/{len(pairs)} par ({time.perf_counter() - start:.0f} s)")
    results.sort(key=lambda r: -r["score"])
    return results


def json_safe(obj):
    """Kopia `obj` z NaN / nieskończonościami zamienionymi na None (w JSON: null).

    Python zapisuje float("nan") jako NaN, a to nie jest poprawny JSON - przeglądarka
    (JSON.parse) odrzuca cały plik. NaN ma np. rmse_au pary bez styku.

    Args:
        obj: słownik / lista / liczba / tekst (zagnieżdżone dowolnie).

    Returns:
        Ten sam obiekt z niekończonymi liczbami zamienionymi na None.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def report(surfaces: list[Surface], results: list[dict], p: SurfaceMatchParams,
           contact_params: ContactParams = ContactParams(), assembly: dict | None = None) -> dict:
    """Słownik do zapisu w matching.json (i do visualization_3d.match_matrix_figure).

    Args:
        surfaces:       powierzchnie z extract_surfaces.
        results:        wyniki match_all dla tych powierzchni.
        p:              ustawienia dopasowania, z którymi liczono wyniki.
        contact_params: ustawienia wykrywania (ważne w trybach "auto" i "manual-auto").

    Returns:
        {"params": ustawienia dopasowania, "contact_params": ustawienia wykrywania,
         "surfaces": [{"key", "bone", "bone_label", "surface", "name", "source",
                       "area_au2", "points"}],
         "pairs": wyniki match_all (liczby zaokrąglone do 4 miejsc),
         przy use_atlas także "atlas": {id kości: kontekst z atlas_context} (do match_tool) i
         "skipped": pary pominięte (split_pairs)}.
    """
    rep = {"params": asdict(p), "contact_params": asdict(contact_params),
           "surfaces": [{"key": s.key, "bone": s.bone.id, "bone_label": s.bone.label, "surface": s.sid,
                         "name": s.contacts.name(s.sid), "source": s.contacts.source,
                         "area_au2": round(s.area_au2, 1), "points": len(s.points),
                         "flat_rms_au": round(s.fit(p)["rms_au"], 3)} for s in surfaces],
           "pairs": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in results]}
    if p.use_atlas:
        rep["atlas"] = {s.bone.id: s.atlas for s in surfaces if s.atlas is not None}
    skipped = split_pairs(surfaces, p)[1]
    if skipped:
        rep["skipped"] = skipped
    if assembly is not None:
        rep["assembly"] = {"chosen": [[r["a"], r["b"]] for r in assembly["chosen"]],
                           "groups": assembly["groups"], "poses": assembly["poses"]}
    return rep

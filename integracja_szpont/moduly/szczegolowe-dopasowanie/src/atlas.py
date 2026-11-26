"""
Atlas anatomiczny: baza modeli 3D KOMPLETNYCH kości jako kontekst przy wykrywaniu powierzchni
styku (przełomów) - algorytm "atlas" (contacts.py, @detector) i rozpoznawanie, z której kości
pochodzi fragment (scripts/identify_bones.py).

POMYSŁ
    Zeskanowany fragment to wycinek jakiejś kości. Jego KORA (naturalna, zewnętrzna powierzchnia)
    leży na zewnętrznej powierzchni tej kości, a PRZEŁOM nie: odsłonięte wnętrze jest POD
    powierzchnią zdrowej kości albo ma zupełnie inną normalną. Kość długa ma w środku kanał
    szpikowy - na skanie przełomu to nie dziura, tylko WKLĘSŁY dół (skaner widzi ściany kanału),
    który też leży głęboko pod powierzchnią kości. Gdy wiadomo, z której kości pochodzi fragment
    i jak na niej leży, wiadomo też, co jest korą - także tam, gdzie kora jest naturalnie
    szorstka (nasady, przyczepy mięśni), a przełom gładki.

PLIKI MODELI (settings.json "paths" -> "atlas", domyślnie atlas/; jednorazowo --atlas):
    atlas/
        kość strzałkowa prawa.stl     <- jeden plik = jeden model, nazwa pliku = nazwa kości
        piszczel/piszczel A.ply       <- podfolder = typ kości (ranking sumuje też typy)
    Formaty .ply / .obj / .stl (ten sam model w kilku = jeden: PLY -> OBJ -> STL); pliki i foldery
    zaczynające się od "." albo "_" są pomijane. Jednostki modelu mogą być inne niż skanu:
    "reference_scale" = ile a.u. skanu to 1 jednostka modelu, "scale_range" = różnice osobnicze.

BAZA = INDEKS (settings.json "paths" -> "atlas_index", domyślnie atlas_index/; sync_index):
    Każdy model jest RAZ wczytywany i zamieniany na mały plik <id>.npz (~1 MB zamiast dziesiątek MB):
    - tylko ZEWNĘTRZNA powierzchnia (ray casting: punkt, z którego żaden promień nie ucieka na
      zewnątrz, leży w środku - np. ściana kanału szpikowego w modelu z tomografii),
    - jej punkty i normalne, skwantyzowane (uint16 / int8),
    - WŁASNOŚCI do sita: długość kości, pole, histogram krzywizn (indeks kształtu x krzywizna),
    - mała siatka do podglądu.
    Deskryptory FPFH NIE są zapisywane - są liczone w locie, tylko dla modeli, które przejdą sito,
    i tylko w potrzebnej skali. Do wykrywania potrzebny jest tylko indeks: oryginalne pliki modeli
    można po zindeksowaniu usunąć albo trzymać gdzie indziej (indeks pamięta model bez pliku;
    usunięcie z indeksu: build_atlas.py --prune). Kolejne uruchomienia czytają gotowy indeks,
    liczone są tylko nowe / zmienione pliki (suma kontrolna), plik o zmienionej nazwie jest
    rozpoznawany po zawartości.

DOPASOWANIE FRAGMENTU (match_fragment; wynik w cache out/3d/<zestaw>/cache/atlas/<id>.npz):
    0. ANALIZA LOKALNA fragmentu: wynik "features" (szorstkość, rozrzut normalnych, gęstość siatki)
       + WGŁĘBIENIA (kanał szpikowy): z wnętrza wklęsłego dołu promienie trafiają w jego ściany
       (osłonięcie >= cavity_occlusion), z kory - uciekają. Daje "prawdopodobną korę" fragmentu.
    1. SITO WŁASNOŚCI (screen) - bez RANSAC, dla każdego modelu w ułamku sekundy:
       - rozmiar: fragment nie może być dłuższy niż kość przy największej skali -> też dolna
         granica skali,
       - krzywizny: kora fragmentu jest częścią powierzchni kości, więc w każdym przedziale
         histogramu (indeks kształtu Koenderinka - nie zależy od skali; krzywizna ogólna - skala
         tylko PRZESUWA histogram) pole kory fragmentu nie może przekraczać pola kości (x s²).
         Część kory, której kość "nie ma" = niewyjaśniona; powyżej max_unexplained przy każdej
         skali -> model wykluczony. Przy okazji wychodzą najbardziej prawdopodobne skale.
       Dalej idzie najwyżej max_models modeli, każdy z scale_candidates skalami.
    2. USTAWIENIE: RANSAC na FPFH (ransac_iterations x ransac_seeds, też dla odbicia lustrzanego =
       druga strona ciała) i odporny ICP ze SKALĄ (point-to-plane, Gauss-Newton w 7 parametrach:
       skala, obrót, przesunięcie): punkty daleko od powierzchni modelu albo z niezgodną normalną
       (przełom, kanał szpikowy) mają wagę ~0, więc nie psują ustawienia.
    3. OCENA (fit): jaka część prawdopodobnej kory leży na powierzchni modelu (bliżej niż
       tolerance_au, normalne zgodne do normal_angle_deg) minus outside_penalty x część fragmentu
       wystająca NA ZEWNĄTRZ modelu (w dobrej kości przełom jest w środku, nic nie wystaje).
    4. PRAWDOPODOBIEŃSTWA: softmax(fit / temperature) z dodatkową możliwością "żadna z bazy"
       o ocenie min_fit - gdy nic nie pasuje (albo sito wykluczyło wszystko), prawdopodobieństwo
       trafia do "żadna z bazy" = niska pewność.

KORA vs PRZEŁOM (dowód z atlasu): dla top_k najbardziej prawdopodobnych modeli każdy wierzchołek
fragmentu dostaje log-szanse "przełom vs kora":
    na powierzchni modelu, normalna zgodna            -> kora,
    pod powierzchnią (wnętrze kości, kanał szpikowy)
    albo na powierzchni z inną normalną               -> przełom,
    na zewnątrz modelu albo daleko                    -> brak dowodu (0) - decyduje analiza lokalna.
Średnia ważona prawdopodobieństwami modeli, razy pewność (suma tych prawdopodobieństw).

ALGORYTM "atlas" (detect): log-szanse analizy lokalnej ((wynik features - feature_threshold) /
local_scale + cavity_weight x wgłębienie) + atlas_weight x pewność x dowód z atlasu; dalej jak
"features" (contacts.regions_from_score): uśrednienie, próg 0, bez pasa przy brzegu skanu,
otwarcie, spójne obszary. Przy pewności < min_confidence - sama analiza lokalna.
Ranking modeli trafia do wyniku: "info" -> "atlas" w contacts/<zestaw>/auto/<id>.json.

    import atlas, config
    p = config.atlas_params()
    db = atlas.load_atlas(p)                         # baza (dolicza nowe pliki modeli)
    m = atlas.cached_match(bone, db, p, config.contact_params(), config.surface_feature_params())
    print("\\n".join(m.ranking_lines()))
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

import contacts as ct
from config import read_json
from data_parsing import compute_fpfh, load_mesh, repair_mesh, sample_surface, slugify, to_cloud, voxel_keep

reg = o3d.pipelines.registration

INDEX_FORMAT = "bones-smm-atlas/1"
MATCH_FORMAT = "bones-smm-atlas-match/2"
MODEL_EXTENSIONS = ("ply", "obj", "stl")     # kolejność = pierwszeństwo, gdy model jest w kilku formatach
MIRROR = np.array([-1.0, 1.0, 1.0])          # odbicie lustrzane (druga strona ciała): x -> -x
SI_BINS = 10                                 # przedziały indeksu kształtu w [-1, 1]
LOGC_RANGE = (-3.5, 1.0)                     # zakres log10 krzywizny ogólnej [1 / jednostka]
LOGC_STEP = 0.05                             # szerokość przedziału log10 krzywizny (~12%) = krok skali sita
OUTER_RAYS = 24                              # promieni na punkt przy szukaniu zewnętrznej powierzchni modelu
CAVITY_RAYS = 16                             # promieni na wierzchołek przy szukaniu wgłębień fragmentu
WINDOW_LENGTHS = (0.2, 0.35, 0.5, 0.75)      # okna wzdłuż osi kości (część długości) - histogramy kawałków kości
WINDOW_STRIDE = 0.1                          # przesunięcie kolejnych okien (część długości)
SCREEN_SLACK = 2.5                           # sito: fragment może mieć do tylu razy więcej danej krzywizny niż okno kości
VIEW_TRIANGLES = 6000                        # trójkątów siatki podglądu modelu
MAX_POINTS = 300000                          # najwięcej zapisanych punktów modelu
CLOUD_CACHE = 12                             # ile chmur FPFH (model x skala x odbicie) trzymać w pamięci
EVIDENCE_EPS = 0.05                          # dowód z atlasu w [-log(1.05/0.05), +...] ~ [-3, 3]
LOCAL_CLIP = 4.0                             # głos analizy lokalnej (szorstkość itd.) najwyżej +-4 log-szans
COARSE_ALONG = 100                           # RANSAC: najwyżej tyle odstępów punktów na długość fragmentu (dłuższy = rzadziej)
SHARED_MIN_P = 0.5                           # wspólna skala: tylko fragmenty, dla których model ma p >= tyle
SHARED_MIN_AGREE = 0.6                       # one_person = "auto": wspólna skala, gdy tyle wagi głosów leży w ±tolerancji
UNKNOWN = "żadna z bazy"
# Ustawienia, które zmieniają tylko łączenie wyników (nie dopasowanie) - bez liczenia dopasowania od nowa.
FUSION_ONLY = ("atlas_weight", "min_confidence")


@dataclass(frozen=True)
class AtlasParams:
    """Ustawienia atlasu (settings.json, sekcja "atlas"). Długości w a.u. skanu."""

    # skala: model w innych jednostkach, inna wielkość osobnicza
    reference_scale: float = 1.0       # ile a.u. skanu to 1 jednostka modelu (model w mm, skan w a.u. ~ mm -> 1)
    scale_range: tuple = (0.75, 1.35)  # dopuszczalna skala względem reference_scale (różnice osobnicze, skaner)
    mirror: bool = True                # sprawdzaj też odbicie lustrzane modelu (druga strona ciała)
    model_scales: dict = field(default_factory=dict)  # {id modelu: skala} wspólna dla zestawu (shared_scales)
    one_person: object = "auto"        # zestaw od jednej osoby: true / false / "auto" (= gdy skale fragmentów się zgadzają)
    shared_scale_tolerance: float = 0.1  # ... model z wpisem: skala tylko w zakresie +- tyle (0.1 = 10%)
    # sito własności (przed RANSAC)
    curvature_radius_au: float = 2.5   # promień liczenia krzywizn [a.u.]
    size_tolerance: float = 0.15       # zapas na różnice osobnicze w rozmiarze i polu (0.15 = 15%)
    max_unexplained: float = 0.2       # wyklucz model, gdy taka część kory fragmentu nie ma odpowiednika w kości
    max_models: int = 5                # najwięcej modeli po sicie do dokładnego dopasowania
    scale_candidates: int = 2          # ile najlepszych skal z sita sprawdzać dla modelu
    # ustawienie (RANSAC + ICP ze skalą)
    voxel_au: float = 1.0              # odstęp punktów do zgrubnego ustawienia [a.u.]
    fpfh_radius_au: float = 5.0        # promień FPFH [a.u.]
    ransac_iterations: int = 200000    # iteracji RANSAC na skalę i odbicie
    ransac_seeds: int = 1              # niezależnych prób RANSAC (wygrywa najlepsza)
    icp_iterations: int = 30           # kroków odpornego ICP ze skalą
    # ocena
    tolerance_au: float = 1.0          # kora = fragment bliżej powierzchni modelu niż to ...
    normal_angle_deg: float = 40.0     # ... z normalnymi zgodnymi z dokładnością do tego kąta
    outside_penalty: float = 1.0       # kara za część fragmentu wystającą na zewnątrz modelu
    min_fit: float = 0.55              # ocena "żadna z bazy": modele dopasowane gorzej są mniej prawdopodobne
    temperature: float = 0.05          # rozmycie rozkładu prawdopodobieństwa (mniej = ostrzej)
    top_k: int = 3                     # ile najbardziej prawdopodobnych modeli rozstrzyga korę / przełom
    # łączenie z analizą lokalną
    atlas_weight: float = 2.0          # waga dowodu z atlasu względem analizy lokalnej
    local_scale: float = 0.3           # tyle wyniku "features" = 1 jednostka log-szans
    min_confidence: float = 0.8        # poniżej tej pewności atlasu - sama analiza lokalna
    cavity_weight: float = 3.0         # log-szanse przełomu dla wgłębienia (kanał szpikowy); 0 = bez
    cavity_occlusion: float = 0.55     # od takiej części zasłoniętych promieni punkt leży we wgłębieniu
    cavity_depth_au: float = 15.0      # promień trafiający dalej niż to nie zasłania [a.u.]


# -----------------
# GEOMETRIA: promienie, krzywizny, podobieństwo
# -----------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Funkcja logistyczna (odporna na duże |x|)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _hemisphere(k: int) -> np.ndarray:
    """k kierunków na półsferze z > 0 (spirala Fibonacciego, gęściej przy biegunie - jak cos(kąta))."""
    i = np.arange(k) + 0.5
    z = np.sqrt(1 - i / k)
    phi = i * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(1 - z ** 2)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def tangent_frames(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dwa jednostkowe wektory styczne prostopadłe do każdej normalnej (N, 3) -> (e1, e2)."""
    a = np.where(np.abs(normals[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e1 = np.cross(normals, a)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    return e1, np.cross(normals, e1)


def ray_hits(mesh: o3d.geometry.TriangleMesh, points: np.ndarray, normals: np.ndarray, rays: int,
             min_dist: float = 0.0, max_dist: float = np.inf) -> np.ndarray:
    """Jaka część promieni wypuszczonych z punktów (półsfera wokół normalnej) trafia w siatkę.

    Promień liczy się jako trafiony, gdy uderza w siatkę w odległości (min_dist, max_dist).
    Na korze (powierzchnia wypukła) promienie uciekają na zewnątrz (~0), we wgłębieniu - np. dole
    po kanale szpikowym na przełomie - trafiają w jego ściany; ze ściany wewnętrznej zamkniętej
    kości nie ucieka żaden (1).

    Args:
        mesh:     siatka.
        points:   (N, 3) punkty na siatce.
        normals:  (N, 3) ich normalne (na zewnątrz).
        rays:     promieni na punkt.
        min_dist: trafienia bliżej nie liczą się (drobna szorstkość) [jednostki siatki].
        max_dist: trafienia dalej nie liczą się [jednostki siatki].

    Returns:
        (N,) w [0, 1].
    """
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    v = np.asarray(mesh.vertices)
    offset = 1e-4 * float(np.linalg.norm(np.ptp(v, axis=0))) + 1e-6
    local = _hemisphere(rays)
    e1, e2 = tangent_frames(normals)
    out = np.zeros(len(points))
    step = max(1, 2_000_000 // rays)
    for a in range(0, len(points), step):
        sl = slice(a, a + step)
        d = (local[None, :, 0, None] * e1[sl, None, :] + local[None, :, 1, None] * e2[sl, None, :]
             + local[None, :, 2, None] * normals[sl, None, :])
        o = np.broadcast_to((points[sl] + offset * normals[sl])[:, None, :], d.shape)
        t = scene.cast_rays(o3d.core.Tensor(np.concatenate([o, d], axis=-1).reshape(-1, 6).astype(np.float32)))
        t = t["t_hit"].numpy().reshape(len(d), rays)
        out[sl] = ((t > min_dist) & (t < max_dist)).mean(axis=1)
    return out


def principal_curvatures(points: np.ndarray, normals: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Krzywizny główne k1 >= k2 w każdym punkcie (z rozkładu normalnych sąsiadów w promieniu `radius`).

    Operator kształtu S (zmiana normalnej przy przesunięciu po powierzchni) dopasowany najmniejszymi
    kwadratami: n_j - n_i ~ S (p_j - p_i) dla sąsiadów j, w płaszczyźnie stycznej. Wartości własne
    S = krzywizny główne; znak: wypukłe > 0 (sfera o promieniu R: k1 = k2 = 1/R), wklęsłe < 0.

    Args:
        points:  (N, 3) równomierne punkty powierzchni.
        normals: (N, 3) ich normalne (na zewnątrz).
        radius:  promień sąsiedztwa.

    Returns:
        (k1, k2, valid) - (N,) każde; valid = dość sąsiadów do oszacowania.
    """
    import surface_features as sf

    p = points - points.mean(axis=0)
    n = normals
    W, counts = sf._neighbours(p, p, radius)
    c = counts.astype(float)[:, None, None]

    def outer(a, b):
        """Iloczyny zewnętrzne wierszy (N, 3, 3)."""
        return np.einsum("ni,nj->nij", a, b)

    Sp, Sn = W @ p, W @ n
    Spp = (W @ outer(p, p).reshape(-1, 9)).reshape(-1, 3, 3)
    Snp = (W @ outer(n, p).reshape(-1, 9)).reshape(-1, 3, 3)
    C = Spp - outer(p, Sp) - outer(Sp, p) + c * outer(p, p)            # sum d d^T
    M = Snp - outer(Sn, p) - outer(n, Sp) + c * outer(n, p)            # sum (n_j - n_i) d^T
    e1, e2 = tangent_frames(n)
    E = np.stack([e1, e2], axis=-1)                                    # (N, 3, 2)
    A = np.einsum("nki,nkl,nlj->nij", E, M, E)
    B = np.einsum("nki,nkl,nlj->nij", E, C, E)
    det = B[:, 0, 0] * B[:, 1, 1] - B[:, 0, 1] * B[:, 1, 0]
    valid = (counts >= 8) & (det > 1e-12 * np.maximum(c[:, 0, 0], 1) ** 2 * radius ** 4)
    Binv = np.zeros_like(B)
    sd = np.where(valid, det, 1.0)
    Binv[:, 0, 0], Binv[:, 1, 1] = B[:, 1, 1] / sd, B[:, 0, 0] / sd
    Binv[:, 0, 1], Binv[:, 1, 0] = -B[:, 0, 1] / sd, -B[:, 1, 0] / sd
    S = A @ Binv
    S = 0.5 * (S + np.transpose(S, (0, 2, 1)))
    m = 0.5 * (S[:, 0, 0] + S[:, 1, 1])
    d = np.sqrt(0.25 * (S[:, 0, 0] - S[:, 1, 1]) ** 2 + S[:, 0, 1] ** 2)
    return m + d, m - d, valid


def curvature_histogram(k1: np.ndarray, k2: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Histogram (indeks kształtu x log10 krzywizny ogólnej) ważony polem.

    Indeks kształtu (Koenderink) SI = 2/pi atan((k1 + k2) / (k1 - k2)) w [-1, 1]: 1 = czasza
    wypukła, 0.5 = walec (trzon kości), 0 = siodło, -0.5 = rynna, -1 = dołek. NIE zależy od skali.
    Krzywizna ogólna C = sqrt((k1² + k2²) / 2) ~ 1 / promień - skala s zmienia log10 C o -log10 s,
    czyli tylko przesuwa histogram wzdłuż tej osi.

    Args:
        k1, k2:  krzywizny główne (k1 >= k2).
        weights: pole przypadające na punkt.

    Returns:
        (SI_BINS, liczba przedziałów log C) - pole w każdym przedziale.
    """
    si = (2 / np.pi) * np.arctan2(k1 + k2, np.maximum(k1 - k2, 0))
    logc = np.log10(np.maximum(np.sqrt(0.5 * (k1 ** 2 + k2 ** 2)), 1e-12))
    nb = int(round((LOGC_RANGE[1] - LOGC_RANGE[0]) / LOGC_STEP))
    a = np.clip(((si + 1) / 2 * SI_BINS).astype(int), 0, SI_BINS - 1)
    b = np.clip(((logc - LOGC_RANGE[0]) / LOGC_STEP).astype(int), 0, nb - 1)
    h = np.zeros((SI_BINS, nb))
    np.add.at(h, (a, b), weights)
    return h


def orient_outward(mesh: o3d.geometry.TriangleMesh) -> bool:
    """Odwraca trójkąty, jeśli normalne siatki patrzą do środka (ujemna objętość). Zwraca True, gdy odwrócono."""
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    c = v.mean(axis=0)
    a, b, d = v[t[:, 0]] - c, v[t[:, 1]] - c, v[t[:, 2]] - c
    if np.einsum("ij,ij->i", a, np.cross(b, d)).sum() < 0:
        mesh.triangles = o3d.utility.Vector3iVector(t[:, [0, 2, 1]])
        return True
    return False


@dataclass
class Pose:
    """Podobieństwo model -> fragment: x_fragment = s R x_modelu + t (model po ewentualnym odbiciu)."""

    s: float
    R: np.ndarray
    t: np.ndarray

    def matrix(self) -> np.ndarray:
        """Macierz 4x4 [[s R, t], [0, 1]]."""
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.s * self.R, self.t
        return T

    @classmethod
    def from_matrix(cls, T) -> Pose:
        """Odwrotność matrix()."""
        T = np.asarray(T, dtype=float)
        s = float(np.cbrt(np.linalg.det(T[:3, :3])))
        return cls(s, T[:3, :3] / s, T[:3, 3].copy())

    def to_model(self, x: np.ndarray) -> np.ndarray:
        """Punkty fragmentu -> układ modelu (po odbiciu): R^T (x - t) / s."""
        return ((x - self.t) @ self.R) / self.s


# -----------------
# PLIKI MODELI I BAZA (INDEKS)
# -----------------

@dataclass
class ModelFile:
    """Plik modelu w folderze atlasu (find_models)."""

    id: str           # nazwa ASCII - nazwa pliku w indeksie
    name: str         # nazwa do wyświetlania = nazwa pliku bez rozszerzenia
    group: str        # podfolder (typ kości), "" = bez
    path: Path


def find_models(source_dir: Path) -> list[ModelFile]:
    """Pliki modeli w folderze atlasu (z podfolderami).

    Args:
        source_dir: folder z modelami (może nie istnieć).

    Returns:
        Lista po jednym pliku na model (PLY -> OBJ -> STL), posortowana po ścieżce.
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        return []
    found: dict[str, list[Path]] = {}
    for p in sorted(source_dir.rglob("*")):
        rel = p.relative_to(source_dir)
        if (p.is_file() and p.suffix.lower().lstrip(".") in MODEL_EXTENSIONS
                and not any(part.startswith((".", "_")) for part in rel.parts)):
            found.setdefault(rel.with_suffix("").as_posix(), []).append(p)
    out, used = [], set()
    for key, paths in sorted(found.items()):
        path = min(paths, key=lambda q: MODEL_EXTENSIONS.index(q.suffix.lower().lstrip(".")))
        rel = Path(key)
        mid = base = slugify(key.replace("/", "__")) or "model"
        n = 2
        while mid in used:
            mid, n = f"{base}_{n}", n + 1
        used.add(mid)
        out.append(ModelFile(mid, rel.name, "" if rel.parent == Path(".") else rel.parent.as_posix(), path))
    return out


def quick_fingerprint(source_dir: Path) -> str:
    """Szybki odcisk folderu modeli (ścieżki, rozmiary, daty zmian) - bez czytania plików."""
    items = []
    for m in find_models(source_dir):
        st = m.path.stat()
        items.append([m.path.relative_to(source_dir).as_posix(), st.st_size, st.st_mtime_ns])
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()[:16]


def index_tag(p: AtlasParams) -> dict:
    """Ustawienia, od których zależy zawartość indeksu modelu (inne = model liczony od nowa)."""
    return {"format": INDEX_FORMAT, "reference_scale": p.reference_scale, "voxel_au": p.voxel_au,
            "curvature_radius_au": p.curvature_radius_au, "si_bins": SI_BINS, "logc": [*LOGC_RANGE, LOGC_STEP],
            "windows": [*WINDOW_LENGTHS, WINDOW_STRIDE]}


def read_index(index_dir: Path) -> dict:
    """index.json bazy ({"format", "models": {id: wpis}}); pusty, gdy brak."""
    path = Path(index_dir) / "index.json"
    if path.exists():
        try:
            data = read_json(path)
            if data.get("format") == INDEX_FORMAT:
                return data
        except (OSError, ValueError):
            pass
    return {"format": INDEX_FORMAT, "models": {}}


def _write_index(index_dir: Path, index: dict) -> None:
    """Zapisuje index.json bazy."""
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    (Path(index_dir) / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")


def index_fingerprint(index: dict) -> str:
    """Odcisk zawartości bazy: modele, ich sumy kontrolne i ustawienia indeksu."""
    items = sorted([mid, e.get("checksum"), e.get("tag")] for mid, e in index.get("models", {}).items())
    return hashlib.sha1(json.dumps(items, sort_keys=True).encode()).hexdigest()[:16]


def _quantize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Punkty -> uint16 w prostopadłościanie otaczającym (dokładność = rozmiar / 65535)."""
    lo = points.min(axis=0)
    span = np.maximum(np.ptp(points, axis=0), 1e-9)
    return np.round((points - lo) / span * 65535).astype(np.uint16), lo, span


def build_model(path: Path, p: AtlasParams) -> tuple[dict, dict]:
    """Zamienia plik modelu w dane indeksu (opis: docstring modułu, "BAZA = INDEKS").

    Args:
        path: plik modelu.
        p:    ustawienia (index_tag).

    Returns:
        (tablice do <id>.npz, opis {"area", "length", "extent", "points", "spacing", "outer_share",
         "flipped", "triangles"} w jednostkach modelu).
    """
    from visualization_3d import decimate

    mesh = repair_mesh(load_mesh(path))
    flipped = orient_outward(mesh)
    mesh.compute_vertex_normals()
    area = float(ct.vertex_areas(mesh).sum())
    spacing = max(0.5 * p.voxel_au / p.reference_scale, math.sqrt(area / MAX_POINTS))
    smp = sample_surface(mesh, spacing, density=3.0)
    outer = ray_hits(mesh, smp.points, smp.normals, OUTER_RAYS) < 1 - 1e-9     # choć jeden promień ucieka
    pts, nrm = smp.points[outer], smp.normals[outer]
    if len(pts) < 50:
        raise ValueError(f"{path.name}: brak zewnętrznej powierzchni (zła siatka?)")
    outer_area = area * float(outer.mean())
    rho = p.curvature_radius_au / p.reference_scale
    sub = voxel_keep(pts, rho / 4)
    k1, k2, ok = principal_curvatures(pts[sub], nrm[sub], rho)
    hist = curvature_histogram(k1[ok], k2[ok], np.full(int(ok.sum()), outer_area / max(len(sub), 1)))
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    extent = np.ptp((pts - c) @ vt.T, axis=0)
    # okna wzdłuż osi kości: fragment to spójny kawałek, więc jego krzywizny mają przypominać JAKIŚ kawałek kości
    axial = (pts[sub] - c) @ vt[0]
    axial = axial - axial.min()
    windows, wmeta = [hist], [[0.0, 1.0]]
    for frac in WINDOW_LENGTHS:
        for a in np.arange(0, 1 - frac + 1e-9, WINDOW_STRIDE):
            sel = ok & (axial >= a * extent[0]) & (axial <= (a + frac) * extent[0])
            windows.append(curvature_histogram(k1[sel], k2[sel], np.full(int(sel.sum()), outer_area / max(len(sub), 1))))
            wmeta.append([round(float(a), 3), frac])
    view = decimate(mesh, VIEW_TRIANGLES)
    vv, vtri = np.asarray(view.vertices), np.asarray(view.triangles)
    near = cKDTree(pts).query(vv[vtri].mean(axis=1))[0] < 3 * spacing           # bez ścian wewnętrznych
    vtri = vtri[near]
    q, lo, span = _quantize(pts)
    vq, vlo, vspan = _quantize(vv)
    arrays = {"points": q, "lo": lo, "span": span, "normals": np.round(nrm * 127).astype(np.int8),
              "hist": hist.astype(np.float32), "windows": np.stack(windows).astype(np.float32),
              "windows_meta": np.asarray(wmeta, dtype=np.float32), "view_v": vq, "view_lo": vlo, "view_span": vspan,
              "view_t": vtri.astype(np.uint32)}
    info = {"area": round(outer_area, 2), "length": round(float(extent[0]), 3),
            "extent": np.round(extent, 3).tolist(), "points": int(len(pts)), "spacing": round(spacing, 5),
            "outer_share": round(float(outer.mean()), 4), "flipped": flipped, "triangles": int(len(mesh.triangles))}
    return arrays, info


def sync_index(p: AtlasParams, source_dir: Path | None = None, index_dir: Path | None = None,
               force: bool = False, prune: bool = False, log: Callable | None = print) -> dict:
    """Uzupełnia bazę (indeks) o nowe i zmienione pliki modeli.

    - nowy plik albo zmieniona zawartość (suma kontrolna) albo inne ustawienia indeksu -> liczony,
    - plik bez zmian -> zostaje (sprawdzane najpierw po rozmiarze i dacie, bez czytania),
    - plik o zmienionej nazwie, ale tej samej zawartości -> wpis przenoszony, bez liczenia,
    - model w bazie bez pliku -> ZOSTAJE (baza nie potrzebuje oryginałów), chyba że prune=True.

    Args:
        p:          ustawienia.
        source_dir: folder plików modeli (domyślnie config.ATLAS_DIR); brak folderu = nic do dodania.
        index_dir:  folder bazy (domyślnie config.ATLAS_INDEX).
        force:      licz wszystkie modele z plikami od nowa.
        prune:      usuń z bazy modele, których plików już nie ma.
        log:        funkcja do komunikatów (None = bez).

    Returns:
        {"new", "changed", "renamed", "kept", "only_index", "outdated", "removed": [id, ...],
         "seconds": {id: czas liczenia}}.
    """
    import config
    from merging import file_checksum

    source_dir = Path(source_dir or config.ATLAS_DIR)
    index_dir = Path(index_dir or config.ATLAS_INDEX)
    index = read_index(index_dir)
    models = index["models"]
    tag = index_tag(p)
    report = {k: [] for k in ("new", "changed", "renamed", "kept", "only_index", "outdated", "removed")}
    report["seconds"] = {}
    sources = find_models(source_dir)
    changed = False
    by_checksum = {e["checksum"]: mid for mid, e in models.items()}
    source_ids = {s.id for s in sources}
    for src in sources:
        st = src.path.stat()
        rel = src.path.relative_to(source_dir).as_posix()
        e = models.get(src.id)
        if e and e.get("size") == st.st_size and e.get("mtime_ns") == st.st_mtime_ns:
            checksum = e["checksum"]
        else:
            checksum = file_checksum(src.path)
        old = by_checksum.get(checksum)
        if e is None and old is not None and old not in source_ids and models[old].get("tag") == tag and not force:
            e = models.pop(old)                                    # zmieniona nazwa pliku - ta sama zawartość
            (index_dir / f"{old}.npz").replace(index_dir / f"{src.id}.npz")
            models[src.id] = e
            report["renamed"].append(f"{old} -> {src.id}")
        if (e and e.get("checksum") == checksum and e.get("tag") == tag and (index_dir / f"{src.id}.npz").exists()
                and not force):
            if [e.get("file"), e.get("name"), e.get("group"), e.get("size"), e.get("mtime_ns")] != \
                    [rel, src.name, src.group, st.st_size, st.st_mtime_ns]:
                e.update(file=rel, name=src.name, group=src.group, size=st.st_size, mtime_ns=st.st_mtime_ns)
                changed = True
            report["kept"].append(src.id)
            continue
        if log:
            log(f"  atlas: indeksuję {rel} ...")
        t0 = time.perf_counter()
        arrays, info = build_model(src.path, p)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(index_dir / f"{src.id}.npz", **arrays)
        report["new" if e is None else "changed"].append(src.id)
        report["seconds"][src.id] = round(time.perf_counter() - t0, 1)
        models[src.id] = {"name": src.name, "group": src.group, "file": rel, "checksum": checksum,
                          "size": st.st_size, "mtime_ns": st.st_mtime_ns, "tag": tag, "info": info,
                          "indexed": time.strftime("%Y-%m-%dT%H:%M:%S")}
        changed = True
    for mid in [m for m in models if m not in source_ids]:
        if prune:
            (index_dir / f"{mid}.npz").unlink(missing_ok=True)
            del models[mid]
            report["removed"].append(mid)
            changed = True
        else:
            report["only_index"].append(mid)
            if models[mid].get("tag") != tag:
                report["outdated"].append(mid)
    if changed or not (index_dir / "index.json").exists() and models:
        _write_index(index_dir, index)
    return report


@dataclass
class Model:
    """Model z bazy wczytany do pamięci (tylko zewnętrzna powierzchnia, jednostki modelu)."""

    id: str
    name: str
    group: str
    entry: dict                 # wpis z index.json
    points: np.ndarray          # (N, 3)
    normals: np.ndarray         # (N, 3)
    hist: np.ndarray            # histogram krzywizn (curvature_histogram) całej kości, w polu modelu
    windows: np.ndarray         # (W, SI, log C) histogramy okien wzdłuż osi kości (pierwsze = cała kość)
    view_v: np.ndarray          # siatka podglądu: wierzchołki
    view_t: np.ndarray          # ... trójkąty
    _tree: cKDTree | None = field(default=None, repr=False)
    _dilated: np.ndarray | None = field(default=None, repr=False)

    @property
    def length(self) -> float:
        """Długość kości (wzdłuż głównej osi) [jednostki modelu]."""
        return float(self.entry["info"]["length"])

    @property
    def area(self) -> float:
        """Pole zewnętrznej powierzchni [jednostki modelu²]."""
        return float(self.entry["info"]["area"])

    @property
    def spacing(self) -> float:
        """Odstęp zapisanych punktów [jednostki modelu]."""
        return float(self.entry["info"]["spacing"])

    def label(self, mirror: bool = False) -> str:
        """Nazwa do wyświetlania (z dopiskiem dla odbicia lustrzanego)."""
        return self.name + (" (odbicie lustrzane)" if mirror else "")

    def tree(self) -> cKDTree:
        """KD-drzewo zapisanych punktów (liczone raz)."""
        if self._tree is None:
            self._tree = cKDTree(self.points)
        return self._tree

    def window_dists(self) -> np.ndarray:
        """Rozkłady krzywizn okien (każde sumuje się do 1), poszerzone o sąsiednie przedziały
        (tolerancja szumu i różnic osobniczych) - (W, SI, log C)."""
        if self._dilated is None:
            w = self.windows / np.maximum(self.windows.sum(axis=(1, 2), keepdims=True), 1e-12)
            self._dilated = maximum_filter(w, size=(1, 3, 3), mode="constant")
        return self._dilated


def load_model(index_dir: Path, mid: str, entry: dict) -> Model:
    """Wczytuje model z <index_dir>/<mid>.npz."""
    with np.load(Path(index_dir) / f"{mid}.npz") as d:
        pts = d["points"].astype(float) / 65535 * d["span"] + d["lo"]
        nrm = d["normals"].astype(float) / 127
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        vv = d["view_v"].astype(float) / 65535 * d["view_span"] + d["view_lo"]
        return Model(mid, entry.get("name", mid), entry.get("group", ""), entry, pts, nrm,
                     d["hist"].astype(float), d["windows"].astype(float), vv, d["view_t"].astype(np.int64))


@dataclass
class Atlas:
    """Baza modeli w pamięci (load_atlas)."""

    index_dir: Path
    source_dir: Path
    models: list[Model]
    fingerprint: str                       # index_fingerprint - zmienia się z zawartością bazy
    report: dict = field(default_factory=dict)            # wynik sync_index
    _clouds: dict = field(default_factory=dict, repr=False)  # chmury + FPFH liczone w locie

    def get(self, mid: str) -> Model:
        """Model o id `mid`."""
        return next(m for m in self.models if m.id == mid)

    def cloud(self, model: Model, s: float, mirror: bool, p: AtlasParams):
        """Chmura modelu w skali s (jednostki fragmentu) z FPFH - liczona w locie i pamiętana (CLOUD_CACHE ostatnich).

        Returns:
            (chmura open3d z normalnymi, FPFH open3d).
        """
        key = (model.id, round(s, 4), mirror, p.voxel_au, p.fpfh_radius_au)
        if key not in self._clouds:
            idx = voxel_keep(model.points, p.voxel_au / s)
            flip = MIRROR if mirror else 1.0
            cloud = to_cloud(model.points[idx] * s * flip, model.normals[idx] * flip)
            self._clouds[key] = (cloud, compute_fpfh(cloud, p.fpfh_radius_au))
            while len(self._clouds) > CLOUD_CACHE:
                self._clouds.pop(next(iter(self._clouds)))
        return self._clouds[key]


_LOADED: dict = {}


def load_atlas(p: AtlasParams, index_dir: Path | None = None, source_dir: Path | None = None,
               sync: bool = True, log: Callable | None = print) -> Atlas:
    """Baza modeli: najpierw dolicza nowe / zmienione pliki (sync_index), potem wczytuje indeks.

    Wczytana baza jest pamiętana do końca programu (te same pliki = bez ponownego czytania).

    Args:
        p:          ustawienia.
        index_dir:  folder bazy (domyślnie config.ATLAS_INDEX).
        source_dir: folder plików modeli (domyślnie config.ATLAS_DIR).
        sync:       dolicz nowe pliki modeli (False = tylko to, co już jest w bazie).
        log:        komunikaty o liczeniu nowych modeli.

    Returns:
        Atlas.

    Raises:
        ValueError: baza jest pusta.
    """
    import config

    index_dir = Path(index_dir or config.ATLAS_INDEX)
    source_dir = Path(source_dir or config.ATLAS_DIR)
    report = sync_index(p, source_dir, index_dir, log=log) if sync else {}
    index = read_index(index_dir)
    fp = index_fingerprint(index)
    key = (str(index_dir.resolve()), fp)
    if key not in _LOADED:
        if not index["models"]:
            raise ValueError(f"Baza modeli kości jest pusta: brak modeli w {config._rel(source_dir)}/ i w "
                             f"{config._rel(index_dir)}/. Wrzuć pliki .ply/.obj/.stl kompletnych kości do "
                             f"{config._rel(source_dir)}/ (albo podaj --atlas FOLDER) i uruchom "
                             "python scripts/build_atlas.py")
        models = [load_model(index_dir, mid, e) for mid, e in sorted(index["models"].items())]
        _LOADED.clear()
        _LOADED[key] = Atlas(index_dir, source_dir, models, fp, report)
    atlas = _LOADED[key]
    atlas.report = report
    return atlas


def data_key() -> str:
    """Od czego zależy wynik algorytmu "atlas" poza sekcją "contacts" (contacts.detector data_key)."""
    import config

    return json.dumps({"index": index_fingerprint(read_index(config.ATLAS_INDEX)),
                       "sources": quick_fingerprint(config.ATLAS_DIR),
                       "atlas": asdict(config.atlas_params()),
                       "features": asdict(config.surface_feature_params())}, sort_keys=True)


# -----------------
# FRAGMENT: ANALIZA LOKALNA I WŁASNOŚCI
# -----------------

@dataclass
class Fragment:
    """Fragment (kość z pipeline) przygotowany do porównania z bazą."""

    bone_id: str
    local_score: np.ndarray      # (V,) wynik "features" (0 ~ typowa kora tej kości)
    cavity: np.ndarray           # (V,) osłonięcie (część zasłoniętych promieni) - wgłębienia, kanał szpikowy
    local_logit: np.ndarray      # (V,) log-szanse przełomu z analizy lokalnej
    coarse: o3d.geometry.PointCloud
    coarse_fpfh: object          # o3d Feature
    points: np.ndarray           # (N, 3) punkty do ICP i oceny (co voxel_au / 2)
    normals: np.ndarray          # (N, 3) ich normalne, uśrednione w promieniu tolerance_au
    vertex_normals: np.ndarray   # (V, 3) normalne wierzchołków uśrednione tak samo
    cortex_w: np.ndarray         # (N,) prawdopodobieństwo kory (analiza lokalna)
    hist: np.ndarray             # histogram krzywizn prawdopodobnej kory (pole w a.u.²)
    length: float                # długość fragmentu wzdłuż głównej osi [a.u.]
    cortex_area: float           # pole prawdopodobnej kory [a.u.²]


def local_evidence(bone, cp, p: AtlasParams, sfp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analiza lokalna fragmentu: (wynik features, osłonięcie, log-szanse przełomu) - każde (V,).

    Log-szanse = (wynik - feature_threshold) / local_scale (przycięte do +-LOCAL_CLIP)
    + cavity_weight x [punkt we wgłębieniu] (osłonięcie >= cavity_occlusion; liczone dla trafień
    od curvature_radius_au / 2 do cavity_depth_au, więc drobna szorstkość kory nie udaje wgłębienia).
    Pas 1.2 x curvature_radius_au przy otwartym brzegu skanu jest pomijany: postrzępione, zawinięte
    brzegi skanu też zasłaniają promienie (na naszych skanach to większość "wgłębień" na korze),
    a kanał szpikowy leży w środku przełomu.
    """
    import surface_features as sf

    mesh = bone.mesh
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    score = ct.bone_scores(sf.cached_features(bone, sfp), ct.vertex_areas(mesh), cp.feature_keys)
    v, vn = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
    if p.cavity_weight > 0:
        cavity = ray_hits(mesh, v, vn, CAVITY_RAYS, 0.5 * p.curvature_radius_au, p.cavity_depth_au)
        cavity[sf._distance_to(v, ct.boundary_vertices(mesh)) < 1.2 * p.curvature_radius_au] = 0.0
    else:
        cavity = np.zeros(len(v))
    # ograniczony głos: bardzo szorstka, ale zdrowa kora (przyczep mięśnia) nie przegłosuje pewnego atlasu;
    # znak bez zmian, więc bez atlasu wynik = "features"
    logit = np.clip((score - cp.feature_threshold) / p.local_scale, -LOCAL_CLIP, LOCAL_CLIP)
    logit = logit + p.cavity_weight * _sigmoid((cavity - p.cavity_occlusion) / 0.08)
    return score, cavity, logit


def prepare_fragment(bone, cp, p: AtlasParams, sfp) -> Fragment:
    """Analiza lokalna, chmury do ustawienia i własności (histogram krzywizn kory, długość) fragmentu.

    Normalne punktów do porównania z modelem są uśredniane w promieniu tolerance_au - szorstka kora
    (przyczep mięśnia) ma normalne trójkątów rozrzucone, ale uśredniona normalna zgadza się z modelem.
    """
    import surface_features as sf

    mesh = bone.mesh
    score, cavity, logit = local_evidence(bone, cp, p, sfp)
    v = np.asarray(mesh.vertices)
    vtree = cKDTree(v)
    cortex_v = 1 - _sigmoid(logit)
    coarse_s = sample_surface(mesh, p.voxel_au)
    coarse = to_cloud(coarse_s.points, coarse_s.normals)
    fine = sample_surface(mesh, p.voxel_au / 2)
    cortex_w = cortex_v[vtree.query(fine.points)[1]]

    def smoothed(at: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        """Normalne punktów `fine` uśrednione wokół punktów `at` (w promieniu tolerance_au)."""
        W, counts = sf._neighbours(at, fine.points, p.tolerance_au)
        n = W @ fine.normals
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        return np.where((counts[:, None] > 0) & (norm > 1e-6), n / np.maximum(norm, 1e-12), fallback)

    fine_n = smoothed(fine.points, fine.normals)
    vert_n = smoothed(v, np.asarray(mesh.vertex_normals))
    # własności: krzywizny prawdopodobnej kory, z dala od otwartego brzegu skanu (tam sąsiedztwo jest ucięte)
    rho = p.curvature_radius_au
    cs = sample_surface(mesh, rho / 4)
    k1, k2, ok = principal_curvatures(cs.points, cs.normals, rho)
    bnd = ct.boundary_vertices(mesh)
    far = cKDTree(v[bnd]).query(cs.points)[0] > rho if bnd.any() else np.ones(len(cs.points), bool)
    cell = float(ct.vertex_areas(mesh).sum()) / max(len(cs.points), 1)
    w = cortex_v[vtree.query(cs.points)[1]] * cell * (ok & far)
    hist = curvature_histogram(k1, k2, w)
    c = fine.points.mean(axis=0)
    _, _, vt = np.linalg.svd(fine.points - c, full_matrices=False)
    length = float(np.ptp((fine.points - c) @ vt[0]))
    return Fragment(bone.id, score, cavity, logit, coarse, compute_fpfh(coarse, p.fpfh_radius_au),
                    fine.points, fine_n, vert_n, cortex_w, hist, length, float(cortex_v @ ct.vertex_areas(mesh)))


# -----------------
# SITO WŁASNOŚCI
# -----------------

def scale_bounds(p: AtlasParams, model_id: str) -> tuple[float, float]:
    """Zakres skali modelu: wspólna skala zestawu +- shared_scale_tolerance, inaczej reference_scale x scale_range."""
    s = p.model_scales.get(model_id)
    if s:
        return s * (1 - p.shared_scale_tolerance), s * (1 + p.shared_scale_tolerance)
    return p.reference_scale * p.scale_range[0], p.reference_scale * p.scale_range[1]


def screen(frag: Fragment, atlas: Atlas, p: AtlasParams) -> list[dict]:
    """Sito własności (opis: docstring modułu, krok 1) - które modele w ogóle warto dopasowywać.

    Returns:
        Lista dla wszystkich modeli (najpierw przepuszczone, od najlepszego): {"model", "name",
        "status": "ok" | "za mała" | "inny kształt" | "poza limitem", "unexplained": najmniejsza
        niewyjaśniona część kory, "scales": skale do sprawdzenia, "reason": opis}.
    """
    total = float(frag.hist.sum())
    hf = frag.hist / total if total > 0 else None
    out = []
    for m in atlas.models:
        lo, hi = scale_bounds(p, m.id)
        e = {"model": m.id, "name": m.name, "scales": [], "unexplained": 1.0}
        s_len = frag.length / (m.length * (1 + p.size_tolerance))
        s_area = math.sqrt(frag.cortex_area / m.area) / (1 + p.size_tolerance)
        s_lo = max(lo, s_len, s_area)
        if s_lo > hi:
            what = (f"fragment ({frag.length:.0f} a.u.) dłuższy niż kość ({m.length * hi:.0f} a.u.)" if s_len > hi else
                    f"kora fragmentu ({frag.cortex_area:.0f} a.u.²) większa niż kość ({m.area * hi * hi:.0f} a.u.²)")
            e.update(status="za mała", reason=what + " przy największej skali")
            out.append(e)
            continue
        ks = list(range(math.ceil(math.log10(s_lo) / LOGC_STEP - 1e-9), math.floor(math.log10(hi) / LOGC_STEP + 1e-9) + 1))
        ks = ks or [int(round(math.log10(s_lo) / LOGC_STEP))]
        ref = m.window_dists()
        viol = []
        for k in ks:
            if hf is None:
                viol.append(0.0)
                continue
            shifted = np.zeros_like(ref)            # przedział fragmentu b <-> przedział modelu b + k
            if k >= 0:
                shifted[:, :, :ref.shape[2] - k] = ref[:, :, k:]
            else:
                shifted[:, :, -k:] = ref[:, :, :ref.shape[2] + k]
            # część rozkładu kory fragmentu, której nie ma w żadnym oknie kości (najlepsze okno)
            viol.append(float(np.maximum(hf[None] - SCREEN_SLACK * shifted, 0).sum(axis=(1, 2)).min()))
        viol = np.array(viol)
        order = np.argsort(viol, kind="stable")
        chosen = []
        for i in order:                              # najlepsze skale, co najmniej 2 przedziały (~25%) od siebie
            if all(abs(ks[i] - ks[j]) >= 2 for j in chosen):
                chosen.append(int(i))
            if len(chosen) >= p.scale_candidates:
                break
        e["scales"] = [round(10 ** (ks[i] * LOGC_STEP), 4) for i in chosen]
        e["unexplained"] = round(float(viol.min()), 3)
        if viol.min() > p.max_unexplained:
            e.update(status="inny kształt", reason=f"{viol.min():.0%} kory fragmentu nie ma odpowiednika "
                                                   "na tej kości (krzywizny)")
        else:
            e.update(status="ok", reason="")
        out.append(e)
    ok = sorted([e for e in out if e["status"] == "ok"], key=lambda e: e["unexplained"])
    for e in ok[p.max_models:]:
        e.update(status="poza limitem", reason=f"gorsze własności niż {p.max_models} innych modeli (max_models)")
    rest = [e for e in out if e["status"] != "ok"]
    return [e for e in ok if e["status"] == "ok"] + sorted(rest, key=lambda e: e["unexplained"])


# -----------------
# USTAWIENIE I OCENA
# -----------------

def surface_terms(points: np.ndarray, normals: np.ndarray, pose: Pose, model: Model, mirror: bool):
    """Położenie punktów fragmentu względem powierzchni modelu ustawionego `pose`.

    Returns:
        (y - punkty w układzie modelu, q - najbliższe punkty modelu, nq - ich normalne,
         signed - odległość ze znakiem od płaszczyzny stycznej modelu [a.u.] (> 0 = na zewnątrz),
         dist - odległość od powierzchni [a.u.], cos - zgodność normalnych).
    """
    flip = MIRROR if mirror else 1.0
    y = pose.to_model(points)
    _, j = model.tree().query(y * flip, workers=-1)
    q, nq = model.points[j] * flip, model.normals[j] * flip
    r = y - q
    dn = np.einsum("ij,ij->i", r, nq)
    tang = np.linalg.norm(r - dn[:, None] * nq, axis=1)
    dist = pose.s * np.sqrt(dn ** 2 + np.maximum(tang - model.spacing, 0) ** 2)
    cos = np.einsum("ij,ij->i", normals @ pose.R, nq)
    return y, q, nq, pose.s * dn, dist, cos


def evidence(signed: np.ndarray, dist: np.ndarray, cos: np.ndarray, p: AtlasParams):
    """Kora / przełom / na zewnątrz wg modelu - każde (N,) w [0, 1] (opis: docstring modułu)."""
    tol = p.tolerance_au
    on = np.exp(-0.5 * (dist / tol) ** 2)
    agree = _sigmoid((cos - math.cos(math.radians(p.normal_angle_deg))) / 0.04)
    inside = _sigmoid((-signed - 1.5 * tol) / (0.3 * tol))
    outside = _sigmoid((signed - 1.5 * tol) / (0.3 * tol))
    return on * agree, np.maximum(inside, on * (1 - agree)), outside


def refine_pose(frag: Fragment, model: Model, mirror: bool, pose: Pose, p: AtlasParams,
                s_bounds: tuple) -> Pose:
    """Odporny ICP ze SKALĄ (point-to-plane, krok Gaussa-Newtona w 7 parametrach: skala, obrót, przesunięcie).

    Każdy punkt fragmentu x ma najbliższy punkt modelu Q z normalną N; minimalizowane jest
    sum w ((x - Q') . N)², gdzie Q' = c + (1 + ds)(I + [dw]x)(Q - c) + dt (c = środek fragmentu).
    Wagi: exp(-(odległość / sigma)² / 2) x [zgodna normalna], sigma od 4 x tolerance_au do
    tolerance_au - punkty przełomu i kanału szpikowego (w środku modelu, inne normalne) mają wagę ~0.

    Args:
        frag, model, mirror: fragment, model i czy odbity lustrzanie.
        pose:     ustawienie startowe (model -> fragment).
        p:        ustawienia (tolerance_au, normal_angle_deg, icp_iterations).
        s_bounds: (min, max) skali.

    Returns:
        Poprawione ustawienie.
    """
    from scipy.spatial.transform import Rotation

    cos_max = math.cos(math.radians(p.normal_angle_deg))
    x = frag.points
    c = x.mean(axis=0)
    for it in range(p.icp_iterations):
        sigma = p.tolerance_au * max(1.0, 4.0 * 0.7 ** it)
        _, q, nq, _, dist, cos = surface_terms(x, frag.normals, pose, model, mirror)
        w = np.exp(-0.5 * (dist / sigma) ** 2) * (cos > cos_max)
        if (w > 0.05).sum() < 10:
            break
        Q = pose.s * q @ pose.R.T + pose.t
        N = nq @ pose.R.T
        r = np.einsum("ij,ij->i", x - Q, N)
        d = Q - c
        J = np.column_stack([np.einsum("ij,ij->i", d, N), np.cross(d, N), N])
        Jw = J * w[:, None]
        H = Jw.T @ J
        H += 1e-6 * np.trace(H) / 7 * np.eye(7)
        theta = np.linalg.solve(H, Jw.T @ r)
        R_inc = Rotation.from_rotvec(theta[1:4]).as_matrix()
        s_new = float(np.clip(pose.s * (1 + theta[0]), *s_bounds))
        k = s_new / pose.s
        pose = Pose(s_new, R_inc @ pose.R, k * R_inc @ (pose.t - c) + c + theta[4:7])
        if sigma <= p.tolerance_au and np.abs(theta[:4]).max() < 1e-5 and np.abs(theta[4:]).max() < 1e-4:
            break
    return pose


def score_pose(frag: Fragment, model: Model, mirror: bool, pose: Pose, p: AtlasParams) -> dict:
    """Ocena ustawienia (fit i składowe; opis: docstring modułu, krok 3)."""
    *_, signed, dist, cos = surface_terms(frag.points, frag.normals, pose, model, mirror)
    cortex, _, outside = evidence(signed, dist, cos, p)
    w = frag.cortex_w if frag.cortex_w.sum() > 0.05 * len(frag.cortex_w) else np.ones(len(cortex))
    cover = float((w * cortex).sum() / w.sum())
    out = float(outside.mean())
    rmse = float(np.sqrt((cortex * dist ** 2).sum() / max(cortex.sum(), 1e-9)))
    return {"fit": cover - p.outside_penalty * out, "cover": cover, "outside": out, "rmse_au": rmse}


def _ransac(frag: Fragment, cloud, fpfh, p: AtlasParams, seed: int) -> np.ndarray:
    """Zgrubne ustawienie fragment -> model (w skali) RANSAC-em na FPFH (jak w matching_surfaces)."""
    dist = 1.5 * p.voxel_au
    o3d.utility.random.seed(seed)
    # open3d ostrzega, gdy po filtrze wzajemności zostaje mało par (i sam bierze wszystkie) - to normalne tutaj
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    try:
        r = reg.registration_ransac_based_on_feature_matching(
            frag.coarse, cloud, frag.coarse_fpfh, fpfh, True, dist,
            reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(dist),
             reg.CorrespondenceCheckerBasedOnNormal(np.deg2rad(p.normal_angle_deg))],
            reg.RANSACConvergenceCriteria(p.ransac_iterations, 1.0))
    finally:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    return r.transformation


# -----------------
# WYNIK DOPASOWANIA
# -----------------

@dataclass
class AtlasMatch:
    """Wynik porównania fragmentu z bazą (match_fragment / cached_match)."""

    bone_id: str
    candidates: list[dict]       # od najbardziej prawdopodobnego: {"model", "name", "group", "mirror", "label",
                                 #  "p", "fit", "cover", "outside", "rmse_au", "scale", "rel_scale", "T" (4x4 model -> fragment)}
    p_unknown: float             # prawdopodobieństwo "żadna z bazy"
    screening: list[dict]        # wynik sita (screen)
    evidence: list[np.ndarray]   # (V,) log-szanse przełom / kora dla candidates[:len(evidence)]
    local_score: np.ndarray      # (V,) wynik "features"
    cavity: np.ndarray           # (V,) osłonięcie (wgłębienia)
    local_logit: np.ndarray      # (V,) log-szanse z analizy lokalnej
    models: int = 0              # ile modeli w bazie
    seconds: float = 0.0
    cached: bool = False

    @property
    def confidence(self) -> float:
        """Pewność dowodu z atlasu = suma prawdopodobieństw modeli, z których wzięto dowód."""
        return float(sum(c["p"] for c in self.candidates[:len(self.evidence)]))

    def atlas_logit(self) -> np.ndarray:
        """(V,) dowód z atlasu: średnia log-szans modeli ważona ich prawdopodobieństwem (0 = brak)."""
        if not self.evidence:
            return np.zeros(len(self.local_logit))
        w = np.array([c["p"] for c in self.candidates[:len(self.evidence)]])
        return np.tensordot(w / max(w.sum(), 1e-12), np.stack([e.astype(float) for e in self.evidence]), axes=1)

    def groups(self) -> list[tuple[str, float]]:
        """Prawdopodobieństwa typów kości (podfoldery bazy), od największego; [] = bez podfolderów."""
        acc: dict[str, float] = {}
        for c in self.candidates:
            if c["group"]:
                acc[c["group"]] = acc.get(c["group"], 0.0) + c["p"]
        return sorted(acc.items(), key=lambda kv: -kv[1])

    def summary(self, n: int = 5) -> dict:
        """Opis do zapisu w wyniku algorytmu ("info" -> "atlas") i w atlas.json (bez dużych tablic)."""
        keys = ("model", "name", "group", "mirror", "label", "p", "fit", "cover", "outside", "rmse_au", "scale",
                "rel_scale")
        return {"ranking": [{k: c[k] for k in keys} for c in self.candidates[:n] if c["p"] >= 0.005],
                "p_unknown": round(self.p_unknown, 4), "confidence": round(self.confidence, 4),
                "groups": [{"group": g, "p": round(q, 4)} for g, q in self.groups()],
                "models": self.models, "checked": sum(e["status"] == "ok" for e in self.screening),
                "excluded": [{"model": e["model"], "name": e["name"], "status": e["status"], "reason": e["reason"]}
                             for e in self.screening if e["status"] != "ok"],
                "seconds": round(self.seconds, 1)}

    def ranking_lines(self, n: int = 3) -> list[str]:
        """Ranking do konsoli (summary_lines)."""
        return summary_lines(self.summary(max(n, 5)), n)


def summary_lines(s: dict, n: int = 3) -> list[str]:
    """Ranking do konsoli z AtlasMatch.summary() (też z pliku wyniku: Contacts.info["atlas"]).

    Args:
        s: opis dopasowania.
        n: ile modeli pokazać.

    Returns:
        Linie: ranking z "żadna z bazy", typy kości (gdy baza ma podfoldery), wynik sita.
    """
    parts = [f"{i}. {c['label']} {c['p']:.0%} (skala ×{c['rel_scale']:.2f}, dopasowanie {c['fit']:.2f})"
             for i, c in enumerate(s.get("ranking", [])[:n], 1)]
    lines = [" · ".join(parts + [f"{UNKNOWN} {s.get('p_unknown', 1.0):.0%}"])]
    if s.get("groups"):
        lines.append("typ kości: " + ", ".join(f"{g['group']} {g['p']:.0%}" for g in s["groups"][:3]))
    excl = s.get("excluded", [])
    line = f"sito: dopasowywane {s.get('checked', 0)} z {s.get('models', 0)} modeli"
    if excl:
        line += ", wykluczone: " + ", ".join(f"{e['name']} ({e['status']})" for e in excl[:6])
        if len(excl) > 6:
            line += f" i {len(excl) - 6} innych"
    return lines + [line]


def probabilities(fits: list[float], p: AtlasParams) -> tuple[np.ndarray, float]:
    """softmax(fit / temperature) z "żadna z bazy" o ocenie min_fit -> (p modeli, p żadnej)."""
    z = np.append(np.asarray(fits, dtype=float), p.min_fit) / max(p.temperature, 1e-6)
    z = np.exp(z - z.max())
    z /= z.sum()
    return z[:-1], float(z[-1])


def match_fragment(bone, atlas: Atlas, p: AtlasParams, cp, sfp, log: Callable | None = None) -> AtlasMatch:
    """Porównuje fragment z bazą: sito własności -> RANSAC + ICP ze skalą -> ocena -> prawdopodobieństwa -> dowód.

    Args:
        bone:  kość (fragment) z pipeline.
        atlas: baza (load_atlas).
        p:     ustawienia atlasu.
        cp:    ustawienia analizy lokalnej (contacts.ContactParams: feature_keys, feature_threshold).
        sfp:   ustawienia cech (surface_features.SurfaceFeatureParams).
        log:   komunikaty o postępie (None = bez).

    Returns:
        AtlasMatch.
    """
    t0 = time.perf_counter()
    frag = prepare_fragment(bone, cp, p, sfp)
    scr = screen(frag, atlas, p)
    # długi fragment (np. cała kość): rzadsze punkty i większa tolerancja RANSAC - kość innej osoby różni
    # się kształtem tym bardziej, im dalej od siebie są punkty, więc trójki punktów nie zgadzają się co do 1 a.u.
    voxel = max(p.voxel_au, frag.length / COARSE_ALONG)
    pr = p
    if voxel > 1.05 * p.voxel_au:
        pr = replace(p, voxel_au=voxel, fpfh_radius_au=p.fpfh_radius_au * voxel / p.voxel_au)
        cs = sample_surface(bone.mesh, voxel)
        frag.coarse = to_cloud(cs.points, cs.normals)
        frag.coarse_fpfh = compute_fpfh(frag.coarse, pr.fpfh_radius_au)
    cands = []
    for e in [e for e in scr if e["status"] == "ok"]:
        model = atlas.get(e["model"])
        lo, hi = scale_bounds(p, model.id)
        s_bounds = (lo, hi) if p.model_scales.get(model.id) else (lo / 1.1, hi * 1.1)
        for mirror in ((False, True) if p.mirror else (False,)):
            best = None
            for s in e["scales"]:
                cloud, fpfh = atlas.cloud(model, s, mirror, pr)
                for seed in range(p.ransac_seeds):
                    T = _ransac(frag, cloud, fpfh, pr, seed)
                    R0 = T[:3, :3]
                    pose = Pose(s, R0.T, -R0.T @ T[:3, 3])        # fragment -> model w skali s  =>  model -> fragment
                    pose = refine_pose(frag, model, mirror, pose, p, s_bounds)
                    sc = score_pose(frag, model, mirror, pose, p)
                    if best is None or sc["fit"] > best["fit"]:
                        best = {**sc, "pose": pose}
            pose = best.pop("pose")
            cands.append({"model": model.id, "name": model.name, "group": model.group, "mirror": mirror,
                          "label": model.label(mirror), **{k: round(v, 4) for k, v in best.items()},
                          "scale": round(pose.s, 4), "rel_scale": round(pose.s / p.reference_scale, 4),
                          "T": np.round(pose.matrix(), 6).tolist()})
            if log:
                log(f"      {model.label(mirror)}: dopasowanie {best['fit']:.2f} (skala ×{pose.s / p.reference_scale:.2f})")
    probs, p_unknown = probabilities([c["fit"] for c in cands], p)
    for c, q in zip(cands, probs):
        c["p"] = round(float(q), 4)
    cands.sort(key=lambda c: -c["p"])
    v = np.asarray(bone.mesh.vertices)
    ev = []
    for c in cands[:p.top_k]:
        if c["p"] < 0.01:
            break
        *_, signed, dist, cos = surface_terms(v, frag.vertex_normals, Pose.from_matrix(c["T"]),
                                              atlas.get(c["model"]), c["mirror"])
        cortex, fracture, _ = evidence(signed, dist, cos, p)
        ev.append(np.log((fracture + EVIDENCE_EPS) / (cortex + EVIDENCE_EPS)).astype(np.float16))
    return AtlasMatch(bone.id, cands, p_unknown, scr, ev, frag.local_score, frag.cavity, frag.local_logit,
                      len(atlas.models), time.perf_counter() - t0)


def fragment_length(mesh) -> float:
    """Długość fragmentu wzdłuż głównej osi (PCA wierzchołków) [a.u.]."""
    v = np.asarray(mesh.vertices)
    c = v.mean(axis=0)
    _, _, vt = np.linalg.svd(v - c, full_matrices=False)
    return float(np.ptp((v - c) @ vt[0]))


def shared_scales(items: list[tuple], p: AtlasParams) -> dict[str, dict]:
    """Wspólna skala każdego modelu dla zestawu - fragmenty jednej osoby mają tę samą skalę względem modelu.

    Każdy fragment, dla którego model (obie strony razem) ma p >= SHARED_MIN_P i ocenę >= min_fit, głosuje
    skalą swojej lepszej strony z wagą p x ocena x długość fragmentu (długi fragment = pewniejsza skala).
    Wynik = mediana ważona (odporna na fragmenty położone na modelu w złym miejscu).

    Args:
        items: [(AtlasMatch, długość fragmentu [a.u.], nazwa fragmentu)].
        p:     ustawienia (min_fit, shared_scale_tolerance).

    Returns:
        {id modelu: {"scale": skala, "name": nazwa modelu, "votes": [(nazwa fragmentu, skala, waga)],
                     "outliers": [(nazwa fragmentu, skala)] - poza zakresem +- shared_scale_tolerance,
                     "agree": część wagi głosów w tym zakresie (1 = wszystkie fragmenty zgodne)}}.
    """
    votes: dict[str, list] = {}
    names: dict[str, str] = {}
    for m, length, label in items:
        per_model: dict[str, list] = {}
        for c in m.candidates:
            per_model.setdefault(c["model"], []).append(c)
        for mid, cs in per_model.items():
            best = max(cs, key=lambda c: c["p"])
            if sum(c["p"] for c in cs) >= SHARED_MIN_P and best["fit"] >= p.min_fit:
                votes.setdefault(mid, []).append((label, float(best["scale"]), best["p"] * best["fit"] * length))
                names[mid] = best["name"]
    out = {}
    for mid, vs in votes.items():
        vs.sort(key=lambda v: v[1])
        w = np.cumsum([v[2] for v in vs])
        s = vs[int(np.searchsorted(w, w[-1] / 2))][1]
        inside = [wt for _, sc, wt in vs if abs(sc / s - 1) <= p.shared_scale_tolerance]
        out[mid] = {"scale": round(s, 4), "name": names[mid], "votes": vs,
                    "outliers": [(lbl, sc) for lbl, sc, _ in vs if abs(sc / s - 1) > p.shared_scale_tolerance],
                    "agree": round(sum(inside) / max(w[-1], 1e-12), 3)}
    return out


def _match_tag(bone, atlas: Atlas, p: AtlasParams, cp, sfp) -> str:
    """Od czego zależy wynik match_fragment (inne = liczone od nowa)."""
    return json.dumps({"format": MATCH_FORMAT, "mesh": ct.mesh_fingerprint(bone.mesh), "atlas": atlas.fingerprint,
                       "params": {k: v for k, v in asdict(p).items() if k not in FUSION_ONLY},
                       "local": {"keys": list(cp.feature_keys), "threshold": cp.feature_threshold},
                       "features": asdict(sfp)}, sort_keys=True)


def cached_match(bone, atlas: Atlas, p: AtlasParams, cp, sfp, cache_dir: Path | None = None,
                 log: Callable | None = None) -> AtlasMatch:
    """match_fragment z cache na dysku: <folder wyników>/cache/atlas/<id>.npz.

    Liczone od nowa, gdy zmieniła się siatka kości, baza modeli albo ustawienia dopasowania
    (zmiana samych atlas_weight / min_confidence nie wymaga liczenia od nowa).

    Args:
        cache_dir: folder cache; domyślnie z bone.folder (out/3d/<zestaw>/cache/atlas), kość bez
                   folderu - bez cache.
        (pozostałe jak w match_fragment)

    Returns:
        AtlasMatch (cached = True, gdy z cache).
    """
    if cache_dir is None and bone.folder is not None:
        cache_dir = Path(bone.folder).parent.parent / "cache" / "atlas"
    tag = _match_tag(bone, atlas, p, cp, sfp)
    path = Path(cache_dir) / f"{bone.id}.npz" if cache_dir is not None else None
    if path is not None and path.exists():
        try:
            with np.load(path) as d:
                if str(d["_tag"]) == tag:
                    meta = json.loads(str(d["meta"]))
                    ev = [d[f"ev{i}"] for i in range(meta["n_evidence"])]
                    return AtlasMatch(bone.id, meta["candidates"], meta["p_unknown"], meta["screening"], ev,
                                      d["local_score"].astype(float), d["cavity"].astype(float),
                                      d["local_logit"].astype(float), meta["models"], meta["seconds"], cached=True)
        except Exception:       # uszkodzony / stary plik - policz od nowa
            pass
    m = match_fragment(bone, atlas, p, cp, sfp, log)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"candidates": m.candidates, "p_unknown": m.p_unknown, "screening": m.screening,
                "models": m.models, "seconds": m.seconds, "n_evidence": len(m.evidence)}
        np.savez_compressed(path, _tag=np.array(tag), meta=np.array(json.dumps(meta, ensure_ascii=False)),
                            local_score=m.local_score.astype(np.float32), cavity=m.cavity.astype(np.float16),
                            local_logit=m.local_logit.astype(np.float32),
                            **{f"ev{i}": e for i, e in enumerate(m.evidence)})
    return m


# -----------------
# ALGORYTM "atlas"
# -----------------

def fused_logit(m: AtlasMatch, p: AtlasParams) -> np.ndarray:
    """(V,) log-szanse przełomu: analiza lokalna + atlas_weight x pewność x dowód z atlasu."""
    z = np.asarray(m.local_logit, dtype=float)
    if m.confidence >= p.min_confidence:
        z = z + p.atlas_weight * m.confidence * m.atlas_logit()
    return z


def detect(bone, cp) -> tuple[np.ndarray, dict]:
    """Algorytm "atlas" (contacts.detect_atlas): etykiety przełomów + ranking modeli.

    Args:
        bone: kość (fragment).
        cp:   contacts.ContactParams (analiza lokalna i obszary).

    Returns:
        ((V,) uint8 etykiety, {"atlas": AtlasMatch.summary()}).

    Raises:
        ValueError: pusta baza modeli.
    """
    import config

    p, sfp = config.atlas_params(), config.surface_feature_params()
    m = cached_match(bone, load_atlas(p), p, cp, sfp)
    labels = ct.regions_from_score(bone.mesh, fused_logit(m, p), 0.0, cp, sfp.boundary_margin_au)
    return labels, {"atlas": m.summary()}


# -----------------
# RAPORT: atlas.json, atlas.csv, atlas.html
# -----------------

def write_reports(out_dir: Path, items: list[dict], atlas: Atlas, p: AtlasParams) -> tuple[Path, Path]:
    """Zapisuje atlas.json (pełne rankingi) i atlas.csv (tabela do Excela) w `out_dir`.

    Args:
        out_dir: folder wyników zestawu.
        items:   [{"nr", "bone", "match": AtlasMatch}, ...].
        atlas:   baza.
        p:       ustawienia.

    Returns:
        (ścieżka atlas.json, ścieżka atlas.csv).
    """
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"format": "bones-smm-atlas-report/1", "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "params": asdict(p), "models": [{"id": m.id, "name": m.name, "group": m.group, **m.entry["info"]}
                                            for m in atlas.models],
            "bones": [{"nr": it["nr"], "bone": it["bone"].id, "label": it["bone"].label,
                       **it["match"].summary(n=len(it["match"].candidates)),
                       "screening": it["match"].screening,
                       "poses": {f"{c['model']}{'|mirror' if c['mirror'] else ''}": c["T"]
                                 for c in it["match"].candidates}} for it in items]}
    jpath = out_dir / "atlas.json"
    jpath.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    cpath = out_dir / "atlas.csv"
    with cpath.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["nr", "kosc", "miejsce", "model", "odbicie", "prawdopodobienstwo", "dopasowanie",
                    "kora_na_modelu", "na_zewnatrz", "skala", "rmse_au"])
        for it in items:
            m = it["match"]
            for i, c in enumerate(m.candidates, 1):
                w.writerow([it["nr"], it["bone"].label, i, c["name"], "tak" if c["mirror"] else "nie",
                            f"{c['p']:.4f}", f"{c['fit']:.3f}", f"{c['cover']:.3f}", f"{c['outside']:.3f}",
                            f"{c['rel_scale']:.3f}", f"{c['rmse_au']:.3f}"])
            w.writerow([it["nr"], it["bone"].label, "-", UNKNOWN, "", f"{m.p_unknown:.4f}", "", "", "", "", ""])
    return jpath, cpath


DECISION = (("kora", "#d9cdb8"), ("przełom", "#e8508f"),
            ("kora wg atlasu (lokalnie wyglądała na przełom)", "#f2a531"),
            ("przełom wg atlasu (lokalnie wyglądał na korę)", "#7b61ff"))


def probability_figure(items: list[dict], view):
    """Mapa prawdopodobieństw: wiersze = fragmenty, kolumny = modele (i odbicia) + "żadna z bazy"."""
    import plotly.graph_objects as go

    from visualization_3d import FONT

    cols = []
    for it in items:
        for c in it["match"].candidates:
            if c["p"] >= 0.01 and c["label"] not in cols:
                cols.append(c["label"])
    cols.append(UNKNOWN)
    z, ylab = [], []
    for it in items:
        m = it["match"]
        row = {c["label"]: c["p"] for c in m.candidates}
        row[UNKNOWN] = m.p_unknown
        z.append([row.get(k, 0.0) for k in cols])
        ylab.append(f"{it['nr']}. {it['bone'].label}")
    z = np.array(z)
    fig = go.Figure(go.Heatmap(z=z, x=cols, y=ylab, zmin=0, zmax=1, xgap=2, ygap=2,
                               colorscale=[[0, "#f4f6f8"], [1, "#e8508f"]],
                               text=[[f"{v:.0%}" if v >= 0.005 else "" for v in r] for r in z], texttemplate="%{text}",
                               colorbar=dict(title="p", tickformat=".0%"),
                               hovertemplate="%{y}<br>%{x}: %{z:.1%}<extra></extra>"))
    fig.update_layout(font=FONT, paper_bgcolor="#ffffff", height=max(300, 36 * len(items) + 220),
                      margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(side="top", tickangle=-30),
                      yaxis=dict(autorange="reversed"))
    return fig


def fragments_figure(items: list[dict], atlas: Atlas, p: AtlasParams, view):
    """Fragment ustawiony na najbardziej prawdopodobnym modelu (półprzezroczysty), kolory wg wyboru:
    decyzja (atlas vs sama analiza lokalna), dowód z atlasu, analiza lokalna, razem."""
    import plotly.graph_objects as go

    from visualization_3d import FONT, LIGHTING, decimate

    per_bone = max(4000, view.triangles_total // max(2 * len(items), 1))
    traces, modes = [], {k: [] for k in ("decision", "atlas", "local", "fused")}
    fig = go.Figure()
    for it in items:
        bone, m = it["bone"], it["match"]
        d = decimate(bone.mesh, per_bone)
        dv, dt = np.asarray(d.vertices), np.asarray(d.triangles)
        near = cKDTree(np.asarray(bone.mesh.vertices)).query(dv)[1]
        fa, ff = it["labels_atlas"] > 0, it["labels_features"] > 0
        cls = np.where(fa & ff, 1, np.where(~fa & ff, 2, np.where(fa & ~ff, 3, 0)))[near]
        c0, c1, c2 = cls[dt[:, 0]], cls[dt[:, 1]], cls[dt[:, 2]]
        modes["decision"].append(np.where(c0 == c1, c0, c2).astype(int).tolist())   # klasa większości wierzchołków

        def face(x: np.ndarray, lim: float) -> list:
            """Wartości wierzchołków -> średnia na trójkąt uproszczonej siatki, przycięta do [-lim, lim]."""
            return np.round(np.clip(x[near][dt].mean(axis=1), -lim, lim), 2).tolist()

        modes["atlas"].append(face(m.confidence * m.atlas_logit(), 3))
        modes["local"].append(face(m.local_logit, 4))
        modes["fused"].append(face(fused_logit(m, p), 4))
        top = m.candidates[0] if m.candidates and m.candidates[0]["p"] >= 0.01 else None
        traces.append((it, d, dv, dt, top))
    first = "decision"
    styles = {
        "decision": dict(colorscale=[[i / 4 + e, c] for i, (_, c) in enumerate(DECISION) for e in (0, 0.25)],
                         cmin=-0.5, cmax=3.5, title="decyzja", ticks=([0, 1, 2, 3], [n for n, _ in DECISION])),
        "atlas": dict(colorscale=[[0, "#2b6fd8"], [0.5, "#f4f6f8"], [1, "#e8508f"]], cmin=-3, cmax=3,
                      title="dowód z atlasu", ticks=([-3, 0, 3], ["kora", "brak dowodu", "przełom"])),
        "local": dict(colorscale=[[0, "#2b6fd8"], [0.5, "#f4f6f8"], [1, "#e8508f"]], cmin=-4, cmax=4,
                      title="analiza lokalna", ticks=([-4, 0, 4], ["kora", "", "przełom"])),
        "fused": dict(colorscale=[[0, "#2b6fd8"], [0.5, "#f4f6f8"], [1, "#e8508f"]], cmin=-4, cmax=4,
                      title="razem", ticks=([-4, 0, 4], ["kora", "próg", "przełom"])),
    }
    n = len(traces)
    for i, (it, d, dv, dt, top) in enumerate(traces):
        st = styles[first]
        fig.add_trace(go.Mesh3d(x=dv[:, 0], y=dv[:, 1], z=dv[:, 2], i=dt[:, 0], j=dt[:, 1], k=dt[:, 2],
                                intensity=modes[first][i], intensitymode="cell", colorscale=st["colorscale"],
                                cmin=st["cmin"], cmax=st["cmax"], flatshading=True, lighting=LIGHTING,
                                visible=i == 0, name=it["bone"].label, hoverinfo="name",
                                colorbar=dict(title=dict(text=st["title"], side="right"), tickvals=st["ticks"][0],
                                              ticktext=st["ticks"][1], len=0.75, thickness=14)))
    for i, (it, d, dv, dt, top) in enumerate(traces):
        if top is None:
            fig.add_trace(go.Scatter3d(x=[], y=[], z=[], visible=i == 0, showlegend=False))
            continue
        model = atlas.get(top["model"])
        pose = Pose.from_matrix(top["T"])
        flip = MIRROR if top["mirror"] else 1.0
        mv = (model.view_v * flip) @ (pose.s * pose.R).T + pose.t
        mt = model.view_t[:, [0, 2, 1]] if top["mirror"] else model.view_t
        fig.add_trace(go.Mesh3d(x=mv[:, 0], y=mv[:, 1], z=mv[:, 2], i=mt[:, 0], j=mt[:, 1], k=mt[:, 2],
                                color="#8a9aab", opacity=0.18, flatshading=False, lighting=LIGHTING,
                                visible=i == 0, name=top["label"], hoverinfo="name", showscale=False))
    buttons = []
    for i, (it, *_rest, top) in enumerate(traces):
        vis = [j == i for j in range(n)] * 2
        name = f"{top['label']} {top['p']:.0%}" if top else f"{UNKNOWN} {it['match'].p_unknown:.0%}"
        buttons.append(dict(label=f"{it['nr']}. {it['bone'].label} → {name}", method="update", args=[{"visible": vis}]))
    mode_buttons = []
    for key, label in (("decision", "decyzja: atlas vs analiza lokalna"), ("atlas", "dowód z atlasu"),
                       ("local", "analiza lokalna"), ("fused", "razem (wynik)")):
        st = styles[key]
        mode_buttons.append(dict(label=label, method="restyle", args=[{
            "intensity": modes[key], "colorscale": [st["colorscale"]] * n, "cmin": [st["cmin"]] * n,
            "cmax": [st["cmax"]] * n, "colorbar.title.text": [st["title"]] * n,
            "colorbar.tickvals": [st["ticks"][0]] * n, "colorbar.ticktext": [st["ticks"][1]] * n},
            list(range(n))]))
    scene = dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                 bgcolor="#ffffff")
    fig.update_layout(font=FONT, paper_bgcolor="#ffffff", height=720, margin=dict(l=0, r=0, t=50, b=0), scene=scene,
                      showlegend=False,
                      updatemenus=[dict(buttons=buttons, x=0.01, y=1.06, xanchor="left", yanchor="top",
                                        bgcolor="#ffffff", bordercolor="#c9d2db"),
                                   dict(buttons=mode_buttons, x=0.99, y=1.06, xanchor="right", yanchor="top",
                                        bgcolor="#ffffff", bordercolor="#c9d2db")])
    return fig


def write_page(path: Path, items: list[dict], atlas: Atlas, p: AtlasParams, view, open_browser: bool = False) -> Path:
    """Strona atlas.html: mapa prawdopodobieństw + fragmenty na modelach w 3D."""
    from visualization_3d import write_page as page

    lead = (f"Baza: {len(atlas.models)} modeli kompletnych kości. Dla każdego fragmentu: rozkład prawdopodobieństwa, "
            f"z której kości pochodzi (z możliwością <b>{UNKNOWN}</b>), oraz podział na korę i przełom "
            "z kontekstu anatomicznego.")
    sections = [
        ("Z której kości pochodzi fragment", "Prawdopodobieństwa z jakości dopasowania (kora fragmentu na powierzchni "
         "modelu, nic nie wystaje). Modele wykluczone przez sito własności (rozmiar, krzywizny) mają 0.",
         probability_figure(items, view)),
        ("Fragment na najbardziej prawdopodobnym modelu", "Szary półprzezroczysty = model z bazy ustawiony na fragmencie "
         "(skala, obrót, odbicie). Kolory (lista po prawej): <b>decyzja</b> - pomarańczowe = szorstka kora, którą sama "
         "analiza lokalna wzięłaby za przełom; fioletowe = przełom (np. gładki albo kanał szpikowy), którego sama analiza "
         "lokalna by nie zaznaczyła.", fragments_figure(items, atlas, p, view)),
    ]
    return page(path, "Atlas anatomiczny - rozpoznawanie kości", lead, sections, view, open_browser)

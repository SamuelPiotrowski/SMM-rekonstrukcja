"""
Powierzchnie styku (przełomy) na kościach: format zapisu, statystyki, algorytmy wykrywania.

Etykiety są przypisane do WIERZCHOŁKÓW siatki kości (out/3d/<zestaw>/bones/<id>/mesh.ply):
    labels[i] = 0  -> wierzchołek nie należy do żadnej powierzchni styku,
    labels[i] = k  -> wierzchołek należy do powierzchni styku nr k (1, 2, ...).
Jedna kość może mieć kilka powierzchni styku (np. przełom na każdym końcu).

Skąd się biorą etykiety (oba rodzaje w jednym folderze contacts/, ten sam format):
    - RĘCZNIE: scripts/mark_contacts.py (przeglądarka) -> contacts/<zestaw>/<id_kości>.json
      To ręczna praca - jest w repozytorium, warto ją commitować.
    - ALGORYTM: scripts/detect_contacts.py -> contacts/<zestaw>/auto/<id_kości>.json
      Pole "source" = "auto:<nazwa algorytmu>", pole "detector" = algorytm i jego ustawienia.
      Generowane (nie ma ich w repozytorium) - auto_contacts() liczy je na nowo, gdy brakuje
      pliku albo zmienił się algorytm, jego ustawienia lub siatka kości.

Algorytmy (DETECTORS): "roughness" (bazowy), "features" (kilka cech lokalnych naraz),
"atlas" (kontekst anatomiczny z bazy modeli kompletnych kości + cechy lokalne - src/atlas.py).

PO KAŻDYM ALGORYTMIE (refine_surfaces, ustawienia surface_fit, split_ends, fit_tolerance_au):
    1. ROZDZIELENIE KOŃCÓW: na wydłużonym fragmencie (>= long_min_elongation razy dłuższy niż szerszy)
       powierzchnia, która ma co najmniej min_surface_au2 po obu stronach środka fragmentu, jest dzielona
       na dwie - przełomy dwóch końców połączone pasem kory to dwie powierzchnie, nie jedna.
    2. PRZYBLIŻENIE (fit_surface): płaszczyzna ("plane") albo funkcja kwadratowa nad nią ("quadric"),
       dopasowana odpornie (punkty daleko od przybliżenia mało ważą); na kości długiej punkty ważone
       (n . oś)², bo twarz przełomu na końcu patrzy wzdłuż osi, a kora w bok.
    3. ODCIĘCIE: zostają wierzchołki bliżej przybliżenia niż fit_tolerance_au - pas kory ciągnący się
       wzdłuż kości od końca odpada; na kości długiej za płaszczyzną zostaje też środek przekroju (do
       MARROW_RADIUS promienia od osi) - wklęsły dół kanału szpikowego to przełom; potem znowu spójne
       obszary >= min_surface_au2.
    4. GŁOWA KOŚCI (head_min_ratio): koniec kości długiej, przy którym fragment jest SZERSZY niż trzon
       (promień skrajnych 12% długości >= head_min_ratio x promień środkowych 40%), to nasada / głowa
       - naturalna, często szorstka powierzchnia. Przełom trzonu zwęża fragment (na grzesiu_test 0.5-0.85
       promienia trzonu, głowy 1.2-1.4). Obszary leżące w tej części (|miejsce| >= head_zone) odpadają.
    5. OSTRE KRAWĘDZIE KORY (crest_max_axial): grzebień / krawędź wzdłuż trzonu ma normalne w bok (średnie
       |n . oś| < crest_max_axial) i jest prawie płaski (odchylenie od płaszczyzny < crest_max_rms_au);
       przełom patrzy częściowo wzdłuż osi i jest nierówny. Takie obszary odpadają.
    Odrzucone obszary (z powodem) trafiają do "info" -> "dropped".
    Opis każdej powierzchni (środek, normalna, odchylenie od przybliżenia rms_au - płaskie spiłowane
    cięcie ma ~0.1 a.u., przełom więcej - i kąt normalnej do osi kości) trafia do "info" -> "fits".

Jak dodać własny algorytm:
    @detector("moja_metoda")
    def moja_metoda(bone: Bone, params: ContactParams) -> np.ndarray:
        ...  # zwróć labels (V,) jak wyżej
        return labels       # albo (labels, info) - słownik info trafia do pliku wyniku ("info")
    potem: python scripts/detect_contacts.py --method moja_metoda
    Jeśli wynik zależy od czegoś poza sekcją "contacts" (np. od ustawień innej sekcji),
    podaj @detector("nazwa", data_key=funkcja) - zmiana jej wyniku = auto_contacts liczy od nowa.
    albo na stałe (też dla dopasowania i narzędzia): settings.json, "contacts" -> "method".
    Skrypt porówna wynik z ręcznymi oznaczeniami (IoU, precyzja, czułość), jeśli są.

Format pliku JSON:
    {
      "format": "bones-smm-contacts/1",
      "bone": "<id kości>", "source": "manual" | "auto:<metoda>", "updated": "<data>",
      "mesh": {"vertices": V, "fingerprint": "..."},    # do wykrycia zmiany siatki
      "scans": {"<suma kontrolna pliku skanu>": T, ...}, # T 4x4: układ skanu -> układ kości (scan_frames)
      "detector": {"method": "...", "params": {...}},   # tylko wynik algorytmu
      "info": {...},                     # tylko wynik algorytmu, jeśli coś zwrócił (np. "atlas": ranking kości)
      "labels_zlib_b64": "...",          # labels (V,) uint8 skompresowane zlib + base64
      "surfaces": [
        {"id": 1, "name": "...", "area_au2": ..., "centroid": [x, y, z], "normal": [...],
         "max_diameter_au": ..., "extent_au": [dł., szer., grub.], "vertices": liczba,
         "points": [[x, y, z], ...]}     # punkty co ~0.5 a.u. - do przeniesienia na nową siatkę
      ]
    }
Najprościej czytać przez load_contacts(). Jeśli siatka kości się zmieni (np. inne ustawienia
prepare_data), etykiety są przenoszone na nową siatkę po położeniu punktów
(najbliższy punkt bliżej niż REMAP_AU). Punkty są najpierw przeliczane przez wspólny skan
("scans"), więc przeniesienie działa też wtedy, gdy kość jest inaczej ustawiona (np. po
włączeniu / wyłączeniu łączenia skanów zmienia się jej środek). Po zmianie nazwy kości
oznaczenia przechodzą na nową nazwę same (adopt_renamed, wywoływane przez prepare_data.py).
"""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

from config import read_json
from data_parsing import voxel_keep

FORMAT = "bones-smm-contacts/1"
POINTS_AU = 0.5       # odstęp zapisanych punktów powierzchni
REMAP_AU = 0.5        # promień przenoszenia etykiet na zmienioną siatkę
# Kolory powierzchni (te same w narzędziu do oznaczania i w wizualizacjach).
PALETTE = ("#e8508f", "#18b0a6", "#f2b134", "#7b61ff", "#2b9be0", "#9bc53d", "#ff8a3d", "#00c2d1")


@dataclass(frozen=True)
class ContactParams:
    """Ustawienia narzędzia do oznaczania i algorytmu wykrywania (settings.json, sekcja "contacts")."""

    method: str = "roughness"         # algorytm wykrywania (detect_contacts.py, narzędzie, dopasowanie "auto")
    brush_au: float = 2.0             # startowy promień pędzla w narzędziu [a.u.]
    wand_angle_deg: float = 25.0      # startowa tolerancja kąta różdżki [stopnie]
    roughness_radius_au: float = 1.0  # algorytm bazowy: promień liczenia szorstkości
    rough_threshold: float = 0.09     # algorytm bazowy: szorstkość, od której powierzchnia = przełom
    min_surface_au2: float = 15.0     # oba algorytmy: mniejsze obszary są pomijane
    feature_keys: tuple = ("spread_2", "rough_1", "edge_au")  # "features": łączone cechy (surface_features)
    feature_threshold: float = 0.15   # "features": o ile wynik ponad typowy dla tej kości (w odchyleniach) = przełom
    feature_smooth_au: float = 1.0    # "features": uśrednianie wyniku w tym promieniu przed progiem
    feature_open_au: float = 1.0      # "features": otwarcie morfologiczne (rozcina wąskie mostki między obszarami)
    surface_fit: str = "none"         # po algorytmie: przybliżenie przełomu "plane" / "quadric" i odcięcie kory; "none" = bez
    split_ends: bool = False          # po algorytmie: rozdziel powierzchnię sięgającą obu końców wydłużonego fragmentu
    fit_tolerance_au: float = 3.0     # odcięcie: zostają punkty bliżej przybliżenia niż to [a.u.]
    long_min_elongation: float = 2.5  # fragment wydłużony (kość długa) = co najmniej tyle razy dłuższy niż szerszy
    head_min_ratio: float = 0.0       # kość długa: koniec szerszy niż trzon tyle razy = głowa, obszary na niej odpadają; 0 = bez
    head_zone: float = 0.3            # ... odpadają obszary od tego miejsca do końca (0 = środek fragmentu, 1 = koniec)
    crest_max_axial: float = 0.0      # kość długa: ostra krawędź kory = średnie |n . oś| poniżej tego ...; 0 = bez
    crest_max_rms_au: float = 0.35    # ... i odchylenie od płaszczyzny poniżej tego [a.u.]
    exclude: tuple = ()               # kości (fragment id / nazwy), dla których algorytm nic nie liczy (np. całe kości)


@dataclass
class Contacts:
    """Etykiety powierzchni styku jednej kości (zawartość jednego pliku contacts/...json).

    Etykiety dotyczą konkretnej siatki kości - labels[i] odpowiada wierzchołkowi i
    siatki mesh.ply (znaczenie wartości: docstring modułu).
    """

    bone_id: str
    labels: np.ndarray                          # (V,) uint8: 0 = brak, k = powierzchnia k
    names: dict[int, str] = field(default_factory=dict)   # {k: nazwa nadana w narzędziu}
    source: str = "manual"                      # "manual" albo "auto:<metoda>"
    updated: str = ""                           # data i czas zapisu (ISO), "" = jeszcze niezapisane
    remapped: bool = False                      # True = przeniesione ze starej wersji siatki
    detector: dict = field(default_factory=dict)  # wynik algorytmu: {"method", "params"}
    info: dict = field(default_factory=dict)    # dodatkowy wynik algorytmu (np. "atlas": ranking kości z bazy)

    @property
    def surface_ids(self) -> list[int]:
        """Numery powierzchni obecnych w etykietach (bez 0), rosnąco."""
        return [int(k) for k in np.unique(self.labels) if k > 0]

    def name(self, k: int) -> str:
        """Nazwa powierzchni k do wyświetlania.

        Args:
            k: numer powierzchni (1, 2, ...).

        Returns:
            Nazwa nadana w narzędziu; gdy nie nadano (albo pusta) - "Powierzchnia k".
        """
        return self.names.get(k) or f"Powierzchnia {k}"


def surface_color(k: int) -> str:
    """Kolor powierzchni nr k - ten sam w narzędziu do oznaczania i w wizualizacjach.

    Args:
        k: numer powierzchni (1, 2, ...); po wyczerpaniu PALETTE kolory się powtarzają.

    Returns:
        Kolor "#rrggbb" z PALETTE.
    """
    return PALETTE[(k - 1) % len(PALETTE)]


# -----------------
# GEOMETRIA
# -----------------

def mesh_fingerprint(mesh: o3d.geometry.TriangleMesh) -> str:
    """Odcisk siatki: SHA-1 (16 znaków) współrzędnych wierzchołków zaokrąglonych do 0.001 a.u.

    Zapisywany razem z etykietami. Inny odcisk przy odczycie = siatka kości się zmieniła
    (np. inne ustawienia prepare_data), więc etykiet nie da się wziąć wprost - trzeba je
    przenieść po położeniu punktów.

    Args:
        mesh: siatka kości.

    Returns:
        16 znaków hex.
    """
    v = np.round(np.asarray(mesh.vertices), 3).astype(np.float32)
    return hashlib.sha1(v.tobytes()).hexdigest()[:16]


def vertex_areas(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Pole przypadające na wierzchołek = 1/3 pól sąsiednich trójkątów [a.u.^2].

    Suma po wierzchołkach = pole całej siatki, więc suma po wierzchołkach powierzchni
    styku = jej pole.

    Args:
        mesh: siatka kości.

    Returns:
        (V,) pola; 0 dla wierzchołków bez trójkątów.
    """
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    a = 0.5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
    return np.bincount(t.ravel(), weights=np.repeat(a / 3, 3), minlength=len(v))


def vertex_adjacency(mesh: o3d.geometry.TriangleMesh):
    """Macierz sąsiedztwa wierzchołków (krawędzie siatki), scipy.sparse CSR (V, V).

    Przykład: (adj @ maska_0_1) > 0 = wierzchołki, które mają zaznaczonego sąsiada.

    Args:
        mesh: siatka kości.

    Returns:
        Macierz int8 (V, V): element [i, j] > 0, gdy i oraz j łączy krawędź (symetryczna;
        wartość = liczba trójkątów z tą krawędzią).
    """
    t = np.asarray(mesh.triangles)
    n = len(mesh.vertices)
    i = np.concatenate([t[:, 0], t[:, 1], t[:, 2], t[:, 1], t[:, 2], t[:, 0]])
    j = np.concatenate([t[:, 1], t[:, 2], t[:, 0], t[:, 0], t[:, 1], t[:, 2]])
    return coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n, n)).tocsr()


def triangle_labels(mesh: o3d.geometry.TriangleMesh, labels: np.ndarray) -> np.ndarray:
    """Etykiety wierzchołków -> etykiety trójkątów.

    Trójkąt dostaje etykietę, którą mają co najmniej 2 z 3 jego wierzchołków (inaczej 0).

    Args:
        mesh:   siatka kości.
        labels: (V,) etykiety wierzchołków.

    Returns:
        (T,) etykiety trójkątów.
    """
    lt = labels[np.asarray(mesh.triangles)]
    out = np.where((lt[:, 0] == lt[:, 1]) | (lt[:, 0] == lt[:, 2]), lt[:, 0], lt[:, 1])
    out[(out != lt[:, 0]) & (out != lt[:, 1]) & (out != lt[:, 2])] = 0
    agree = (lt == out[:, None]).sum(axis=1) >= 2
    return np.where(agree, out, 0)


def max_diameter(points: np.ndarray) -> float:
    """Największa odległość między dwoma punktami ("średnica" powierzchni) [a.u.].

    Liczona tylko na wierzchołkach otoczki wypukłej (najdalsza para zawsze na niej leży).
    Gdy wierzchołków otoczki jest ponad 3000, brana jest ich losowa próbka (stałe ziarno) -
    wynik może być wtedy minimalnie zaniżony.

    Args:
        points: (N, 3) punkty.

    Returns:
        Średnica; 0 dla mniej niż 2 punktów.
    """
    if len(points) < 2:
        return 0.0
    try:
        pts = points[ConvexHull(points).vertices] if len(points) > 4 else points
    except Exception:           # punkty współliniowe / współpłaszczyznowe
        pts = points
    if len(pts) > 3000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 3000, replace=False)]
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    return float(d.max())


def surface_stats(mesh: o3d.geometry.TriangleMesh, labels: np.ndarray) -> list[dict]:
    """Miary każdej powierzchni styku - do wstępnej selekcji par kości (podproblem 5).

    Środek i normalna są ważone polem wierzchołków, wymiary liczone wzdłuż osi
    głównych (PCA) punktów powierzchni. Jeśli siatka nie ma normalnych wierzchołków,
    są liczone (siatka jest zmieniana).

    Args:
        mesh:   siatka kości.
        labels: (V,) etykiety wierzchołków.

    Returns:
        Lista słowników {"id", "area_au2", "centroid", "normal" (średnia ważona polem),
        "max_diameter_au" (największa odległość dwóch punktów), "extent_au" (długość,
        szerokość, grubość wzdłuż osi głównych), "vertices" (liczba)}.
    """
    v = np.asarray(mesh.vertices)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    n = np.asarray(mesh.vertex_normals)
    va = vertex_areas(mesh)
    out = []
    for k in [int(x) for x in np.unique(labels) if x > 0]:
        m = labels == k
        w = va[m]
        pts = v[m]
        c = (pts * w[:, None]).sum(axis=0) / max(w.sum(), 1e-12)
        nn = (n[m] * w[:, None]).sum(axis=0)
        nn = nn / max(np.linalg.norm(nn), 1e-12)
        if len(pts) >= 3:
            _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
            proj = (pts - c) @ vt.T
            extent = np.ptp(proj, axis=0)
        else:
            extent = np.zeros(3)
        out.append({"id": k, "area_au2": round(float(w.sum()), 2), "centroid": np.round(c, 3).tolist(),
                     "normal": np.round(nn, 4).tolist(), "max_diameter_au": round(max_diameter(pts), 2),
                     "extent_au": np.round(extent, 2).tolist(), "vertices": int(m.sum())})
    return out


# -----------------
# ZAPIS / ODCZYT
# -----------------

def scan_frames(bone) -> dict[str, list]:
    """Przekształcenia skanów źródłowych kości: {suma kontrolna pliku skanu: T 4x4 układ skanu -> kości}.

    Zapisywane w pliku oznaczeń - dzięki nim punkty da się przenieść na kość, która powstała
    z tego samego skanu, ale jest inaczej ustawiona (inne łączenie, inna nazwa).

    Args:
        bone: kość (bones.Bone) z meta.json; None = brak danych.

    Returns:
        Słownik (pusty dla kości bez sum kontrolnych - przygotowanych starszą wersją).
    """
    if bone is None:
        return {}
    return {s["checksum"]: s["transform_to_bone"] for s in bone.meta.get("scans", []) if s.get("checksum")}


def _frame_shift(saved: dict, current: dict) -> np.ndarray | None:
    """Przekształcenie 4x4 starego układu kości na nowy przez wspólny skan (None = brak wspólnego)."""
    common = sorted(set(saved) & set(current))
    if not common:
        return None
    return np.asarray(current[common[0]], dtype=float) @ np.linalg.inv(np.asarray(saved[common[0]], dtype=float))


def contacts_path(bone_id: str, source: str = "manual", root: Path | None = None) -> Path:
    """Ścieżka pliku etykiet kości.

    Args:
        bone_id: id kości.
        source:  "manual" -> <root>/<id>.json, cokolwiek innego (np. "auto")
                 -> <root>/auto/<id>.json.
        root:    folder oznaczeń (domyślnie CONTACTS_DIR z settings.json, np. inny w testach).

    Returns:
        Ścieżka pliku (plik nie musi istnieć).
    """
    if root is None:
        from config import CONTACTS_DIR as root
    root = Path(root)
    return root / f"{bone_id}.json" if source == "manual" else root / "auto" / f"{bone_id}.json"


def save_contacts(mesh: o3d.geometry.TriangleMesh, contacts: Contacts, path: Path | None = None,
                  frames: dict | None = None) -> Path | None:
    """Zapisuje etykiety do JSON (format: docstring modułu) i zwraca ścieżkę.

    Ręczne bez żadnej powierzchni = plik jest usuwany (zwraca None). Wynik algorytmu jest
    zapisywany zawsze, także pusty - żeby nie liczyć go na nowo przy każdym uruchomieniu.
    Oprócz etykiet zapisuje statystyki każdej powierzchni (surface_stats) i jej punkty
    co POINTS_AU - po nich etykiety da się przenieść na zmienioną siatkę.

    Args:
        mesh:     siatka kości, do której należą etykiety.
        contacts: etykiety; contacts.source decyduje o domyślnym folderze.
        path:     plik docelowy (domyślnie contacts_path(...) wg contacts.source).
        frames:   scan_frames(kość) - układy skanów źródłowych (do przeniesienia na inaczej
                  ustawioną kość); None = bez nich.

    Returns:
        Ścieżka zapisanego pliku albo None (usunięte puste ręczne oznaczenia).

    Raises:
        ValueError: liczba etykiet różni się od liczby wierzchołków siatki.
    """
    manual = contacts.source == "manual"
    path = Path(path or contacts_path(contacts.bone_id, "manual" if manual else "auto"))
    labels = np.asarray(contacts.labels, dtype=np.uint8)
    if len(labels) != len(mesh.vertices):
        raise ValueError(f"{contacts.bone_id}: {len(labels)} etykiet, a siatka ma {len(mesh.vertices)} wierzchołków")
    if manual and not labels.any():
        if path.exists():
            path.unlink()
        return None
    v = np.asarray(mesh.vertices)
    stats = {s["id"]: s for s in surface_stats(mesh, labels)}
    surfaces = []
    for k in sorted(stats):
        idx = np.flatnonzero(labels == k)
        pts = v[idx][voxel_keep(v[idx], POINTS_AU)]
        surfaces.append({"id": k, "name": contacts.name(k), **{a: b for a, b in stats[k].items() if a != "id"},
                         "points": np.round(pts, 2).tolist()})
    data = {"format": FORMAT, "bone": contacts.bone_id, "source": contacts.source,
            "updated": datetime.now().isoformat(timespec="seconds"),
            "mesh": {"vertices": len(v), "fingerprint": mesh_fingerprint(mesh)},
            **({"scans": frames} if frames else {}),
            **({"detector": contacts.detector} if contacts.detector else {}),
            **({"info": contacts.info} if contacts.info else {}),
            "labels_zlib_b64": base64.b64encode(zlib.compress(labels.tobytes(), 9)).decode("ascii"),
            "surfaces": surfaces}
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=1)
    # listy liczb w jednej linii - plik jest dużo krótszy i czytelniejszy w gicie
    text = _compact_lists(text)
    path.write_text(text, encoding="utf-8")
    return path


def _compact_lists(text: str) -> str:
    """Skraca tekst JSON: listy liczb i listy punktów [[x, y, z], ...] zapisuje w jednej linii.

    Args:
        text: JSON z json.dumps(..., indent=...).

    Returns:
        Ten sam JSON (te same dane), krótszy.
    """
    import re

    text = re.sub(r"\[\s*([-0-9.,\s]+?)\s*\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", text)
    # lista punktów [[x, y, z], ...] też w jednej linii
    return re.sub(r"\[\s*(\[[^\[\]]*\](?:\s*,\s*\[[^\[\]]*\])*)\s*\]",
                  lambda m: "[" + re.sub(r"\]\s*,\s*\[", "], [", m.group(1)) + "]", text)


def load_contacts(bone_id: str, mesh: o3d.geometry.TriangleMesh, source: str = "manual",
                  path: Path | None = None, root: Path | None = None,
                  frames: dict | None = None) -> Contacts | None:
    """Wczytuje zapisane etykiety kości: "manual" (ręczne) albo "auto" (wynik algorytmu).

    Jeśli siatka zmieniła się od zapisu (inny fingerprint), etykiety są przenoszone
    po położeniu punktów, a `remapped` = True: wierzchołek dostaje etykietę najbliższego
    zapisanego punktu, jeśli ten jest bliżej niż REMAP_AU (inaczej 0). Plik nie jest
    przy tym zmieniany - przeniesione etykiety trzeba zapisać samemu. Gdy plik i kość mają
    wspólny skan źródłowy (`frames`), punkty są najpierw przeliczane do nowego układu kości.

    Args:
        bone_id: id kości.
        mesh:    aktualna siatka kości (out/3d/<zestaw>/bones/<id>/mesh.ply).
        source:  "manual" albo "auto" - który plik czytać (gdy nie podano `path`).
        path:    konkretny plik zamiast contacts_path(bone_id, source, root).
        root:    folder oznaczeń zamiast CONTACTS_DIR.
        frames:  scan_frames(kość) - do przeliczenia punktów, gdy kość jest inaczej ustawiona.

    Returns:
        Contacts z etykietami (V,) dla `mesh` albo None, gdy pliku nie ma.
    """
    path = Path(path or contacts_path(bone_id, source, root))
    if not path.exists():
        return None
    data = read_json(path)
    n = len(mesh.vertices)
    labels = np.zeros(n, dtype=np.uint8)
    names = {int(s["id"]): s.get("name", "") for s in data.get("surfaces", [])}
    same = data.get("mesh", {}).get("fingerprint") == mesh_fingerprint(mesh)
    if same:
        labels = np.frombuffer(zlib.decompress(base64.b64decode(data["labels_zlib_b64"])), dtype=np.uint8).copy()
    else:
        pts, ids = [], []
        for s in data["surfaces"]:
            p = np.asarray(s.get("points", []), dtype=float).reshape(-1, 3)
            pts.append(p)
            ids.append(np.full(len(p), int(s["id"])))
        if pts and sum(len(p) for p in pts):
            pts = np.vstack(pts)
            shift = _frame_shift(data.get("scans", {}), frames or {})
            if shift is not None:
                pts = pts @ shift[:3, :3].T + shift[:3, 3]
            ids = np.concatenate(ids)
            v = np.asarray(mesh.vertices)
            saved = np.frombuffer(zlib.decompress(base64.b64decode(data["labels_zlib_b64"])), dtype=np.uint8)
            dv, vi = cKDTree(v).query(pts) if len(saved) == n else (None, None)
            if dv is not None and dv.max() < 0.02 and np.array_equal(saved[vi], ids):
                labels = saved.copy()       # ta sama siatka, tylko przesunięta (np. inny środek) - etykiety wprost
            else:
                d, idx = cKDTree(pts).query(v, distance_upper_bound=REMAP_AU)
                ok = np.isfinite(d)
                labels[ok] = ids[idx[ok]]
    return Contacts(bone_id=bone_id, labels=labels, names=names, source=data.get("source", source),
                    updated=data.get("updated", ""), remapped=not same, detector=data.get("detector", {}),
                    info=data.get("info", {}))


# -----------------
# ZMIANA NAZWY KOŚCI
# -----------------

ADOPT_SHARE = 0.8     # część zapisanych punktów, która musi leżeć na siatce, żeby uznać plik za oznaczenia tej kości


def _file_points(data: dict) -> np.ndarray:
    """Wszystkie zapisane punkty powierzchni z pliku oznaczeń (M, 3)."""
    pts = [np.asarray(s.get("points", []), dtype=float).reshape(-1, 3) for s in data.get("surfaces", [])]
    return np.vstack(pts) if pts else np.zeros((0, 3))


def adopt_renamed(bones, root: Path | None = None, extra_roots=(), log: Callable | None = print) -> list[dict]:
    """Po zmianie nazwy kości (np. folderu ze skanem) przenosi jej oznaczenia na nową nazwę.

    Oznaczenia są zapisane pod id kości (<root>/<id>.json), a id pochodzi z nazwy folderu / pliku
    skanu. Po zmianie nazwy i ponownym prepare_data.py plik zostaje pod starą nazwą ("osierocony").
    Każda kość BEZ własnych ręcznych oznaczeń dostaje osierocony plik, który do niej pasuje:
        1. ten sam odcisk siatki (zmieniła się tylko nazwa), albo
        2. ten sam plik skanu źródłowego (suma kontrolna w "scans") - nazwa i ustawienie kości
           zmieniły się naraz (np. inne ustawienia łączenia), albo
        3. (starsze pliki bez "scans") co najmniej ADOPT_SHARE zapisanych punktów powierzchni leży na siatce (bliżej niż
           REMAP_AU) - nazwa i siatka zmieniły się naraz (np. inne ustawienia łączenia);
           etykiety przeniesie potem load_contacts, jak przy każdej zmianie siatki.
    Plik z `root` jest przenoszony pod nową nazwę (z poprawionym polem "bone"), plik z
    `extra_roots` (np. oznaczenia zestawu o zmienionej nazwie) - kopiowany, oryginał zostaje.
    Wyniki algorytmu (<root>/auto/) z tym samym odciskiem siatki też są przenoszone, a pozostałe
    osierocone wyniki algorytmu - usuwane (to tylko cache, policzą się od nowa).

    Args:
        bones:       wszystkie kości zestawu (z siatkami) - nazwy spoza tej listy to pliki osierocone.
        root:        folder oznaczeń zestawu (domyślnie CONTACTS_DIR z settings.json).
        extra_roots: dodatkowe foldery z oznaczeniami do przejrzenia (kopiowane, nie przenoszone).
        log:         funkcja do komunikatów (None = bez komunikatów).

    Returns:
        Lista przeniesień {"from": ścieżka, "to": ścieżka, "how": jak rozpoznano, "copy": bool}.
    """
    if root is None:
        from config import CONTACTS_DIR as root
    root = Path(root)
    ids = {b.id for b in bones}
    orphans = [(f, False) for f in sorted(root.glob("*.json")) if f.stem not in ids]
    for extra in extra_roots:
        orphans += [(f, True) for f in sorted(Path(extra).glob("*.json"))]
    moves = []
    needy = [b for b in bones if not contacts_path(b.id, "manual", root).exists()]
    if orphans and needy:
        prints = {b.id: mesh_fingerprint(b.mesh) for b in needy}
        trees: dict[str, cKDTree] = {}
        cands = []                                  # (pewność, kolejność, plik, kopia, kość, sposób)
        for n, (f, copy) in enumerate(orphans):
            try:
                data = read_json(f)
            except (OSError, ValueError):
                continue
            if data.get("format") != FORMAT or data.get("source", "manual") != "manual":
                continue
            fp = data.get("mesh", {}).get("fingerprint")
            saved_frames = data.get("scans", {})
            pts = _file_points(data)
            for b in needy:
                if fp == prints[b.id]:
                    cands.append((3.0, n, f, copy, b, "odcisk siatki"))
                elif set(saved_frames) & set(scan_frames(b)):
                    cands.append((2.0, n, f, copy, b, "ten sam plik skanu"))
                elif len(pts):
                    if b.id not in trees:
                        trees[b.id] = cKDTree(np.asarray(b.mesh.vertices))
                    share = float(np.isfinite(trees[b.id].query(pts, distance_upper_bound=REMAP_AU)[0]).mean())
                    if share >= ADOPT_SHARE:
                        cands.append((share, n, f, copy, b, f"punkty ({share:.0%} na siatce)"))
        used_files, used_bones = set(), set()
        for _, _, f, copy, b, how in sorted(cands, key=lambda c: (-c[0], c[1])):
            if f in used_files or b.id in used_bones:
                continue
            used_files.add(f)
            used_bones.add(b.id)
            data = read_json(f)
            data["bone"] = b.id
            target = contacts_path(b.id, "manual", root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_compact_lists(json.dumps(data, ensure_ascii=False, indent=1)), encoding="utf-8")
            if not copy:
                f.unlink()
            moves.append({"from": f, "to": target, "how": how, "copy": copy})
            if log:
                verb = "skopiowano" if copy else "przeniesiono"
                log(f"  oznaczenia styku: {verb} {f.parent.name}/{f.name} -> {target.name} "
                    f"(zmiana nazwy kości; rozpoznane po: {how})")
    for f, _ in orphans:
        if f.exists() and f.parent == root and f not in {m["from"] for m in moves} and log:
            log(f"  UWAGA: oznaczenia {f.name} nie pasują do żadnej kości (zostają w {root.name}/; "
                "jeśli kość zmieniła nazwę i kształt, zmień nazwę pliku ręcznie)")

    auto_dir = root / "auto"
    auto_orphans = [f for f in auto_dir.glob("*.json") if f.stem not in ids] if auto_dir.is_dir() else []
    if auto_orphans:
        by_print = {mesh_fingerprint(b.mesh): b.id for b in bones if not contacts_path(b.id, "auto", root).exists()}
        for f in auto_orphans:
            try:
                data = read_json(f)
                new_id = by_print.pop(data.get("mesh", {}).get("fingerprint"), None)
            except (OSError, ValueError):
                new_id = None
            if new_id:
                data["bone"] = new_id
                contacts_path(new_id, "auto", root).write_text(
                    _compact_lists(json.dumps(data, ensure_ascii=False, indent=1)), encoding="utf-8")
            f.unlink()
    return moves


# -----------------
# ALGORYTMY WYKRYWANIA
# -----------------

DETECTORS: dict[str, Callable] = {}
DATA_KEYS: dict[str, Callable] = {}     # algorytm -> funkcja (bone, params) -> tekst (patrz detector)


def detector(name: str, data_key: Callable | None = None):
    """Dekorator rejestrujący algorytm wykrywania w DETECTORS pod nazwą `name`.

    Funkcja dostaje (bone: Bone, params: ContactParams) i zwraca labels (V,) dla
    wierzchołków bone.mesh. Przykład: docstring modułu, "Jak dodać własny algorytm".

    Args:
        name:     nazwa algorytmu (argument --method, klucz "method" w settings.json).
        data_key: dla algorytmów, których wynik zależy od czegoś poza sekcją "contacts" (np. od
                  ustawień innej sekcji): (bone, params) -> krótki tekst, który się zmienia,
                  gdy te dane się zmienią. Zapisywany w wyniku ("detector" -> "data").

    Returns:
        Dekorator, który rejestruje funkcję i zwraca ją bez zmian.
    """
    def wrap(fn):
        """Rejestruje `fn` w DETECTORS pod nazwą `name` i zwraca ją bez zmian."""
        DETECTORS[name] = fn
        if data_key is not None:
            DATA_KEYS[name] = data_key
        return fn
    return wrap


def split_regions(mesh: o3d.geometry.TriangleMesh, mask: np.ndarray, min_area: float) -> np.ndarray:
    """Maska wierzchołków -> etykiety 1..K spójnych obszarów (od największego), małe pominięte.

    Obszar = wierzchołki z maski połączone krawędziami siatki. Przydatne na końcu
    każdego algorytmu: zamienia "te wierzchołki wyglądają na przełom" na osobne powierzchnie.

    Args:
        mesh:     siatka kości.
        mask:     (V,) bool.
        min_area: obszary o mniejszym polu [a.u.²] dostają 0.

    Returns:
        (V,) uint8 etykiety; najwyżej 255 obszarów (największych).
    """
    adj = vertex_adjacency(mesh)
    idx = np.flatnonzero(mask)
    labels = np.zeros(len(mask), dtype=np.uint8)
    if len(idx) == 0:
        return labels
    sub = adj[idx][:, idx]
    _, comp = connected_components(sub, directed=False)
    va = vertex_areas(mesh)[idx]
    areas = np.bincount(comp, weights=va)
    order = [c for c in np.argsort(-areas) if areas[c] >= min_area][:255]
    for k, c in enumerate(order, start=1):
        labels[idx[comp == c]] = k
    return labels


def local_shape(mesh: o3d.geometry.TriangleMesh, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Lokalne miary kształtu dla każdego wierzchołka (sąsiedzi w promieniu `radius`).

    Sąsiedzi to punkty równomiernej próbki powierzchni (co radius / 3), więc wynik nie
    zależy od gęstości siatki. Wierzchołki z mniej niż 6 sąsiadami dostają 0.

    Args:
        mesh:   siatka kości.
        radius: promień sąsiedztwa [a.u.].

    Returns:
        (szorstkość: odchylenie sąsiadów od dopasowanej płaszczyzny / promień,
         rozrzut normalnych: 1 - |średnia normalnych sąsiadów|, 0 = płasko, rośnie na krawędziach)
    """
    from data_parsing import sample_surface

    v = np.asarray(mesh.vertices)
    s = sample_surface(mesh, radius / 3)            # sąsiedzi z równomiernej próbki (~30 na wierzchołek)
    p, n = s.points, s.normals
    nb = cKDTree(p).query_ball_point(v, radius, workers=-1)
    counts = np.fromiter((len(x) for x in nb), dtype=np.int64, count=len(v))
    rows = np.repeat(np.arange(len(v)), counts)
    cols = np.fromiter((j for x in nb for j in x), dtype=np.int64, count=counts.sum())
    W = coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(v), len(p))).tocsr()
    cnt = np.maximum(counts, 1)[:, None].astype(float)
    mean = (W @ p) / cnt
    outer = np.einsum("ni,nj->nij", p, p).reshape(len(p), 9)
    cov = ((W @ outer) / cnt).reshape(-1, 3, 3) - np.einsum("ni,nj->nij", mean, mean)
    smallest = np.linalg.eigvalsh(cov)[:, 0]
    rough = np.sqrt(np.clip(smallest, 0, None)) / radius
    spread = 1 - np.linalg.norm((W @ n) / cnt, axis=1)
    rough[counts < 6] = 0
    spread[counts < 6] = 0
    return rough, spread


def _rings(mesh: o3d.geometry.TriangleMesh, dist: float) -> int:
    """Ile pierścieni sąsiadów siatki odpowiada odległości `dist` (wg średniej długości krawędzi).

    Średnia jest liczona z jednej krawędzi każdego trójkąta (v0-v1) - to wystarcza.

    Args:
        mesh: siatka kości.
        dist: odległość po powierzchni [a.u.].

    Returns:
        Liczba pierścieni, co najmniej 1.
    """
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    edge = np.linalg.norm(v[t[:, 0]] - v[t[:, 1]], axis=1).mean()
    return max(1, int(round(dist / edge)))


def morph(mesh: o3d.geometry.TriangleMesh, mask: np.ndarray, dist: float, grow: bool) -> np.ndarray:
    """Dylatacja (grow=True) albo erozja (grow=False) maski wierzchołków o ~`dist` po siatce.

    Jeden krok = jeden pierścień sąsiadów (krawędzie siatki); liczba kroków z _rings.

    Args:
        mesh: siatka kości.
        mask: (V,) bool - nie jest zmieniana.
        dist: o ile poszerzyć / zwęzić [a.u.].
        grow: True = poszerz, False = zwęź.

    Returns:
        Nowa maska (V,) bool.
    """
    adj = vertex_adjacency(mesh)
    m = mask.copy()
    for _ in range(_rings(mesh, dist)):
        hits = adj @ (m if grow else ~m).astype(np.int8)
        m = (m | (hits > 0)) if grow else (m & (hits == 0))
    return m


@detector("roughness")
def detect_roughness(bone, params: ContactParams) -> np.ndarray:
    """ALGORYTM BAZOWY (punkt wyjścia, nie rozwiązanie): przełom jest szorstki, kora kości gładka.

    1. Szorstkość każdego wierzchołka: odchylenie powierzchni od płaszczyzny w promieniu
       roughness_radius_au, względem promienia (kora ~0.04, przełom gąbczasty ~0.15).
    2. Szorstkie = powyżej rough_threshold, z pominięciem pasa ~roughness_radius_au
       wzdłuż otwartego brzegu skanu (brzeg jest postrzępiony, ale to nie przełom).
    3. Otwarcie morfologiczne (erozja, potem dylatacja o ~0.6 x promień): znikają cienkie
       linie ostrych krawędzi, zostają szerokie ściany przełomu.
    4. Spójne obszary >= min_surface_au2 = wykryte powierzchnie (od największej).
    Słabo działa na gładkich, świeżych przełomach kości zbitej - tu potrzebny lepszy algorytm.

    Args:
        bone:   kość (wystarczy bone.mesh).
        params: roughness_radius_au, rough_threshold, min_surface_au2.

    Returns:
        (V,) uint8 etykiety wierzchołków bone.mesh.
    """
    mesh = bone.mesh
    rough, _ = local_shape(mesh, params.roughness_radius_au)
    mask = rough > params.rough_threshold
    mask &= ~morph(mesh, boundary_vertices(mesh), params.roughness_radius_au, grow=True)   # brzeg skanu
    r = 0.6 * params.roughness_radius_au
    mask = morph(mesh, morph(mesh, mask, r, grow=False), r, grow=True)
    return split_regions(mesh, mask, params.min_surface_au2)


def _features_data_key(bone, params: ContactParams) -> str:
    """Ustawienia cech (settings.json, sekcja "surface_features") - od nich zależy wynik "features"."""
    import config

    return json.dumps(asdict(config.surface_feature_params()), sort_keys=True)


def bone_scores(features: dict[str, np.ndarray], areas: np.ndarray, keys) -> np.ndarray:
    """Wynik "jak bardzo to przełom" każdego wierzchołka - bez żadnych oznaczeń.

    Każda cecha jest porównywana z typową wartością na TEJ SAMEJ kości: większość kości to
    kora, więc mediana cechy (ważona polem) to "poziom kory". Cecha (log) jest zamieniana na
    liczbę odchyleń od mediany: (log x - mediana) / (1.4826 * MAD) - odporne na przełom
    i niezależne od skanera (inna gęstość siatki, inna skala szorstkości = te same liczby).
    Cechy mniejsze na przełomie (np. długość krawędzi siatki) dostają znak minus
    (feature_info "sign"). Wynik = średnia po cechach: ~0 na typowej korze, > 0 na przełomie.

    Args:
        features: wynik surface_features.vertex_features.
        areas:    (V,) pola wierzchołków (wagi mediany).
        keys:     które cechy połączyć.

    Returns:
        (V,) wynik.

    Raises:
        ValueError: nieznana cecha albo cecha bez stałego kierunku (np. jasność koloru).
    """
    import surface_features as sf

    info = sf.feature_info()
    out = np.zeros(len(areas))
    for k in keys:
        if k not in features:
            raise ValueError(f"Nieznana cecha {k!r}, dostępne: {', '.join(features)}")
        sign = info.get(k, {}).get("sign", 0)
        if sign == 0:
            raise ValueError(f"Cecha {k!r} nie ma stałego kierunku (na różnych kościach raz większa, raz mniejsza "
                             f"na przełomie) - nie nadaje się do \"features\"")
        x = np.log(features[k] + 1e-3)
        med = sf.weighted_quantiles(x, areas, [0.5])[0]
        mad = 1.4826 * sf.weighted_quantiles(np.abs(x - med), areas, [0.5])[0]
        out += sign * (x - med) / max(mad, 1e-9)
    return out / max(len(keys), 1)


@detector("features", data_key=_features_data_key)
def detect_features(bone, params: ContactParams) -> np.ndarray:
    """KILKA CECH NARAZ, BEZ OZNACZEŃ: szorstkość, rozrzut normalnych i gęstość siatki.

    1. Cechy każdego wierzchołka (src/surface_features.py; domyślnie rozrzut normalnych 2 a.u.,
       szorstkość 1 a.u., długość krawędzi siatki - lista: feature_keys, promienie:
       settings.json -> "surface_features"; opis: raporty/przelom_vs_kora.md).
    2. Każda cecha względem typowej wartości na tej samej kości (bone_scores): przełom jest
       bardziej szorstki, ma większy rozrzut normalnych i gęstszą siatkę niż reszta kości.
       Wynik = średnia tych odchyleń.
    3. Uśrednienie wyniku w promieniu feature_smooth_au (pojedyncze szorstkie punkty kory
       i gładkie punkty przełomu znikają) i próg feature_threshold.
    4. Bez pasu przy otwartym brzegu skanu, otwarcie morfologiczne o feature_open_au (znikają
       cienkie linie krawędzi i wąskie mostki - bez tego np. oba końce długiej kości potrafią
       połączyć się w jeden obszar), spójne obszary >= min_surface_au2 (od największego).
    Założenie: większość zeskanowanej powierzchni to kora (mediana = kora). Na kości, która
    jest prawie cała przełomem, algorytm zaznaczy za mało.

    Args:
        bone:   kość.
        params: feature_keys, feature_threshold, feature_smooth_au, feature_open_au, min_surface_au2.

    Returns:
        (V,) uint8 etykiety wierzchołków bone.mesh.

    Raises:
        ValueError: zła nazwa cechy w feature_keys.
    """
    import config
    import surface_features as sf

    sfp = config.surface_feature_params()
    score = bone_scores(sf.cached_features(bone, sfp), vertex_areas(bone.mesh), params.feature_keys)
    return regions_from_score(bone.mesh, score, params.feature_threshold, params, sfp.boundary_margin_au)


def regions_from_score(mesh: o3d.geometry.TriangleMesh, score: np.ndarray, threshold: float,
                       params: ContactParams, boundary_margin_au: float) -> np.ndarray:
    """Wynik "jak bardzo to przełom" wierzchołków -> etykiety powierzchni (wspólny koniec "features" i "atlas").

    1. Uśrednienie wyniku w promieniu feature_smooth_au (ważone polem; 0 = bez).
    2. Przełom = wynik > threshold, z pominięciem pasa boundary_margin_au przy otwartym brzegu skanu.
    3. Otwarcie morfologiczne o feature_open_au (rozcina wąskie mostki, usuwa cienkie linie).
    4. Spójne obszary >= min_surface_au2, od największego (split_regions).

    Args:
        mesh:               siatka kości.
        score:              (V,) wynik wierzchołków (większy = bardziej przełom).
        threshold:          próg wyniku.
        params:             feature_smooth_au, feature_open_au, min_surface_au2.
        boundary_margin_au: szerokość pomijanego pasa przy brzegu skanu [a.u.].

    Returns:
        (V,) uint8 etykiety wierzchołków.
    """
    import surface_features as sf

    v, va = np.asarray(mesh.vertices), vertex_areas(mesh)
    if params.feature_smooth_au > 0:
        W, _ = sf._neighbours(v, v, params.feature_smooth_au)
        score = (W @ (score * va)) / np.maximum(W @ va, 1e-12)
    mask = score > threshold
    mask &= sf._distance_to(v, boundary_vertices(mesh)) > boundary_margin_au
    if params.feature_open_au > 0:
        r = params.feature_open_au
        mask = morph(mesh, morph(mesh, mask, r, grow=False), r, grow=True)
    return split_regions(mesh, mask, params.min_surface_au2)


# -----------------
# PRZYBLIŻENIE POWIERZCHNI PRZEŁOMU (płaszczyzna / funkcja kwadratowa)
# -----------------

SURFACE_FITS = ("none", "plane", "quadric")
MARROW_RADIUS = 0.6     # odcięcie: za płaszczyzną zostaje środek przekroju (do tylu promieni kości od osi) - kanał szpikowy


def bone_axis(mesh: o3d.geometry.TriangleMesh) -> dict:
    """Oś główna kości (PCA wierzchołków).

    Returns:
        {"c": środek, "d": kierunek osi (jednostkowy), "elong": wydłużenie (odchylenie wzdłuż osi /
         największe w poprzek), "r": promień (mediana odległości wierzchołków od osi) [a.u.],
         "lo", "hi": zasięg wierzchołków wzdłuż osi od środka [a.u.]}.
    """
    v = np.asarray(mesh.vertices)
    c = v.mean(axis=0)
    w, U = np.linalg.eigh(np.cov((v - c).T))
    d = U[:, -1]
    t = (v - c) @ d
    radial = np.linalg.norm((v - c) - np.outer(t, d), axis=1)
    return {"c": c, "d": d, "elong": float(np.sqrt(w[-1] / max(w[-2], 1e-12))),
            "r": float(max(np.median(radial), 1e-6)), "lo": float(t.min()), "hi": float(t.max())}


def fit_surface(points: np.ndarray, normals: np.ndarray, weights: np.ndarray | None = None,
                axis: np.ndarray | None = None, mode: str = "plane", iterations: int = 6) -> dict:
    """Odporne przybliżenie powierzchni przełomu płaszczyzną albo funkcją kwadratową nad płaszczyzną.

    Ważone PCA, potem kilka razy wagi x exp(-(odległość / sigma)² / 2), sigma = 2.5 x odporne
    odchylenie (MAD) - punkty daleko od płaszczyzny (kora, odpryski) mało ważą. Z osią kości
    wagi startowe x (n . oś)²: twarz przełomu na końcu kości długiej patrzy wzdłuż osi, kora w bok.
    "quadric": na koniec h = a u² + b uv + c v² + d u + e v + f (h = odległość od płaszczyzny,
    u, v = współrzędne w płaszczyźnie) metodą najmniejszych kwadratów z tymi samymi wagami.

    Args:
        points, normals: (N, 3) punkty powierzchni i ich normalne (na zewnątrz kości).
        weights:         (N,) pole przypadające na punkt; None = równe.
        axis:            kierunek osi kości długiej; None = bez ważenia normalnymi.
        mode:            "plane" albo "quadric".
        iterations:      kroków odpornego ważenia.

    Returns:
        {"mode", "center", "normal" (na zewnątrz kości), "u", "v" (osie w płaszczyźnie, u = najdłuższa),
         "coef" (6 współczynników funkcji kwadratowej; zera dla płaszczyzny), "rms_au": ważone odchylenie
         punktów od przybliżenia (bez punktów odrzuconych), "axis_deg": kąt normalnej do osi (None bez osi),
         "extent_au": (dł., szer.) w płaszczyźnie}.
    """
    P = np.asarray(points, dtype=float)
    N = np.asarray(normals, dtype=float)
    w0 = np.ones(len(P)) if weights is None else np.asarray(weights, dtype=float).copy()
    if axis is not None:
        w0 = w0 * ((N @ axis) ** 2 + 1e-3)
    w = w0.copy()
    for _ in range(max(iterations, 1)):
        c = (w[:, None] * P).sum(axis=0) / max(w.sum(), 1e-12)
        X = P - c
        C = (X * w[:, None]).T @ X / max(w.sum(), 1e-12)
        _, E = np.linalg.eigh(C)
        nrm = E[:, 0]
        dist = X @ nrm
        mad = _weighted_median(np.abs(dist), w)
        sigma = max(2.5 * 1.4826 * mad, 0.3)
        w = w0 * np.exp(-0.5 * (dist / sigma) ** 2)
    if (w[:, None] * N).sum(axis=0) @ nrm < 0:
        nrm = -nrm
    u, v_ = E[:, 2], np.cross(nrm, E[:, 2])
    X = P - c
    pu, pv, h = X @ u, X @ v_, X @ nrm
    coef = np.zeros(6)
    if mode == "quadric" and len(P) >= 6:
        A = np.column_stack([pu ** 2, pu * pv, pv ** 2, pu, pv, np.ones(len(P))])
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(A * sw[:, None], h * sw, rcond=None)[0]
    fit = {"mode": mode, "center": c, "normal": nrm, "u": u, "v": v_, "coef": coef}
    res = surface_residual(fit, P)
    keep = w > 0.05 * w.max() if w.max() > 0 else np.ones(len(P), bool)
    fit["rms_au"] = float(np.sqrt((w[keep] * res[keep] ** 2).sum() / max(w[keep].sum(), 1e-12)))
    fit["axis_deg"] = None if axis is None else float(np.degrees(np.arccos(min(1.0, abs(float(nrm @ axis))))))
    fit["extent_au"] = (float(np.ptp(pu[keep])) if keep.any() else 0.0, float(np.ptp(pv[keep])) if keep.any() else 0.0)
    return fit


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    """Mediana ważona (0 dla pustych danych)."""
    if len(x) == 0 or w.sum() <= 0:
        return 0.0
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, cw[-1] / 2)])


def surface_residual(fit: dict, points: np.ndarray) -> np.ndarray:
    """Odległość punktów od przybliżenia (ze znakiem: + = na zewnątrz kości) [a.u.]."""
    X = np.asarray(points, dtype=float) - fit["center"]
    pu, pv, h = X @ fit["u"], X @ fit["v"], X @ fit["normal"]
    a, b, c, d, e, f = fit["coef"]
    return h - (a * pu ** 2 + b * pu * pv + c * pv ** 2 + d * pu + e * pv + f)


def fit_info(fit: dict) -> dict:
    """Opis przybliżenia do zapisu w JSON ("info" -> "fits"): liczby zaokrąglone, bez tablic numpy."""
    r = lambda x: [round(float(t), 4) for t in x]  # noqa: E731
    return {"mode": fit["mode"], "center": r(fit["center"]), "normal": r(fit["normal"]), "u": r(fit["u"]),
            "v": r(fit["v"]), "coef": r(fit["coef"]), "rms_au": round(fit["rms_au"], 4),
            "axis_deg": None if fit["axis_deg"] is None else round(fit["axis_deg"], 2),
            "extent_au": r(fit["extent_au"])}


def refine_surfaces(mesh: o3d.geometry.TriangleMesh, labels: np.ndarray,
                    params: ContactParams) -> tuple[np.ndarray, dict]:
    """Po algorytmie wykrywania: rozdzielenie końców, przybliżenie przełomu i odcięcie kory (docstring modułu).

    Args:
        mesh:   siatka kości.
        labels: (V,) etykiety z algorytmu.
        params: surface_fit, split_ends, fit_tolerance_au, long_min_elongation, min_surface_au2.

    Returns:
        ((V,) uint8 nowe etykiety 1..K od największej, {"fits": {"<k>": fit_info} - przybliżenie każdej
         powierzchni (puste przy surface_fit = "none"), "dropped": [{"area_au2", "pos", "why"}] - obszary
         odrzucone jako głowa kości albo ostra krawędź}; {} gdy nic nie robi).

    Raises:
        ValueError: nieznane surface_fit.
    """
    if params.surface_fit not in SURFACE_FITS:
        raise ValueError(f"contacts.surface_fit = {params.surface_fit!r}, dostępne: {', '.join(SURFACE_FITS)}")
    labels = np.asarray(labels, dtype=np.uint8)
    fitting = params.surface_fit != "none"
    rules = params.head_min_ratio > 0 or params.crest_max_axial > 0
    if not labels.any() or (not fitting and not params.split_ends and not rules):
        return labels, {}
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    v, n, va = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals), vertex_areas(mesh)
    ax = bone_axis(mesh)
    long = ax["elong"] >= params.long_min_elongation
    t = (v - ax["c"]) @ ax["d"]
    mid = 0.5 * (ax["lo"] + ax["hi"])
    groups = []
    for k in [int(k) for k in np.unique(labels) if k > 0]:
        idx = np.flatnonzero(labels == k)
        parts = [idx]
        if params.split_ends and long:
            low, high = idx[t[idx] < mid], idx[t[idx] >= mid]
            if min(va[low].sum(), va[high].sum()) >= params.min_surface_au2:
                parts = [low, high]
        for g in parts:
            if fitting:
                f = fit_surface(v[g], n[g], va[g], ax["d"] if long else None, params.surface_fit)
                res = surface_residual(f, v[g])
                keep = np.abs(res) <= params.fit_tolerance_au
                if long:          # kanał szpikowy: wklęsły dół W ŚRODKU przekroju, za płaszczyzną - zostaje
                    rel = v[g] - ax["c"]
                    radial = np.linalg.norm(rel - np.outer(rel @ ax["d"], ax["d"]), axis=1)
                    keep |= (res < 0) & (radial < MARROW_RADIUS * ax["r"])
                g = g[keep]
            groups.append(g)
    found = []                                     # (pole, wierzchołki) spójnych obszarów po odcięciu
    for g in groups:
        mask = np.zeros(len(v), bool)
        mask[g] = True
        sub = split_regions(mesh, mask, params.min_surface_au2)
        found += [(float(va[sub == j].sum()), np.flatnonzero(sub == j)) for j in range(1, int(sub.max()) + 1)]
    found.sort(key=lambda x: -x[0])
    dropped = []
    if long and rules:
        radial = np.linalg.norm((v - ax["c"]) - np.outer(t, ax["d"]), axis=1)
        length = ax["hi"] - ax["lo"]
        shaft = np.median(radial[(t > ax["lo"] + 0.3 * length) & (t < ax["hi"] - 0.3 * length)])
        end_ratio = {-1: np.median(radial[t < ax["lo"] + 0.12 * length]) / shaft,
                     1: np.median(radial[t > ax["hi"] - 0.12 * length]) / shaft}
        keep = []
        for area, g in found:
            pos = float((t[g].mean() - mid) / max(length / 2, 1e-9))
            side = 1 if pos > 0 else -1
            why = ""
            if params.head_min_ratio > 0 and end_ratio[side] >= params.head_min_ratio and abs(pos) >= params.head_zone:
                why = f"głowa kości (koniec {end_ratio[side]:.2f} x szerszy niż trzon)"
            elif params.crest_max_axial > 0:
                axial = float(np.abs(n[g] @ ax["d"]).mean())
                rms = fit_surface(v[g], n[g], va[g], ax["d"], "plane")["rms_au"]
                if axial < params.crest_max_axial and rms < params.crest_max_rms_au:
                    why = f"ostra krawędź kory (|n . oś| {axial:.2f}, odchylenie {rms:.2f} a.u.)"
            if why:
                dropped.append({"area_au2": round(area, 1), "pos": round(pos, 2), "why": why})
            else:
                keep.append((area, g))
        found = keep
    out = np.zeros(len(v), dtype=np.uint8)
    fits = {}
    for k, (_, g) in enumerate(found[:255], start=1):
        out[g] = k
        if fitting:
            fits[str(k)] = fit_info(fit_surface(v[g], n[g], va[g], ax["d"] if long else None, params.surface_fit))
    return out, {"fits": fits, "dropped": dropped}


def _atlas_data_key(bone, params: ContactParams) -> str:
    """Od czego poza sekcją "contacts" zależy wynik "atlas": modele w bazie, sekcje "atlas" i "surface_features"."""
    import atlas

    return atlas.data_key()


@detector("atlas", data_key=_atlas_data_key)
def detect_atlas(bone, params: ContactParams):
    """ATLAS ANATOMICZNY: kontekst z bazy modeli kompletnych kości + analiza lokalna jak w "features".

    Szuka w bazie (settings.json "paths" -> "atlas") kości, z której pochodzi fragment (z inną
    skalą, odbiciem lustrzanym, tylko częściowo), liczy rozkład prawdopodobieństwa modeli
    (z "żadna z bazy") i na najbardziej prawdopodobnych modelach rozstrzyga, co jest korą
    (leży na powierzchni modelu), a co przełomem (wnętrze kości). Łączy to z wynikiem "features",
    więc naturalnie szorstka kora (nasady, przyczepy mięśni) nie jest brana za przełom.
    Opis: src/atlas.py, ustawienia: settings.json -> "atlas".

    Args:
        bone:   kość (fragment).
        params: ustawienia analizy lokalnej (feature_*) i obszarów (min_surface_au2).

    Returns:
        ((V,) uint8 etykiety, {"atlas": ranking modeli z prawdopodobieństwami}).

    Raises:
        ValueError: pusta baza modeli albo zła nazwa cechy w feature_keys.
    """
    import atlas

    return atlas.detect(bone, params)


def boundary_vertices(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Maska wierzchołków na otwartym brzegu siatki (krawędź tylko jednego trójkąta).

    Skan ma brzeg tam, gdzie skaner przestał widzieć powierzchnię - to NIE jest przełom,
    a brzeg bywa postrzępiony, więc wygląda na szorstki.

    Args:
        mesh: siatka kości.

    Returns:
        (V,) bool; same False dla siatki zamkniętej.
    """
    t = np.asarray(mesh.triangles)
    e = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    uniq, count = np.unique(e, axis=0, return_counts=True)
    mask = np.zeros(len(mesh.vertices), dtype=bool)
    mask[uniq[count == 1].ravel()] = True
    return mask


def detection_params(params: ContactParams) -> dict:
    """Ustawienia wpływające na wynik algorytmu (bez ustawień narzędzia i nazwy algorytmu).

    Zapisywane w pliku wyniku ("detector" -> "params"); auto_contacts porównuje je
    z bieżącymi, żeby wiedzieć, czy zapisany wynik jest aktualny.

    Args:
        params: wszystkie ustawienia sekcji "contacts".

    Returns:
        {nazwa: wartość} bez method, brush_au, wand_angle_deg i exclude (krotki jako listy - jak po odczycie JSON).
    """
    return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(params).items()
            if k not in ("method", "brush_au", "wand_angle_deg", "exclude")}


def _fold(text: str) -> str:
    """Tekst do porównywania nazw: bez polskich znaków i wielkości liter, spacje = "_" (jak id kości)."""
    from data_parsing import slugify
    return slugify(text).lower()


def is_excluded(bone, params: ContactParams) -> bool:
    """Czy kość jest na liście "exclude" sekcji "contacts" (algorytm jej nie liczy).

    Pasuje, gdy id albo nazwa kości zawiera któryś z tekstów - bez wielkości liter
    i polskich znaków, spacja = "_" (więc "cała kość" pasuje do id "Test_cala_kosc_strzalkowa").

    Args:
        bone:   kość (id, label).
        params: ustawienia sekcji "contacts".

    Returns:
        True = pominąć.
    """
    names = (_fold(bone.id), _fold(bone.label))
    return any(p and any(p in n for n in names) for p in map(_fold, params.exclude))


def detector_info(bone, method: str, params: ContactParams) -> dict:
    """Opis algorytmu zapisywany z wynikiem ("detector"): nazwa, ustawienia i - jeśli algorytm
    ma data_key - dane spoza sekcji "contacts", od których zależy.

    Args:
        bone:   kość.
        method: nazwa algorytmu.
        params: ustawienia.

    Returns:
        {"method", "params"[, "data"]}.
    """
    info = {"method": method, "params": detection_params(params)}
    if method in DATA_KEYS:
        info["data"] = DATA_KEYS[method](bone, params)
    return info


def detect_contacts(bone, method: str | None = None, params: ContactParams = ContactParams()) -> Contacts:
    """Uruchamia algorytm `method` na kości. Nic nie zapisuje (zapis: save_contacts).

    Args:
        bone:   kość.
        method: nazwa z DETECTORS; None = params.method.
        params: ustawienia algorytmu.

    Po algorytmie: refine_surfaces (rozdzielenie końców, przybliżenie przełomu, odcięcie kory).

    Returns:
        Contacts z source = "auto:<method>", nazwami "Wykryta k" i opisem algorytmu
        w `detector` (żeby auto_contacts mógł sprawdzić aktualność). Algorytm może zwrócić
        (labels, info) - wtedy `info` (słownik JSON, np. ranking kości z atlasu) trafia do Contacts.info;
        przybliżenia powierzchni: info["fits"].

    Raises:
        ValueError: nieznany algorytm.
    """
    method = method or params.method
    if method not in DETECTORS:
        raise ValueError(f"Nieznany algorytm {method!r}, dostępne: {', '.join(DETECTORS)}")
    result = DETECTORS[method](bone, params)
    labels, info = result if isinstance(result, tuple) else (result, {})
    labels, refined = refine_surfaces(bone.mesh, np.asarray(labels, dtype=np.uint8), params)
    info = dict(info or {})
    info.update({k: val for k, val in refined.items() if val})
    names = {int(k): f"Wykryta {k}" for k in np.unique(labels) if k > 0}
    return Contacts(bone_id=bone.id, labels=labels, names=names, source=f"auto:{method}",
                    detector=detector_info(bone, method, params), info=dict(info or {}))


def auto_contacts(bone, params: ContactParams = ContactParams(),
                  root: Path | None = None) -> tuple[Contacts, bool]:
    """Wynik algorytmu params.method dla kości: z cache (contacts/<zestaw>/auto/) albo policzony i zapisany.

    Liczy na nowo, gdy pliku nie ma albo powstał innym algorytmem, z innymi ustawieniami,
    na innej siatce kości lub (algorytmy z data_key, np. "features") z innymi danymi spoza sekcji.

    Args:
        bone:   kość.
        params: algorytm (params.method) i jego ustawienia.
        root:   folder oznaczeń zamiast CONTACTS_DIR (wynik w <root>/auto/).

    Kość z listy "exclude" (is_excluded): puste etykiety, nic nie liczy ani nie zapisuje.

    Returns:
        (etykiety - puste, gdy algorytm nic nie znalazł; True = właśnie policzone, False = z cache).
    """
    if is_excluded(bone, params):
        return Contacts(bone.id, np.zeros(len(bone.mesh.vertices), np.uint8), source=f"auto:{params.method}"), False
    want = detector_info(bone, params.method, params)
    cached = load_contacts(bone.id, bone.mesh, "auto", root=root, frames=scan_frames(bone))
    if cached is not None and not cached.remapped and cached.detector == want:
        return cached, False
    found = detect_contacts(bone, params.method, params)
    save_contacts(bone.mesh, found, contacts_path(bone.id, "auto", root), scan_frames(bone))
    return found, True


# -----------------
# OCENA ALGORYTMU WZGLĘDEM RĘCZNYCH OZNACZEŃ
# -----------------

def compare(mesh: o3d.geometry.TriangleMesh, predicted: np.ndarray, truth: np.ndarray) -> dict:
    """Porównuje wynik algorytmu z ręcznymi oznaczeniami (ważone polem wierzchołków).

    IoU = część wspólna / suma, precyzja = jaka część wykrytego to naprawdę styk,
    czułość = jaka część prawdziwego styku została wykryta.

    Args:
        mesh:      siatka kości.
        predicted: (V,) etykiety z algorytmu.
        truth:     (V,) etykiety ręczne.

    Returns:
        {"iou", "precision", "recall"} dla podziału styk / nie-styk oraz "surfaces":
        dla każdej ręcznej powierzchni najlepsze IoU z którąkolwiek wykrytą.
    """
    va = vertex_areas(mesh)
    p, t = predicted > 0, truth > 0
    inter, union = va[p & t].sum(), va[p | t].sum()
    out = {"iou": inter / union if union else 1.0,
           "precision": inter / va[p].sum() if p.any() else 0.0,
           "recall": inter / va[t].sum() if t.any() else 1.0, "surfaces": []}
    for k in [int(x) for x in np.unique(truth) if x > 0]:
        tk = truth == k
        best = 0.0
        for j in [int(x) for x in np.unique(predicted[tk]) if x > 0]:
            pj = predicted == j
            best = max(best, va[pj & tk].sum() / va[pj | tk].sum())
        out["surfaces"].append({"id": k, "best_iou": best, "area_au2": float(va[tk].sum())})
    return out

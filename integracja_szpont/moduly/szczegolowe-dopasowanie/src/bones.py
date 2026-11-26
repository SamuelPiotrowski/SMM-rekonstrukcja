"""
Kość = wynik przygotowania danych: połączone skany jednego fragmentu, gotowe do
oznaczania powierzchni styku i dopasowywania.

Każda kość jest zapisana w DATA_OUT/bones/<id>/:
    mesh.ply        - siatka kości (wyśrodkowana w (0, 0, 0)), z kolorami skanera
    source.npy      - (T,) z którego skanu pochodzi każdy trójkąt (indeks w meta["scans"])
    cloud.npz       - równomierna chmura punktów: points, normals, colors (odstęp ~voxel_size)
    fpfh.npy        - deskryptory FPFH chmury (N, 33) - do dopasowania fragmentów
    meta.json       - skąd pochodzi: skany, ich przekształcenia, usunięty szum, statystyki
    scans/*.ply     - każdy skan ustawiony w układzie kości (do wizualizacji łączenia);
                      *_noise.ply - kawałki usunięte jako szum

Wczytanie w innym module:
    from bones import load_bones
    for bone in load_bones():
        bone.mesh, bone.cloud, bone.fpfh, bone.meta
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from data_parsing import compute_fpfh, mesh_stats, sample_surface, to_cloud
from config import read_json


@dataclass(frozen=True)
class FeatureParams:
    """Ustawienia chmury punktów i deskryptorów FPFH kości (settings.json, sekcja "features")."""

    center: bool = True               # przesuń kość tak, żeby jej środek był w (0, 0, 0)
    voxel_size: float = 0.5           # odstęp punktów chmury [a.u.]
    sampling_density: float = 4.0     # punkty losowane na voxel^2 przed przerzedzeniem
    fpfh_radius_factor: float = 5.0   # promień FPFH = ta liczba x voxel_size


@dataclass
class Bone:
    """Jedna kość (fragment) gotowa do dalszej analizy.

    Attributes:
        id:     nazwa ASCII = nazwa folderu w DATA_OUT/bones/.
        label:  czytelna nazwa do wyświetlania.
        mesh:   siatka (open3d) z kolorami i normalnymi.
        source: (T,) indeks skanu źródłowego każdego trójkąta (meta["scans"][i]).
        cloud:  chmura punktów z normalnymi (open3d) albo None, jeśli nie wczytano.
        fpfh:   (N, 33) deskryptory FPFH chmury (ta sama kolejność co punkty) albo None.
        meta:   słownik z meta.json.
        folder: DATA_OUT/bones/<id>/ albo None, jeśli kości jeszcze nie zapisano.

    Wszystkie współrzędne są w układzie kości (jak w mesh.ply), w a.u.
    """

    id: str
    label: str
    mesh: o3d.geometry.TriangleMesh
    source: np.ndarray
    cloud: o3d.geometry.PointCloud | None = None
    fpfh: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
    folder: Path | None = None

    @property
    def n_scans(self) -> int:
        """Z ilu skanów powstała kość (co najmniej 1)."""
        return len(self.meta.get("scans", [])) or 1


# -----------------
# ZAPIS / ODCZYT SIATEK (trimesh - działa też z polskimi znakami w ścieżce)
# -----------------

def write_mesh(path: Path, mesh: o3d.geometry.TriangleMesh) -> None:
    """Zapisuje siatkę open3d do pliku (format wg rozszerzenia, np. .ply) przez trimesh.

    Zapisywane są wierzchołki, trójkąty i kolory wierzchołków (jako 0-255). Normalne
    liczy na nowo read_mesh. Brakujące foldery są tworzone, istniejący plik jest nadpisywany.

    Args:
        path: plik docelowy; rozszerzenie wybiera format (.ply, .stl, .obj, ...).
        mesh: siatka do zapisania (nie jest zmieniana).
    """
    colors = None
    if mesh.has_vertex_colors():
        colors = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).round().astype(np.uint8)
    tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), vertex_colors=colors,
                         process=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tm.export(path)


def read_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    """Wczytuje siatkę zapisaną przez write_mesh i liczy normalne wierzchołków i trójkątów.

    Args:
        path: plik siatki (zwykle DATA_OUT/bones/<id>/mesh.ply).

    Returns:
        Siatka open3d z kolorami (jeśli były w pliku) i normalnymi.
    """
    from data_parsing import load_mesh

    mesh = load_mesh(path)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


# -----------------
# BUDOWA KOŚCI
# -----------------

def finalize_bone(merged, params: FeatureParams = FeatureParams()) -> Bone:
    """Zamienia wynik łączenia skanów (merging.MergedBone) w gotową kość.

    1. Przesuwa kość tak, żeby jej środek (średnia wierzchołków) był w (0, 0, 0) -
       jeśli params.center. Tak samo przesuwa siatki skanów i szumu.
    2. Losuje równomierną chmurę punktów (odstęp params.voxel_size) i liczy FPFH
       (promień params.fpfh_radius_factor x voxel_size).
    3. Uzupełnia meta: przesunięcie, przekształcenia skanów do układu kości,
       statystyki siatki, liczbę punktów chmury i użyte ustawienia.

    Uwaga: siatki i meta w `merged` są zmieniane w miejscu (merged.mesh to bone.mesh).

    Args:
        merged: wynik merging.merge_group.
        params: ustawienia chmury i FPFH.

    Returns:
        Kość z chmurą i FPFH, jeszcze niezapisana (zapis: save_bone).
    """
    mesh = merged.mesh
    center = mesh.get_center() if params.center else np.zeros(3)
    shift = np.eye(4)
    shift[:3, 3] = -center
    mesh.transform(shift)
    for m in (*merged.scan_meshes.values(), *merged.noise_meshes.values()):
        if m is not mesh:           # kość z jednego skanu: siatka kości = siatka skanu, bez podwójnego przesunięcia
            m.transform(shift)

    s = sample_surface(mesh, params.voxel_size, params.sampling_density)
    cloud = to_cloud(s.points, s.normals, s.colors)
    fpfh = np.asarray(compute_fpfh(cloud, params.voxel_size * params.fpfh_radius_factor).data).T

    meta = dict(merged.meta)
    meta["translation_applied"] = (-center).round(6).tolist()
    for sc in meta["scans"]:
        sc["transform_to_bone"] = np.round(shift @ np.asarray(sc["transform_to_bone"]), 6).tolist()
    meta["stats"] = mesh_stats(mesh)
    meta["cloud_points"] = len(s.points)
    meta["feature_params"] = asdict(params)
    return Bone(id=merged.id, label=merged.label, mesh=mesh, source=merged.source, cloud=cloud,
                fpfh=fpfh, meta=meta)


def save_bone(bone: Bone, merged, out_dir: Path) -> Path:
    """Zapisuje kość do out_dir/bones/<id>/ (pliki opisane w docstringu modułu).

    Istniejące pliki są nadpisywane, a folder scans/ najpierw czyszczony, żeby nie
    zostały w nim skany z poprzedniego uruchomienia. Ustawia bone.folder.

    Args:
        bone:    kość z finalize_bone (z chmurą i FPFH).
        merged:  ten sam MergedBone - stąd siatki pojedynczych skanów i szumu.
        out_dir: folder wyników (DATA_OUT).

    Returns:
        Folder kości.
    """
    target = out_dir / "bones" / bone.id
    if (target / "scans").exists():
        shutil.rmtree(target / "scans")
    target.mkdir(parents=True, exist_ok=True)
    write_mesh(target / "mesh.ply", bone.mesh)
    np.save(target / "source.npy", bone.source.astype(np.int32))
    np.savez_compressed(target / "cloud.npz", points=np.asarray(bone.cloud.points, dtype=np.float32),
                        normals=np.asarray(bone.cloud.normals, dtype=np.float32),
                        colors=np.asarray(bone.cloud.colors, dtype=np.float32))
    np.save(target / "fpfh.npy", bone.fpfh.astype(np.float32))
    for scan_id, m in merged.scan_meshes.items():
        write_mesh(target / "scans" / f"{scan_id}.ply", m)
    for scan_id, m in merged.noise_meshes.items():
        write_mesh(target / "scans" / f"{scan_id}_noise.ply", m)
    (target / "meta.json").write_text(json.dumps(bone.meta, indent=2, ensure_ascii=False), encoding="utf-8")
    bone.folder = target
    return target


def remove_stale(out_dir: Path, keep_ids: set[str]) -> list[str]:
    """Usuwa foldery kości z poprzednich uruchomień, których już nie ma.

    Np. po zmianie grupowania skanów kość może dostać inne id - stary folder by został.
    Usuwane są tylko foldery z meta.json (czyli utworzone przez save_bone); inne pliki
    w out_dir/bones/ nie są ruszane.

    Args:
        out_dir:  folder wyników (DATA_OUT).
        keep_ids: id kości z bieżącego uruchomienia.

    Returns:
        Id usuniętych kości.
    """
    removed = []
    for meta in (out_dir / "bones").glob("*/meta.json"):
        if meta.parent.name not in keep_ids:
            shutil.rmtree(meta.parent)
            removed.append(meta.parent.name)
    return removed


def load_bones(out_dir: Path | None = None, features: bool = False) -> list[Bone]:
    """Wczytuje kości zapisane przez scripts/prepare_data.py.

    Args:
        out_dir:  folder wyników (domyślnie DATA_OUT z settings.json).
        features: wczytaj też chmurę punktów i FPFH (wolniej; potrzebne do dopasowania
                  i kolorów "fpfh"). False = bone.cloud i bone.fpfh są None.

    Returns:
        Lista kości posortowana po id - ta kolejność wyznacza numery kości (1, 2, ...)
        we wszystkich skryptach. Pusta, jeśli nic jeszcze nie przygotowano.
    """
    if out_dir is None:
        from config import DATA_OUT
        out_dir = DATA_OUT
    return [load_bone(p.parent, features) for p in sorted((Path(out_dir) / "bones").glob("*/meta.json"))]


def load_bone(folder: Path, features: bool = False) -> Bone:
    """Wczytuje jedną kość z jej folderu DATA_OUT/bones/<id>/ (jak load_bones, ale jedną).

    Args:
        folder:   folder kości (z meta.json i mesh.ply).
        features: wczytaj też chmurę punktów i FPFH.

    Returns:
        Kość; id = nazwa folderu.
    """
    d = Path(folder)
    meta = read_json(d / "meta.json")
    mesh = read_mesh(d / "mesh.ply")
    source = np.load(d / "source.npy") if (d / "source.npy").exists() else np.zeros(len(mesh.triangles), int)
    bone = Bone(id=d.name, label=meta.get("label", d.name), mesh=mesh, source=source, meta=meta, folder=d)
    if features:
        c = np.load(d / "cloud.npz")
        bone.cloud = to_cloud(c["points"], c["normals"], c["colors"])
        bone.fpfh = np.load(d / "fpfh.npy")
    return bone


def select_bones(bones: list[Bone], patterns: list[str] | None) -> list[Bone]:
    """Kości, których id lub nazwa zawiera którykolwiek z tekstów (bez wielkości liter).

    Numer (np. "3") wybiera kość o tym numerze na liście (od 1); numer spoza listy
    jest traktowany jak zwykły tekst. Tak działają argumenty skryptów, np.
    `python scripts/show_bones.py 3 strzalkowa`.

    Args:
        bones:    wszystkie kości (z load_bones - numery wg tej listy).
        patterns: numery albo fragmenty nazw; pusta lista / None = wszystkie kości.

    Returns:
        Wybrane kości w kolejności wzorców, bez powtórzeń.

    Raises:
        ValueError: jeśli któryś wzorzec nie pasuje do żadnej kości.
    """
    if not patterns:
        return list(bones)
    chosen = []
    for pat in patterns:
        if pat.isdigit() and 1 <= int(pat) <= len(bones):
            hit = [bones[int(pat) - 1]]
        else:
            hit = [b for b in bones if pat.lower() in b.id.lower() or pat.lower() in b.label.lower()]
        if not hit:
            listing = "\n".join(f"  {i}. {b.label}  ({b.id})" for i, b in enumerate(bones, 1))
            raise ValueError(f"Żadna kość nie pasuje do \"{pat}\". Dostępne:\n{listing}")
        chosen += [b for b in hit if b not in chosen]
    return chosen


def write_summary(bones: list[Bone], out_dir: Path) -> Path:
    """Zapisuje tabelę kości out_dir/summary.csv - do otwarcia w Excelu.

    Jeden wiersz na kość: numer, id, nazwa, skany, skan referencyjny, sposób sklejenia,
    liczba trójkątów, pole, wymiary, usunięty szum, liczba punktów chmury. Separator ";"
    i UTF-8 z BOM - tak polski Excel od razu rozdziela kolumny i pokazuje polskie znaki.

    Args:
        bones:   kości z kompletnym meta (z finalize_bone albo load_bones).
        out_dir: folder wyników.

    Returns:
        Ścieżka do summary.csv.
    """
    path = out_dir / "summary.csv"
    cols = ["nr", "id", "nazwa", "skany", "skan_referencyjny", "polaczenie", "trojkaty", "pole_au2",
            "wymiar_x_au", "wymiar_y_au", "wymiar_z_au", "usuniety_szum_au2", "punkty_chmury"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(cols)
        for i, b in enumerate(bones, 1):
            m, st = b.meta, b.meta["stats"]
            w.writerow([i, b.id, b.label, ", ".join(s["id"] for s in m["scans"]), m["reference_scan"],
                        m["fuse_method"], st["triangles"], st["area_au2"], *st["extent_au"],
                        m["noise_removed_au2"], m["cloud_points"]])
    return path

"""
Syntetyczne kości 3D pocięte na fragmenty - osobny zestaw danych z PEWNĄ prawdą:
które fragmenty do siebie pasują, gdzie dokładnie są przełomy i jak fragmenty leżą
względem siebie. Do testowania wykrywania przełomów i dopasowania bez ręcznych oznaczeń.

    from synthetic_bones import SyntheticParams, generate, write_dataset
    synth = generate(SyntheticParams(n_bones=6, seed=1))
    write_dataset(synth, config.SCANS_ROOT / "synthetic", config.CONTACTS_ROOT / "synthetic")
(najprościej: python scripts/make_synthetic.py)

Jak powstaje kość (wszystko w a.u., oś kości = Z, kość od z = 0 do z = długość):
    - kora: "rurka" o promieniu r(kąt, z) - trzon + poszerzone nasady na końcach, lekko
      eliptyczna, wygięta oś, drobne nierówności (szum Fouriera = suma kilku sinusów),
    - końce kości: zaokrąglone kopułki (to też kora, nie przełom),
    - przełom: powierzchnia z = f(x, y) - pochyła płaszczyzna (złamanie skośne) + duże
      nierówności (relief) + drobna szorstkość. Oba sąsiednie fragmenty dostają DOKŁADNIE
      tę samą powierzchnię przełomu (z przeciwnych stron), ale każdy z inną siatką
      (inny obrót siatki), tak jak dwa niezależne skany.
    - płaski odłamek (cięcie "chip"): dwa przełomy w odległości chip_au - górny to dolny
      przesunięty w górę z własnymi, małymi nierównościami - między nimi cienka "płytka".
Fragment = część kości między dwiema powierzchniami (przełom albo koniec kości) - zamknięta,
szczelna siatka: kora z boku + dwie "zatyczki" na dole i na górze.
Cięcia są losowe (n_bones, fragments) albo podane ręcznie (layout, gotowe układy: PRESETS).

Jak "skan" fragmentu (write_dataset):
    - szum skanera (przesunięcie wierzchołków wzdłuż normalnej), opcjonalnie dziury
      (miejsca, których skaner nie widział), kolory (kora jasna, przełom ciemniejszy),
    - losowe położenie i obrót - każdy fragment w swoim układzie, jak prawdziwy skan,
    - numeracja frag_01, frag_02, ... wymieszana między kośćmi (nazwa nie zdradza kości).

Co zapisuje write_dataset:
    <skany zestawu>/frag_XX/frag_XX.ply   - skany fragmentów (jeden folder = jedna kość w pipeline),
    <skany zestawu>/ground_truth.json     - prawda (format niżej),
    <skany zestawu>/settings.json         - ustawienia zestawu (łączenie skanów wyłączone:
                                            każdy fragment to osobna kość),
    <oznaczenia zestawu>/frag_XX.json     - prawdziwe przełomy jako "ręczne" oznaczenia styku
                                            (contacts.py) - od razu widać je w show_bones.py,
                                            detect_contacts.py porównuje z nimi algorytm,
                                            a match_surfaces.py (source "manual") ich używa.

ground_truth.json:
    {
      "format": "bones-smm-synthetic/1", "params": {...},
      "bones": [{"id": "kosc_A", "length_au", "radius_au", "cuts": ["fracture", "chip", ...],
                 "fragments": [id od dołu do góry]}],
      "fragments": {
        "frag_03": {"bone": "kosc_A", "index": 0 (od dołu), "file": "frag_03/frag_03.ply",
                    "pose_to_bone": T 4x4 (układ skanu -> układ złożonej kości),
                    "surfaces": {"1": {"partner": "frag_07", "partner_surface": 1,
                                       "fracture": "kosc_A/1", "area_au2": ...}}}},
      "pairs": [{"a", "b", "surface_a", "surface_b", "fracture", "area_au2",
                 "transform_b_to_a": T 4x4 (skan b -> skan a)}]
    }
Numery powierzchni ("surfaces") = numery w oznaczeniach styku (contacts/<zestaw>/frag_XX.json).
Prawdziwe ustawienie par po przygotowaniu kości (inne środki): true_transform().
"""

from __future__ import annotations

import json
import string
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from config import read_json

FORMAT = "bones-smm-synthetic/1"
GROUND_TRUTH = "ground_truth.json"
DATASET_SETTINGS = "settings.json"
MIN_FRAGMENT_AU = 6.0       # najkrótszy kawałek kory między dwoma przełomami (wzdłuż osi)
PREVIEW_EDGE_AU = 0.8       # krawędź siatek podglądu (preview_figure) - małe pliki HTML


@dataclass(frozen=True)
class SyntheticParams:
    """Ustawienia generatora (settings.json, sekcja "synthetic"). Długości w a.u.

    Pary (od, do) = wartość losowana z tego przedziału osobno dla każdej kości.
    """

    n_bones: int = 6                    # ile kości w zestawie
    fragments: tuple = (2, 4)           # ile fragmentów na kość (od, do)
    length_au: tuple = (50.0, 90.0)     # długość kości (od, do)
    radius_au: tuple = (4.0, 7.0)       # promień trzonu (od, do)
    flare: float = 0.35                 # poszerzenie nasad na końcach (0.35 = o 35% szersze niż trzon)
    ellipticity: float = 0.15           # spłaszczenie przekroju (0 = okrągły)
    bend_au: float = 2.0                # wygięcie osi kości (w środku długości)
    bump_au: float = 0.25               # nierówności kory (RMS)
    tilt_deg: float = 35.0              # największe pochylenie przełomu względem przekroju poprzecznego
    relief_au: float = 1.2              # duże nierówności przełomu (RMS, fale 3-10 a.u.)
    rough_au: float = 0.12              # drobna szorstkość przełomu (RMS, fale 0.8-2 a.u.)
    edge_au: float = 0.3                # długość krawędzi siatki (gęstość "skanu")
    scan_noise_au: float = 0.02         # szum skanera wzdłuż normalnej (odchylenie standardowe)
    holes: int = 0                      # dziury w korze na fragment (miejsca niewidoczne dla skanera)
    hole_au: tuple = (1.5, 3.0)         # promień dziury (od, do)
    colors: bool = True                 # kolory wierzchołków (kora jasna, przełom ciemniejszy)
    chip_au: tuple = (2.0, 4.0)         # grubość płaskiego odłamka (cięcie "chip") (od, do)
    # Ręczny układ cięć: jedna lista na kość, cięcia od dołu do góry: "fracture" = zwykły przełom,
    # "chip" = płaski odłamek (dwa bliskie, prawie równoległe przełomy). () = losowo wg n_bones
    # i fragments; podany układ wyznacza liczbę kości i fragmentów.
    layout: tuple = ()
    seed: int = 0                       # ziarno losowania - to samo = ten sam zestaw


CUT_KINDS = ("fracture", "chip")
# Gotowe zestawy (make_synthetic.py --preset <nazwa>): zmiany względem ustawień z settings.json.
PRESETS = {
    # 1 kość, 4 części: koniec | zwykły przełom | środek | płaski odłamek między dwoma przełomami | koniec
    "chip": {"n_bones": 1, "layout": (("fracture", "chip"),), "length_au": (80.0, 80.0)},
}


# -----------------
# KSZTAŁT
# -----------------

class FourierNoise:
    """Gładki losowy szum: suma `n` sinusów o losowych kierunkach i długościach fali.

    Wartość średnia ~0, odchylenie standardowe (RMS) = `amp`.
    """

    def __init__(self, rng: np.random.Generator, dim: int, amp: float, wavelengths: tuple, n: int = 12):
        """Losuje sinusy.

        Args:
            rng:         generator liczb losowych.
            dim:         wymiar punktów (2 = (x, y), 3 = (x, y, z)).
            amp:         RMS szumu [a.u.].
            wavelengths: (najkrótsza, najdłuższa) długość fali [a.u.] - losowane logarytmicznie.
            n:           liczba sinusów (więcej = mniej regularny wzór).
        """
        k = rng.normal(size=(n, dim))
        k /= np.linalg.norm(k, axis=1, keepdims=True)
        lam = np.exp(rng.uniform(np.log(wavelengths[0]), np.log(wavelengths[1]), n))
        self.k = k * (2 * np.pi / lam)[:, None]
        self.phase = rng.uniform(0, 2 * np.pi, n)
        self.a = amp * np.sqrt(2.0 / n)

    def __call__(self, p: np.ndarray) -> np.ndarray:
        """Szum w punktach `p` (..., dim) -> (...)."""
        return self.a * np.sin(p @ self.k.T + self.phase).sum(axis=-1)


@dataclass
class BoneShape:
    """Kształt kory jednej kości: oś Z od 0 do `length`, promień r(kąt, z)."""

    length: float
    radius: float
    flare: float
    ellipticity: float
    ellipse_phase: float
    bend: np.ndarray                     # (2,) przesunięcie osi w połowie długości
    bumps: FourierNoise                  # nierówności kory (3D)

    @property
    def max_radius(self) -> float:
        """Górne oszacowanie promienia (nasada, dłuższa oś elipsy, nierówności)."""
        return self.radius * (1 + self.flare) * (1 + self.ellipticity) + 3 * self.bumps.a * len(self.bumps.phase) ** 0.5

    def axis(self, z: np.ndarray) -> np.ndarray:
        """Punkt osi kości (x, y) na wysokości z: (...,) -> (..., 2)."""
        return self.bend * np.sin(np.pi * np.asarray(z) / self.length)[..., None]

    def radius_at(self, theta: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Promień kory w kierunku `theta` na wysokości `z` (tablice tego samego kształtu)."""
        w = 0.12 * self.length
        profile = 1 + self.flare * (np.exp(-(z / w) ** 2) + np.exp(-((z - self.length) / w) ** 2))
        r = self.radius * profile * (1 + self.ellipticity * np.cos(2 * (theta - self.ellipse_phase)))
        p = np.stack([self.radius * np.cos(theta), self.radius * np.sin(theta), z], axis=-1)
        return r + self.bumps(p)

    def lateral_xy(self, theta: np.ndarray, z: np.ndarray) -> np.ndarray:
        """(x, y) punktu kory w kierunku `theta` na wysokości `z` -> (..., 2)."""
        r = self.radius_at(theta, z)
        return self.axis(z) + r[..., None] * np.stack([np.cos(theta), np.sin(theta)], axis=-1)


@dataclass
class CutSurface:
    """Powierzchnia z = height(x, y) ograniczająca fragment z dołu albo z góry.

    kind = "fracture" (przełom między dwoma fragmentami) albo "end" (koniec kości - kopułka).
    """

    kind: str
    z_ref: float                          # wysokość w środku przekroju
    center: np.ndarray                    # (2,) środek przekroju (oś kości na z_ref)
    gradient: np.ndarray = field(default_factory=lambda: np.zeros(2))   # pochylenie (przełom)
    relief: FourierNoise | None = None
    rough: FourierNoise | None = None
    dome: float = 0.0                     # koniec: wysokość kopułki (+ w górę, - w dół)
    dome_radius: float = 1.0              # koniec: promień kopułki
    extent: float = 0.0                   # przełom: o ile najwyżej odchodzi od z_ref [a.u.]
    parent: CutSurface | None = None      # górna ściana odłamka: = parent + offset + własne nierówności
    offset: float = 0.0                   # górna ściana odłamka: grubość odłamka [a.u.]

    def height(self, xy: np.ndarray) -> np.ndarray:
        """Wysokość z powierzchni nad punktami (..., 2)."""
        d = xy - self.center
        if self.kind == "end":
            rho2 = np.clip((d ** 2).sum(axis=-1) / self.dome_radius ** 2, 0, 1)
            return self.z_ref + self.dome * (1 - rho2)
        if self.parent is not None:
            z = self.parent.height(xy) + self.offset
        else:
            z = self.z_ref + d @ self.gradient
        if self.relief is not None:
            z = z + self.relief(xy)
        if self.rough is not None:
            z = z + self.rough(xy)
        return z


def _uniform(rng: np.random.Generator, span) -> float:
    """Liczba losowa z przedziału (od, do); pojedyncza liczba = bez losowania."""
    if np.isscalar(span):
        return float(span)
    return float(rng.uniform(span[0], span[1]))


def random_shape(rng: np.random.Generator, params: SyntheticParams) -> BoneShape:
    """Losuje kształt kory jednej kości."""
    angle = rng.uniform(0, 2 * np.pi)
    return BoneShape(length=_uniform(rng, params.length_au), radius=_uniform(rng, params.radius_au),
                     flare=params.flare * rng.uniform(0.7, 1.3),
                     ellipticity=params.ellipticity * rng.uniform(0.5, 1.5),
                     ellipse_phase=rng.uniform(0, np.pi),
                     bend=params.bend_au * rng.uniform(0.3, 1.0) * np.array([np.cos(angle), np.sin(angle)]),
                     bumps=FourierNoise(rng, 3, params.bump_au, (4.0, 15.0)))


def random_fracture(rng: np.random.Generator, shape: BoneShape, z: float, params: SyntheticParams,
                    tilt_scale: float = 1.0) -> CutSurface:
    """Losuje powierzchnię przełomu na wysokości `z` (pochylenie do tilt_deg x tilt_scale)."""
    tilt = np.radians(rng.uniform(0, params.tilt_deg * tilt_scale))
    phi = rng.uniform(0, 2 * np.pi)
    relief_amp = params.relief_au * rng.uniform(0.6, 1.2)
    extent = np.tan(tilt) * shape.max_radius + 3 * (relief_amp + params.rough_au)
    return CutSurface(kind="fracture", z_ref=z, center=shape.axis(np.array(z)),
                      gradient=np.tan(tilt) * np.array([np.cos(phi), np.sin(phi)]),
                      relief=FourierNoise(rng, 2, relief_amp, (3.0, 10.0)),
                      rough=FourierNoise(rng, 2, params.rough_au, (0.8, 2.0), n=24), extent=extent)


def chip_face(rng: np.random.Generator, lower: CutSurface, thickness: float,
              params: SyntheticParams) -> CutSurface:
    """Górna ściana płaskiego odłamka: dolny przełom `lower` przesunięty o `thickness` w górę
    + własne, małe nierówności (odłamek ma grubość thickness +- ~15%, ściany się nie przecinają)."""
    relief_amp = 0.12 * thickness
    return CutSurface(kind="fracture", z_ref=lower.z_ref + thickness, center=lower.center, parent=lower,
                      offset=thickness, relief=FourierNoise(rng, 2, relief_amp, (8.0, 20.0)),
                      rough=FourierNoise(rng, 2, params.rough_au, (0.8, 2.0), n=24),
                      extent=lower.extent + 3 * (relief_amp + params.rough_au))


def cut_surfaces(rng: np.random.Generator, shape: BoneShape, kinds: list[str],
                 params: SyntheticParams, strict: bool = False) -> list[CutSurface]:
    """Powierzchnie dzielące kość: [koniec dolny, przełom 1, ..., koniec górny].

    Cięcia `kinds` (od dołu) są rozłożone mniej więcej równo wzdłuż kości, tak żeby się nie
    przecinały i między nimi zostało co najmniej MIN_FRAGMENT_AU kory. "fracture" = jeden
    przełom, "chip" = płaski odłamek: dwa przełomy w odległości chip_au (chip_face). Gdy kość
    jest za krótka, przełomy są mniej pochylone, a w ostateczności (strict=False) ostatnie
    cięcia są pomijane.

    Returns:
        Lista powierzchni od dołu; fragment k leży między [k] a [k + 1].

    Raises:
        ValueError: strict=True, a cięcia nie mieszczą się na kości.
    """
    L, rmax = shape.length, shape.max_radius
    ends = [CutSurface(kind="end", z_ref=0.0, center=shape.axis(np.array(0.0)), dome=-0.4 * shape.radius,
                       dome_radius=rmax),
            CutSurface(kind="end", z_ref=L, center=shape.axis(np.array(L)), dome=0.4 * shape.radius,
                       dome_radius=rmax)]
    kinds = list(kinds)
    while kinds:
        n = len(kinds) + 1
        for attempt in range(30):
            scale = 1.0 - attempt / 30
            step = L / n
            zs = step * np.arange(1, n) + rng.uniform(-0.15, 0.15, n - 1) * step
            surfaces, lo, hi = [], [0.0], []
            for z, kind in zip(zs, kinds):
                if kind == "chip":
                    t = _uniform(rng, params.chip_au)
                    f = random_fracture(rng, shape, float(z) - t / 2, params, scale)
                    top = chip_face(rng, f, t, params)
                    surfaces += [f, top]
                    hi.append(f.z_ref - f.extent)
                    lo.append(top.z_ref + top.extent)
                else:
                    f = random_fracture(rng, shape, float(z), params, scale)
                    surfaces.append(f)
                    hi.append(f.z_ref - f.extent)
                    lo.append(f.z_ref + f.extent)
            hi.append(L)
            if all(h - l >= MIN_FRAGMENT_AU for l, h in zip(lo, hi)):
                return [ends[0], *surfaces, ends[1]]
        if strict:
            raise ValueError(f"Cięcia {kinds} nie mieszczą się na kości długości {L:.0f} a.u. - "
                             "wydłuż kość (length_au) albo zmniejsz tilt_deg / relief_au")
        kinds = kinds[:-1]
    return ends                                         # kość w jednym kawałku


# -----------------
# SIATKA FRAGMENTU
# -----------------

def _boundary_z(shape: BoneShape, surface: CutSurface, theta: np.ndarray) -> np.ndarray:
    """Wysokość, na której powierzchnia `surface` przecina korę w kierunkach `theta`.

    Rozwiązuje z = surface.height(kora(theta, z)) iteracją (kora zmienia się wolno wzdłuż z).
    """
    z = np.full(len(theta), surface.z_ref, dtype=float)
    for _ in range(60):
        z_new = surface.height(shape.lateral_xy(theta, z))
        if np.abs(z_new - z).max() < 1e-7:
            return z_new
        z = z_new
    return z


def _ring_strip(a_idx, a_ang, b_idx, b_ang) -> list[tuple[int, int, int]]:
    """Trójkąty między dwoma zamkniętymi pierścieniami wierzchołków (a = wewnętrzny, b = zewnętrzny).

    Kąty obu pierścieni rosną i zaczynają się od tego samego kąta; pierścienie mogą mieć
    różną liczbę wierzchołków. Trójkąty są skierowane tak, że normalna = (na zewnątrz) x
    (rosnący kąt), czyli +Z dla zatyczki patrzącej w górę.
    """
    na, nb = len(a_idx), len(b_idx)
    a_next = np.append(a_ang[1:], a_ang[0] + 2 * np.pi)
    b_next = np.append(b_ang[1:], b_ang[0] + 2 * np.pi)
    tris, i, j = [], 0, 0
    while i < na or j < nb:
        if j < nb and (i >= na or b_next[j] <= a_next[i]):
            tris.append((a_idx[i % na], b_idx[j], b_idx[(j + 1) % nb]))
            j += 1
        else:
            tris.append((a_idx[i], b_idx[j % nb], a_idx[(i + 1) % na]))
            i += 1
    return tris


def build_fragment(shape: BoneShape, lower: CutSurface, upper: CutSurface, edge: float,
                   theta0: float = 0.0, lower_id: int = 0, upper_id: int = 0):
    """Zamknięta siatka fragmentu kości między powierzchniami `lower` i `upper` (układ kości).

    Kora z boku: siatka (kąt x wysokość), na dole i na górze "zatyczki" - pierścienie od środka
    przekroju do krawędzi kory, z liczbą wierzchołków rosnącą z promieniem. Zatyczka i kora
    mają wspólne wierzchołki na krawędzi, więc siatka jest szczelna.

    Args:
        shape:    kształt kory.
        lower, upper: powierzchnie ograniczające fragment z dołu i z góry.
        edge:     docelowa długość krawędzi [a.u.].
        theta0:   obrót siatki wokół osi - różny dla sąsiednich fragmentów, żeby ten sam przełom
                  miał na nich różne siatki (jak dwa niezależne skany).
        lower_id, upper_id: numer powierzchni styku dla wierzchołków zatyczki (0 = kora,
                  np. dla końca kości).

    Returns:
        (wierzchołki (V, 3), trójkąty (F, 3), etykiety (V,) uint8: 0 = kora, k = przełom k).
    """
    r_mid = float(shape.radius_at(np.linspace(0, 2 * np.pi, 64, endpoint=False),
                                  np.full(64, (lower.z_ref + upper.z_ref) / 2)).max())
    n_t = max(24, int(np.ceil(2 * np.pi * r_mid / edge)))
    theta = theta0 + 2 * np.pi * np.arange(n_t) / n_t
    z_lo, z_hi = _boundary_z(shape, lower, theta), _boundary_z(shape, upper, theta)
    n_z = max(2, int(np.ceil((z_hi - z_lo).max() / edge)))
    t = np.linspace(0, 1, n_z + 1)[:, None]
    Z = z_lo[None, :] + (z_hi - z_lo)[None, :] * t                         # (n_z + 1, n_t)
    TH = np.broadcast_to(theta, Z.shape)
    verts = [np.column_stack([shape.lateral_xy(TH, Z).reshape(-1, 2), Z.reshape(-1)])]
    labels = [np.zeros(Z.size, dtype=np.uint8)]
    labels[0][:n_t] = lower_id
    labels[0][-n_t:] = upper_id
    lat = np.arange(Z.size).reshape(Z.shape)
    a, b = lat[:-1], np.roll(lat[:-1], -1, axis=1)
    c, d = np.roll(lat[1:], -1, axis=1), lat[1:]
    faces = [np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])]
    count = Z.size

    for surface, boundary, sid, up in ((lower, lat[0], lower_id, False), (upper, lat[-1], upper_id, True)):
        bxy = shape.lateral_xy(theta, z_lo if not up else z_hi)
        center = bxy.mean(axis=0)
        n_s = max(2, int(np.ceil(np.linalg.norm(bxy - center, axis=1).max() / edge)))
        cap_v = [np.append(center, surface.height(center))[None, :]]
        rings = [(np.array([count]), None)]
        count += 1
        for m in range(1, n_s):
            n_m = max(6, int(round(n_t * m / n_s)))
            ang = theta0 + 2 * np.pi * np.arange(n_m) / n_m
            rxy = shape.lateral_xy(ang, _boundary_z(shape, surface, ang))
            xy = center + (m / n_s) * (rxy - center)
            cap_v.append(np.column_stack([xy, surface.height(xy)]))
            rings.append((count + np.arange(n_m), ang))
            count += n_m
        rings.append((boundary, theta))
        tris = [(rings[0][0][0], r1[k], r1[(k + 1) % len(r1)]) for r1 in [rings[1][0]] for k in range(len(r1))]
        for (ia, aa), (ib, ab) in zip(rings[1:-1], rings[2:]):
            tris += _ring_strip(ia, aa, ib, ab)
        tris = np.asarray(tris, dtype=np.int64)
        if not up:
            tris = tris[:, [0, 2, 1]]                  # zatyczka dolna patrzy w dół
        cap_v = np.vstack(cap_v)
        verts.append(cap_v)
        labels.append(np.full(len(cap_v), sid, dtype=np.uint8))
        faces.append(tris)
    return np.vstack(verts), np.vstack(faces).astype(np.int64), np.concatenate(labels)


# -----------------
# ZESTAW
# -----------------

@dataclass
class SyntheticFragment:
    """Jeden fragment: siatka "skanu" (układ skanu) + prawda o nim."""

    id: str                               # np. "frag_03"
    bone: str                             # np. "kosc_A"
    index: int                            # numer fragmentu w kości, od dołu (0, 1, ...)
    vertices: np.ndarray                  # (V, 3) w układzie skanu
    faces: np.ndarray                     # (F, 3)
    labels: np.ndarray                    # (V,) uint8: 0 = kora, k = powierzchnia styku k
    colors: np.ndarray | None             # (V, 3) w [0, 1] albo None
    pose_to_bone: np.ndarray              # 4x4: układ skanu -> układ złożonej kości
    surfaces: dict = field(default_factory=dict)   # {k: {"partner", "partner_surface", "fracture", "area_au2"}}
    preview: tuple = ()                   # (wierzchołki, trójkąty, etykiety) - rzadsza siatka w układzie kości


@dataclass
class SyntheticSet:
    """Wynik generate(): fragmenty i prawda (ground_truth.json)."""

    params: SyntheticParams
    fragments: list[SyntheticFragment]
    bones: list[dict]                      # [{"id", "length_au", "radius_au", "fragments": [...]}]

    def ground_truth(self) -> dict:
        """Zawartość ground_truth.json (format: docstring modułu)."""
        frags = {f.id: f for f in self.fragments}
        pairs = []
        for f in self.fragments:
            for k, s in sorted(f.surfaces.items()):
                if f.id < s["partner"]:
                    other = frags[s["partner"]]
                    T = np.linalg.inv(f.pose_to_bone) @ other.pose_to_bone
                    pairs.append({"a": f.id, "b": other.id, "surface_a": k, "surface_b": s["partner_surface"],
                                  "fracture": s["fracture"],
                                  "area_au2": round((s["area_au2"] + other.surfaces[s["partner_surface"]]["area_au2"]) / 2, 2),
                                  "transform_b_to_a": np.round(T, 6).tolist()})
        pairs.sort(key=lambda p: (p["a"], p["b"]))
        return {"format": FORMAT, "params": asdict(self.params), "bones": self.bones,
                "fragments": {f.id: {"bone": f.bone, "index": f.index, "file": f"{f.id}/{f.id}.ply",
                                     "pose_to_bone": np.round(f.pose_to_bone, 6).tolist(),
                                     "surfaces": {str(k): v for k, v in sorted(f.surfaces.items())}}
                              for f in sorted(self.fragments, key=lambda f: f.id)},
                "pairs": pairs}


def _random_pose(rng: np.random.Generator) -> np.ndarray:
    """Losowe przekształcenie sztywne 4x4 (dowolny obrót, przesunięcie do 40 a.u.)."""
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=rng).as_matrix()
    T[:3, 3] = rng.uniform(-40, 40, 3)
    return T


def _triangle_labels(faces: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Etykieta trójkąta = etykieta co najmniej 2 z 3 wierzchołków (jak contacts.triangle_labels)."""
    lt = labels[faces]
    out = np.where(lt[:, 0] == lt[:, 1], lt[:, 0], np.where(lt[:, 0] == lt[:, 2], lt[:, 0], lt[:, 1]))
    return np.where((lt == out[:, None]).sum(axis=1) >= 2, out, 0)


def _areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Pola trójkątów [a.u.²]."""
    return 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)


def _scan(rng: np.random.Generator, v: np.ndarray, f: np.ndarray, labels: np.ndarray,
          params: SyntheticParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """"Skanowanie": szum wzdłuż normalnych i dziury w korze. Zwraca (v, f, labels)."""
    if params.scan_noise_au > 0:
        normals = trimesh.Trimesh(v, f, process=False).vertex_normals
        v = v + normals * rng.normal(0, params.scan_noise_au, len(v))[:, None]
    if params.holes > 0:
        cortex = np.flatnonzero(labels == 0)
        drop = np.zeros(len(f), dtype=bool)
        only_cortex = (labels[f] == 0).all(axis=1)
        centroids = v[f].mean(axis=1)
        for c in rng.choice(cortex, size=min(params.holes, len(cortex)), replace=False):
            r = _uniform(rng, params.hole_au)
            drop |= only_cortex & (np.linalg.norm(centroids - v[c], axis=1) < r)
        f = f[~drop]
        used = np.unique(f)
        remap = np.full(len(v), -1)
        remap[used] = np.arange(len(used))
        v, labels, f = v[used], labels[used], remap[f]
    return v, f, labels


def _colors(rng: np.random.Generator, v: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Kolory wierzchołków: kora jasna, kremowa; przełom ciemniejszy, czerwonawy, nakrapiany."""
    tint = FourierNoise(rng, 3, 0.035, (3.0, 15.0))(v)
    cortex = np.array([0.86, 0.80, 0.67]) + tint[:, None] + rng.normal(0, 0.008, (len(v), 1))
    fracture = np.array([0.76, 0.60, 0.50]) + tint[:, None] + rng.normal(0, 0.04, (len(v), 3))
    return np.clip(np.where((labels > 0)[:, None], fracture, cortex), 0, 1)


def generate(params: SyntheticParams = SyntheticParams()) -> SyntheticSet:
    """Losuje zestaw kości, tnie je na fragmenty i "skanuje" każdy fragment. Nic nie zapisuje.

    Ten sam params.seed = ten sam zestaw.

    Args:
        params: ustawienia generatora.

    Returns:
        SyntheticSet - zapis: write_dataset, podgląd: preview_figure.

    Raises:
        ValueError: nieprawidłowe ustawienia (np. 0 kości, fragments od > do).
    """
    lo, hi = (params.fragments, params.fragments) if np.isscalar(params.fragments) else params.fragments
    layout = [list(cuts) for cuts in params.layout]
    bad = sorted({c for cuts in layout for c in cuts} - set(CUT_KINDS))
    if bad:
        raise ValueError(f"Złe ustawienia: nieznane cięcia {bad} w layout, dostępne: {', '.join(CUT_KINDS)}")
    if not layout and (params.n_bones < 1 or lo < 1 or hi < lo):
        raise ValueError(f"Złe ustawienia: n_bones={params.n_bones}, fragments={params.fragments}")
    rng = np.random.default_rng(params.seed)
    raw, bones = [], []
    for b in range(len(layout) or params.n_bones):
        bone_id = "kosc_" + (string.ascii_uppercase[b] if b < 26 else f"{b + 1:02d}")
        shape = random_shape(rng, params)
        if layout:
            kinds = layout[b]
            cuts = cut_surfaces(rng, shape, kinds, params, strict=True)
        else:
            kinds = ["fracture"] * (int(rng.integers(lo, hi + 1)) - 1)
            cuts = cut_surfaces(rng, shape, kinds, params)
            kinds = kinds[:len(cuts) - 2]
        bones.append({"id": bone_id, "length_au": round(shape.length, 2), "radius_au": round(shape.radius, 2),
                      "cuts": kinds, "fragments": []})
        for k in range(len(cuts) - 1):
            lower_id = 1 if cuts[k].kind == "fracture" else 0
            upper_id = (lower_id + 1) if cuts[k + 1].kind == "fracture" else 0
            theta0 = rng.uniform(0, 2 * np.pi)
            mesh = build_fragment(shape, cuts[k], cuts[k + 1], params.edge_au, theta0, lower_id, upper_id)
            preview = build_fragment(shape, cuts[k], cuts[k + 1], max(PREVIEW_EDGE_AU, params.edge_au),
                                     theta0, lower_id, upper_id)
            raw.append((bone_id, k, lower_id, upper_id, mesh, preview))

    order = rng.permutation(len(raw))                    # numery fragmentów wymieszane między kośćmi
    width = max(2, len(str(len(raw))))
    frag_ids = {int(i): f"frag_{n:0{width}d}" for n, i in enumerate(order, 1)}
    fragments = []
    for i, (bone_id, k, lower_id, upper_id, (v, f, lab), preview) in enumerate(raw):
        v, f, lab = _scan(rng, v, f, lab, params)
        pose = _random_pose(rng)                         # układ kości -> układ skanu
        vs = v @ pose[:3, :3].T + pose[:3, 3]
        colors = _colors(rng, vs, lab) if params.colors else None
        fa, tl = _areas(v, f), _triangle_labels(f, lab)
        surfaces = {}
        for sid, nb in ((lower_id, i - 1), (upper_id, i + 1)):
            if sid:
                surfaces[sid] = {"partner": frag_ids[nb], "fracture": f"{bone_id}/{k + (sid == upper_id)}",
                                 "area_au2": round(float(fa[tl == sid].sum()), 2)}
        fragments.append(SyntheticFragment(id=frag_ids[i], bone=bone_id, index=k, vertices=vs, faces=f, labels=lab,
                                           colors=colors, pose_to_bone=np.linalg.inv(pose), surfaces=surfaces,
                                           preview=preview))
    by_id = {fr.id: fr for fr in fragments}
    bones_by_id = {b["id"]: b for b in bones}
    for fr in fragments:
        for s in fr.surfaces.values():
            partner = by_id[s["partner"]]
            s["partner_surface"] = next(k for k, t in partner.surfaces.items() if t["partner"] == fr.id)
        bones_by_id[fr.bone]["fragments"].append((fr.index, fr.id))
    for b in bones:
        b["fragments"] = [fid for _, fid in sorted(b["fragments"])]
    return SyntheticSet(params=params, fragments=sorted(fragments, key=lambda f: f.id), bones=bones)


# -----------------
# ZAPIS / ODCZYT
# -----------------

SETTINGS_TEXT = """// Ustawienia TEGO zestawu - nadpisują settings.json z głównego folderu (te same sekcje i klucze).
// Plik utworzył scripts/make_synthetic.py.
{
  // Każdy plik to inny fragment - nie ma czego łączyć (i porównanie każdej pary skanów trwałoby długo).
  "merging": {"enabled": false}
}
"""


def write_dataset(synth: SyntheticSet, data_in: Path, contacts_dir: Path | None = None) -> dict:
    """Zapisuje skany, ground_truth.json, settings.json zestawu i (opcjonalnie) prawdziwe przełomy.

    Args:
        synth:        wynik generate().
        data_in:      folder skanów zestawu (np. scanned_bones/synthetic) - tworzony.
        contacts_dir: folder oznaczeń styku zestawu (np. contacts/synthetic); None = bez oznaczeń.
                      Przełomy są zapisywane jako "ręczne" oznaczenia (contacts.save_contacts) -
                      przy prepare_data przechodzą na przygotowane kości same (wspólny plik skanu).

    Returns:
        Zawartość zapisanego ground_truth.json.
    """
    data_in = Path(data_in)
    data_in.mkdir(parents=True, exist_ok=True)
    for fr in synth.fragments:
        path = data_in / fr.id / f"{fr.id}.ply"
        path.parent.mkdir(parents=True, exist_ok=True)
        colors = None if fr.colors is None else (fr.colors * 255).round().astype(np.uint8)
        trimesh.Trimesh(fr.vertices, fr.faces, vertex_colors=colors, process=False).export(path)
    gt = synth.ground_truth()
    (data_in / GROUND_TRUTH).write_text(json.dumps(gt, indent=1, ensure_ascii=False), encoding="utf-8")
    (data_in / DATASET_SETTINGS).write_text(SETTINGS_TEXT, encoding="utf-8")
    if contacts_dir is not None:
        write_contacts(synth, data_in, contacts_dir)
    return gt


def write_contacts(synth: SyntheticSet, data_in: Path, contacts_dir: Path) -> list[Path]:
    """Prawdziwe przełomy -> contacts_dir/<id fragmentu>.json (format contacts.py, source "manual").

    Etykiety dotyczą siatki skanu; "scans" = {suma kontrolna pliku: macierz jednostkowa}, więc
    load_contacts przeliczy je na przygotowaną kość (wyśrodkowaną) bez zgadywania.

    Returns:
        Ścieżki zapisanych plików.
    """
    import open3d as o3d

    from contacts import Contacts, contacts_path, save_contacts
    from merging import file_checksum

    out = []
    for fr in synth.fragments:
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(fr.vertices.astype(np.float64)),
                                         o3d.utility.Vector3iVector(fr.faces.astype(np.int32)))
        names = {k: f"przełom z {s['partner']}" for k, s in fr.surfaces.items()}
        contacts = Contacts(bone_id=fr.id, labels=fr.labels, names=names, source="manual")
        frames = {file_checksum(Path(data_in) / fr.id / f"{fr.id}.ply"): np.eye(4).tolist()}
        path = save_contacts(mesh, contacts, contacts_path(fr.id, "manual", contacts_dir), frames)
        if path:
            out.append(path)
    return out


def load_ground_truth(data_in: Path | None = None) -> dict | None:
    """Wczytuje ground_truth.json zestawu (domyślnie bieżącego - config.DATA_IN); None = brak."""
    if data_in is None:
        from config import DATA_IN as data_in
    path = Path(data_in) / GROUND_TRUTH
    return read_json(path) if path.exists() else None


def true_transform(gt: dict, a, b) -> np.ndarray:
    """Prawdziwe ustawienie kości b względem kości a (przygotowanych - układ out/3d/.../mesh.ply).

    Args:
        gt:   ground_truth.json (load_ground_truth).
        a, b: kości (bones.Bone, z meta.json) albo id fragmentów (wtedy układ skanów).

    Returns:
        T 4x4: punkt kości b -> to samo miejsce w układzie kości a (gdy pasują, nakłada je
        przełom na przełom; dla fragmentów z różnych kości wynik nie ma sensu fizycznego).
    """
    def frames(x):
        """(id fragmentu, przekształcenie skan -> układ przygotowanej kości)."""
        if isinstance(x, str):
            return x, np.eye(4)
        return x.id, np.asarray(x.meta["scans"][0]["transform_to_bone"], dtype=float)

    (ia, Ta), (ib, Tb) = frames(a), frames(b)
    Pa = np.asarray(gt["fragments"][ia]["pose_to_bone"], dtype=float)
    Pb = np.asarray(gt["fragments"][ib]["pose_to_bone"], dtype=float)
    return Ta @ np.linalg.inv(Pa) @ Pb @ np.linalg.inv(Tb)


# -----------------
# PODGLĄD
# -----------------

def preview_figure(synth: SyntheticSet, explode_au: float = 4.0):
    """Figura plotly: kości złożone z fragmentów (lekko rozsuniętych), przełomy w kolorach par.

    Każda kość leży wzdłuż osi X, kolejne kości jedna za drugą. Ten sam kolor przełomu na dwóch
    fragmentach = para, która do siebie pasuje. Najechanie myszą pokazuje nazwę fragmentu.

    Args:
        synth:      wynik generate().
        explode_au: odstęp między sąsiednimi fragmentami kości [a.u.].

    Returns:
        plotly.graph_objects.Figure.
    """
    import plotly.graph_objects as go

    from contacts import surface_color
    from visualization_3d import COLORS, FONT, LIGHTING

    fracture_nr: dict[str, int] = {}
    fig = go.Figure()
    y = 0.0
    for nb, bone in enumerate(synth.bones):
        frags = [next(f for f in synth.fragments if f.id == fid) for fid in bone["fragments"]]
        width = 0.0
        for k, fr in enumerate(frags):
            v, f, lab = fr.preview
            width = max(width, np.ptp(v[:, :2], axis=0).max())
            p = np.column_stack([v[:, 2] + k * explode_au, v[:, 1], -v[:, 0]])     # oś kości -> X
            tl = _triangle_labels(f, lab)
            base = COLORS[nb % len(COLORS)] if k % 2 == 0 else "#d9c7a7"
            fc = np.array([base] * len(f), dtype=object)
            for sid, s in fr.surfaces.items():
                n = fracture_nr.setdefault(s["fracture"], len(fracture_nr) + 1)
                fc[tl == sid] = surface_color(n)
            partners = ", ".join(f"{s['partner']} ({s['area_au2']:.0f} a.u.²)" for s in fr.surfaces.values())
            fig.add_trace(go.Mesh3d(x=p[:, 0], y=p[:, 1] + y, z=p[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                    facecolor=fc, flatshading=False, lighting=LIGHTING, name=fr.id,
                                    hovertext=f"<b>{fr.id}</b> ({bone['id']}, {k + 1}. od lewej)<br>"
                                              f"pasuje do: {partners or '-'}", hoverinfo="text",
                                    showlegend=False))
        fig.add_trace(go.Scatter3d(x=[-8], y=[y], z=[0], mode="text", text=[f"<b>{bone['id']}</b>"],
                                   textfont=dict(size=14, color="#1f2a36"), hoverinfo="skip", showlegend=False))
        y += width + 10
    axis = dict(visible=False)
    fig.update_layout(scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data",
                                 camera=dict(eye=dict(x=0.0, y=-0.3, z=1.6), up=dict(x=0, y=1, z=0))),
                      margin=dict(l=0, r=0, t=40, b=0), font=FONT, paper_bgcolor="#f4f6f8",
                      title=dict(text="Syntetyczne kości: fragmenty złożone; ten sam kolor przełomu = para", x=0.02))
    return fig

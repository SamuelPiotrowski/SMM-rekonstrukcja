"""Run the current solver on local FRACTURA and PENGWIN data with held-out poses."""

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import h5py
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from case_solver import outward_mesh, solve_case, wire_parts
from fracture_library import FRACTURA, GROUPS, PENGWIN, mesh_from_arrays, read_fragment_groups
from osteo_engine_v6 import moved

OUT = ROOT / "out" / "testy-rzeczywiste"
TESTS = {
    "fractura-limb3": ("FRACTURA · limb3", "limb3", (), None),
    "pengwin-017": ("PENGWIN 017 · 3 fragmenty", "017", ("left",), None),
    "pengwin-008": ("PENGWIN 008 · 5 fragmentów", "008", ("left",), None),
    "pengwin-178": ("PENGWIN 178 · 10 fragmentów", "178", ("right",), None),
    "pengwin-005-mix": ("PENGWIN 005 · 2 kości", "005", ("left", "right"), None),
    "pengwin-008-missing": ("PENGWIN 008 · bez fragmentu 103", "008", ("left",), "103"),
}


def load_inputs(case, sides, omit):
    rng = np.random.default_rng(41)
    if not sides:
        with h5py.File(FRACTURA / "fractura_real-001.hdf5", "r") as archive:
            entries = [(f"{case}/{key}", mesh_from_arrays(piece["vertices"][:], piece["faces"][:]), "bone")
                       for key, piece in archive[f"bones/{case}/pieces"].items()]
        matrices = {}
        for name, mesh, _ in entries:
            pose = np.eye(4)
            pose[:3, :3] = Rotation.random(random_state=rng).as_matrix()
            pose[:3, 3] = rng.uniform(-50, 50, 3) - pose[:3, :3] @ mesh.get_center()
            mesh.transform(pose)
            matrices[name] = np.linalg.inv(pose)
    else:
        folder = PENGWIN / case
        entries = [(key, mesh, side) for side in sides
                   for key, mesh in read_fragment_groups(folder / "peripelvic-fracture-fragments.obj", *GROUPS[side]).items()
                   if key != omit]
        matrices = {str(row["fragment_id"]): np.asarray(row["transformation"])
                    for row in json.loads((folder / "plan_pl_gt.json").read_text(encoding="utf-8"))}
        if len(sides) > 1:
            rng.shuffle(entries)
    meshes = [(name, outward_mesh(mesh)) for name, mesh, _ in entries]
    reference = {str(i): matrices[name] for i, (name, _) in enumerate(meshes, 1)}
    groups = [tuple(str(i) for i, (_, _, side) in enumerate(entries, 1) if side == group)
              for group in (sides or ("bone",))]
    return meshes, reference, groups


def measure(meshes, poses, predicted, reference, expected):
    true_pairs = {frozenset(pair) for group in expected for pair in combinations(group, 2)}
    predicted_pairs = {frozenset(pair) for group in predicted for pair in combinations(group, 2)}
    errors = []
    areas = {str(i): mesh.get_surface_area() for i, (_, mesh) in enumerate(meshes, 1)}
    for group in predicted:
        for target in expected:
            shared = sorted(set(group).intersection(target), key=int)
            if len(shared) < 2:
                continue
            anchor = max(shared, key=areas.get)
            align = reference[anchor] @ np.linalg.inv(poses[anchor])
            for key in shared:
                if key == anchor:
                    continue
                label, mesh = meshes[int(key) - 1]
                vertices = np.asarray(mesh.vertices)
                vertices = vertices[::max(1, len(vertices) // 5000)]
                computed = align @ poses[key]
                diff = moved(vertices, computed) - moved(vertices, reference[key])
                rmse = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
                angle = Rotation.from_matrix(computed[:3, :3] @ reference[key][:3, :3].T).magnitude()
                errors.append({"fragment": label, "anchor": meshes[int(anchor) - 1][0],
                               "rmse": float(rmse), "angle": float(np.degrees(angle))})
    return {"correct_pairs": len(true_pairs & predicted_pairs), "expected_pairs": len(true_pairs),
            "cross_bone_pairs": len(predicted_pairs - true_pairs), "errors": errors}


def preview(meshes, poses, groups):
    result = []
    for part in wire_parts(meshes, poses, groups, detail=2600):
        part["vertices"] = moved(np.array(part["vertices"]), np.array(part.pop("matrix"))).round(5).tolist()
        result.append(part)
    return result


def run(test, output_dir=OUT):
    title, case, sides, omit = TESTS[test]
    print(test, "loading", flush=True)
    meshes, reference, expected = load_inputs(case, sides, omit)
    started = perf_counter()
    last = ""

    def progress(message):
        nonlocal last
        if "/" in message:
            prefix, number = message.rsplit(" ", 1)
            current = int(number.split("/")[0])
            if prefix == last and current % 20:
                return
            last = prefix
        print(test, message, flush=True)

    poses, groups, links, comparisons, _ = solve_case(meshes, progress, mixed=len(sides) > 1)
    metrics = measure(meshes, poses, groups, reference, expected)
    print(test, "groups", groups, json.dumps(metrics), flush=True)
    # Compare assembled and reference views in the same orientation for a completely connected bone.
    visible_poses = poses
    if len(groups) == len(expected) == 1:
        anchor = max(groups[0], key=lambda key: meshes[int(key) - 1][1].get_surface_area())
        shift = reference[anchor] @ np.linalg.inv(poses[anchor])
        visible_poses = {key: shift @ pose for key, pose in poses.items()}
    report = {"test": test, "title": title, "case": case, "reference_kind": "PENGWIN" if sides else "FRACTURA",
              "groups": groups, "expected": expected, "fragments": [name for name, _ in meshes],
              "faces": [len(mesh.triangles) for _, mesh in meshes], "comparisons": comparisons,
              "seconds": round(perf_counter() - started, 1), "links": [[edge["a"], edge["b"]] for edge in links],
              "shared_frame_fit": any(edge["hypothesis_id"].startswith("near-") for edge in links),
              **metrics, "predicted_meshes": preview(meshes, visible_poses, groups),
              "reference_meshes": preview(meshes, reference, expected)}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (test + ".json")
    path.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def write_report(output_dir=OUT, *, case_ids=TESTS):
    from plotly.offline import get_plotlyjs

    tests = [json.loads(path.read_text(encoding="utf-8")) for key in case_ids
             if (path := output_dir / (key + ".json")).is_file()]
    if not tests:
        raise ValueError("Uruchom co najmniej jeden test przed utworzeniem raportu.")
    payload = json.dumps(tests, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    output = output_dir / "porownanie.html"
    output.write_text(REPORT_HTML.replace("__RESULTS__", payload, 1)
                      .replace("__PLOTLY__", get_plotlyjs(), 1), encoding="utf-8")
    print(output)


REPORT_HTML = """<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OSTEOFORM · Porównanie z referencją</title>
<style>
* { box-sizing: border-box; }
body { margin: 0; color: #233543; background: #edf3f6; font: 16px/1.5 "Segoe UI", system-ui, sans-serif; }
main { max-width: 1250px; margin: auto; padding: 22px; }
header { background: #18374b; color: #f5fafc; border-radius: 14px; padding: 21px 24px; }
header p { margin: 0 0 2px; color: #b1dce4; font-size: .85rem; font-weight: 700; letter-spacing: .09em; }
h1 { margin: 0; font-size: clamp(1.4rem, 4vw, 2.15rem); line-height: 1.2; }
section { background: #fff; border: 1px solid #d9e4eb; border-radius: 12px; margin-top: 14px; padding: 15px 18px; }
label { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-weight: 600; }
select { font: inherit; max-width: 100%; padding: 7px 9px; border: 1px solid #a9bfcd; border-radius: 6px; }
dt { color: #536a7a; font-size: .89rem; }
dd { margin: 1px 0 0; font-size: 1.1rem; font-weight: 700; }
dl { margin-bottom: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }
dl div { padding: 9px 11px; background: #f4f8fa; border-radius: 7px; }
#comparison { width: 100%; height: 510px; }
p { margin: 8px 0; }
a { color: #075b78; }
table { width: 100%; border-collapse: collapse; text-align: left; }
th, td { padding: 8px; vertical-align: top; border-bottom: 1px solid #e0e8ed; }
.table-wrap { overflow: auto; }
.chart { padding: 4px 8px; }
h2 { margin: 0 0 7px; font-size: 1.12rem; }
@media (max-width: 700px) { main { padding: 10px; } #comparison { height: 700px; } }
</style>
<script>__PLOTLY__</script>
</head>
<body><main>
<header><p>OSTEOFORM</p><h1>Porównanie z referencją 3D</h1></header>
<section aria-label="Wynik wybranego testu">
<label>Przypadek <select id="case"></select></label>
<dl id="summary"></dl>
<p id="note"></p>
</section>
<section class="chart" aria-label="Porównanie modeli 3D"><div id="comparison" aria-label="Rekonstrukcja i referencja"></div></section>
<section><h2>Ocena ułożenia</h2><div class="table-wrap" id="errors"></div>
<p>Błąd obliczono względem największego wspólnego fragmentu w każdej porównywalnej grupie. Pary w podsumowaniu oznaczają przynależność do jednej kości.</p></section>
<section><h2>Materiały</h2>
<p id="inputSource" hidden></p>
<div id="datasetSources">
<p>Fragmenty FRACTURA przestawiono niezależnie. Punktem odniesienia jest układ zapisany w zbiorze. <a href="https://ai4ce.github.io/GARF/">FRACTURA i publikacja GARF</a></p>
<p>W PENGWIN wykorzystano wejściowe siatki z CT i referencyjne macierze autorów. W teście z brakiem referencja pokazuje pozostałe cztery fragmenty. <a href="https://zenodo.org/records/19734383">PENGWIN 2026 Task 3</a></p>
</div>
</section>
</main>
<script type="application/json" id="data">__RESULTS__</script>
<script>
const reports = JSON.parse(document.getElementById('data').textContent);
const selector = document.getElementById('case');
const fmt = new Intl.NumberFormat('pl', { maximumFractionDigits: 4 });
const colors = ['#d6a447', '#5f9fc9', '#82b47d', '#c77d88', '#aa95d0', '#d79667', '#7aafb5', '#bd8d55', '#8297c6', '#a9a36b'];
const make = (tag, value) => Object.assign(document.createElement(tag), { textContent: value });
for (const report of reports) selector.add(new Option(report.title, report.test));
if (reports.some(report => report.test === 'pengwin-008')) selector.value = 'pengwin-008';

function trace(part, scene) {
  return { type: 'mesh3d', scene, name: part.label, legendgroup: part.id, showlegend: scene === 'scene',
    x: part.vertices.map(v => v[0]), y: part.vertices.map(v => v[1]), z: part.vertices.map(v => v[2]),
    i: part.triangles.map(t => t[0]), j: part.triangles.map(t => t[1]), k: part.triangles.map(t => t[2]),
    color: colors[(Number(part.id) - 1) % colors.length], flatshading: true, hoverinfo: 'name',
    lighting: {ambient: .72, diffuse: .78, roughness: .9} };
}

function chart(report) {
  const narrow = matchMedia('(max-width: 700px)').matches;
  const first = narrow ? {x: [0, 1], y: [.52, 1]} : {x: [0, .49], y: [0, 1]};
  const second = narrow ? {x: [0, 1], y: [0, .48]} : {x: [.51, 1], y: [0, 1]};
  const axis = {visible: false};
  const camera = {eye: {x: .8, y: -1.42, z: 1.03}, projection: {type: 'orthographic'}};
  const scene = domain => ({domain, xaxis: axis, yaxis: axis, zaxis: axis, camera,
    aspectmode: 'data', bgcolor: '#f3f7f9'});
  const caption = (text, domain) => ({text, x: (domain.x[0] + domain.x[1]) / 2,
    y: domain.y[1], showarrow: false, yanchor: 'bottom', xref: 'paper', yref: 'paper', font: {size: 15}});
  Plotly.react('comparison', [...report.predicted_meshes.map(p => trace(p, 'scene')),
    ...report.reference_meshes.map(p => trace(p, 'scene2'))], {
    scene: scene(first), scene2: scene(second),
    annotations: [caption('Wynik algorytmu', first), caption('Ułożenie referencyjne', second)],
    margin: {l: 2, r: 2, t: 36, b: 42}, legend: { orientation: 'h', y: -.045 },
    font: {family: 'Segoe UI, sans-serif', color: '#233543'}
  }, {responsive: true, displaylogo: false});
}

function render() {
  const report = reports.find(report => report.test === selector.value);
  const inputSource = document.getElementById('inputSource');
  inputSource.textContent = report.source_note || '';
  inputSource.hidden = !report.source_note;
  document.getElementById('datasetSources').hidden = Boolean(report.source_note);
  const groups = report.groups.map(group => group.length).sort((a, b) => b - a).join(' + ');
  const values = [['Fragmenty na wejściu', report.fragments.length], ['Rozmiary grup', groups],
    ['Pary z tej samej kości w jednej grupie', report.correct_pairs + ' / ' + report.expected_pairs],
    ['Pary z różnych kości w jednej grupie', report.cross_bone_pairs]];
  document.getElementById('summary').replaceChildren(...values.map(([label, value]) => {
    const box = make('div', ''); box.append(make('dt', label), make('dd', value)); return box;
  }));
  const note = report.shared_frame_fit ? 'Algorytm uwzględnił położenie wejściowe oraz zgodność krawędzi. ' : '';
  document.getElementById('note').textContent = note + (report.groups.length > 1 ?
    'Oddzielne grupy pokazano obok siebie. Kolory identyfikują te same fragmenty w obu widokach.' :
    'Kolory identyfikują te same fragmenty w obu widokach.');
  const errors = document.getElementById('errors');
  if (report.errors.length) {
    const unit = report.reference_kind === 'PENGWIN' ? 'mm' : 'jedn. siatki';
    const table = make('table', '');
    const header = table.createTHead().insertRow();
    header.append(...['Fragment', 'Fragment odniesienia', 'RMSE położenia (' + unit + ')', 'Błąd obrotu (°)'].map(text => make('th', text)));
    for (const error of report.errors) {
      const row = table.insertRow();
      row.append(...[error.fragment, error.anchor, fmt.format(error.rmse), fmt.format(error.angle)].map(text => make('td', text)));
    }
    errors.replaceChildren(table);
  } else errors.textContent = 'Brak połączonej pary z tej samej kości, dla której można policzyć błąd wzajemnego ułożenia.';
  chart(report);
}

selector.onchange = render;
matchMedia('(max-width: 700px)').addEventListener('change', render);
render();
</script>
</body></html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", metavar="TEST", help=", ".join(TESTS))
    parser.add_argument("--report", action="store_true", help="write the HTML comparison from available results")
    parser.add_argument("--output-dir", type=Path, default=OUT, help="directory for results and the report")
    args = parser.parse_args()
    unknown = set(args.cases) - TESTS.keys()
    if unknown or not (args.cases or args.report):
        parser.error(f"Choose one of {', '.join(TESTS)} or --report.")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    for test in args.cases:
        run(test, args.output_dir)
    if args.report:
        write_report(args.output_dir)

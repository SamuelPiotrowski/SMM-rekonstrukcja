import { Matrix4, Quaternion, Vector3 } from 'three';

const element = id => document.getElementById(id);
const number = new Intl.NumberFormat('pl', { maximumFractionDigits: 2 });
const percent = new Intl.NumberFormat('pl', { style: 'percent', maximumFractionDigits: 1 });
const status = element('resultStatus');
const selector = element('candidate');
const toggle = element('togglePosition');
const motion = element('motion');
const play = element('play');
const scene = element('scene');
const groups = element('groupSelect');
const inspect = element('inspection');
const wireframe = element('wireframe');
const resultColors = element('resultColors');
const scanFragment = element('scanFragment');
const scanSource = element('scanSource');
const scanDetail = element('scanDetail');
const scanStatus = element('scanStatus');
const scanScene = element('scanScene');
const scanDownload = element('scanDownload');
const query = new URLSearchParams(location.search);
const isReference = query.get('reference') === '1';
const job = query.get('job');
const isCase = Boolean(job);
const isDemo = ['demo-fibula', 'demo-mixed', 'demo-fibula-pair', 'demo-fibula-rejection', 'demo-fibula-gap'].includes(job);
let presentation = isDemo;
const colors = [['#e9c37f', '#83633b'], ['#8dc6e5', '#174f87'], ['#a5cf9e', '#176d58'],
  ['#e6a6b6', '#8e443b'], ['#bdaae2', '#624786'], ['#edb595', '#a96022']];
const demoColors = ['#2683bf', '#dd9737', '#32a77f', '#cd6079', '#8067c6',
  '#29a9b9', '#e3734c', '#92a43a', '#4e64ba', '#c77ab0', '#a7865b'];
const demoBackground = '#eaf1f7';
const scanLighting = { ambient: .72, diffuse: .73, roughness: .92, specular: .06, fresnel: .03 };
const demoLighting = { ambient: .78, diffuse: .84, roughness: .89, specular: .09, fresnel: .04 };
let output;
let combinedGroup;
let objects = [];
let demoBounds;
let demoFit = 0;
let playing = false;
let rendering = false;
let pendingFrame = null;
let latestDraw = Promise.resolve();
const markerColors = { contact: '#009787', collision: '#d12d4d' };
const sourceCache = new Map();
let scanRequest = 0;
let fragmentListKey = '';
let pickingEnabled = false;

function selectedGroup() {
  if (groups.value === 'combined') return combinedGroup;
  return output.mixed ? output.sections[Number(groups.value)] : null;
}

function gapPositions(frame) {
  const gap = output?.gap;
  if (!gap) return null;
  const group = selectedGroup();
  if (group?.unassigned) return null;
  const points = group?.combined ? gap.combined_points : gap.points;
  return coords(frame >= 1 ? points : [], null, 0);
}

function combineGroups(data) {
  const parts = data.sections.flatMap((section, sectionIndex) => section.parts.map(part => {
    const { offset, matrix } = data.combined_layout[part.id];
    const shift = points => points.map(point => point.map((value, axis) => value + offset[axis]));
    const combined = { ...part, vertices: shift(part.vertices), matrix, demoSection: sectionIndex };
    for (const name of Object.keys(markerColors)) {
      if (part[name]?.length) combined[name] = shift(part[name]);
    }
    return combined;
  }));
  return { parts, diagnostics: data.diagnostics, combined: true };
}

function displayMarks(part, progress) {
  return Object.keys(markerColors).filter(name => part[name]?.length).map(name => ({
    name, points: inspect.checked && progress >= .99 ? part[name] : [],
  }));
}

function transforms(part, matrix) {
  if (!matrix) return null;
  const target = new Matrix4().set(...matrix.flat());
  const center = new Vector3();
  for (const point of part.vertices) center.add(new Vector3(...point));
  center.divideScalar(part.vertices.length);
  const result = { center, end: center.clone().applyMatrix4(target),
    rotation: new Quaternion().setFromRotationMatrix(target) };
  if (part.coarse_matrix) {
    const intermediate = new Matrix4().set(...part.coarse_matrix.flat());
    result.midpoint = center.clone().applyMatrix4(intermediate);
    result.coarseRotation = new Quaternion().setFromRotationMatrix(intermediate);
  }
  return result;
}

function coords(points, transform, progress) {
  const coordinates = { x: [], y: [], z: [] };
  if (transform && progress > 0) {
    let position, rotation;
    if (!transform.midpoint) {
      position = transform.center.clone().lerp(transform.end, progress);
      rotation = new Quaternion().slerp(transform.rotation, progress);
    } else if (progress <= .5) {
      position = transform.center.clone().lerp(transform.midpoint, 2 * progress);
      rotation = new Quaternion().slerp(transform.coarseRotation, 2 * progress);
    } else {
      position = transform.midpoint.clone().lerp(transform.end, 2 * progress - 1);
      rotation = transform.coarseRotation.clone().slerp(transform.rotation, 2 * progress - 1);
    }
    const temporary = new Vector3();
    for (const p of points) {
      temporary.fromArray(p).sub(transform.center).applyQuaternion(rotation).add(position);
      coordinates.x.push(temporary.x);
      coordinates.y.push(temporary.y);
      coordinates.z.push(temporary.z);
    }
  } else {
    for (const [x, y, z] of points) {
      coordinates.x.push(x);
      coordinates.y.push(y);
      coordinates.z.push(z);
    }
  }
  return coordinates;
}

function meshTrace(part, positions, color, detailed = false) {
  return {
    type: 'mesh3d', name: part.label, showlegend: true, color,
    ...positions,
    i: part.triangles.map(t => t[0]), j: part.triangles.map(t => t[1]), k: part.triangles.map(t => t[2]),
    hoverinfo: 'name', flatshading: detailed,
    lighting: { ambient: .51, diffuse: .84, roughness: .92, specular: .08, fresnel: .03 },
    lightposition: { x: 150, y: -125, z: 220 },
  };
}

function triangleWire({ vertices, triangles }, triangleLimit) {
  const points = [];
  const count = Math.min(triangles.length, triangleLimit);
  const a = new Vector3(), b = new Vector3(), c = new Vector3();
  const ab = new Vector3(), ac = new Vector3(), normal = new Vector3();
  for (let i = 0; i < count; i++) {
    const [first, second, third] = triangles[Math.floor(i * triangles.length / count)];
    a.fromArray(vertices[first]);
    b.fromArray(vertices[second]);
    c.fromArray(vertices[third]);
    ab.subVectors(b, a);
    ac.subVectors(c, a);
    // Lift the decoration slightly off the face so it remains visible at the same depth.
    normal.crossVectors(ab, ac).normalize().multiplyScalar(.10 * Math.min(ab.length(), ac.length()));
    points.push(a.add(normal).toArray(), b.add(normal).toArray(), c.add(normal).toArray());
  }
  return points;
}

function edgeCoordinates(wire, transform, progress) {
  const lines = { x: [], y: [], z: [] };
  if (!wireframe.checked) return lines;
  const positions = coords(wire, transform, progress);
  for (const axis of ['x', 'y', 'z']) {
    const values = positions[axis];
    for (let i = 0; i < values.length; i += 3) {
      lines[axis].push(values[i], values[i + 1], values[i + 2], values[i], null);
    }
  }
  return lines;
}

function hasScanColors(part) {
  return Boolean(part && part.vertex_colors?.length === part.vertices.length);
}

function demoColor(part) {
  return demoColors[(Number(part.id) - 1) % demoColors.length];
}

function shadedColors(source, base) {
  const luminance = hex => {
    const rgb = Number.parseInt(hex.slice(1), 16);
    return .21 * ((rgb >> 16) & 255) + .72 * ((rgb >> 8) & 255) + .07 * (rgb & 255);
  };
  const brightness = source.map(luminance);
  const average = brightness.reduce((sum, value) => sum + value, 0) / brightness.length;
  const rgb = Number.parseInt(base.slice(1), 16);
  const channels = [16, 8, 0].map(shift => (rgb >> shift) & 255);
  return brightness.map(value => {
    const factor = Math.min(1.19, Math.max(.78, 1 + (value - average) / 300));
    return '#' + channels.map(channel => Math.min(255, Math.round(factor * channel)).toString(16).padStart(2, '0')).join('');
  });
}

function arrangeDemoMotion() {
  const sections = new Map();
  for (const object of objects) {
    if (!object.transform) continue;
    const key = object.part.demoSection ?? 0;
    if (!sections.has(key)) sections.set(key, []);
    sections.get(key).push(object);
  }
  for (const section of sections.values()) {
    const center = section.reduce((sum, object) => sum.add(object.transform.end), new Vector3());
    center.divideScalar(section.length);
    section.sort((a, b) => a.transform.end.distanceToSquared(center) - b.transform.end.distanceToSquared(center));
    for (const [index, object] of section.entries()) {
      object.delay = .27 * index / Math.max(1, section.length - 1);
    }
  }
}

function smoothStep(progress) {
  return progress * progress * (3 - 2 * progress);
}

function fragmentProgress(object, progress) {
  if (!presentation) return progress;
  const delay = object.delay ?? 0;
  return smoothStep(Math.max(0, (progress - delay) / (1 - delay)));
}

function traces(object, index) {
  const { part, transform, wire } = object;
  const [color, patchColor] = colors[index % colors.length];
  const frame = Number(motion.value) / 100;
  const progress = fragmentProgress(object, frame);
  const positions = coords(part.vertices, transform, progress);
  const surface = meshTrace(part, positions, color, true);
  surface.lighting = { ambient: .78, diffuse: .72, roughness: .86, specular: .10, fresnel: .03 };
  if (presentation) {
    const tone = demoColor(part);
    surface.showlegend = false;
    surface.color = resultColors.checked ? '#ddceb6' : tone;
    if (hasScanColors(part)) {
      surface.vertexcolor = resultColors.checked ? part.vertex_colors : shadedColors(part.vertex_colors, tone);
    }
    surface.lighting = demoLighting;
    surface.lightposition = { x: 85, y: 280, z: 320 };
  } else if (resultColors.checked && hasScanColors(part)) {
    surface.vertexcolor = part.vertex_colors;
    surface.lighting = scanLighting;
  }
  surface.opacity = inspect.checked && (part.contact?.length || part.collision?.length) ? .79 : 1;
  const series = [surface, {
    type: 'scatter3d', mode: 'lines', name: part.label,
    ...edgeCoordinates(wire, transform, progress), line: { color: 'rgba(27, 45, 60, .62)', width: 2 },
    hoverinfo: 'none', showlegend: false,
  }];
  if (part.patch?.length) series.push({
    type: 'scatter3d', mode: 'markers', name: part.label + ' · przełom',
    ...coords(part.patch, transform, progress), marker: { color: patchColor, size: 2.1 },
    hoverinfo: 'name', visible: element('patches').checked,
  });
  for (const { name, points } of displayMarks(part, frame)) series.push({
    type: 'scatter3d', mode: 'markers', name: name === 'contact' ? 'Styk' : 'Możliwe przenikanie',
    ...coords(points, transform, progress), marker: { color: markerColors[name], size: name === 'contact' ? 2.6 : 4.2 },
    hoverinfo: 'name', showlegend: false,
  });
  return series.map(trace => ({ ...trace, legendgroup: String(index), meta: { fragment: index } }));
}

function setupScanPanel(parts) {
  const key = JSON.stringify([groups.value, parts.map(part => [part.id, part.label])]);
  if (key === fragmentListKey) return;
  fragmentListKey = key;
  scanRequest++;
  element('scanPanel').hidden = presentation;
  scanFragment.replaceChildren(new Option('Wybierz fragment', ''), ...parts.map((part, i) => new Option(part.label, i)));
  scanFragment.value = '';
  element('scanSourceLabel').hidden = true;
  element('scanDetailLabel').hidden = true;
  scanDetail.checked = false;
  scanScene.hidden = true;
  scanDownload.hidden = true;
  scanStatus.textContent = 'Kliknij fragment w widoku 3D lub wybierz go z listy.';
}

function demoButtons() {
  return Array.from(element('demoFragments').querySelectorAll('button'));
}

function updateDemoSelection() {
  for (const [index, button] of demoButtons().entries()) {
    button.setAttribute('aria-pressed', String(String(index) === scanFragment.value));
  }
}

function renderDemoStage(parts, group) {
  element('demoStage').hidden = false;
  element('scanClose').hidden = false;
  const count = parts.length + (parts.length === 4 ? ' odłamki' : ' odłamków');
  const target = output.target;
  element('demoStageDetail').textContent = group?.combined ?
    (target ? output.placed + (output.gap ? ' do ułożenia' : ' do złożenia') +
      ' · ' + target.excluded.length + ' poza grupą' : '2 kości · ' + count) : count;
  const legend = element('demoFragments');
  const buttons = parts.map((part, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.title = part.id === target?.anchor ? 'Fragment odniesienia · ' + part.label : part.label;
    button.setAttribute('aria-label', 'Pokaż skan · ' + button.title);
    const swatch = document.createElement('span');
    swatch.className = 'demo-swatch';
    const scanColor = part.vertex_colors?.[Math.floor(part.vertices.length / 2)] ?? '#ddceb6';
    swatch.style.setProperty('--swatch', resultColors.checked ? scanColor : demoColor(part));
    button.append(swatch, String(part.id).padStart(2, '0'));
    button.onclick = () => chooseFragment(index);
    return button;
  });
  if (group?.combined) {
    legend.replaceChildren(...output.sections.map((section, index) => {
      const row = document.createElement('div');
      row.className = 'demo-fragment-group';
      const name = document.createElement('strong');
      name.textContent = section.unassigned ? 'Odłożone odłamki' :
        section.bone_type?.name || output.bone_type?.name || 'Wybrana kość';
      row.append(name, ...buttons.filter((_, partIndex) => parts[partIndex].demoSection === index));
      return row;
    }));
  } else {
    legend.replaceChildren(...buttons);
  }
  legend.hidden = false;
  updateDemoSelection();
}

async function drawScan() {
  const serial = ++scanRequest;
  scanDownload.hidden = true;
  const index = Number(scanFragment.value);
  const part = objects[index]?.part;
  if (!part || scanFragment.value === '') {
    scanScene.hidden = true;
    element('scanSourceLabel').hidden = true;
    element('scanDetailLabel').hidden = true;
    scanStatus.textContent = 'Kliknij fragment w widoku 3D lub wybierz go z listy.';
    return;
  }
  const uploaded = part.uploaded_scan;
  const source = uploaded || (scanSource.value ? part.source_scans?.[Number(scanSource.value)] : null);
  const fullDetail = Boolean(source && (uploaded || scanDetail.checked));
  if (source?.download) {
    scanDownload.href = source.download;
    scanDownload.hidden = false;
  }
  element('scanDetailLabel').hidden = !source || Boolean(uploaded);
  let geometry = part;
  let caption = 'Skan przed złożeniem · obróć myszką lub dotykiem.';
  if (source) {
    const url = source.url + (!uploaded && fullDetail ? '?detail=full' : '');
    scanStatus.textContent = fullDetail ? 'Wczytuję pełną siatkę…' : 'Wczytuję podgląd skanu…';
    scanScene.hidden = true;
    try {
      if (!sourceCache.has(url)) {
        const response = await fetch(url);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error);
        if (sourceCache.size >= 3) sourceCache.delete(sourceCache.keys().next().value);
        sourceCache.set(url, payload);
      }
      geometry = sourceCache.get(url);
      caption = (fullDetail ? 'Pełna siatka źródłowa' : 'Siatka podglądowa') + ' · ' +
        number.format(geometry.triangles.length) + ' trójkątów' + (source.label ? ' · ' + source.label : '');
    } catch {
      if (serial === scanRequest) scanDetail.checked = false;
      caption = 'Ujęcie źródłowe jest niedostępne. Wyświetlam siatkę fragmentu przed złożeniem.';
    }
  }
  if (serial !== scanRequest) return;
  scanStatus.textContent = caption;
  scanScene.hidden = false;
  const trace = meshTrace(geometry, coords(geometry.vertices, null, 0), '#b9bbc0', fullDetail && geometry !== part);
  trace.hoverinfo = 'skip';
  trace.showlegend = false;
  if (hasScanColors(geometry)) {
    trace.vertexcolor = geometry.vertex_colors;
    trace.lighting = scanLighting;
  }
  const axis = { visible: false, autorange: true };
  await Plotly.react(scanScene, [trace], {
    autosize: true, paper_bgcolor: '#f3f6fa', margin: { l: 0, r: 0, t: 0, b: 0 },
    uirevision: fragmentListKey + ':' + index + ':' + scanSource.value,
    scene: { xaxis: axis, yaxis: axis, zaxis: axis, aspectmode: 'data', bgcolor: '#f3f6fa',
      camera: { eye: { x: 1.3, y: 1.28, z: .82 }, projection: { type: 'orthographic' } } },
  }, { responsive: true, displaylogo: false });
}

function chooseFragment(index) {
  const part = index === '' ? null : objects[Number(index)]?.part;
  scanFragment.value = part ? String(index) : '';
  const sources = part?.source_scans || [];
  scanSource.replaceChildren(new Option('Siatka przed złożeniem', ''), ...sources.map((source, i) => new Option(source.label, i)));
  element('scanSourceLabel').hidden = !sources.length;
  scanSource.value = sources.length ? '0' : '';
  if (presentation) {
    const panel = element('scanPanel');
    const changed = panel.hidden !== !part;
    panel.hidden = !part;
    updateDemoSelection();
    if (changed) Plotly.Plots.resize(scene);
  }
  return drawScan();
}

function metric(name, value) {
  const box = document.createElement('div');
  const label = document.createElement('dt');
  const data = document.createElement('dd');
  label.textContent = name;
  data.textContent = value;
  box.append(label, data);
  return box;
}

function candidate() {
  return isReference || isCase || selector.value === '' ? null : output.candidates[Number(selector.value)];
}

function updatePlaybackControls() {
  const frame = Number(motion.value);
  toggle.textContent = frame === 100 ? 'Pokaż przed złożeniem' : 'Pokaż po złożeniu';
  play.textContent = playing ? 'Wstrzymaj' : 'Odtwórz składanie';
  if (presentation) {
    element('demoPhase').textContent = frame === 0 ? 'Rozrzucone odłamki' :
      frame === 100 ? (output?.gap && !selectedGroup()?.unassigned ? 'Rekonstrukcja niepełna' : 'Po złożeniu') :
        playing ? 'Składanie' : 'W trakcie składania';
    element('demoProgress').textContent = frame + '%';
  }
  if (output?.open3d_stages) {
    for (const [name, position] of [['registrationStart', 0], ['registrationGlobal', 50], ['registrationRefined', 100]]) {
      element(name).setAttribute('aria-pressed', String(frame === position));
    }
    element('registrationStage').textContent = frame === 0 ? 'Skany przed obliczeniem dopasowania.'
      : frame < 50 ? 'Ruch do ustawienia obliczonego przez FPFH i RANSAC.'
      : frame === 50 ? 'Po rejestracji globalnej FPFH + RANSAC.'
      : frame < 100 ? 'Ruch do ustawienia doprecyzowanego przez ICP.'
      : 'Po lokalnym dopasowaniu ICP point-to-plane.';
  }
}

function showDiagnostics(pairs, parts) {
  element('pairDetails').hidden = !pairs.length;
  element('inspectionLabel').hidden = !pairs.length;
  element('inspectionLegend').hidden = !pairs.length || !inspect.checked;
  const names = new Map(parts.map(part => [part.id, part.label]));
  const available = pairs.filter(pair => pair.near_samples);
  const body = element('pairList');
  body.replaceChildren();
  for (const pair of available) {
    const row = body.insertRow();
    const heading = document.createElement('th');
    heading.scope = 'row';
    heading.textContent = names.get(pair.a) + ' + ' + names.get(pair.b);
    if (pair.method === 'local') {
      const note = document.createElement('small');
      note.textContent = 'Szacunek lokalny';
      heading.append(note);
    }
    row.append(heading);
    row.insertCell().textContent = pair.contact_coverage?.map(value => percent.format(value)).join(' / ') || '—';
    for (const value of [pair.contact_samples, pair.gap_rmse, pair.collision_samples, pair.collision_p95]) {
      row.insertCell().textContent = Number.isFinite(value) ? number.format(value) : '—';
    }
  }
  element('pairTable').hidden = !available.length;
  element('pairNote').textContent = available.length ?
    'Kontakt A / B to odsetek próbek z całej powierzchni każdego fragmentu. Odległości podano w jednostkach skanu.' :
    'Brak próbek w zasięgu sprawdzania styku.';
}

function renderGroupingEvidence(target, separation) {
  const entries = target?.excluded ?? separation ?? [];
  element('rejectionDetails').hidden = !entries.length;
  element('rejectionLink').hidden = !entries.length;
  if (!entries.length) return;
  element('rejectionTitle').textContent = target ? 'Dlaczego odłożono te fragmenty?' : 'Jak rozdzielono odłamki?';
  element('rejectionLink').textContent = target ? 'Powody odrzuceń' : 'Podstawa podziału';
  element('rejectionIntro').textContent = target ? (target.basis || 'Fragment odniesienia') + ' · ' + target.label :
    'Obie grupy to kości strzałkowe. O ich rozdzieleniu zadecydowały potwierdzone styki.';
  element('rejectionList').replaceChildren(...entries.map(entry => {
    const row = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'ghost';
    button.textContent = entry.label;
    button.onclick = async () => {
      if (!objects.some(object => object.part.id === entry.id)) {
        groups.value = 'combined';
        await groups.onchange();
      }
      await chooseFragment(objects.findIndex(object => object.part.id === entry.id));
      scanFragment.focus({ preventScroll: true });
      element('scanPanel').scrollIntoView({ block: 'nearest' });
    };
    const text = document.createElement('div');
    const reason = document.createElement('strong');
    const detail = document.createElement('p');
    reason.textContent = entry.reason;
    detail.textContent = entry.detail;
    text.append(reason, detail);
    if (entry.source) {
      const source = document.createElement('p');
      source.textContent = entry.source;
      text.append(source);
    }
    row.append(button, text);
    return row;
  }));
}

function viewport(ranges) {
  const axis = key => ({ title: '', range: ranges[key], visible: false });
  const eye = presentation ? { x: .19, y: 1.88, z: .31 } : { x: 1.09, y: 1.16, z: .74 };
  return { aspectmode: 'data', xaxis: axis('x'), yaxis: axis('y'), zaxis: axis('z'),
    uirevision: 'fragment-view-' + groups.value, bgcolor: presentation ? demoBackground : '#f3f6fa',
    camera: { eye, projection: { type: 'orthographic' } } };
}

async function render() {
  await latestDraw;
  const match = candidate();
  const group = selectedGroup();
  const parts = group?.parts || (isReference || isCase ? output.parts : [output.a, { ...output.b, matrix: match?.matrix }]);
  element('resultColorsLabel').hidden = !presentation && !parts.some(hasScanColors);
  const triangleLimit = Math.max(160, Math.floor(6200 / parts.length));
  objects = parts.map(part => ({ part, transform: transforms(part, part.matrix),
    wire: triangleWire(part, triangleLimit) }));
  setupScanPanel(parts);
  if (presentation) {
    arrangeDemoMotion();
    demoBounds = { wide: sceneRanges(), final: sceneRanges([1]) };
    renderDemoStage(parts, group);
  }
  const available = isReference || (isCase && !group?.unassigned &&
    ((group && !output.target) || output.placed > 1)) || Boolean(match);
  element('focusAssembly').hidden = !available;
  toggle.disabled = !available;
  element('export').disabled = !match;
  play.disabled = motion.disabled = !available;
  updatePlaybackControls();
  status.textContent = group?.unassigned ? 'Fragmenty bez dostatecznie mocnego dopasowania.' : group
    ? 'Liczba odłamków w wybranej grupie · ' + parts.length
    : isCase ? output.reference_view ? 'Ułożenie referencyjne ze zbioru FRACTURA.' : (
    (output.placed === output.total ? 'Wszystkie fragmenty tworzą jedną grupę.' :
      'Program zostawił osobno fragmenty bez dostatecznie mocnego dopasowania.') +
    (output.labelled_surfaces ? (output.source_label ? ' Użyto oznaczeń przełomów ' + output.source_label + '.'
      : ' Powierzchnie przełomu podano z FRACTURA.') : ' Dopasowanie obliczono z samych siatek.')
  ) : isReference ? 'Animacja referencyjna. Położenie końcowe pochodzi z bazy PENGWIN.'
    : !available ? 'Nie znaleziono żadnej propozycji dla tych zaznaczeń.'
    : match.eligible ? 'Propozycja spełnia robocze kryteria kontaktu.'
    : 'Ta propozycja ma zbyt słaby kontakt lub przenikanie przekracza próg.';
  if (isCase && output.shared_frame_fit) {
    status.textContent = 'Wstępne ułożenie ze wspólnego układu skanów. Sprawdź styk i ciągłość zewnętrznej powierzchni.';
  }
  if (output.target) {
    status.textContent = (output.target.basis || 'Fragment odniesienia') + ' · ' + output.target.label +
      '. Przypisano ' + output.placed + ' z ' + output.total + ' odłamków.';
  }
  if (output.gap) {
    status.textContent += ' Wzdłuż trzonu pozostała luka. Żaden z dostępnych fragmentów jej nie wypełnia.';
  }
  if (output.case_note) status.textContent = output.target ? status.textContent + ' ' + output.case_note : output.case_note;
  const mm = v => v === null ? '—' : number.format(v) + ' mm';
  const metrics = !match ? [] : [
    metric('Kontakt przełomu A', percent.format(match.contact_a)),
    metric('Kontakt przełomu B', percent.format(match.contact_b)),
    metric('Szacowana powierzchnia styku', number.format(match.contact_area) + ' mm²'),
    metric('RMSE w strefie styku', mm(match.contact_rmse)),
    metric('P95 w strefie styku', mm(match.contact_p95)),
    metric('Wskaźnik przenikania', percent.format(match.penetration)),
    metric('Tolerancja styku', mm(output.tolerance)),
  ];
  if (match?.reference) {
    metrics.push(metric('Błąd położenia względem referencji', mm(match.reference.pose_rmse_mm)),
                 metric('Błąd obrotu względem referencji', number.format(match.reference.rotation_deg) + '°'));
  }
  if (isCase && !output.reference_view) {
    if (output.target) {
      metrics.push(metric(output.gap ? 'Ułożone według wzorca' : 'Złożone odłamki', output.placed + ' / ' + output.total),
                   metric('Poza wybraną kością', output.target.excluded.length));
      if (output.gap) {
        metrics.push(metric('Potwierdzone połączenia', output.gap.contacts),
                     metric('Luka wzdłuż osi kości', number.format(output.gap.length) + ' j. skanu'));
      }
    } else if (output.mixed) {
      const assembled = output.sections.filter(section => !section.unassigned);
      const assigned = assembled.reduce((count, section) => count + section.parts.length, 0);
      metrics.push(metric('Złożone grupy', assembled.length),
                   metric('Przypisane fragmenty', assigned + ' / ' + output.total));
    } else {
      metrics.push(metric('Największa grupa', output.placed + ' / ' + output.total));
    }
    metrics.push(metric('Porównania geometrii', output.pairs));
    const atlas = group?.atlas || output.atlas;
    const boneType = group?.bone_type || output.bone_type;
    if (group?.combined && !output.target) {
      metrics.push(metric('Kości w pokazie', output.sections.map(section => section.bone_type.name).join(' i ')));
    } else if (boneType) {
      metrics.push(metric('Typ kości', boneType.name),
                   metric('Podstawa identyfikacji', boneType.source));
    } else if (atlas) {
      metrics.push(metric('Typ sugerowany przez atlas', atlas.clear ? atlas.suggestion :
        'Niepewny · najbliżej: ' + atlas.suggestion));
    } else if (isCase && !output.mixed) {
      metrics.push(metric('Typ kości', 'Brak rozstrzygnięcia'));
    }
    if (Number.isFinite(output.tolerance)) {
      metrics.push(metric('Tolerancja styku', number.format(output.tolerance) + ' j. skanu'));
    }
    if (output.reference_check) {
      const check = output.reference_check;
      metrics.push(metric('Największe RMSE względem skanu wejściowego', number.format(check.rmse) + ' j. skanu'),
                   metric('Największy błąd obrotu względem skanu wejściowego', number.format(check.rotation_deg) + '°'));
    }
    if (output.benchmark) {
      metrics.push(metric('Grupy zgodne z referencją', output.benchmark.correct_groups + ' / ' + output.benchmark.expected_groups),
                   metric('Prawidłowo nieprzypisane', output.benchmark.correct_unassigned + ' / ' + output.benchmark.expected_unassigned),
                   metric('Błędne grupy', output.benchmark.false_groups));
      if (typeof output.benchmark.worst_pose_rmse_percent === 'number') {
        metrics.push(metric('Największy błąd położenia w poprawnej grupie',
          number.format(output.benchmark.worst_pose_rmse_percent) + '% rozmiaru modelu'));
      }
    }
    for (const error of output.reference_errors || []) {
      metrics.push(metric(error.fragment + ' · błąd położenia / obrotu',
        number.format(error.rmse_percent) + '% rozmiaru modelu / ' + number.format(error.angle_deg) + '°'));
    }
    if (output.coarse_reference_errors?.length && output.reference_errors?.length) {
      const worst = rows => Math.max(...rows.map(row => row.rmse_percent));
      metrics.push(metric('Błąd po RANSAC → po ICP',
        number.format(worst(output.coarse_reference_errors)) + '% → ' +
        number.format(worst(output.reference_errors)) + '% rozmiaru modelu'));
    }
  }
  element('metrics').replaceChildren(...metrics);
  showDiagnostics(group?.diagnostics || output.diagnostics || [], parts);
  if (presentation) demoFit = demoFocus(Number(motion.value) / 100);
  const ranges = presentation ? demoRange(demoFit) : sceneRanges(group?.unassigned ? [1] : undefined);
  const series = objects.flatMap(traces);
  const gap = gapPositions(Number(motion.value) / 100);
  if (gap) {
    series.push({ type: 'scatter3d', mode: 'lines+text', name: 'Brak odłamka',
      ...gap, line: { color: '#a45b16', width: 6, dash: 'dash' },
      text: ['', 'Brak odłamka', ''], textposition: 'top center',
      textfont: { color: '#774418', size: 15 }, hoverinfo: 'name', showlegend: false });
  }
  await Plotly.react(scene, series, {
    autosize: true, paper_bgcolor: presentation ? demoBackground : '#fff', font: { family: 'Segoe UI, sans-serif', color: '#1f2a36' },
    margin: { l: 0, r: 0, b: 0, t: presentation ? 0 : 8 }, legend: { orientation: 'h', bgcolor: '#fff', y: .04 },
    scene: viewport(ranges),
  }, { responsive: true, displaylogo: false });
  if (!pickingEnabled) {
    scene.on('plotly_click', event => {
      const hit = event.points?.[0];
      const meta = hit?.data?.meta || hit?.fullData?.meta || scene.data?.[hit?.curveNumber]?.meta;
      if (Number.isInteger(meta?.fragment)) chooseFragment(meta.fragment);
    });
    pickingEnabled = true;
  }
}

function sceneRanges(steps = [0, .25, .5, .75, 1]) {
  const ranges = { x: [Infinity, -Infinity], y: [Infinity, -Infinity], z: [Infinity, -Infinity] };
  for (const object of objects) {
    const { part, transform } = object;
    for (const frame of transform ? steps : [0]) {
      const axes = coords(part.vertices, transform, fragmentProgress(object, frame));
      for (const [axis, values] of Object.entries(axes)) {
        for (const v of values) {
          ranges[axis][0] = Math.min(ranges[axis][0], v);
          ranges[axis][1] = Math.max(ranges[axis][1], v);
        }
      }
    }
  }
  return Object.fromEntries(Object.entries(ranges).map(([key, [min, max]]) => {
    const width = Math.max(...Object.values(ranges).map(([a, b]) => b - a));
    const margin = Math.max(width * (presentation ? .037 : .045), (max - min) * .08);
    return [key, [min - margin, max + margin]];
  }));
}

function demoFocus(progress) {
  return smoothStep(Math.max(0, (progress - .73) / .27));
}

function demoRange(focus) {
  return Object.fromEntries(Object.entries(demoBounds.wide).map(([axis, limits]) =>
    [axis, limits.map((limit, i) => limit + focus * (demoBounds.final[axis][i] - limit))]));
}

async function drawProgress(value) {
  pendingFrame = value;
  if (rendering) return;
  rendering = true;
  try {
    while (pendingFrame !== null) {
      const frame = pendingFrame / 100;
      pendingFrame = null;
      const series = objects.flatMap(object => {
        const { part, transform, wire } = object;
        const progress = fragmentProgress(object, frame);
        const positions = coords(part.vertices, transform, progress);
        return [positions, edgeCoordinates(wire, transform, progress),
          ...(part.patch?.length ? [coords(part.patch, transform, progress)] : []),
          ...displayMarks(part, frame).map(({ points }) => coords(points, transform, progress))];
      });
      const gap = gapPositions(frame);
      if (gap) series.push(gap);
      const update = Object.fromEntries(['x', 'y', 'z'].map(key => [key, series.map(s => s[key])]));
      const focus = presentation ? demoFocus(frame) : demoFit;
      if (focus !== demoFit) {
        demoFit = focus;
        latestDraw = Plotly.update(scene, update, rangePatch(demoRange(focus)));
      } else {
        latestDraw = Plotly.restyle(scene, update);
      }
      await latestDraw;
    }
  } finally { rendering = false; }
}

function setFrame(value) {
  motion.value = Math.round(value);
  updatePlaybackControls();
  return drawProgress(presentation ? value : Number(motion.value));
}

let playbackId = 0;
function stopPlayback() {
  playing = false;
  playbackId++;
  updatePlaybackControls();
}

function startPlayback() {
  if (playing) { stopPlayback(); return; }
  const start = Number(motion.value) === 100 ? 0 : Number(motion.value);
  if (!presentation) adjustRange(sceneRanges());
  const millisecondsPerPercent = presentation ? 64 : 52;
  const time = performance.now() + (presentation && start === 0 ? 330 : 0);
  const id = ++playbackId;
  playing = true;
  updatePlaybackControls();
  if (presentation && Number(motion.value) !== start) setFrame(start);
  let last = start;
  async function tick(now) {
    if (id !== playbackId) return;
    const value = Math.min(100, start + Math.max(0, now - time) / millisecondsPerPercent);
    if (!presentation || value === 100 || value - last >= .4) {
      last = value;
      await setFrame(value);
    }
    if (id !== playbackId) return;
    if (value < 100) requestAnimationFrame(tick);
    else {
      stopPlayback();
      if (!presentation) adjustRange(sceneRanges([1]));
    }
  }
  requestAnimationFrame(tick);
}

function rangePatch(ranges) {
  return Object.fromEntries(Object.entries(ranges).map(([axis, range]) => ['scene.' + axis + 'axis.range', range]));
}

function adjustRange(ranges) {
  return Plotly.relayout(scene, rangePatch(ranges));
}

selector.onchange = () => { stopPlayback(); motion.value = '100'; render(); };
scanFragment.onchange = () => chooseFragment(scanFragment.value);
element('scanClose').onclick = () => {
  const index = Number(scanFragment.value);
  chooseFragment('');
  demoButtons()[index]?.focus();
};
scanSource.onchange = drawScan;
scanDetail.onchange = drawScan;
wireframe.onchange = () => drawProgress(Number(motion.value));
resultColors.onchange = () => { stopPlayback(); render(); };
toggle.onclick = async () => {
  stopPlayback();
  const target = Number(motion.value) === 100 ? 0 : 100;
  await setFrame(target);
  if (!presentation) await adjustRange(sceneRanges([target / 100]));
};
motion.oninput = () => {
  stopPlayback();
  if (!presentation) adjustRange(sceneRanges());
  return setFrame(Number(motion.value));
};
play.onclick = startPlayback;
groups.onchange = async () => {
  stopPlayback(); inspect.checked = false;
  const group = selectedGroup();
  motion.value = group.unassigned ? '100' : '0';
  await render();
  if (presentation && output.placed > 1 && !group.unassigned && selectedGroup() === group && !playing &&
      !window.matchMedia('(prefers-reduced-motion: reduce)').matches) startPlayback();
};
element('focusAssembly').onclick = async () => {
  stopPlayback();
  await setFrame(100);
  await adjustRange(sceneRanges([1]));
};
for (const [name, position] of [['registrationStart', 0], ['registrationGlobal', 50], ['registrationRefined', 100]]) {
  element(name).onclick = async () => {
    stopPlayback();
    await setFrame(position);
    await adjustRange(sceneRanges([position / 100]));
  };
}
inspect.onchange = async () => {
  stopPlayback();
  motion.value = '100';
  await render();
  await adjustRange(sceneRanges([1]));
};
element('patches').onchange = () => { stopPlayback(); render(); };
element('export').onclick = () => {
  const match = candidate();
  const file = new Blob([JSON.stringify({ convention: 'B to A', a: output.a.label, b: output.b.label,
    tolerance_mm: output.tolerance, ...match }, null, 2)], { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(file);
  link.download = 'dopasowanie-B-do-A.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
};

(async () => {
  try {
    const key = query.get('id') || '';
    const selectedCase = query.get('case');
    const caseQuery = selectedCase ? '?' + new URLSearchParams({ case: selectedCase }) : '';
    element('markLink').href = '/' + caseQuery;
    element('markLink').hidden = isCase;
    if (isDemo) {
      const libraryLink = element('libraryLink');
      libraryLink.href = '/case-start#nasze-skany';
      libraryLink.textContent = 'Wróć do naszych skanów';
    }
    const url = isCase ? '/api/solve/' + encodeURIComponent(job) :
      isReference ? '/api/reference' + caseQuery : '/api/match/' + encodeURIComponent(key);
    let response = await fetch(url);
    let data = await response.json();
    while (response.ok && isCase && data.status === 'running') {
      status.textContent = data.message;
      await new Promise(resolve => setTimeout(resolve, 900));
      response = await fetch(url);
      data = await response.json();
    }
    if (!response.ok) {
      if (isDemo && response.status === 404 && data.error === 'Przypadek nie jest już dostępny.') {
        throw new Error('Ta wersja serwera nie obsługuje zapisanego pokazu. Uruchom ponownie OSTEOFORM i wybierz pokaz w zakładce Nasze skany.');
      }
      throw new Error(data.error);
    }
    if (isCase) {
      if (data.status === 'error') throw new Error(data.message);
      data = data.result;
    }
    output = data;
    presentation = isDemo || Boolean(data.target);
    if (presentation) {
      document.body.classList.add('assembly-demo');
      resultColors.checked = false;
    }
    element('registrationStages').hidden = !data.open3d_stages;
    if (data.mixed) {
      const options = data.sections.map((group, i) => {
        const type = group.bone_type?.name || (group.atlas?.clear ? group.atlas.suggestion : 'typ niepewny');
        const label = data.target ? (group.unassigned ? 'Odłożone odłamki' : group.bone_type?.name || 'Wybrana kość') :
          group.unassigned ? 'Nieprzypisane' : 'Grupa ' + (i + 1) + ' · ' + type;
        return new Option(label + ' · ' + group.parts.length, i);
      });
      if (data.combined_layout) {
        combinedGroup = combineGroups(data);
        const label = data.target ? 'Cały zestaw' : 'Obie kości';
        options.unshift(new Option(label + ' · ' + combinedGroup.parts.length + ' odłamków', 'combined'));
      }
      groups.replaceChildren(...options);
      groups.value = combinedGroup ? 'combined' : '0';
      element('groupLabel').hidden = false;
      if (selectedGroup().unassigned) motion.value = '100';
    }
    if (job?.startsWith('benchmark-') && !data.mixed) {
      const compare = element('benchmarkLink');
      compare.hidden = false;
      compare.textContent = data.reference_view ? 'Wynik automatyczny' : 'Pokaż ułożenie referencyjne';
      compare.href = '/assembly?job=' + (data.reference_view ? 'benchmark-auto' : 'benchmark-reference');
      element('labelledLink').hidden = job !== 'benchmark-auto';
    }
    if (data.reference_scan_url) {
      const referenceLink = element('benchmarkLink');
      referenceLink.href = data.reference_scan_url;
      referenceLink.textContent = 'Pełny skan źródłowy i Testy 1–4';
      referenceLink.hidden = false;
    }
    if (isReference || isCase) {
      element('resultTitle').textContent = isCase ? data.title : 'PENGWIN · ' + data.title;
      if (isDemo) document.title = data.title + ' · OSTEOFORM';
      element('candidateLabel').hidden = element('patchesLabel').hidden = element('export').hidden = true;
      if (!(data.mixed ? data.sections.length : data.parts.length)) throw new Error('Brak siatek do wyświetlenia.');
    } else {
      selector.replaceChildren(...data.candidates.map((m, i) =>
        new Option('Propozycja ' + (i + 1) + (m.eligible ? '' : ' · odrzucona'), i)));
      if (!data.candidates.length) selector.add(new Option('Brak propozycji', ''));
    }
    const initialParts = selectedGroup()?.parts || data.parts || [data.a, data.b];
    if (presentation || initialParts.some(hasScanColors)) wireframe.checked = false;
    renderGroupingEvidence(data.target, data.separation);
    await render();
    element('resultControls').hidden = false;
    if (isCase && output.placed > 1 && !selectedGroup()?.unassigned &&
        !window.matchMedia('(prefers-reduced-motion: reduce)').matches) startPlayback();
  } catch (error) {
    element('assemblyWorkspace')?.setAttribute('hidden', '');
    status.textContent = error.message || 'Nie można wczytać wyniku.';
  }
})();

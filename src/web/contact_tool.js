// Narzędzie do oznaczania powierzchni styku (serwer: src/contact_tool.py).
// Etykiety są przypisane do wierzchołków siatki: 0 = brak, k = powierzchnia k.
import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const MAX_UNDO = 60;
const BONE_COLOR = [0.86, 0.82, 0.75];
const MIX = 0.78;                      // jak mocno kolor powierzchni przykrywa kolor skanu

const S = {
  bones: [], palette: [], detectors: [],
  cur: null,                           // aktualna kość
  tool: 'brush', brush: 2.0, wand: 25, active: 1, scanColors: true,
  dirty: false, undo: [], redo: [], painting: false, lastHit: null, busy: false,
};

// ---------------------------------------------------------------- three.js
const canvas = $('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 5000);
scene.add(camera);
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a99a8, 1.6));
const headlight = new THREE.DirectionalLight(0xffffff, 1.9);
headlight.position.set(0.4, 0.6, 1);
camera.add(headlight);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.mouseButtons = { LEFT: -1, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.ROTATE };
controls.zoomToCursor = true;

const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.78, metalness: 0.0, side: THREE.DoubleSide });
let mesh = null;

const cursor = new THREE.Mesh(
  new THREE.RingGeometry(0.9, 1.0, 48),
  new THREE.MeshBasicMaterial({ color: 0x1f2a36, transparent: true, opacity: 0.85, depthTest: false, side: THREE.DoubleSide }));
cursor.renderOrder = 10;
cursor.visible = false;
scene.add(cursor);

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(r.height, 1);
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

// ---------------------------------------------------------------- pomocnicze
const SRGB_LIN = new Float32Array(256).map((_, i) => { const c = i / 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; });
const hexLin = (hex) => { const c = new THREE.Color(hex); return [c.r, c.g, c.b]; };   // THREE.Color zamienia sRGB -> liniowy
const boneLin = BONE_COLOR.map((c) => SRGB_LIN[Math.round(c * 255)]);
const surfColor = (k) => S.palette[(k - 1) % S.palette.length];

function b64ToBytes(b64) {
  const bin = atob(b64); const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function bytesToB64(bytes) {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `${r.status} ${r.statusText}`);
  return data;
}
let toastTimer;
function toast(msg, kind = '') {
  const t = $('toast'); t.textContent = msg; t.className = `toast show ${kind}`;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.className = 'toast'), kind === 'error' ? 6000 : 2600);
}
const fmt = (x, d = 0) => x.toLocaleString('pl-PL', { maximumFractionDigits: d, minimumFractionDigits: d });

// ---------------------------------------------------------------- kość
function buildAdjacency(nV, idx) {
  const deg = new Uint32Array(nV + 1);
  for (let t = 0; t < idx.length; t += 3) { deg[idx[t]] += 2; deg[idx[t + 1]] += 2; deg[idx[t + 2]] += 2; }
  const start = new Uint32Array(nV + 1);
  for (let i = 0; i < nV; i++) start[i + 1] = start[i] + deg[i];
  const fill = start.slice(0, nV); const nb = new Uint32Array(start[nV]);
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t], b = idx[t + 1], c = idx[t + 2];
    nb[fill[a]++] = b; nb[fill[a]++] = c; nb[fill[b]++] = a; nb[fill[b]++] = c; nb[fill[c]++] = a; nb[fill[c]++] = b;
  }
  return { start, nb };
}

function smoothNormals(N, adj, iters) {
  let cur = N.slice();
  for (let it = 0; it < iters; it++) {
    const nxt = new Float32Array(cur.length);
    for (let i = 0; i < cur.length / 3; i++) {
      let x = cur[3 * i], y = cur[3 * i + 1], z = cur[3 * i + 2];
      for (let k = adj.start[i]; k < adj.start[i + 1]; k++) { const j = adj.nb[k]; x += cur[3 * j]; y += cur[3 * j + 1]; z += cur[3 * j + 2]; }
      const l = Math.hypot(x, y, z) || 1; nxt[3 * i] = x / l; nxt[3 * i + 1] = y / l; nxt[3 * i + 2] = z / l;
    }
    cur = nxt;
  }
  return cur;
}

function vertexAreas(P, idx, nV) {
  const A = new Float32Array(nV);
  for (let t = 0; t < idx.length; t += 3) {
    const a = 3 * idx[t], b = 3 * idx[t + 1], c = 3 * idx[t + 2];
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const area = 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) / 3;
    A[idx[t]] += area; A[idx[t + 1]] += area; A[idx[t + 2]] += area;
  }
  return A;
}

async function loadBone(id) {
  if (S.busy) return;
  if (S.cur && S.dirty) await save(true);
  S.busy = true; $('loading').hidden = false;
  try {
    const d = await api(`/api/bones/${encodeURIComponent(id)}`);
    const P = new Float32Array(b64ToBytes(d.positions).buffer);
    const N = new Float32Array(b64ToBytes(d.normals).buffer);
    const I = new Uint32Array(b64ToBytes(d.indices).buffer);
    const C = b64ToBytes(d.colors);
    const nV = P.length / 3;
    const adj = buildAdjacency(nV, I);
    S.cur = {
      id: d.id, label: d.label, nV, P, N, I, C, adj,
      Ns: smoothNormals(N, adj, 3), area: vertexAreas(P, I, nV),
      labels: b64ToBytes(d.labels), names: Object.fromEntries(Object.entries(d.names).map(([k, v]) => [+k, v])),
    };
    S.undo = []; S.redo = []; setDirty(false);
    const ids = surfaceIds();
    S.active = ids.length ? ids[0] : 1;

    if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(P, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(N, 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(nV * 3), 3));
    g.setIndex(new THREE.BufferAttribute(I, 1));
    g.computeBoundingSphere();
    mesh = new THREE.Mesh(g, material);
    const r = d.rotation;   // najdłuższa oś poziomo, najcieńsza do kamery
    mesh.matrixAutoUpdate = false;
    mesh.matrix.set(r[0][0], r[0][1], r[0][2], 0, r[1][0], r[1][1], r[1][2], 0, r[2][0], r[2][1], r[2][2], 0, 0, 0, 0, 1);
    mesh.updateMatrixWorld(true);
    scene.add(mesh);
    resetView();
    recolor(); renderSurfaces(); renderBoneList(); updateStats();
    const b = S.bones.find((x) => x.id === id);
    $('boneTitle').innerHTML = `<b>${b.nr}. ${escapeHtml(d.label)}</b> &nbsp;·&nbsp; ${b.scans} skan(y) · ${fmt(b.area_mm2)} mm²`;
    history.replaceState(null, '', `#${encodeURIComponent(id)}`);
    if (d.remapped) toast('Siatka kości zmieniła się od zapisu - oznaczenia przeniesiono po położeniu. Sprawdź je i zapisz.', 'error');
  } catch (e) {
    toast(`Nie udało się wczytać kości: ${e.message}`, 'error');
  } finally {
    S.busy = false; $('loading').hidden = true;
  }
}

function resetView(flip = false) {
  if (!mesh) return;
  const s = mesh.geometry.boundingSphere;
  const dist = s.radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.05;
  const center = s.center.clone().applyMatrix4(mesh.matrixWorld);
  camera.position.set(center.x, center.y, center.z + (flip ? -dist : dist));
  camera.up.set(0, 1, 0);
  camera.near = dist / 100; camera.far = dist * 10; camera.updateProjectionMatrix();
  controls.target.copy(center); controls.update();
}

// ---------------------------------------------------------------- kolory i statystyki
function recolor() {
  const c = S.cur; if (!c) return;
  const col = mesh.geometry.attributes.color.array;
  const surf = {}; for (const k of surfaceIds(true)) surf[k] = hexLin(surfColor(k));
  for (let i = 0; i < c.nV; i++) {
    let r, g, b;
    if (S.scanColors) { r = SRGB_LIN[c.C[3 * i]]; g = SRGB_LIN[c.C[3 * i + 1]]; b = SRGB_LIN[c.C[3 * i + 2]]; }
    else { [r, g, b] = boneLin; }
    const k = c.labels[i];
    if (k) { const s = surf[k] || hexLin(surfColor(k)); r += (s[0] - r) * MIX; g += (s[1] - g) * MIX; b += (s[2] - b) * MIX; }
    col[3 * i] = r; col[3 * i + 1] = g; col[3 * i + 2] = b;
  }
  mesh.geometry.attributes.color.needsUpdate = true;
}

function surfaceIds(includeNamed = true) {
  const c = S.cur; if (!c) return [];
  const seen = new Set();
  for (let i = 0; i < c.nV; i++) if (c.labels[i]) seen.add(c.labels[i]);
  if (includeNamed) for (const k of Object.keys(c.names)) seen.add(+k);
  return [...seen].sort((a, b) => a - b);
}

function surfaceAreas() {
  const c = S.cur; const out = {};
  for (let i = 0; i < c.nV; i++) { const k = c.labels[i]; if (k) out[k] = (out[k] || 0) + c.area[i]; }
  return out;
}

function updateStats() {
  const c = S.cur; if (!c) return;
  const areas = surfaceAreas(); const marked = Object.values(areas).reduce((a, b) => a + b, 0);
  const total = c.area.reduce((a, b) => a + b, 0);
  $('stats').textContent = `${fmt(c.nV)} wierzchołków · pole ${fmt(total)} mm² · oznaczono ${fmt(marked)} mm² (${fmt(100 * marked / total)}%)`;
}

// ---------------------------------------------------------------- listy
const escapeHtml = (s) => String(s).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));

function renderBoneList() {
  const q = $('boneSearch').value.trim().toLowerCase();
  const ul = $('boneList'); ul.innerHTML = '';
  for (const b of S.bones) {
    if (q && !(`${b.nr} ${b.label} ${b.id}`.toLowerCase().includes(q))) continue;
    const li = document.createElement('li');
    if (S.cur && b.id === S.cur.id) li.className = 'active';
    const n = b.surfaces.length;
    const [title, part] = b.label.includes(' / ') ? b.label.split(' / ', 2) : [b.label, ''];
    li.title = `${b.label}\n${b.id}`;
    li.innerHTML = `<span class="nr">${b.nr}.</span>
      <span class="name"><b>${escapeHtml(title)}</b><small>${part ? `${escapeHtml(part)} · ` : ''}${b.scans} skan(y) · ${n ? `${n} pow. styku` : 'brak oznaczeń'}</small></span>
      <i class="dot ${n ? 'done' : ''}"></i>`;
    li.onclick = () => { if (!S.cur || b.id !== S.cur.id) loadBone(b.id); };
    ul.appendChild(li);
  }
  $('boneCount').textContent = `${S.bones.filter((b) => b.surfaces.length).length}/${S.bones.length} oznaczonych`;
}

function renderSurfaces() {
  const ul = $('surfaceList'); ul.innerHTML = '';
  const c = S.cur; if (!c) return;
  const areas = surfaceAreas();
  const ids = surfaceIds();
  if (!ids.includes(S.active)) ids.push(S.active);
  ids.sort((a, b) => a - b);
  for (const k of ids) {
    const li = document.createElement('li');
    if (k === S.active) li.className = 'active';
    const name = c.names[k] || `Powierzchnia ${k}`;
    li.innerHTML = `<span class="chip" style="background:${surfColor(k)}"></span>
      <span class="label" title="Kliknij dwa razy, żeby zmienić nazwę">${escapeHtml(name)}</span>
      <span class="area">${areas[k] ? `${fmt(areas[k], 1)} mm²` : 'pusta'}</span>
      <button class="del" title="Usuń tę powierzchnię">✕</button>`;
    li.onclick = (e) => { if (e.target.closest('.del') || e.target.tagName === 'INPUT') return; S.active = k; renderSurfaces(); updateCursorColor(); };
    li.querySelector('.label').ondblclick = () => renameSurface(li, k);
    li.querySelector('.del').onclick = () => deleteSurface(k);
    ul.appendChild(li);
  }
}

function renameSurface(li, k) {
  const span = li.querySelector('.label');
  const input = document.createElement('input');
  input.value = S.cur.names[k] || `Powierzchnia ${k}`;
  span.replaceWith(input); input.focus(); input.select();
  const done = (ok) => {
    if (ok && input.value.trim()) { pushUndo(); S.cur.names[k] = input.value.trim(); setDirty(true); }
    renderSurfaces();
  };
  input.onkeydown = (e) => { e.stopPropagation(); if (e.key === 'Enter') done(true); if (e.key === 'Escape') done(false); };
  input.onblur = () => done(true);
}

function newSurface() {
  if (!S.cur) return;
  const ids = surfaceIds();
  const k = Math.max(0, ...ids, S.active) + 1;
  if (k > 255) return toast('Za dużo powierzchni', 'error');
  pushUndo();
  S.cur.names[k] = `Powierzchnia ${k}`;
  S.active = k; setDirty(true); renderSurfaces(); updateCursorColor();
}

function deleteSurface(k) {
  const c = S.cur;
  const area = surfaceAreas()[k] || 0;
  if (area > 0 && !confirm(`Usunąć „${c.names[k] || `Powierzchnia ${k}`}” (${fmt(area, 1)} mm²)?`)) return;
  pushUndo();
  for (let i = 0; i < c.nV; i++) if (c.labels[i] === k) c.labels[i] = 0;
  delete c.names[k];
  if (S.active === k) S.active = surfaceIds()[0] || 1;
  setDirty(true); afterEdit();
}

// ---------------------------------------------------------------- cofanie
function snapshot() { return { labels: S.cur.labels.slice(), names: { ...S.cur.names } }; }
function pushUndo() { S.undo.push(snapshot()); if (S.undo.length > MAX_UNDO) S.undo.shift(); S.redo = []; }
function restore(from, to) {
  if (!S.cur || !from.length) return;
  to.push(snapshot());
  const s = from.pop(); S.cur.labels = s.labels; S.cur.names = s.names;
  if (!surfaceIds().includes(S.active)) S.active = surfaceIds()[0] || 1;
  setDirty(true); afterEdit();
}
function afterEdit() { recolor(); renderSurfaces(); updateStats(); updateButtons(); }
function setDirty(v) { S.dirty = v; $('dirtyBadge').hidden = !v; updateButtons(); }
function updateButtons() {
  $('undoBtn').disabled = !S.undo.length; $('redoBtn').disabled = !S.redo.length;
  $('saveBtn').disabled = !S.cur;
}

// ---------------------------------------------------------------- malowanie
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const invMatrix = new THREE.Matrix4();

function pick(ev) {
  if (!mesh) return null;
  const r = canvas.getBoundingClientRect();
  ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const hit = raycaster.intersectObject(mesh, false)[0];
  if (!hit) return null;
  invMatrix.copy(mesh.matrixWorld).invert();
  const local = hit.point.clone().applyMatrix4(invMatrix);
  return { world: hit.point, local, normal: hit.face.normal.clone(), face: hit.face, viewNormal: hit.face.normal.clone().transformDirection(mesh.matrixWorld) };
}

function paintSphere(p, n, label) {
  const c = S.cur, P = c.P, N = c.N, L = c.labels, r2 = S.brush * S.brush;
  let changed = false;
  for (let i = 0, k = 0; i < c.nV; i++, k += 3) {
    const dx = P[k] - p.x, dy = P[k + 1] - p.y, dz = P[k + 2] - p.z;
    if (dx * dx + dy * dy + dz * dz > r2) continue;
    if (N[k] * n.x + N[k + 1] * n.y + N[k + 2] * n.z < -0.2) continue;   // druga ścianka cienkiej kości
    if (L[i] !== label) { L[i] = label; changed = true; }
  }
  return changed;
}

function strokeTo(hit) {
  const label = S.tool === 'eraser' ? 0 : S.active;
  let changed = false;
  const last = S.lastHit;
  if (last) {
    const d = last.local.distanceTo(hit.local);
    if (d > S.brush * 0.5 && d < S.brush * 6) {         // uzupełnij przerwy przy szybkim ruchu
      const steps = Math.ceil(d / (S.brush * 0.5));
      for (let s = 1; s < steps; s++) {
        const t = s / steps;
        changed = paintSphere(last.local.clone().lerp(hit.local, t), last.normal.clone().lerp(hit.normal, t), label) || changed;
      }
    }
  }
  changed = paintSphere(hit.local, hit.normal, label) || changed;
  S.lastHit = hit;
  if (changed) { recolor(); setDirty(true); }
}

function wandAt(hit, erase) {
  const c = S.cur;
  const f = hit.face;
  let seed = f.a, best = Infinity;
  for (const v of [f.a, f.b, f.c]) {
    const d = (c.P[3 * v] - hit.local.x) ** 2 + (c.P[3 * v + 1] - hit.local.y) ** 2 + (c.P[3 * v + 2] - hit.local.z) ** 2;
    if (d < best) { best = d; seed = v; }
  }
  const cosT = Math.cos(THREE.MathUtils.degToRad(S.wand));
  const label = erase ? 0 : S.active;
  const L = c.labels, Ns = c.Ns, adj = c.adj;
  const mine = (v) => erase ? L[v] !== 0 : (L[v] === 0 || L[v] === label);   // nie nadpisuje innych powierzchni
  if (!mine(seed)) return toast('Tu jest już inna powierzchnia - wybierz ją na liście albo użyj gumki');
  pushUndo();
  const seen = new Uint8Array(c.nV); const queue = new Uint32Array(c.nV);
  let head = 0, tail = 0; queue[tail++] = seed; seen[seed] = 1;
  while (head < tail) {
    const u = queue[head++];
    L[u] = label;
    for (let k = adj.start[u]; k < adj.start[u + 1]; k++) {
      const v = adj.nb[k];
      if (seen[v] || !mine(v)) continue;
      const dot = Ns[3 * u] * Ns[3 * v] + Ns[3 * u + 1] * Ns[3 * v + 1] + Ns[3 * u + 2] * Ns[3 * v + 2];
      if (dot < cosT) continue;
      seen[v] = 1; queue[tail++] = v;
    }
  }
  setDirty(true); afterEdit();
  const frac = tail / c.nV;
  if (!erase && frac > 0.4) toast(`Różdżka zaznaczyła ${fmt(100 * frac)}% kości - za dużo? Ctrl+Z i zmniejsz tolerancję.`, 'error');
}

canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('pointerdown', (ev) => {
  if (ev.button !== 0 || !S.cur || S.busy) return;
  const hit = pick(ev);
  if (!hit) return;
  if (S.tool === 'wand') return wandAt(hit, ev.shiftKey || ev.altKey);
  canvas.setPointerCapture(ev.pointerId);
  pushUndo(); S.painting = true; S.lastHit = null;
  strokeTo(hit);
});
canvas.addEventListener('pointermove', (ev) => {
  const hit = pick(ev);
  placeCursor(hit);
  if (S.painting && hit) strokeTo(hit);
});
const endStroke = () => {
  if (!S.painting) return;
  S.painting = false; S.lastHit = null;
  if (S.undo.length && S.cur && arraysEqual(S.undo[S.undo.length - 1].labels, S.cur.labels)) S.undo.pop();  // pusty ruch
  renderSurfaces(); updateStats(); updateButtons();
};
canvas.addEventListener('pointerup', endStroke);
canvas.addEventListener('pointercancel', endStroke);
canvas.addEventListener('pointerleave', () => (cursor.visible = false));

function arraysEqual(a, b) { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; }

function placeCursor(hit) {
  if (!hit) { cursor.visible = false; return; }
  cursor.visible = true;
  const r = S.tool === 'wand' ? Math.max(0.3, mesh.geometry.boundingSphere.radius * 0.015) : S.brush;
  cursor.scale.setScalar(r);
  cursor.position.copy(hit.world);
  cursor.lookAt(hit.world.clone().add(hit.viewNormal));
}

function updateCursorColor() {
  cursor.material.color.set(S.tool === 'eraser' ? '#1f2a36' : surfColor(S.active));
}

// ---------------------------------------------------------------- narzędzia
const TIPS = {
  brush: 'Maluj lewym przyciskiem. Obracaj prawym. Pędzel maluje też po innych powierzchniach.',
  eraser: 'Usuwa oznaczenia lewym przyciskiem (dowolnej powierzchni).',
  wand: 'Kliknij przełom: zaznacza obszar aż do ostrej krawędzi. Shift + klik usuwa obszar. Za dużo? Ctrl+Z i zmniejsz tolerancję.',
};
const HINTS = {
  brush: '<b>Lewy</b>: maluj · <b>Prawy</b>: obracaj · <b>Środkowy</b>: przesuń · <b>Kółko</b>: zoom · <b>[ ]</b>: rozmiar',
  eraser: '<b>Lewy</b>: usuń oznaczenie · <b>Prawy</b>: obracaj · <b>Kółko</b>: zoom · <b>[ ]</b>: rozmiar',
  wand: '<b>Klik</b>: zaznacz obszar · <b>Shift+klik</b>: usuń obszar · <b>Prawy</b>: obracaj · <b>[ ]</b>: tolerancja',
};
function setTool(t) {
  S.tool = t;
  document.querySelectorAll('.tool').forEach((b) => b.classList.toggle('active', b.dataset.tool === t));
  $('brushRow').hidden = t === 'wand'; $('wandRow').hidden = t !== 'wand';
  $('toolTip').textContent = TIPS[t]; $('hint').innerHTML = HINTS[t];
  updateCursorColor();
}
function setBrush(v) { S.brush = Math.min(10, Math.max(0.2, v)); $('brushSize').value = S.brush; $('brushVal').textContent = `${fmt(S.brush, 1)} mm`; }
function setWand(v) { S.wand = Math.min(70, Math.max(5, Math.round(v))); $('wandAngle').value = S.wand; $('wandVal').textContent = `${S.wand}°`; }

document.querySelectorAll('.tool').forEach((b) => (b.onclick = () => setTool(b.dataset.tool)));
$('brushSize').oninput = (e) => setBrush(+e.target.value);
$('wandAngle').oninput = (e) => setWand(+e.target.value);
$('scanColors').onchange = (e) => { S.scanColors = e.target.checked; recolor(); };
$('resetView').onclick = () => resetView(false);
$('flipView').onclick = () => resetView(camera.position.z > controls.target.z);
$('newSurface').onclick = newSurface;
$('undoBtn').onclick = () => restore(S.undo, S.redo);
$('redoBtn').onclick = () => restore(S.redo, S.undo);
$('clearBtn').onclick = () => {
  if (!S.cur || !surfaceIds(false).length) return;
  if (!confirm('Usunąć wszystkie oznaczenia tej kości? (Ctrl+Z cofa)')) return;
  pushUndo(); S.cur.labels.fill(0); S.cur.names = {}; S.active = 1; setDirty(true); afterEdit();
};
$('saveBtn').onclick = () => save(false);
$('detectBtn').onclick = detect;
$('boneSearch').oninput = renderBoneList;
$('helpBtn').onclick = () => ($('help').hidden = false);
$('helpClose').onclick = () => { $('help').hidden = true; try { localStorage.setItem('contactToolHelpSeen', '1'); } catch (e) { /* brak dostępu */ } };
$('help').onclick = (e) => { if (e.target.id === 'help') $('help').hidden = true; };

async function save(silent) {
  const c = S.cur; if (!c) return;
  const ids = new Set(surfaceIds(false));
  const names = {}; for (const [k, v] of Object.entries(c.names)) if (ids.has(+k)) names[k] = v;
  try {
    const r = await api(`/api/bones/${encodeURIComponent(c.id)}/contacts`, { method: 'PUT', body: JSON.stringify({ labels: bytesToB64(c.labels), names }) });
    const b = S.bones.find((x) => x.id === c.id);
    b.surfaces = r.surfaces;
    setDirty(false); renderBoneList();
    const total = r.surfaces.reduce((a, s) => a + s.area_mm2, 0);
    toast(r.saved ? `Zapisano ✓  ${r.surfaces.length} pow. styku, ${fmt(total, 1)} mm²` : 'Zapisano ✓  (brak oznaczeń - plik usunięty)', 'ok');
  } catch (e) {
    toast(`Błąd zapisu: ${e.message}`, 'error');
    if (silent) throw e;
  }
}

async function detect() {
  const c = S.cur; if (!c || S.busy) return;
  if (surfaceIds(false).length && !confirm('Zastąpić obecne oznaczenia propozycją algorytmu? (Ctrl+Z cofa)')) return;
  S.busy = true; $('loading').hidden = false; $('loading').querySelector('span').textContent = 'Algorytm szuka przełomów…';
  try {
    const r = await api(`/api/bones/${encodeURIComponent(c.id)}/detect`, { method: 'POST', body: '{}' });
    pushUndo();
    c.labels = b64ToBytes(r.labels);
    c.names = Object.fromEntries(Object.entries(r.names).map(([k, v]) => [+k, v]));
    S.active = surfaceIds()[0] || 1; setDirty(true); afterEdit();
    const n = surfaceIds(false).length;
    toast(n ? `Algorytm „${r.method}” zaproponował ${n} pow. - popraw i zapisz` : 'Algorytm niczego nie znalazł', n ? 'ok' : '');
  } catch (e) {
    toast(`Błąd algorytmu: ${e.message}`, 'error');
  } finally {
    S.busy = false; $('loading').hidden = true; $('loading').querySelector('span').textContent = 'Wczytywanie kości…';
  }
}

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' && e.target.type !== 'range' && e.target.type !== 'checkbox') return;
  const k = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && k === 's') { e.preventDefault(); save(false); return; }
  if ((e.ctrlKey || e.metaKey) && k === 'z') { e.preventDefault(); e.shiftKey ? restore(S.redo, S.undo) : restore(S.undo, S.redo); return; }
  if ((e.ctrlKey || e.metaKey) && k === 'y') { e.preventDefault(); restore(S.redo, S.undo); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (k === 'b') setTool('brush');
  else if (k === 'e') setTool('eraser');
  else if (k === 'w') setTool('wand');
  else if (k === '[') S.tool === 'wand' ? setWand(S.wand - 3) : setBrush(S.brush * 0.8);
  else if (k === ']') S.tool === 'wand' ? setWand(S.wand + 3) : setBrush(S.brush * 1.25);
  else if (k === 'n') newSurface();
  else if (k === 'r') resetView(false);
  else if (k === 'f') resetView(camera.position.z > controls.target.z);
  else if (k === 'c') { $('scanColors').checked = !$('scanColors').checked; $('scanColors').onchange({ target: $('scanColors') }); }
  else if (k === 'h' || k === '?') $('help').hidden = !$('help').hidden;
  else if (k === 'escape') $('help').hidden = true;
  else if (/^[1-9]$/.test(k)) { S.active = +k; renderSurfaces(); updateCursorColor(); }
  else if (e.key === 'PageDown' || e.key === 'PageUp') {
    e.preventDefault();
    const i = S.bones.findIndex((b) => S.cur && b.id === S.cur.id);
    const j = Math.min(S.bones.length - 1, Math.max(0, i + (e.key === 'PageDown' ? 1 : -1)));
    if (j !== i) loadBone(S.bones[j].id);
  }
});
window.addEventListener('beforeunload', (e) => { if (S.dirty) { e.preventDefault(); e.returnValue = ''; } });

// ---------------------------------------------------------------- start
(async function init() {
  try {
    const d = await api('/api/bones');
    S.bones = d.bones; S.palette = d.palette; S.detectors = d.detectors;
    setBrush(d.params.brush_mm); setWand(d.params.wand_angle_deg); setTool('brush');
    $('detectBtn').hidden = !S.detectors.length;
    renderBoneList(); updateButtons();
    const fromHash = decodeURIComponent(location.hash.slice(1));
    const first = S.bones.find((b) => b.id === fromHash) || S.bones.find((b) => !b.surfaces.length) || S.bones[0];
    let seen = false; try { seen = !!localStorage.getItem('contactToolHelpSeen'); } catch (e) { /* brak dostępu */ }
    if (!seen) $('help').hidden = false;
    if (first) await loadBone(first.id);
  } catch (e) {
    $('loading').hidden = true;
    toast(`Nie można połączyć z serwerem: ${e.message}. Czy python scripts/mark_contacts.py nadal działa?`, 'error');
  }
})();

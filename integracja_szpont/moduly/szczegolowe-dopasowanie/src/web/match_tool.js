// Podgląd dopasowania z suwakami (serwer: src/match_tool.py).
//
// Przebieg: main() pobiera pary i ustawienia (GET /api/pairs) -> buildSliders() ->
// selectPair() rysuje parę (GET /api/geometry dla obu powierzchni) -> zmiana suwaka ->
// requestCompute() -> runPending() (POST /api/match) -> draw() + renderResult().
//
// Stan (obiekt `state`):
//   pairs       - pary z matching.json, od najlepszej (wyniki i T z pliku),
//   base        - ustawienia, z którymi policzono matching.json (startowe położenie suwaków),
//   params      - obecne położenie suwaków,
//   computed    - {klucz pary: {result, params}} - wyniki przeliczone na tej stronie,
//   geometry    - {klucz powierzchni: siatki} - pobierane raz,
//   current     - indeks pokazanej pary w `pairs`,
//   busy        - trwa zapytanie POST /api/match, pending - ostatnie żądanie czekające na swoją kolej,
//   ruler       - linijka w widoku 3D (src/web/ruler.js; przycisk "📏 Linijka" albo klawisz L).
// Zmiana suwaka przelicza TYLKO pokazaną parę. Gdy zmieniło się któreś z ustawień
// szukania (SEARCH), para jest dopasowywana od nowa; gdy tylko ustawienia oceny -
// oceniane jest obecne ustawienie (szybko). Zapytania nie nakładają się: w trakcie
// liczenia zapamiętywane jest tylko ostatnie żądanie.

const COLORS = { a: "#6f9fd8", b: "#e0c9a0", surf_a: "#e8508f", surf_b: "#18b0a6" };
const LIGHTING = { ambient: 0.55, diffuse: 0.75, specular: 0.12, roughness: 0.65, fresnel: 0.05 };
const ITERATIONS = [10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2000000];   // wartości suwaka iteracji

// Suwaki ustawień z sekcji "matching" (klucze = pola SurfaceMatchParams):
//   group: "search" = zmiana wymaga dopasowania od nowa, "score" = wystarczy nowa ocena,
//   min / max / step - zakres suwaka (rozszerzany, gdy wartość z matching.json jest poza nim),
//   values - zamiast min/max: lista dozwolonych wartości (suwak wybiera indeks),
//   unit, help - opis pod suwakiem, needs - suwak widoczny tylko, gdy któreś z tych ustawień jest włączone
//   (np. ["use_atlas"] - atlas, ["axis_check"] - osie).
//   buildSliders dopisuje: input (element <input>), out (pole wartości).
const SLIDERS = [
  { group: "search", key: "sample_au", label: "Odstęp punktów powierzchni", unit: "a.u.", min: 0.15, max: 1, step: 0.05,
    help: "Mniej = dokładniej i wolniej. Wpływa też na ocenę." },
  { group: "search", key: "coarse_voxel_au", label: "Odstęp punktów RANSAC", unit: "a.u.", min: 0.4, max: 3, step: 0.1,
    help: "Rzadsze punkty zgrubnego ustawienia = szybciej, mniej dokładnie." },
  { group: "search", key: "fpfh_radius_au", label: "Promień FPFH", unit: "a.u.", min: 1, max: 10, step: 0.25,
    help: "≈ wielkość nierówności przełomu, po których rozpoznawany jest kształt." },
  { group: "search", key: "ransac_iterations", label: "Iteracje RANSAC", values: ITERATIONS,
    help: "Więcej = pewniej trafia dobre ustawienie, czas rośnie liniowo." },
  { group: "search", key: "ransac_seeds", label: "Próby RANSAC + ICP", min: 1, max: 8, step: 1,
    help: "Niezależne próby, wygrywa najlepsza." },
  { group: "search", key: "ransac_normal_angle_deg", label: "RANSAC: maks. różnica normalnych", unit: "°",
    min: 5, max: 90, step: 1, help: "Odrzuca ustawienia z bardziej rozbieżnymi normalnymi." },
  { group: "search", key: "icp_start_au", label: "ICP: promień startowy", unit: "a.u.", min: 0.5, max: 6, step: 0.1,
    help: "Do jakiej odległości ICP szuka par punktów (potem połowa, na koniec 2 × odstęp)." },
  { group: "search", key: "axis_starts", label: "Osie: starty koniec do końca", min: 0, max: 36, step: 1,
    needs: ["axis_check"], help: "Oś A na osi B, przełom na przełomie, obrót wokół osi co 360° / tyle; 0 = bez." },
  { group: "search", key: "start_icp_au", label: "Starty z osi / atlasu: promień ICP", unit: "a.u.",
    min: 1, max: 15, step: 0.5, needs: ["axis_check", "use_atlas"],
    help: "Taki start jest przybliżony - pierwszy krok ICP szuka par punktów dalej." },
  { group: "score", key: "tolerance_au", label: "Styk: maks. odległość", unit: "a.u.", min: 0.1, max: 2, step: 0.05,
    help: "Punkty bliżej niż to liczą się do styku…" },
  { group: "score", key: "normal_angle_deg", label: "Styk: tolerancja kąta normalnych", unit: "°",
    min: 5, max: 90, step: 1, help: "…jeśli ich normalne są przeciwne z tą dokładnością." },
  { group: "score", key: "collision_au", label: "Kolizja: głębokość", unit: "a.u.", min: 0.2, max: 4, step: 0.1,
    help: "Punkt kości głębiej pod powierzchnią drugiej = kolizja." },
  { group: "score", key: "collision_search_au", label: "Kolizja: zasięg szukania", unit: "a.u.", min: 1, max: 10,
    step: 0.5, help: "Sprawdzane tylko tak blisko drugiej kości." },
  { group: "score", key: "axis_max_deg", label: "Osie: maks. kąt bez kary", unit: "°", min: 5, max: 60, step: 1,
    needs: ["axis_check"], help: "Kawałki kości długiej leżą w jednej linii; przy 2× tym kącie styk się nie liczy." },
  { group: "score", key: "axis_max_offset", label: "Osie: maks. rozsunięcie bez kary", unit: "promienia",
    min: 0.25, max: 3, step: 0.05, needs: ["axis_check"],
    help: "Odległość osi w miejscu styku w promieniach kości; koniec do boku ≈ 2." },
  { group: "score", key: "atlas_boost", label: "Atlas: premia za tę samą kość", min: 0, max: 1, step: 0.05,
    needs: ["use_atlas"], help: "Styk liczy się o tyle mocniej, gdy atlas mówi, że oba fragmenty są z tej samej kości." },
];
const SEARCH = SLIDERS.filter((s) => s.group === "search").map((s) => s.key);   // ustawienia szukania
const ALL = SLIDERS.map((s) => s.key);                                          // wszystkie ustawienia z suwaków

/** Element strony o danym id. */
const $ = (id) => document.getElementById(id);
const state = { pairs: [], base: {}, params: {}, computed: {}, geometry: {}, current: 0,
                busy: false, pending: null, ruler: null };

// ---------- pomocnicze ----------

/**
 * Zapytanie JSON do serwera podglądu.
 * @param {string} url      np. '/api/pairs'
 * @param {object} options  opcje fetch (method, headers, body, ...)
 * @returns {Promise<object>} odpowiedź serwera
 * @throws {Error} z komunikatem serwera ({"error": ...}), gdy odpowiedź nie jest OK
 */
async function api(url, options) {
  const r = await fetch(url, options);
  const data = await r.json();
  if (!r.ok || data.error) throw new Error(data.error || r.statusText);
  return data;
}

/**
 * Base64 z serwera (surowe bajty tablicy numpy, match_tool._b64) -> tablica typu `Type`.
 * @param {string} s        tekst base64
 * @param {Function} Type   np. Float32Array, Uint32Array - musi zgadzać się z typem po stronie Pythona
 * @returns {ArrayBufferView}
 */
function b64(s, Type) {
  const bin = atob(s);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}

let toastTimer;
/**
 * Dymek z komunikatem na dole strony (błąd wisi dłużej: 8 s zamiast 2,5 s).
 * @param {string} msg
 * @param {boolean} error  true = czerwony dymek błędu
 */
function toast(msg, error = false) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast" + (error ? " error" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), error ? 8000 : 2500);
}

/** Klucz pary "<klucz A>|<klucz B>" - indeks w state.computed. */
const pairKey = (p) => `${p.a}|${p.b}`;
/** Liczba z `d` miejscami po przecinku; brak wartości (null / NaN) -> "–". */
const fmt = (x, d = 0) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
/** Liczba miejsc po przecinku kroku suwaka, np. 0.05 -> 2. */
const decimals = (step) => (String(step).split(".")[1] || "").length;
/** Czy ustawienia `p` i `q` mają te same wartości dla wszystkich kluczy `keys`. */
const sameParams = (p, q, keys) => keys.every((k) => Number(p[k]) === Number(q[k]));

/**
 * Wartość suwaka do wyświetlenia: z jednostką i liczbą miejsc wg kroku
 * (iteracje - z separatorem tysięcy, bez jednostki).
 * @param {object} s  suwak z SLIDERS
 * @param {number} v  wartość
 * @returns {string}
 */
function fmtValue(s, v) {
  if (s.values) return Number(v).toLocaleString("pl-PL");
  return `${Number(v).toFixed(decimals(s.step))}${s.unit ? " " + s.unit : ""}`;
}

/**
 * Wynik pary pokazywany teraz: przeliczony na stronie albo z matching.json.
 * @param {number} k  indeks pary w state.pairs
 * @returns {{result: object, params: object}} wynik (score, T, ...) i ustawienia, z którymi go policzono
 */
function shown(k) {
  const p = state.pairs[k];
  return state.computed[pairKey(p)] || { result: p, params: state.base };
}

// ---------- suwaki ----------

/**
 * Tworzy suwaki z SLIDERS w panelach "Szukanie ustawienia" i "Ocena" i ustawia je na state.base.
 * Przesuwanie zmienia tylko wyświetlaną wartość; puszczenie suwaka zapisuje ją w state.params
 * i zleca przeliczenie pokazanej pary (requestCompute).
 */
function buildSliders() {
  for (const s of SLIDERS) {
    const box = document.createElement("div");
    box.className = "slider";
    const v0 = state.base[s.key];
    const input = document.createElement("input");
    input.type = "range";
    if (s.values) {
      if (!s.values.includes(v0)) s.values = [...s.values, v0].sort((x, y) => x - y);
      Object.assign(input, { min: 0, max: s.values.length - 1, step: 1 });
    } else {
      Object.assign(input, { min: Math.min(s.min, v0), max: Math.max(s.max, v0), step: s.step });
    }
    input.id = "s-" + s.key;
    box.innerHTML = `<div class="row"><label for="${input.id}">${s.label}</label><span class="v"></span></div>`;
    box.appendChild(input);
    box.insertAdjacentHTML("beforeend", `<div class="help">${s.help}</div>`);
    box.hidden = Boolean(s.needs && !s.needs.some((k) => state.base[k]));   // np. suwaki atlasu tylko z atlasem
    // obecna wartość suwaka (z listy `values` albo zaokrąglona do kroku)
    const value = () => (s.values ? s.values[Number(input.value)] : Number(Number(input.value).toFixed(decimals(s.step))));
    input.addEventListener("input", () => showValue(s, value()));
    input.addEventListener("change", () => {
      state.params[s.key] = value();
      updateSnippet();
      requestCompute(false);
    });
    s.input = input;
    s.out = box.querySelector(".v");
    $(s.group === "search" ? "slidersSearch" : "slidersScore").appendChild(box);
  }
  setSliders(state.base);
}

/**
 * Wyświetla wartość suwaka; wyróżnia ją, gdy różni się od matching.json.
 * @param {object} s  suwak z SLIDERS
 * @param {number} v  wartość
 */
function showValue(s, v) {
  s.out.textContent = fmtValue(s, v);
  s.out.classList.toggle("changed", Number(v) !== Number(state.base[s.key]));
}

/**
 * Ustawia wszystkie suwaki i state.params na podane wartości (bez przeliczania).
 * @param {object} values  {klucz ustawienia: wartość}, np. state.base
 */
function setSliders(values) {
  for (const s of SLIDERS) {
    const v = values[s.key];
    s.input.value = s.values ? s.values.indexOf(v) : v;
    showValue(s, v);
  }
  state.params = Object.fromEntries(ALL.map((k) => [k, values[k]]));
  updateSnippet();
}

/** Wypisuje zmienione ustawienia jako linie gotowe do wklejenia w settings.json (i pokazuje "Kopiuj"). */
function updateSnippet() {
  const changed = ALL.filter((k) => Number(state.params[k]) !== Number(state.base[k]));
  $("snippet").textContent = changed.length
    ? changed.map((k) => `"${k}": ${state.params[k]},`).join("\n")
    : "(bez zmian względem matching.json)";
  $("copy").hidden = !changed.length;
}

// ---------- lista par ----------

/**
 * Rysuje listę par (z filtrem z pola wyszukiwania). Przy parach przeliczonych na stronie:
 * stary wynik → nowy, z kolorem wzrostu / spadku (różnica co najmniej 0,5 jednostki wyniku).
 */
function renderList() {
  const q = $("pairSearch").value.trim().toLowerCase();
  const ul = $("pairList");
  ul.innerHTML = "";
  state.pairs.forEach((p, k) => {
    if (q && !`${p.a_name} ${p.b_name}`.toLowerCase().includes(q)) return;
    const li = document.createElement("li");
    li.classList.toggle("active", k === state.current);
    const c = state.computed[pairKey(p)];
    let score = `<b>${fmt(p.score)}</b>`;
    if (c) {
      const cls = c.result.score > p.score + 0.5 ? "up" : c.result.score < p.score - 0.5 ? "down" : "";
      score = `<span class="muted">${fmt(p.score)}</span><span class="new ${cls}">→ ${fmt(c.result.score)}</span>`;
    }
    li.innerHTML = `<span class="nr">${k + 1}.</span>
      <span class="names"><span title="${p.a_name}">${p.a_name}</span><span class="sep">↔</span>
      <span title="${p.b_name}">${p.b_name}</span></span><span class="score">${score}</span>`;
    li.addEventListener("click", () => selectPair(k));
    ul.appendChild(li);
  });
}

// ---------- 3D ----------

/**
 * Siatka kości i nakładka powierzchni (GET /api/geometry) - pobierana raz i pamiętana.
 * @param {string} key  klucz powierzchni "<id kości>#<numer>"
 * @returns {Promise<{name, label, pos: Float32Array, idx: Uint32Array, spos: Float32Array, sidx: Uint32Array}>}
 *          pos / idx - kość, spos / sidx - nakładka powierzchni; wszystko w układzie kości
 */
async function getGeometry(key) {
  if (!state.geometry[key]) {
    const g = await api(`/api/geometry?key=${encodeURIComponent(key)}`);
    state.geometry[key] = {
      name: g.name, label: g.label,
      pos: b64(g.positions, Float32Array), idx: b64(g.indices, Uint32Array),
      spos: b64(g.surf_positions, Float32Array), sidx: b64(g.surf_indices, Uint32Array),
    };
  }
  return state.geometry[key];
}

/**
 * Rozbija pozycje (V, 3) na osobne tablice x, y, z (format plotly), opcjonalnie przekształcone.
 * @param {Float32Array} pos  pozycje x0, y0, z0, x1, ...
 * @param {number[][]|null} T  macierz 4x4 (obrót + przesunięcie) albo null = bez zmian
 * @returns {{x: Float32Array, y: Float32Array, z: Float32Array}}
 */
function xyz(pos, T) {
  const n = pos.length / 3;
  const x = new Float32Array(n), y = new Float32Array(n), z = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const px = pos[3 * i], py = pos[3 * i + 1], pz = pos[3 * i + 2];
    if (T) {
      x[i] = T[0][0] * px + T[0][1] * py + T[0][2] * pz + T[0][3];
      y[i] = T[1][0] * px + T[1][1] * py + T[1][2] * pz + T[1][3];
      z[i] = T[2][0] * px + T[2][1] * py + T[2][2] * pz + T[2][3];
    } else {
      x[i] = px; y[i] = py; z[i] = pz;
    }
  }
  return { x, y, z };
}

/**
 * Ślad plotly mesh3d jednej siatki.
 * @param {Float32Array} pos     pozycje wierzchołków (V, 3)
 * @param {Uint32Array} idx      trójkąty (T, 3)
 * @param {number[][]|null} T    przekształcenie 4x4 (kość A) albo null (kość B)
 * @param {string} color         kolor "#rrggbb"
 * @param {string} name          nazwa w podpowiedzi po najechaniu
 * @returns {object} ślad dla Plotly.react
 */
function meshTrace(pos, idx, T, color, name) {
  const n = idx.length / 3;
  const i = new Uint32Array(n), j = new Uint32Array(n), k = new Uint32Array(n);
  for (let t = 0; t < n; t++) { i[t] = idx[3 * t]; j[t] = idx[3 * t + 1]; k[t] = idx[3 * t + 2]; }
  return { type: "mesh3d", ...xyz(pos, T), i, j, k, color, name, showlegend: false, flatshading: false,
           lighting: LIGHTING, lightposition: { x: -1e4, y: -2e4, z: 3e4 },
           hovertemplate: `<b>${name}</b><extra></extra>` };
}

/**
 * Rysuje pokazaną parę: kość B w swoim układzie, kość A przeniesiona przekształceniem T
 * z obecnego wyniku (shown), obie z nakładkami powierzchni. Kamera zostaje przy
 * przeliczaniu tej samej pary (uirevision).
 * @returns {Promise<void>}
 */
async function draw() {
  const k = state.current;
  const p = state.pairs[k];
  const [ga, gb] = await Promise.all([getGeometry(p.a), getGeometry(p.b)]);
  if (k !== state.current) return;                    // w międzyczasie wybrano inną parę
  const T = shown(k).result.T;
  const traces = [meshTrace(gb.pos, gb.idx, null, COLORS.b, "B: " + gb.label)];
  if (gb.sidx.length) traces.push(meshTrace(gb.spos, gb.sidx, null, COLORS.surf_b, gb.name));
  traces.push(meshTrace(ga.pos, ga.idx, T, COLORS.a, "A: " + ga.label));
  if (ga.sidx.length) traces.push(meshTrace(ga.spos, ga.sidx, T, COLORS.surf_a, ga.name));
  const axis = { visible: false };
  await Plotly.react("plot", traces, {
    paper_bgcolor: "#f4f6f8", margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: { xaxis: axis, yaxis: axis, zaxis: axis, aspectmode: "data" },
    uirevision: pairKey(p),                           // kamera zostaje przy przeliczaniu, zmienia się z parą
  }, { displaylogo: false, scrollZoom: true, responsive: true });
  // linijka (src/web/ruler.js): po każdym przerysowaniu pomiary znikają - kość A mogła się przesunąć
  if (!state.ruler) state.ruler = attachRuler($("plot"), { place: "right" });
  else state.ruler.clear();
}

// ---------- wynik ----------

/**
 * Karta wyniku pokazanej pary (wynik, styk, pokrycie, RMSE, kolizja) z poprzednimi
 * wartościami z matching.json, gdy się zmieniły, i tytuł pary. Pokazuje ramkę
 * "Przelicz z obecnymi", gdy suwaki różnią się od ustawień pokazanego wyniku.
 */
function renderResult() {
  const k = state.current;
  const p = state.pairs[k];
  const { result: r, params } = shown(k);
  const c = state.computed[pairKey(p)];
  // "było X" - tylko dla przeliczonej pary i tylko gdy zmianę widać po zaokrągleniu do `d` miejsc
  const was = (now, old, d = 0, unit = "") =>
    c && Math.abs((now ?? 0) - (old ?? 0)) >= 0.5 * 10 ** -d
      ? `<span class="was">było ${fmt(old, d)}${unit}</span>` : "";
  const row = (lbl, val) => `<span class="lbl">${lbl}</span><span class="val">${val}</span>`;
  const unit = r.score_unit ?? p.score_unit ?? "a.u.²";        // "%" = udział w obu przełomach, "a.u.²" = pole
  $("resultCard").innerHTML = [
    row("wynik = (styk" + (r.axis_weight !== undefined ? " × waga osi" : "")
        + (r.atlas_same !== undefined ? " × premia" : "") + " − kolizja)"
        + (unit === "%" ? " / średnie pole przełomów" : ""),
        `<span class="big">${fmt(r.score)}</span> ${unit}${was(r.score, p.score)}`),
    row("styk", `${fmt(r.contact_au2)} a.u.²${was(r.contact_au2, p.contact_au2)}`),
    row("pokrycie A / B", `${fmt(100 * r.cover_a)}% / ${fmt(100 * r.cover_b)}%`),
    row("RMSE w styku", `${fmt(r.rmse_au, 2)} a.u.${was(r.rmse_au, p.rmse_au, 2)}`),
    row("kolizja kości", `${fmt(r.collision_au2)} a.u.²${was(r.collision_au2, p.collision_au2)}`),
    row("pola powierzchni A / B", `${fmt(p.a_area_au2)} / ${fmt(p.b_area_au2)} a.u.²`),
    ...(r.axis_weight !== undefined ? [
      row("osie: kąt / rozsunięcie", `${fmt(r.axis_deg)}° / ${fmt(r.axis_offset, 1)} promienia → waga ${fmt(100 * r.axis_weight)}%`)]
      : []),
    ...(r.atlas_same !== undefined ? [
      row("atlas: ta sama kość", `${fmt(100 * r.atlas_same)}% (różne ${fmt(100 * r.atlas_diff)}%)`),
      row("premia z atlasu", `+${fmt(r.atlas_bonus)} ${unit}`)] : []),
    ...(r.start ? [row("start", `${r.start}${r.ransac === false ? " (bez RANSAC)" : ""}`)] : []),
    `<span class="foot">${c
      ? (c.result.mode === "full" ? "Dopasowane od nowa" : "Nowa ocena tego samego ustawienia")
        + ` w ${fmt(c.result.seconds, 1)} s` + (c.result.seed !== null && c.result.seed !== undefined
          ? `, wygrała próba ${c.result.seed + 1}` + (c.result.start ? ` (${c.result.start})` : "") : "")
      : "Wynik z matching.json."}</span>`,
  ].join("");
  $("stale").hidden = sameParams(params, state.params, ALL) || state.busy;
  $("pairTitle").innerHTML = `${k + 1}. <b>${p.a_name}</b> ↔ <b>${p.b_name}</b>`;
}

// ---------- przeliczanie ----------

/**
 * Zleca przeliczenie pokazanej pary. Gdy trwa inne liczenie, zapamiętywane jest tylko
 * ostatnie żądanie (force z wcześniejszego żądania tej samej pary nie ginie).
 * @param {boolean} force  dopasowanie od nowa nawet bez zmiany ustawień szukania (RANSAC jest losowy)
 */
function requestCompute(force) {
  const prev = state.pending;
  state.pending = { k: state.current, force: force || (prev && prev.k === state.current && prev.force) };
  if (!state.busy) runPending();
}

/**
 * Wykonuje czekające żądanie (POST /api/match), potem kolejne, jeśli przyszło w trakcie.
 * Tryb "full" (dopasowanie od nowa), gdy force albo zmieniły się ustawienia szukania
 * względem pokazanego wyniku; inaczej "score" - ocena obecnego T z nowymi ustawieniami oceny.
 * Błąd jest pokazywany w dymku, a stary wynik zostaje.
 * @returns {Promise<void>}
 */
async function runPending() {
  const job = state.pending;
  state.pending = null;
  if (!job) return;
  const p = state.pairs[job.k];
  const cur = shown(job.k);
  const params = { ...state.params };
  const full = job.force || !sameParams(cur.params, params, SEARCH);
  const body = { a: p.a, b: p.b, params, mode: full ? "full" : "score" };
  if (!full) Object.assign(body, { T: cur.result.T, seed: cur.result.seed, start: cur.result.start });
  setBusy(true, full
    ? `Dopasowuję od nowa… (RANSAC ${params.ransac_iterations.toLocaleString("pl-PL")} × ${params.ransac_seeds})`
    : "Oceniam…");
  try {
    const r = await api("/api/match", { method: "POST", headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify(body) });
    state.computed[pairKey(p)] = { result: r, params };
    if (job.k === state.current) await draw();
  } catch (e) {
    toast("Błąd przeliczania: " + e.message, true);
  } finally {
    setBusy(false);
    renderList();
    renderResult();
    if (state.pending) runPending();
  }
}

/**
 * Włącza / wyłącza stan "liczę…" (nakładka na wykresie, status, zablokowany "Przelicz od nowa").
 * @param {boolean} on
 * @param {string} text  opis na nakładce, np. "Oceniam…"
 */
function setBusy(on, text = "") {
  state.busy = on;
  $("busy").hidden = !on;
  $("busyText").textContent = text;
  $("status").hidden = !on;
  $("status").textContent = "liczę…";
  $("recompute").disabled = on;
  if (on) $("stale").hidden = true;
}

/**
 * Pokazuje parę k: lista, karta wyniku i wykres 3D (błąd wczytania -> dymek).
 * @param {number} k  indeks pary w state.pairs
 * @returns {Promise<void>}
 */
async function selectPair(k) {
  state.current = k;
  renderList();
  renderResult();
  try {
    await draw();
  } catch (e) {
    toast("Błąd wczytywania kości: " + e.message, true);
  }
}

// ---------- start ----------

/**
 * Start strony: pobiera pary i ustawienia, buduje suwaki, podpina przyciski
 * (Przelicz od nowa, Przywróć ustawienia, Kopiuj) i strzałki góra / dół, pokazuje pierwszą parę.
 * @returns {Promise<void>}
 */
async function main() {
  const data = await api("/api/pairs");
  state.pairs = data.pairs;
  state.base = data.params;
  $("pairCount").textContent = `(${state.pairs.length})`;
  const warnings = data.warnings || [];                 // matching.json nieaktualny (MatchState.warnings)
  $("warnings").hidden = !warnings.length;
  $("warnings").textContent = warnings.map((w) => "⚠ " + w).join("\n");
  buildSliders();
  $("pairSearch").addEventListener("input", renderList);
  $("recompute").addEventListener("click", () => requestCompute(true));
  $("recomputeStale").addEventListener("click", () => requestCompute(false));
  $("reset").addEventListener("click", () => {          // suwaki i pokazana para wracają do matching.json
    setSliders(state.base);
    if (!state.busy) delete state.computed[pairKey(state.pairs[state.current])];
    renderList();
    renderResult();
    draw();
  });
  $("copy").addEventListener("click", async () => {
    await navigator.clipboard.writeText($("snippet").textContent);
    toast("Skopiowano - wklej do settings.json, sekcja \"matching\"");
  });
  document.addEventListener("keydown", (e) => {         // strzałki góra / dół = poprzednia / następna para
    if (e.target.tagName === "INPUT" || !["ArrowUp", "ArrowDown"].includes(e.key)) return;
    const step = e.key === "ArrowDown" ? 1 : -1;
    selectPair(Math.min(Math.max(state.current + step, 0), state.pairs.length - 1));
    e.preventDefault();
  });
  if (!state.pairs.length) {
    $("pairTitle").textContent = "Brak par w matching.json";
    return;
  }
  await selectPair(0);
}

main().catch((e) => {
  $("pairTitle").textContent = "Błąd";
  toast(e.message, true);
});

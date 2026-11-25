// Linijka do widoków 3D plotly: odległość [mm] między dwoma klikniętymi punktami kości.
//
// Używana przez:
//   - statyczne strony HTML (src/visualization_3d.py: save_figure i write_page - np. bones.html
//     z show_bones.py, merging.html, matching.html z tune_matching.py --html),
//   - stronę z suwakami (src/web/match_tool.js, tune_matching.py).
//
// Użycie: const ruler = attachRuler(divWykresu);   // po pierwszym Plotly.newPlot / Plotly.react
//   Przycisk "📏 Linijka" (albo klawisz L) włącza pomiar. Klik w kość = punkt, co dwa punkty
//   powstaje odcinek z długością w mm. Esc - anuluj zaczęty odcinek, "Cofnij" - usuń ostatni
//   odcinek, "Wyczyść" - wszystkie. Obracanie / przesuwanie widoku działa normalnie
//   (przeciągnięcie myszą nie stawia punktu).
//   ruler.clear() - usuwa pomiary (np. gdy strona rysuje nową scenę przez Plotly.react).
//
// Punkt przyciąga się do najbliższego wierzchołka siatki (tak plotly podaje kliknięcie w mesh3d),
// więc dokładność ≈ odstęp wierzchołków pokazanej (uproszczonej) siatki. Liczą się tylko siatki
// (mesh3d) z podpowiedzią po najechaniu - stół, napisy itp. są pomijane.

(function () {
  const COLOR = "#7048e8";                       // inny niż kolory powierzchni styku i kości
  const UIDS = ["ruler-lines", "ruler-labels"];   // ślady linijki dopisywane do wykresu
  const DRAG_PX = 5;                              // przesunięcie myszy większe niż to = obracanie, nie klik
  const CSS = `
.ruler-box { position: absolute; left: 50%; bottom: 12px; transform: translateX(-50%); z-index: 5;
  display: flex; gap: 8px; align-items: center; flex-wrap: wrap; justify-content: center;
  max-width: calc(100% - 24px); background: rgba(255, 255, 255, .93); border: 1px solid #c9d2db;
  border-radius: 10px; padding: 5px 8px; font: 13px "Segoe UI", system-ui, Arial, sans-serif; color: #1f2a36;
  box-shadow: 0 2px 10px rgba(31, 42, 54, .08); }
.ruler-box.ruler-right { left: auto; right: 12px; transform: none; justify-content: flex-end; }
.ruler-box button { font: inherit; color: inherit; border: 1px solid #c9d2db; background: #fff; border-radius: 7px;
  padding: 3px 10px; cursor: pointer; }
.ruler-box button:hover { background: #f4f6f8; }
.ruler-box .ruler-toggle.on { background: ${COLOR}; border-color: ${COLOR}; color: #fff; }
.ruler-box .ruler-status { color: #6b7b8c; font-size: 12px; }
.ruler-box .ruler-list { font-variant-numeric: tabular-nums; }
.ruler-box .ruler-list b { color: ${COLOR}; }
.ruler-on canvas { cursor: crosshair !important; }
`;
  let active = null;   // linijka, której dotyczą skróty klawiszowe (ostatnio użyta / wskazana myszą)

  /** Dopisuje style linijki do strony (raz). */
  function injectStyle() {
    if (document.getElementById("ruler-style")) return;
    const s = document.createElement("style");
    s.id = "ruler-style";
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  /**
   * Długość w mm jako tekst: 1 miejsce po przecinku, poniżej 10 mm - 2.
   * @param {number} d  odległość [mm]
   * @returns {string}
   */
  function fmt(d) {
    return `${d.toFixed(d < 10 ? 2 : 1)} mm`;
  }

  /**
   * Dodaje linijkę do wykresu 3D plotly.
   * @param {HTMLElement} gd           div wykresu (już narysowany przez Plotly.newPlot / react)
   * @param {object} [options]
   * @param {"center"|"right"} [options.place="center"]  gdzie na dole wykresu stoi pasek linijki
   * @returns {{clear: function(): void, toggle: function(boolean=): void}}
   */
  function attachRuler(gd, options = {}) {
    if (gd._ruler) return gd._ruler;
    injectStyle();
    const host = gd.parentElement;
    if (getComputedStyle(host).position === "static") host.style.position = "relative";

    const box = document.createElement("div");
    box.className = "ruler-box" + (options.place === "right" ? " ruler-right" : "");
    box.innerHTML = `
      <button class="ruler-toggle" title="Pomiar odległości między dwoma punktami (klawisz L)">📏 Linijka</button>
      <span class="ruler-status" hidden></span>
      <span class="ruler-list" hidden></span>
      <button class="ruler-undo" title="Usuń ostatni odcinek" hidden>Cofnij</button>
      <button class="ruler-clear" title="Usuń wszystkie odcinki" hidden>Wyczyść</button>`;
    host.appendChild(box);
    const el = (c) => box.querySelector("." + c);

    // segs - gotowe odcinki [[p, q], ...], first - pierwszy punkt zaczętego odcinka,
    // down - gdzie wciśnięto mysz, cand - punkt z plotly_click od wciśnięcia, upAt - kiedy puszczono
    const st = { on: false, segs: [], first: null, down: null, moved: false, cand: null, upAt: 0 };

    /** Indeksy śladów linijki w gd.data (-1 = brak). */
    const indices = () => UIDS.map((u) => (gd.data || []).findIndex((t) => t.uid === u));

    /** Rysuje odcinki (ślady linijki) i odświeża pasek. */
    function render() {
      const L = { x: [], y: [], z: [] }, T = { x: [], y: [], z: [], text: [] };
      for (const [p, q] of st.segs) {
        for (const a of "xyz") {
          L[a].push(p[a], q[a], null);
          T[a].push((p[a] + q[a]) / 2);
        }
        T.text.push(`<b>${fmt(dist(p, q))}</b>`);
      }
      if (st.first) for (const a of "xyz") L[a].push(st.first[a], null);
      const idx = indices();
      if (idx[0] < 0 || idx[1] < 0) {
        if (L.x.length) {
          Plotly.addTraces(gd, [
            { type: "scatter3d", mode: "lines+markers", uid: UIDS[0], name: "linijka", ...L,
              line: { color: COLOR, width: 6 }, marker: { size: 4, color: COLOR },
              hoverinfo: "skip", showlegend: false },
            { type: "scatter3d", mode: "text", uid: UIDS[1], name: "linijka", ...T,
              textposition: "top center", textfont: { size: 14, color: COLOR },
              hoverinfo: "skip", showlegend: false },
          ]);
        }
      } else {
        Plotly.restyle(gd, { x: [L.x, T.x], y: [L.y, T.y], z: [L.z, T.z], text: [[], T.text], visible: true }, idx);
      }

      el("ruler-toggle").classList.toggle("on", st.on);
      host.classList.toggle("ruler-on", st.on);
      el("ruler-status").hidden = !st.on;
      el("ruler-status").textContent = st.first ? "kliknij drugi punkt (Esc = anuluj)" : "kliknij pierwszy punkt";
      el("ruler-list").hidden = !st.segs.length;
      el("ruler-list").innerHTML = st.segs.map(([p, q], i) => `${i + 1}: <b>${fmt(dist(p, q))}</b>`).join(" · ");
      el("ruler-undo").hidden = el("ruler-clear").hidden = !st.segs.length;
    }

    /** Odległość dwóch punktów {x, y, z} [mm]. */
    function dist(p, q) {
      return Math.hypot(p.x - q.x, p.y - q.y, p.z - q.z);
    }

    /** Dodaje kliknięty punkt: pierwszy albo drugi koniec odcinka. */
    function addPoint(p) {
      if (st.first) {
        st.segs.push([st.first, p]);
        st.first = null;
      } else {
        st.first = p;
      }
      render();
    }

    /** Punkt kliknięcia z danych plotly_click albo null (nie siatka kości). */
    function pickedPoint(e) {
      const pt = e && e.points && e.points[0];
      if (!pt || pt.x === undefined || !pt.data || pt.data.type !== "mesh3d" || pt.data.hoverinfo === "skip") return null;
      return { x: +pt.x, y: +pt.y, z: +pt.z };
    }

    // W 3D plotly zgłasza plotly_click już przy wciśnięciu przycisku, także przed obracaniem -
    // punkt jest przyjmowany dopiero po puszczeniu myszy bez przesunięcia.
    gd.addEventListener("pointerdown", (e) => {
      active = api;
      st.down = e.button === 0 ? { x: e.clientX, y: e.clientY } : null;
      st.moved = false;
      st.cand = null;
      st.upAt = 0;
    }, true);
    gd.addEventListener("pointerup", (e) => {
      if (!st.down) return;
      st.moved = Math.hypot(e.clientX - st.down.x, e.clientY - st.down.y) > DRAG_PX;
      st.down = null;
      if (!st.on || st.moved) return;
      if (st.cand) addPoint(st.cand);
      else st.upAt = Date.now();      // plotly_click może przyjść chwilę po puszczeniu
      st.cand = null;
    }, true);
    gd.on("plotly_click", (e) => {
      if (!st.on) return;
      const p = pickedPoint(e);
      if (!p) return;
      if (st.upAt && Date.now() - st.upAt < 400) {
        st.upAt = 0;
        addPoint(p);
      } else if (st.down) {
        st.cand = p;
      }
    });
    // przyciski "pokaż / ukryj" i legenda zmieniają widoczność wszystkich śladów - linijka zostaje widoczna
    const keepVisible = () => {
      const idx = indices().filter((i) => i >= 0 && gd.data[i].visible !== true);
      if (idx.length) Plotly.restyle(gd, { visible: true }, idx);
    };
    gd.on("plotly_restyle", keepVisible);
    gd.on("plotly_update", keepVisible);
    host.addEventListener("pointerenter", () => { active = api; });

    const api = {
      /** Usuwa wszystkie odcinki (tryb pomiaru zostaje). */
      clear() {
        st.segs = [];
        st.first = null;
        render();
      },
      /** Włącza / wyłącza tryb pomiaru (bez argumentu - przełącza). */
      toggle(on) {
        st.on = on === undefined ? !st.on : !!on;
        st.first = null;
        active = api;
        render();
      },
      /** Klawisze: L - włącz / wyłącz, Esc - anuluj zaczęty odcinek. */
      key(e) {
        if (e.key === "l" || e.key === "L") api.toggle();
        else if (e.key === "Escape" && st.first) { st.first = null; render(); }
        else return false;
        return true;
      },
    };
    el("ruler-toggle").addEventListener("click", () => api.toggle());
    el("ruler-undo").addEventListener("click", () => { st.segs.pop(); render(); });
    el("ruler-clear").addEventListener("click", () => api.clear());
    gd._ruler = api;
    if (!active) active = api;
    render();
    return api;
  }

  document.addEventListener("keydown", (e) => {
    const t = e.target;
    if (!active || e.ctrlKey || e.altKey || e.metaKey || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName)) return;
    if (active.key(e)) e.preventDefault();
  });

  window.attachRuler = attachRuler;
})();

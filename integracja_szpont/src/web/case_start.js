'use strict';

const form = document.getElementById('caseForm');
const input = document.getElementById('scans');
const button = document.getElementById('solveCase');
const saveButton = document.getElementById('saveCase');
const message = document.getElementById('caseMessage');
const selected = document.getElementById('fileList');
const selectedFiles = document.getElementById('selectedFiles');
const filesSummary = document.getElementById('filesSummary');
const drop = document.getElementById('scanDrop');
const mode = document.getElementById('caseMode');
const anchor = document.getElementById('scanAnchor');
const uploadFolder = document.getElementById('uploadFolder');
const historyFolder = document.getElementById('historyFolder');
const folderForm = document.getElementById('folderForm');
const historyCases = document.getElementById('historyCases');
const historyMessage = document.getElementById('historyMessage');
const replayExport = document.getElementById('replayExport');
const replayImport = document.getElementById('replayImportFile');
const replayImportButton = document.getElementById('replayImportButton');
const detailedFrame = document.getElementById('detailedMatchingFrame');
const boneReport = document.getElementById('boneReport');
const boneReportFrame = document.getElementById('boneReportFrame');
const boneReportLink = document.getElementById('boneReportLink');
const boneMaterialsStatus = document.getElementById('boneMaterialsStatus');
const dateFormat = new Intl.DateTimeFormat('pl', { dateStyle: 'short', timeStyle: 'short' });
const fileSizeFormat = new Intl.NumberFormat('pl', { maximumFractionDigits: 1 });
const tabs = Array.from(document.querySelectorAll('.case-tabs [role="tab"]'));
let folders = [];
let savedCases = [];
let boneMaterialsLoaded = false;

async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Operacja się nie powiodła.');
  return data;
}

function jsonBody(method, data) {
  return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) };
}

function navigateCase(id) {
  location.assign('/assembly?job=' + encodeURIComponent(id));
}

function folderOptions(select, all = false, value = select.value) {
  select.replaceChildren(...(all ? [new Option('Wszystkie przypadki', 'all')] : []),
    new Option('Bez folderu', ''), ...folders.map(folder => new Option(folder.name, folder.id)));
  select.value = Array.from(select.options).some(option => option.value === value) ? value : all ? 'all' : '';
}

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function showBoneReport() {
  const option = boneReport.selectedOptions[0];
  if (!option) return;
  boneReportFrame.title = option.textContent;
  boneReportFrame.src = option.value;
  boneReportLink.href = option.value;
}

function showBonePhotos(photos) {
  document.getElementById('bonePhotoSection').hidden = !photos.length;
  document.getElementById('bonePhotoSummary').textContent = 'Zdjęcia kości (' + photos.length + ')';
  document.getElementById('bonePhotoGrid').replaceChildren(...photos.map((photo, index) => {
    const link = node('a');
    link.href = photo.url;
    link.target = '_blank';
    link.rel = 'noopener';
    link.title = photo.name;
    const image = node('img');
    image.src = photo.preview;
    image.alt = 'Zdjęcie kości ' + (index + 1);
    image.loading = 'lazy';
    link.append(image);
    return link;
  }));
}

function showBoneFiles(groups) {
  document.getElementById('boneFileSection').hidden = !groups.length;
  document.getElementById('boneFileGroups').replaceChildren(...groups.map(group => {
    const heading = node('h3', group.label);
    const list = node('ul');
    for (const file of group.files) {
      const entry = node('li');
      const link = node('a', file.label);
      link.href = file.url;
      entry.append(link);
      list.append(entry);
    }
    const section = node('section');
    section.append(heading, list);
    return section;
  }));
}

async function loadBoneMaterials() {
  if (boneMaterialsLoaded) return;
  boneMaterialsLoaded = true;
  boneMaterialsStatus.hidden = false;
  boneMaterialsStatus.textContent = 'Wczytuję materiały…';
  try {
    const materials = await request('/api/bone-materials');
    boneReport.replaceChildren(...materials.reports.map(report => new Option(report.label, report.url)));
    showBoneReport();
    document.getElementById('boneReportControls').hidden = boneReportFrame.hidden = !materials.reports.length;
    showBonePhotos(materials.photos);
    showBoneFiles(materials.groups);
    boneMaterialsStatus.hidden = Boolean(materials.reports.length || materials.photos.length || materials.groups.length);
    if (!boneMaterialsStatus.hidden) boneMaterialsStatus.textContent = 'Brak materiałów do wyświetlenia.';
  } catch (error) {
    boneMaterialsLoaded = false;
    boneMaterialsStatus.textContent = error.message;
  }
}

boneReport.addEventListener('change', showBoneReport);

function updateTab() {
  const active = tabs.find(tab => tab.hash === location.hash) ?? tabs[0];
  for (const tab of tabs) {
    const selected = tab === active;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
    document.getElementById(tab.getAttribute('aria-controls')).hidden = !selected;
  }
  if (active.id === 'detailedMatchingTab' && !detailedFrame.hasAttribute('src')) {
    detailedFrame.src = detailedFrame.dataset.src;
  }
  if (active.id === 'boneMaterialsTab') loadBoneMaterials();
}

for (const [index, tab] of tabs.entries()) {
  tab.onclick = event => {
    event.preventDefault();
    if (location.hash !== tab.hash) history.pushState(null, '', tab.hash);
    updateTab();
    if (tab.hash === '#historia') refreshHistory().catch(error => { historyMessage.textContent = error.message; });
  };
  tab.onkeydown = event => {
    const next = { ArrowLeft: (index + tabs.length - 1) % tabs.length, ArrowRight: (index + 1) % tabs.length,
      Home: 0, End: tabs.length - 1 }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    tabs[next].click();
    tabs[next].focus();
  };
}
window.addEventListener('hashchange', updateTab);
window.addEventListener('popstate', updateTab);
updateTab();

function historyRow(entry) {
  const key = encodeURIComponent(entry.id);
  const item = node('li');
  const row = node('div', '', 'history-row');
  const summary = node('div');
  const folder = folders.find(folder => folder.id === entry.folder)?.name || 'Bez folderu';
  const state = entry.running ? 'Analiza trwa' : entry.done ? 'Wynik gotowy' : entry.error ? 'Analiza nieukończona' : 'Skany zapisane';
  summary.append(node('h3', entry.title), node('p', [folder, dateFormat.format(new Date(entry.created)), state].join(' · '), 'history-meta'));
  const actions = node('div', '', 'history-actions');
  const label = node('label', 'Folder zapisu', 'history-move');
  const move = node('select');
  move.setAttribute('aria-label', 'Folder zapisu – ' + entry.title);
  folderOptions(move, false, entry.folder ?? '');
  move.onchange = async () => {
    move.disabled = true;
    try {
      await request('/api/history/' + key, jsonBody('PUT', { folder: move.value }));
      await refreshHistory();
    } catch (error) {
      move.value = entry.folder ?? '';
      historyMessage.textContent = error.message;
    } finally { move.disabled = false; }
  };
  label.append(move);
  actions.append(label);
  if (entry.done || entry.running) {
    const open = node('a', entry.done ? 'Otwórz wynik' : 'Zobacz postęp', 'library-link');
    open.href = '/assembly?job=' + key;
    actions.append(open);
    if (entry.done) {
      const replay = node('a', 'Pobierz złożenie', 'library-link');
      replay.href = '/api/history/replays/' + key;
      replay.download = '';
      actions.append(replay);
    }
  } else if (entry.files.length >= 2) {
    const retry = node('button', 'Uruchom analizę', 'ghost');
    retry.type = 'button';
    retry.onclick = async () => {
      retry.disabled = true;
      try {
        await request('/api/history/' + key + '/solve', { method: 'POST' });
        navigateCase(entry.id);
      } catch (error) {
        historyMessage.textContent = error.message;
        retry.disabled = false;
      }
    };
    actions.append(retry);
  }
  const files = node('details');
  files.append(node('summary', 'Oryginalne pliki (' + entry.files.length + ')'));
  for (const file of entry.files) {
    const link = node('a', file.name);
    link.href = '/api/history/' + key + '/scan/' + file.number;
    files.append(link);
  }
  if (entry.error && !entry.running && !entry.done) files.append(node('p', entry.error));
  row.append(summary, actions, files);
  item.append(row);
  return item;
}

function showHistory() {
  const value = historyFolder.value;
  const visible = value === 'all' ? savedCases : savedCases.filter(entry => (entry.folder ?? '') === value);
  historyCases.replaceChildren(...visible.map(historyRow));
  replayExport.hidden = !savedCases.some(entry => entry.done);
  historyMessage.textContent = visible.length ? '' : 'Brak zapisanych przypadków w tym widoku.';
}

async function refreshHistory(newFolder) {
  const listing = await request('/api/history');
  folders = listing.folders;
  savedCases = listing.cases;
  folderOptions(uploadFolder, false, newFolder ?? uploadFolder.value);
  folderOptions(historyFolder, true);
  showHistory();
}

historyFolder.onchange = showHistory;
replayImportButton.onclick = () => replayImport.click();
replayImport.onchange = async () => {
  const file = replayImport.files[0];
  if (!file) return;
  replayImportButton.disabled = true;
  historyMessage.textContent = 'Wczytuję zapisane złożenia…';
  try {
    const result = await request('/api/history/replays', {
      method: 'POST', headers: { 'Content-Type': 'application/zip' }, body: file
    });
    historyFolder.value = 'all';
    await refreshHistory();
    historyMessage.textContent = 'Dodano: ' + result.imported + ' · Już w historii: ' + result.existing;
  } catch (error) {
    historyMessage.textContent = error.message;
  } finally {
    replayImport.value = '';
    replayImportButton.disabled = false;
  }
};
folderForm.onsubmit = async event => {
  event.preventDefault();
  const submit = event.submitter;
  submit.disabled = true;
  try {
    const folder = await request('/api/history/folders', jsonBody('POST', { name: folderForm.elements.name.value }));
    folderForm.reset();
    await refreshHistory(folder.id);
    historyMessage.textContent = 'Utworzono folder „' + folder.name + '” i wybrano go w formularzu nowego przypadku.';
  } catch (error) {
    historyMessage.textContent = error.message;
  } finally { submit.disabled = false; }
};
refreshHistory().catch(error => { historyMessage.textContent = error.message; });
mode.onchange = () => {
  button.textContent = mode.value === 'mixed' ? 'Pogrupuj i złóż fragmenty' :
    mode.value === 'target' ? 'Złóż wybraną kość' : 'Złóż fragmenty';
  document.getElementById('scanAnchorLabel').hidden = anchor.disabled = mode.value !== 'target';
  message.textContent = '';
};

function showFiles() {
  anchor.replaceChildren(new Option('Wybierz odłamek', ''),
    ...Array.from(input.files, (file, index) => new Option((index + 1) + ' · ' + file.name, index + 1)));
  const rows = document.createDocumentFragment();
  let size = 0;
  for (const file of input.files) {
    rows.append(node('li', file.name));
    size += file.size;
  }
  selected.replaceChildren(rows);
  selectedFiles.hidden = input.files.length === 0;
  const amount = size >= 1024 * 1024 ? fileSizeFormat.format(size / (1024 * 1024)) + ' MB' :
    fileSizeFormat.format(size / 1024) + ' kB';
  filesSummary.textContent = 'Wybrane pliki (' + input.files.length + ') · ' + amount;
}

input.onchange = () => {
  showFiles();
  message.textContent = '';
};
document.getElementById('clearFiles').onclick = () => {
  input.value = '';
  input.onchange();
  input.focus();
};
drop.ondragover = event => {
  event.preventDefault();
  drop.classList.add('is-dragging');
};
drop.ondragleave = () => drop.classList.remove('is-dragging');
drop.ondrop = event => {
  event.preventDefault();
  drop.classList.remove('is-dragging');
  const files = event.dataTransfer.files;
  if (!files.length) return;
  if (Array.from(files).some(file => !/\.(obj|ply|stl)$/i.test(file.name))) {
    message.textContent = 'Upuść pliki STL, OBJ lub PLY.';
    return;
  }
  input.files = files;
  input.onchange();
};
showFiles();
mode.onchange();

form.onsubmit = async event => {
  event.preventDefault();
  const solve = event.submitter !== saveButton;
  const minimum = solve ? 2 : 1;
  if (input.files.length < minimum) {
    message.textContent = solve ? 'Wybierz co najmniej dwa pliki.' : 'Wybierz co najmniej jeden plik.';
    return;
  }
  if (mode.value === 'target' && !anchor.value) {
    message.textContent = 'Wybierz fragment odniesienia.';
    anchor.focus();
    return;
  }
  button.disabled = saveButton.disabled = true;
  message.textContent = 'Zapisuję pliki skanów…';
  try {
    const data = await request(solve ? '/api/solve' : '/api/history', { method: 'POST', body: new FormData(form) });
    if (solve) navigateCase(data.id);
    else {
      input.value = '';
      showFiles();
      message.textContent = 'Pliki zapisano w historii skanów.';
      refreshHistory().catch(error => { historyMessage.textContent = error.message; });
    }
  } catch (error) {
    message.textContent = error.message;
  } finally { button.disabled = saveButton.disabled = false; }
};

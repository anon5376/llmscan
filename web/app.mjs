// llmscan workbench: reads one CLI output folder in the browser and renders it. Nothing is uploaded.
import { readReport } from './report.mjs';
import { h } from './dom.mjs';
import {
  TIERS, buildWorkbench, filterTensors, formatCount, formatNumber, formatPercent, formatShape,
  pageOf, sortTensors, statusLabel, tierInfo,
} from './model.mjs';
import { extent, kindOrder, kindSpectraChart, layerGradient, PALETTE, rankByLayerChart, tensorSpectrumChart } from './charts.mjs';

const PAGE_SIZE = 100;
const DEMO_FILES = ['pair.txt', 'inventory.csv', 'ranking.csv', 'delta.csv'];
const TABS = [
  { id: 'ranking', label: 'Ranking', count: (m) => m.ranking.length },
  { id: 'inventory', label: 'Inventory', count: (m) => m.inventory.length },
  { id: 'delta', label: 'ΔW stats', count: (m) => m.measured.length },
];
const CAVEAT = 'llmscan ranks suspicion. It never certifies a model clean: a tensor low in the ranking, '
  + 'or absent from it, has only passed the checks listed here.';

const main = document.getElementById('main');
const notice = document.getElementById('notice');
const live = document.getElementById('status');
const dialog = document.getElementById('detail-dialog');
const folderInput = document.getElementById('folder-input');
const filesInput = document.getElementById('files-input');
const wide = window.matchMedia('(min-width: 1100px)');

const state = { model: null, source: null, tab: 'ranking', selected: null, view: null, rows: [], refresh: null };

function freshView() {
  return {
    ranking: { tiers: new Set(), query: '', sort: { key: 'rank', direction: 'asc' }, page: 0 },
    inventory: { statuses: new Set(), kind: '', query: '', sort: { key: null, direction: 'asc' }, page: 0 },
    delta: { kind: '', query: '', sort: { key: 'effectiveRank', direction: 'desc' }, page: 0 },
  };
}

// ---------------------------------------------------------------- loading

function announce(message) {
  live.textContent = message;
}

function showNotice(title, message) {
  notice.replaceChildren(
    h('div', null, h('strong', null, title), h('p', null, message)),
    h('button', { type: 'button', class: 'button quiet', onclick: () => { notice.hidden = true; } }, 'Dismiss'),
  );
  notice.hidden = false;
}

async function load(source, getFiles) {
  const controls = document.querySelectorAll('[data-action]');
  controls.forEach((b) => { b.disabled = true; });
  main.setAttribute('aria-busy', 'true');
  announce(`Reading ${source.label}…`);
  try {
    const model = buildWorkbench(await readReport(await getFiles()));
    Object.assign(state, { model, source, view: freshView(), selected: null });
    if (wide.matches) state.selected = model.ranking[0]?.name ?? null;
    notice.hidden = true;
    if (dialog.open) dialog.close();
    render();
    announce(`${source.label} loaded: ${model.ranking.length} ranked tensors, ${model.inventory.length} in the inventory.`);
  } catch (error) {
    showNotice(source.demo ? 'The demo report could not be opened' : 'This report could not be imported',
      error instanceof Error ? error.message : String(error));
    announce('Import failed.');
    if (!state.model) renderWelcome();
  } finally {
    controls.forEach((b) => { b.disabled = false; });
    main.removeAttribute('aria-busy');
  }
}

async function demoFiles() {
  const files = [];
  for (const name of DEMO_FILES) {
    const response = await fetch(new URL(`demo/${name}`, import.meta.url));
    if (response.ok) files.push(new File([await response.blob()], name));
  }
  return files;
}

const openDemo = () => load({ demo: true, label: 'Demo report' }, demoFiles);

function importFiles(files, folder) {
  if (!files.length) return;
  const label = folder ? `Folder ${folder}` : `${files.length} selected file${files.length === 1 ? '' : 's'}`;
  load({ demo: false, label }, () => files);
}

const entryFile = (entry) => new Promise((resolve, reject) => { entry.file(resolve, reject); });

async function directoryFiles(directory) {
  const reader = directory.createReader();
  const files = [];
  for (;;) {
    const batch = await new Promise((resolve, reject) => { reader.readEntries(resolve, reject); });
    if (!batch.length) return files;
    for (const entry of batch) if (entry.isFile) files.push(await entryFile(entry));
  }
}

/** Files of a drop: loose files, or the top level of a dropped folder (the CLI writes a flat directory). */
async function droppedFiles(dataTransfer) {
  const entries = [...dataTransfer.items].map((item) => item.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return { files: [...dataTransfer.files] };
  const files = [];
  for (const entry of entries) files.push(...(entry.isDirectory ? await directoryFiles(entry) : [await entryFile(entry)]));
  return { files, folder: entries.find((entry) => entry.isDirectory)?.name };
}

function wireImport() {
  const actions = {
    demo: openDemo,
    folder: () => folderInput.click(),
    files: () => filesInput.click(),
  };
  document.addEventListener('click', (event) => {
    const action = event.target.closest('[data-action]')?.dataset.action;
    if (action) actions[action]();
  });
  for (const input of [folderInput, filesInput]) {
    input.addEventListener('change', () => {
      const files = [...input.files];
      input.value = '';
      importFiles(files, files[0]?.webkitRelativePath.split('/')[0]);
    });
  }
  let depth = 0;
  const hasFiles = (event) => event.dataTransfer?.types.includes('Files');
  window.addEventListener('dragenter', (event) => {
    if (hasFiles(event) && ++depth === 1) document.body.classList.add('dropping');
  });
  window.addEventListener('dragleave', (event) => {
    if (hasFiles(event) && --depth === 0) document.body.classList.remove('dropping');
  });
  window.addEventListener('dragover', (event) => { if (hasFiles(event)) event.preventDefault(); });
  window.addEventListener('drop', async (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    depth = 0;
    document.body.classList.remove('dropping');
    try {
      const { files, folder } = await droppedFiles(event.dataTransfer);
      importFiles(files, folder);
    } catch (error) {
      showNotice('The dropped folder could not be read', error instanceof Error ? error.message : String(error));
    }
  });
}

// ---------------------------------------------------------------- shared pieces

function tierBadge(id, { full = false } = {}) {
  if (!id) return h('span', { class: 'unranked' }, full ? 'not ranked' : [h('span', { 'aria-hidden': 'true' }, '–'), h('span', { class: 'sr-only' }, 'not ranked')]);
  const tier = tierInfo(id);
  return h('span', { class: 'tier', 'data-tier': id, title: full ? null : tier.label },
    h('span', { class: 'tier-id' }, id), full && tier.label !== id && h('span', null, tier.label));
}

function statusBadge(status) {
  return h('span', { class: 'status', 'data-status': status }, statusLabel(status));
}

function chip(key, label, count, pressed, onclick, attrs = {}) {
  return h('button', { type: 'button', class: 'chip', 'data-key': `chip:${key}`, 'aria-pressed': String(pressed), onclick, ...attrs },
    label, h('span', { class: 'chip-count' }, formatCount(count)));
}

function searchBox(id, value, oninput) {
  return h('label', { class: 'field', for: id }, h('span', null, 'Filter'),
    h('input', {
      id, type: 'search', value, placeholder: 'tensor name, kind, reason…', autocomplete: 'off', spellcheck: 'false',
      oninput: (event) => oninput(event.target.value),
    }));
}

function kindSelect(id, kinds, value, onchange) {
  return h('label', { class: 'field', for: id }, h('span', null, 'Kind'),
    h('select', { id, onchange: (event) => onchange(event.target.value) },
      h('option', { value: '' }, 'All kinds'),
      kinds.map((kind) => h('option', { value: kind, selected: kind === value }, kind))));
}

function emptyState(title, ...body) {
  return h('div', { class: 'empty' }, h('h3', null, title), body);
}

function tensorButton(t) {
  return h('button', {
    type: 'button', class: 'tensor-link', 'aria-current': String(t.name === state.selected),
    onclick: (event) => { event.stopPropagation(); select(t.name, { open: true }); },
  }, t.name);
}

/** A sortable table of tensors. Clicking a row, or its name button, opens the tensor's detail. */
function tensorTable({ caption, columns, rows, sort, onSort }) {
  const header = columns.map((c) => {
    const active = sort.key === c.key;
    const next = active && sort.direction === (c.first ?? 'asc') ? (c.first === 'desc' ? 'asc' : 'desc') : (c.first ?? 'asc');
    return h('th', {
      scope: 'col', class: c.class,
      'aria-sort': c.key && active ? (sort.direction === 'asc' ? 'ascending' : 'descending') : null,
    }, c.key
      ? h('button', { type: 'button', class: 'sort', 'data-key': `sort:${c.key}`, onclick: () => onSort({ key: c.key, direction: next }) },
        c.label, h('span', { class: 'sort-mark', 'aria-hidden': 'true' }, active ? (sort.direction === 'asc' ? '▲' : '▼') : ''))
      : c.label);
  });
  return h('div', { class: 'table-wrap' },
    h('table', null,
      h('caption', { class: 'sr-only' }, caption),
      h('thead', null, h('tr', null, header)),
      h('tbody', null, rows.map((t) => h('tr', {
        'data-tensor': t.name, class: t.name === state.selected ? 'selected' : null,
        onclick: () => select(t.name, { open: true }),
      }, columns.map((c) => h('td', { class: c.class }, c.cell(t))))))));
}

function pager(page, total, onPage) {
  if (page.pages < 2) return null;
  const step = (label, to, enabled) => h('button', { type: 'button', class: 'button quiet', 'data-key': `page:${label}`, disabled: !enabled, onclick: () => onPage(to) }, label);
  return h('nav', { class: 'pager', 'aria-label': 'Pages' },
    step('Previous', page.page - 1, page.page > 0),
    h('span', null, `${formatCount(page.start + 1)}–${formatCount(page.end)} of ${formatCount(total)}`),
    step('Next', page.page + 1, page.page < page.pages - 1));
}

const COLUMNS = {
  rank: { key: 'rank', label: '#', class: 'num narrow', cell: (t) => formatCount(t.rank) },
  tier: { key: 'tier', label: 'Tier', cell: (t) => tierBadge(t.tier) },
  reason: { key: 'name', label: 'Tensor and reason', class: 'grow', cell: (t) => [tensorButton(t), h('p', { class: 'reason' }, t.reason)] },
  name: { key: 'name', label: 'Tensor', class: 'grow', cell: tensorButton },
  layer: { key: 'layer', label: 'Layer', class: 'num optional', cell: (t) => (t.layer == null ? '–' : String(t.layer)) },
  kind: { key: 'kind', label: 'Kind', class: 'optional', cell: (t) => h('code', null, t.kind || '–') },
  status: { key: 'status', label: 'Status', cell: (t) => statusBadge(t.status) },
  shape: {
    label: 'Shape', class: 'roomy nowrap',
    cell: (t) => (t.suspectShape && t.baseShape && t.suspectShape !== t.baseShape
      ? `${formatShape(t.baseShape)} → ${formatShape(t.suspectShape)}` : formatShape(t.baseShape || t.suspectShape)),
  },
  numel: { key: 'numel', label: 'Parameters', class: 'num optional', first: 'desc', cell: (t) => formatCount(t.numel) },
  effectiveRank: { key: 'effectiveRank', label: 'Eff. rank', class: 'num', first: 'desc', cell: (t) => formatNumber(t.effectiveRank) },
  sv1: { key: 'sv1', label: 'σ₁', class: 'num optional', first: 'desc', cell: (t) => formatNumber(t.sv1) },
  gap: { key: 'gap', label: 'σ₁ / σ₂', class: 'num optional', first: 'desc', cell: (t) => formatNumber(t.gap) },
};

/** Redraw `container`, then put keyboard focus back on the control that had it (matched by data-key or id). */
function keepFocus(container, redraw) {
  const active = container.contains(document.activeElement) ? document.activeElement : null;
  const key = active?.dataset.key ?? active?.id;
  redraw();
  if (!active || container.contains(document.activeElement)) return;
  const again = key && [...container.querySelectorAll('[data-key], [id]')].find((el) => (el.dataset.key ?? el.id) === key);
  (again && !again.disabled ? again : container.closest('[tabindex]') ?? container).focus();
}

/** Controls stay mounted while typing; only the results below them are redrawn. */
function listing({ controls, results }) {
  const region = h('div', { class: 'results' });
  state.refresh = () => {
    keepFocus(region, () => region.replaceChildren(...[results()].flat(Infinity).filter(Boolean)));
    renderDetail(); // Previous and Next follow the list as it is now filtered and sorted
  };
  state.refresh();
  return [h('div', { class: 'controls' }, controls), region];
}

function resultsFor(view, rows, table, { empty, onClear }) {
  const sorted = view.sort.key ? sortTensors(rows, view.sort.key, view.sort.direction) : rows;
  state.rows = sorted;
  const page = pageOf(sorted, view.page, PAGE_SIZE);
  view.page = page.page;
  if (!sorted.length) {
    return emptyState('No tensor matches these filters',
      h('p', null, empty), h('button', { type: 'button', class: 'button', onclick: onClear }, 'Clear filters'));
  }
  const repaint = (change) => { Object.assign(view, change); state.refresh(); };
  return [
    table(page.rows, (sort) => repaint({ sort, page: 0 })),
    pager(page, sorted.length, (to) => { repaint({ page: to }); main.querySelector('.results')?.scrollIntoView({ block: 'nearest' }); }),
  ];
}

// ---------------------------------------------------------------- views

function rankingView(model) {
  const view = state.view.ranking;
  if (!model.ranking.length) {
    state.rows = [];
    return emptyState(model.inventory.length ? 'Nothing was ranked' : 'This report lists no tensors',
      h('p', null, model.inventory.length
        ? 'ranking.csv has no rows: no tensor in this comparison differs from the base.'
        : 'inventory.csv and ranking.csv are both empty.'),
      h('p', null, 'That is not a certificate that the model is clean. llmscan compares weights tensor by tensor and '
        + 'ranks what differs; it does not test behaviour.'));
  }
  const setTiers = (tiers) => { view.tiers = tiers; view.page = 0; renderPanel(); };
  const controls = [
    h('div', { class: 'chips', role: 'group', 'aria-label': 'Filter by tier' },
      chip('all', 'All', model.ranking.length, view.tiers.size === 0, () => setTiers(new Set())),
      [...model.tierCounts].map(([id, count]) => chip(id,
        [h('span', { class: 'tier-id', 'data-tier': id }, id), tierInfo(id).label !== id && tierInfo(id).label], count, view.tiers.has(id),
        () => { const next = new Set(view.tiers); if (!next.delete(id)) next.add(id); setTiers(next); },
        { disabled: count === 0 && !view.tiers.has(id) }))),
    searchBox('ranking-filter', view.query, (query) => { view.query = query; view.page = 0; state.refresh(); }),
  ];
  const columns = [COLUMNS.rank, COLUMNS.tier, COLUMNS.reason, COLUMNS.layer, COLUMNS.kind, COLUMNS.effectiveRank, COLUMNS.sv1];
  return [
    !model.hasDelta && h('p', { class: 'note' },
      h('strong', null, 'No ΔW stats. '),
      'delta.csv is missing or lists no tensor, so no row here has singular values behind it: '
      + 'this ranking rests on structure and location and cannot tell a rank-1 edit from a diffuse one.'),
    listing({
      controls,
      results: () => {
        const rows = filterTensors(model.ranking, view);
        return [
          h('p', { class: 'count', role: 'status' }, `${formatCount(rows.length)} of ${formatCount(model.ranking.length)} ranked tensors, most suspicious first`),
          resultsFor(view, rows, (pageRows, onSort) => tensorTable({ caption: 'Ranked tensors', columns, rows: pageRows, sort: view.sort, onSort }), {
            empty: 'Try another tier or a shorter search.',
            onClear: () => { view.query = ''; setTiers(new Set()); },
          }),
        ];
      },
    }),
  ];
}

function kindMatrix(model, view, rerender) {
  const cell = (k, status) => {
    const count = k.counts.get(status) ?? 0;
    if (!count) return h('td', { class: 'num zero' }, '0');
    const active = view.kind === k.kind && view.statuses.size === 1 && view.statuses.has(status);
    return h('td', { class: 'num' }, h('button', {
      type: 'button', class: 'cell-link', 'data-key': `cell:${k.kind}:${status}`, 'aria-pressed': String(active),
      'aria-label': `${count} ${statusLabel(status)} ${k.kind} tensors: ${active ? 'clear this filter' : 'filter the list'}`,
      onclick: () => { Object.assign(view, active ? { kind: '', statuses: new Set() } : { kind: k.kind, statuses: new Set([status]) }); view.page = 0; rerender(); },
    }, formatCount(count)));
  };
  view.matrixOpen ??= wide.matches;
  return h('details', { class: 'matrix', open: view.matrixOpen, ontoggle: (event) => { view.matrixOpen = event.target.open; } },
    h('summary', null, 'Tensors by kind and status'),
    h('div', { class: 'table-wrap' }, h('table', { class: 'compact' },
      h('thead', null, h('tr', null, h('th', { scope: 'col' }, 'Kind'),
        model.statuses.map((s) => h('th', { scope: 'col', class: 'num' }, statusLabel(s))), h('th', { scope: 'col', class: 'num' }, 'Total'))),
      h('tbody', null, model.kinds.map((k) => h('tr', null, h('th', { scope: 'row' }, h('code', null, k.kind || '–')),
        model.statuses.map((s) => cell(k, s)), h('td', { class: 'num' }, formatCount(k.total))))))));
}

function inventoryView(model) {
  const view = state.view.inventory;
  if (!model.inventory.length) {
    state.rows = [];
    return emptyState('This report lists no tensors', h('p', null, 'inventory.csv has no rows: neither checkpoint contained a tensor llmscan could read.'));
  }
  const rerender = () => renderPanel();
  const columns = [COLUMNS.name, COLUMNS.layer, COLUMNS.kind, COLUMNS.status, COLUMNS.tier, COLUMNS.shape, COLUMNS.numel];
  const controls = [
    h('div', { class: 'chips', role: 'group', 'aria-label': 'Filter by status' },
      chip('all', 'All', model.inventory.length, view.statuses.size === 0, () => { view.statuses = new Set(); view.page = 0; rerender(); }),
      model.statuses.map((status) => chip(status, [h('span', { class: 'dot', 'data-status': status }), statusLabel(status)], model.statusCounts.get(status), view.statuses.has(status),
        () => { const next = new Set(view.statuses); if (!next.delete(status)) next.add(status); view.statuses = next; view.page = 0; rerender(); }))),
    kindSelect('inventory-kind', model.kinds.map((k) => k.kind), view.kind, (kind) => { view.kind = kind; view.page = 0; rerender(); }),
    searchBox('inventory-filter', view.query, (query) => { view.query = query; view.page = 0; state.refresh(); }),
  ];
  return [
    kindMatrix(model, view, rerender),
    listing({
      controls,
      results: () => {
        const rows = filterTensors(model.inventory, view);
        return [
          h('p', { class: 'count', role: 'status' }, `${formatCount(rows.length)} of ${formatCount(model.inventory.length)} tensors`),
          resultsFor(view, rows, (pageRows, onSort) => tensorTable({ caption: 'Tensor inventory', columns, rows: pageRows, sort: view.sort, onSort }), {
            empty: 'No tensor has this combination of status, kind and text.',
            onClear: () => { Object.assign(view, { statuses: new Set(), kind: '', query: '' }); rerender(); },
          }),
        ];
      },
    }),
  ];
}

const tensorCount = (n) => `${formatCount(n)} tensor${n === 1 ? '' : 's'}`;

function tile(label, value, detail) {
  return h('div', { class: 'tile' }, h('dt', null, label), h('dd', null, h('span', { class: 'tile-value' }, value), detail && h('span', { class: 'tile-detail' }, detail)));
}

function deltaCharts(model) {
  const box = h('div', { class: 'charts' });
  const draw = () => {
    const options = { selected: state.selected, onPick: (name) => select(name, { open: true }) };
    const withLayer = model.measured.filter((t) => t.layer != null);
    const spectra = (kind) => model.measured.filter((t) => t.kind === kind && t.spectrum.length);
    const kinds = kindOrder(model.measured.filter((t) => t.spectrum.length).map((t) => t.kind));
    const layerRange = extent(withLayer.map((t) => t.layer));
    const byLayer = withLayer.length ? rankByLayerChart(withLayer, options) : null;
    const offStack = model.measured.length - withLayer.length;
    box.replaceChildren(
      h('figure', { class: 'card' },
        h('figcaption', null, h('h3', null, 'Effective rank of ΔW by layer'),
          h('p', null, 'One line per matrix kind, side by side within a layer. A rank-1 edit scores 1; r equal directions score r. Select a point to open its tensor.')),
        byLayer ? [byLayer.chart, h('ul', { class: 'legend' }, byLayer.legend.map(({ kind, color }) =>
          h('li', null, h('span', { class: 'swatch', style: { background: color } }), h('code', null, kind))))]
          : h('p', { class: 'note' }, 'No changed tensor in this report belongs to a numbered layer.'),
        offStack > 0 && h('p', { class: 'fine' }, `${tensorCount(offStack)} outside the layer stack ${offStack === 1 ? 'is' : 'are'} only in the table below.`)),
      h('figure', { class: 'card' },
        h('figcaption', null, h('h3', null, 'Singular values of ΔW by kind'),
          h('p', null, 'One line per changed tensor, log scale. An abliteration-like edit is one spike above a low floor.')),
        h('div', { class: 'multiples' }, kinds.map((kind) => h('div', null,
          h('h4', null, h('code', null, kind), ` · ${tensorCount(spectra(kind).length)}`),
          kindSpectraChart(kind, spectra(kind), { ...options, layerRange })))),
        withLayer.length > 0 && h('p', { class: 'ramp' }, `layer ${layerRange[0]}`,
          h('span', { class: 'ramp-bar', style: { background: layerGradient } }), `layer ${layerRange[1]}`)),
    );
  };
  draw();
  return { box, draw };
}

function deltaView(model) {
  const view = state.view.delta;
  state.rows = [];
  if (!model.hasDelta) {
    return emptyState('No ΔW stats in this report',
      h('p', null, 'delta.csv is missing or lists no tensor. ', h('code', null, 'llmscan --inventory-only'),
        ' and a scan that stops early write no delta.csv; a scan that finds no 2-D float tensor to analyse writes an empty one.'),
      h('p', null, 'The ranking and inventory are complete, but no tensor has singular values to compare with the others of its kind. '
        + 'If the run was inventory-only, rerun llmscan without ', h('code', null, '--inventory-only'), ' on the same pair to add them.'));
  }
  const s = model.deltaSummary;
  const tiles = h('dl', { class: 'tiles' },
    tile('2-D tensors scanned', formatCount(s.scanned), `${formatCount(s.scanned - s.measured)} without usable stats (ΔW = 0 or unusable)`),
    tile('Changed, with ΔW stats', formatCount(s.measured)),
    tile('Median effective rank', formatNumber(s.medianRank)),
    tile('Highest effective rank', formatNumber(s.top?.effectiveRank ?? null), s.top && tensorButton(s.top)),
    tile('Median σ₁ / σ₂', formatNumber(s.medianGap), 'gap between the first two directions'));
  const charts = model.measured.length ? deltaCharts(model) : null;
  state.redrawCharts = charts?.draw ?? null;
  const listed = model.delta.filter((t) => t.status !== 'identical');
  const columns = [COLUMNS.name, COLUMNS.layer, COLUMNS.kind, COLUMNS.tier, COLUMNS.shape, COLUMNS.effectiveRank, COLUMNS.sv1, COLUMNS.gap];
  return [
    tiles,
    charts?.box ?? emptyState('No changed 2-D tensor to plot',
      h('p', null, 'delta.csv lists tensors, but none carries a usable effective rank: every scanned tensor is identical to base, or its stats are unusable.')),
    listed.length > 0 && h('h3', { class: 'section-title' }, 'Changed tensors in delta.csv'),
    listed.length > 0 && listing({
      controls: [
        kindSelect('delta-kind', kindOrder(listed.map((t) => t.kind)), view.kind, (kind) => { view.kind = kind; view.page = 0; state.refresh(); }),
        searchBox('delta-filter', view.query, (query) => { view.query = query; view.page = 0; state.refresh(); }),
      ],
      results: () => {
        const rows = filterTensors(listed, view);
        return [
          h('p', { class: 'count', role: 'status' }, `${formatCount(rows.length)} of ${formatCount(listed.length)} changed tensors. `
            + `${formatCount(model.delta.length - listed.length)} identical 2-D tensors have ΔW = 0 and are not listed.`),
          resultsFor(view, rows, (pageRows, onSort) => tensorTable({ caption: 'ΔW stats per changed tensor', columns, rows: pageRows, sort: view.sort, onSort }), {
            empty: 'No changed tensor has this kind and text.',
            onClear: () => { Object.assign(view, { kind: '', query: '' }); renderPanel(); },
          }),
        ];
      },
    }),
  ];
}

const VIEWS = { ranking: rankingView, inventory: inventoryView, delta: deltaView };

// ---------------------------------------------------------------- detail

function fact(label, value) {
  return [h('dt', null, label), h('dd', null, value)];
}

function sides(base, suspect, format = (v) => v) {
  if (base === suspect) return format(base) || '–';
  return `base ${base ? format(base) : 'absent'} → suspect ${suspect ? format(suspect) : 'absent'}`;
}

function deltaFacts(model, t) {
  if (!model.hasDelta) return h('p', null, 'delta.csv is missing or lists no tensor, so no tensor in this report has ΔW stats.');
  if (!t.scanned) return h('p', null, 'Not in delta.csv. The scan covers 2-D float tensors that exist in both models with the same shape.');
  if (!t.spectrum.length && t.effectiveRank == null) {
    return h('p', null, t.status === 'identical' ? 'ΔW = 0: identical tensors are listed in delta.csv without stats.' : 'delta.csv lists this tensor without stats.');
  }
  const kind = model.kindStats.get(t.kind);
  const ratio = kind && t.effectiveRank > 0 ? t.effectiveRank / kind.median : null;
  const peers = model.measured.filter((p) => p.kind === t.kind && p !== t && p.spectrum.length);
  return [
    h('dl', { class: 'facts' },
      fact('Effective rank', formatNumber(t.effectiveRank)),
      fact('σ₁', formatNumber(t.sv1)),
      fact('σ₁ / σ₂', formatNumber(t.gap)),
      kind && fact(['Median of ', formatCount(kind.count), ' changed ', h('code', null, t.kind)],
        [formatNumber(kind.median), ratio != null && Number.isFinite(ratio) && h('p', { class: 'fine' }, `this tensor is ${formatNumber(ratio)}× that`)])),
    t.spectrum.length > 0 && h('figure', null, tensorSpectrumChart(t, peers, PALETTE.ink),
      h('figcaption', { class: 'fine' }, `Top ${t.spectrum.length === 1 ? 'singular value' : `${t.spectrum.length} singular values`} of ΔW`
        + (peers.length ? `; gray: the other ${formatCount(peers.length)} changed ${t.kind} tensors.` : '.'))),
  ];
}

function detailContent(model, t, { modal }) {
  const index = state.rows.indexOf(t);
  const step = (label, direction, to) => h('button', {
    type: 'button', class: 'button quiet', 'data-step': direction, disabled: to == null,
    onclick: () => { select(to.name, { open: true }); document.querySelector(`${modal ? '#detail-dialog' : '#detail'} [data-step="${direction}"]`)?.focus(); },
  }, label);
  const tier = t.tier ? tierInfo(t.tier) : null;
  return [
    h('div', { class: 'detail-bar' },
      h('p', { class: 'detail-rank' }, tierBadge(t.tier, { full: true }),
        t.rank != null && h('span', null, `Rank ${formatCount(t.rank)} of ${formatCount(model.ranking.length)}`)),
      h('div', { class: 'detail-steps' },
        index >= 0 && [step('Previous', 'previous', state.rows[index - 1]), step('Next', 'next', state.rows[index + 1])],
        modal && h('button', { type: 'button', class: 'button quiet', onclick: () => dialog.close() }, 'Close'))),
    h('h2', { class: 'tensor-name', id: modal ? 'detail-dialog-title' : 'detail-title', tabindex: '-1' }, t.name),
    h('section', null,
      h('h3', null, t.tier ? 'Why it is ranked here' : 'Not ranked'),
      t.tier
        ? [h('p', { class: 'reason-text' }, t.reason || 'ranking.csv gives no reason for this row.'),
          tier.meaning && h('p', { class: 'fine' }, h('strong', null, `${tier.id} · ${tier.label}. `), tier.meaning)]
        : h('p', null, t.status === 'identical'
          ? 'Identical to base (same bytes, or the same values after a float re-save), so the ranking leaves it out. Identical weights say nothing about the rest of the model.'
          : 'ranking.csv has no row for this tensor.')),
    h('section', null, h('h3', null, 'Tensor'),
      h('dl', { class: 'facts' },
        fact('Status', statusBadge(t.status)),
        fact('Kind', h('code', null, t.kind || '–')),
        fact('Layer', t.layer == null ? 'outside the layer stack' : String(t.layer)),
        fact('Shape', sides(t.baseShape, t.suspectShape, formatShape)),
        fact('Dtype', sides(t.baseDtype, t.suspectDtype)),
        fact('Parameters', formatCount(t.numel)))),
    h('section', null, h('h3', null, 'ΔW = W suspect − W base'), deltaFacts(model, t)),
  ];
}

function renderDetail({ open = false } = {}) {
  const aside = document.getElementById('detail');
  if (!aside) return;
  const t = state.model.tensors.get(state.selected);
  aside.replaceChildren(h('div', { class: 'detail-body' }, t ? detailContent(state.model, t, { modal: false })
    : h('p', { class: 'placeholder' }, 'Select a tensor to see why it is ranked where it is, its shapes and its ΔW spectrum.')));
  if (wide.matches || !t || !(open || dialog.open)) return;
  dialog.replaceChildren(h('div', { class: 'detail-body' }, detailContent(state.model, t, { modal: true })));
  if (!dialog.open) dialog.showModal();
}

function select(name, { open = false } = {}) {
  state.selected = name;
  const index = state.rows.findIndex((t) => t.name === name);
  const view = state.view[state.tab];
  if (index >= 0 && Math.floor(index / PAGE_SIZE) !== view.page) {
    view.page = Math.floor(index / PAGE_SIZE);
    state.refresh?.();
  }
  for (const row of main.querySelectorAll('tr[data-tensor]')) {
    const current = row.dataset.tensor === name;
    row.classList.toggle('selected', current);
    row.querySelector('.tensor-link')?.setAttribute('aria-current', String(current));
  }
  state.redrawCharts?.();
  renderDetail({ open });
}

// ---------------------------------------------------------------- page

function renderPanel() {
  const panel = document.getElementById('panel');
  state.refresh = null;
  state.redrawCharts = null;
  panel.setAttribute('aria-labelledby', `tab-${state.tab}`);
  keepFocus(panel, () => panel.replaceChildren(...[VIEWS[state.tab](state.model)].flat(Infinity).filter(Boolean)));
  for (const tab of main.querySelectorAll('[role="tab"]')) {
    const active = tab.id === `tab-${state.tab}`;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
  }
  renderDetail();
}

function showTab(id, { focus = false } = {}) {
  state.tab = id;
  history.replaceState(null, '', `#${id}`);
  renderPanel();
  if (focus) document.getElementById(`tab-${id}`).focus();
}

function summary(model, source) {
  const structural = model.inventory.filter((t) => !['identical', 'changed'].includes(t.status)).length;
  return h('section', { class: 'summary', 'aria-labelledby': 'pair-title' },
    h('div', { class: 'pair' },
      h('p', { class: 'eyebrow' }, h('span', { class: source.demo ? 'source demo' : 'source' }, source.demo ? 'Demo' : 'Imported'), source.label),
      h('h1', { id: 'pair-title' }, h('span', { class: 'model' }, model.suspectName), h('span', { class: 'vs' }, 'compared with base'), h('span', { class: 'model' }, model.baseName)),
      h('dl', { class: 'paths' }, h('div', null, h('dt', null, 'suspect'), h('dd', null, model.suspect || '–')), h('div', null, h('dt', null, 'base'), h('dd', null, model.base || '–')))),
    h('ul', { class: 'tier-cards', 'aria-label': 'Ranked tensors by tier' }, TIERS.map((tier) => {
      const count = model.tierCounts.get(tier.id) ?? 0;
      return h('li', null, h('button', {
        type: 'button', class: 'tier-card', 'data-tier': tier.id, disabled: count === 0,
        'aria-label': `${count} ranked in ${tier.id}, ${tier.label}: show them`,
        onclick: () => { Object.assign(state.view.ranking, { tiers: new Set([tier.id]), query: '', page: 0 }); showTab('ranking', { focus: true }); },
      }, h('span', { class: 'tier-count' }, formatCount(count)), h('span', { class: 'tier-name' }, h('span', { class: 'tier-id' }, tier.id), tier.label)));
    })),
    h('dl', { class: 'tiles' },
      tile('Tensors compared', formatCount(model.inventory.length)),
      tile('Changed values', formatCount(model.statusCounts.get('changed') ?? 0)),
      tile('Structural mismatches', formatCount(structural), 'missing, added, reshaped or retyped'),
      tile('Parameters in identical tensors', formatPercent(model.identicalParams, model.totalParams))),
    h('p', { class: 'caveat' }, CAVEAT));
}

function render() {
  const { model, source } = state;
  if (!TABS.some((t) => t.id === state.tab)) state.tab = 'ranking';
  const tabs = h('div', {
    class: 'tabs', role: 'tablist', 'aria-label': 'Report sections',
    onkeydown: (event) => {
      const move = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      const at = TABS.findIndex((t) => t.id === state.tab);
      const to = event.key === 'Home' ? 0 : event.key === 'End' ? TABS.length - 1 : move ? (at + move + TABS.length) % TABS.length : null;
      if (to == null) return;
      event.preventDefault();
      showTab(TABS[to].id, { focus: true });
    },
  }, TABS.map((tab) => h('button', {
    type: 'button', role: 'tab', id: `tab-${tab.id}`, 'aria-controls': 'panel', onclick: () => showTab(tab.id),
  }, tab.label, h('span', { class: 'chip-count' }, formatCount(tab.count(model))))));
  main.replaceChildren(
    summary(model, source), tabs,
    h('div', { class: 'workspace' },
      h('div', { id: 'panel', class: 'panel', role: 'tabpanel', tabindex: '0' }),
      h('aside', { id: 'detail', class: 'detail', 'aria-label': 'Selected tensor' })));
  renderPanel();
}

function renderWelcome() {
  main.replaceChildren(h('section', { class: 'welcome' },
    h('h1', null, 'Inspect an llmscan comparison'),
    h('p', null, 'Open the folder llmscan wrote for a base and suspect pair to read its ranked tensors, the reason behind each rank, the tensor inventory and the ΔW stats.'),
    h('div', { class: 'welcome-actions' },
      h('button', { type: 'button', class: 'button primary', 'data-action': 'folder' }, 'Import report folder'),
      h('button', { type: 'button', class: 'button', 'data-action': 'files' }, 'Pick report files'),
      h('button', { type: 'button', class: 'button', 'data-action': 'demo' }, 'Open the demo')),
    h('p', { class: 'fine' }, 'You can also drop the folder anywhere on this page. A report folder is ', h('code', null, 'out/<suspect>-vs-<base>/'),
      ' and holds pair.txt, inventory.csv, ranking.csv and, unless the run was inventory-only, delta.csv.'),
    h('p', { class: 'fine' }, 'Files are read inside this browser tab. Nothing is uploaded.')));
}

wide.addEventListener('change', () => {
  if (wide.matches && dialog.open) dialog.close();
  if (state.model) renderDetail();
});
dialog.addEventListener('click', (event) => { if (event.target === dialog) dialog.close(); });
window.addEventListener('hashchange', () => {
  const id = location.hash.slice(1);
  if (state.model && id !== state.tab && TABS.some((tab) => tab.id === id)) showTab(id);
});

wireImport();
state.tab = location.hash.slice(1) || 'ranking';
main.replaceChildren(h('p', { class: 'loading' }, 'Opening the demo report…'));
openDemo();

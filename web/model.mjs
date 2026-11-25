// Pure view-model for a parsed llmscan report: no DOM, so it runs under `node --test`.
// Input is what readReport() resolves to: {base, suspect, inventory, ranking, delta}, with the
// arrays keyed by the CLI's CSV headers and every cell a string.

export const TIERS = [
  {
    id: 'T1',
    label: 'Structure',
    meaning: 'Present in only one model, a shape mismatch, or a non-float dtype mismatch. '
      + 'An abliteration edits values in place; it does not add, drop or resize tensors.',
  },
  {
    id: 'T2',
    label: 'Unexpected location',
    meaning: 'Changed, but not one of the matrix kinds an abliteration is expected to edit.',
  },
  {
    id: 'T3',
    label: 'Outlier',
    meaning: 'An expected kind whose ΔW stats stand out from the other changed tensors of the same '
      + 'kind in this model, or whose stats could not be used.',
  },
  {
    id: 'T4',
    label: 'Consistent with abliteration',
    meaning: 'Changed the way an abliteration would. Least suspicious in this ranking, not certified clean.',
  },
];

// Reporting order of inventory.csv statuses; anything else a report carries is listed after these.
export const STATUSES = ['identical', 'changed', 'dtype_mismatch', 'shape_mismatch', 'only_base', 'only_suspect'];

const SV_COLUMN = /^sv_(\d+)$/;

// rank.py's MIN_GROUP: changed tensors of one kind needed before a median means anything
export const MIN_GROUP = 3;

/**
 * The kind rank.py groups a tensor under (its `canonical_kind`): fused MoE `experts.down_proj` counts as
 * `down_proj`, GGUF `ffn_down_exps` and `ffn_down_shexp` as `ffn_down`.
 */
export function canonicalKind(kind) {
  return kind.replace(/^experts\./, '').replace(/_(exps|shexp)$/, '');
}

export function tierInfo(id) {
  return TIERS.find((t) => t.id === id) ?? { id, label: id, meaning: '' };
}

export function statusLabel(status) {
  return status ? status.replaceAll('_', ' ') : 'unknown';
}

/** A CSV cell as a number; blank or non-numeric -> null. Python's `nan` / `inf` spellings are kept. */
export function parseNumber(cell) {
  const text = (cell ?? '').trim().toLowerCase();
  if (text === '') return null;
  if (text === 'nan') return NaN;
  if (/^[+-]?inf(inity)?$/.test(text)) return text.startsWith('-') ? -Infinity : Infinity;
  const value = Number(text);
  return Number.isNaN(value) ? null : value;
}

const usable = (v) => v != null && Number.isFinite(v) && v > 0;

/** rank.py's test for a row it can judge: a stat is given, and every given stat is finite and positive. */
const judgeable = (t) => (t.effectiveRank != null || t.sv1 != null)
  && [t.effectiveRank, t.sv1].every((v) => v == null || usable(v));

export function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** `layers.2` before `layers.10`, like the CLI's natural name order. */
export function naturalCompare(a, b) {
  const pa = a.split(/(\d+)/), pb = b.split(/(\d+)/);
  for (let i = 0; i < Math.min(pa.length, pb.length); i++) {
    if (pa[i] === pb[i]) continue;
    if (i % 2) return Number(pa[i]) - Number(pb[i]) || (pa[i] < pb[i] ? -1 : 1);
    return pa[i] < pb[i] ? -1 : 1;
  }
  return pa.length - pb.length;
}

/** The last path segment of a model path or Hugging Face id, the way the CLI names its output. */
export function shortName(model) {
  const parts = String(model ?? '').replaceAll('\\', '/').split('/').filter(Boolean);
  return parts.at(-1) ?? '';
}

function spectrumOf(row) {
  const columns = Object.keys(row)
    .map((key) => [key, SV_COLUMN.exec(key)])
    .filter(([, match]) => match)
    .sort((a, b) => Number(a[1][1]) - Number(b[1][1]));
  const values = [];
  for (const [key] of columns) {
    const value = parseNumber(row[key]);
    if (value == null) break;
    values.push(value);
  }
  return values;
}

function blankTensor(name) {
  return {
    name, layer: null, kind: '', status: '',
    baseDtype: '', suspectDtype: '', baseShape: '', suspectShape: '', numel: null,
    rank: null, tier: null, reason: '', yardstick: null,
    rankEffectiveRank: null, rankSv1: null,
    scanned: false, effectiveRank: null, sv1: null, spectrum: [], gap: null, seconds: null,
  };
}

const hasRankStats = (tensor) => tensor.rankEffectiveRank != null || tensor.rankSv1 != null;

function scanProvenance(rows, tensors, ranking, hasRankingStats) {
  if (!rows.length) return 'absent';
  // An inventory-only rerun leaves the prior delta.csv untouched. Check the current inventory and
  // both summary columns before joining, so an incompatible scan cannot supply any tensor facts.
  const seen = new Set();
  let unverified = false;
  for (const row of rows) {
    const t = tensors.get(row.name);
    if (!t || t.status !== row.status || t.kind !== row.kind || t.layer !== parseNumber(row.layer) || t.baseShape !== row.shape) return 'incompatible';
    const er = parseNumber(row.effective_rank), sv1 = parseNumber(row.sv_001);
    if (!Object.is(t.rankEffectiveRank, er) || !Object.is(t.rankSv1, sv1)) {
      if (hasRankingStats || t.status === 'identical') return 'incompatible';
      // The CLI writes the same ranking both when it skips a scan and when it stops one early.
      unverified = true;
    }
    seen.add(row.name);
  }
  if (ranking.some((t) => hasRankStats(t) && !seen.has(t.name))) return 'incompatible';
  return unverified ? 'unverified' : 'compatible';
}

/**
 * Join the three tables by tensor name into one record per tensor.
 * `ranking` keeps the CLI's order, `inventory` the inventory's, `delta` the scan's.
 */
export function buildWorkbench(report) {
  const tensors = new Map();
  const tensor = (row) => {
    const name = row.name ?? '';
    if (!tensors.has(name)) tensors.set(name, blankTensor(name));
    const t = tensors.get(name);
    t.layer ??= parseNumber(row.layer);
    t.kind ||= row.kind ?? '';
    t.status ||= row.status ?? '';
    return t;
  };

  const inventory = (report.inventory ?? []).map((row) => Object.assign(tensor(row), {
    baseDtype: row.base_dtype ?? '', suspectDtype: row.suspect_dtype ?? '',
    baseShape: row.base_shape ?? '', suspectShape: row.suspect_shape ?? '',
    numel: parseNumber(row.numel),
  }));

  const ranking = (report.ranking ?? []).map((row, i) => {
    const t = tensor(row);
    t.rank = parseNumber(row.rank) ?? i + 1;
    t.tier = row.tier ?? '';
    t.reason = row.reason ?? '';
    t.rankEffectiveRank = t.effectiveRank = parseNumber(row.effective_rank);
    t.rankSv1 = t.sv1 = parseNumber(row.sv_001);
    return t;
  });

  const deltaRows = report.delta ?? [];
  const hasRankingStats = ranking.some(hasRankStats);
  const deltaSource = scanProvenance(deltaRows, tensors, ranking, hasRankingStats);
  const delta = (deltaSource === 'incompatible' ? [] : deltaRows).map((row) => {
    const spectrum = spectrumOf(row);
    return Object.assign(tensors.get(row.name), {
      scanned: true,
      effectiveRank: parseNumber(row.effective_rank),
      sv1: parseNumber(row.sv_001),
      spectrum,
      gap: usable(spectrum[0]) && usable(spectrum[1]) ? spectrum[0] / spectrum[1] : null,
      seconds: parseNumber(row.seconds),
    });
  });

  const tierCounts = new Map(TIERS.map((t) => [t.id, 0]));
  for (const t of ranking) tierCounts.set(t.tier, (tierCounts.get(t.tier) ?? 0) + 1);

  const statusCounts = new Map();
  const kinds = new Map();
  let totalParams = 0, identicalParams = 0;
  for (const t of inventory) {
    statusCounts.set(t.status, (statusCounts.get(t.status) ?? 0) + 1);
    if (!kinds.has(t.kind)) kinds.set(t.kind, { kind: t.kind, total: 0, counts: new Map() });
    const k = kinds.get(t.kind);
    k.total += 1;
    k.counts.set(t.status, (k.counts.get(t.status) ?? 0) + 1);
    totalParams += t.numel ?? 0;
    if (t.status === 'identical') identicalParams += t.numel ?? 0;
  }
  const statuses = [
    ...STATUSES.filter((s) => statusCounts.has(s)),
    ...[...statusCounts.keys()].filter((s) => !STATUSES.includes(s)).sort(),
  ];
  const changedOf = (k) => k.counts.get('changed') ?? 0;

  // What the charts draw: changed tensors whose row rank.py can judge and that have an effective rank.
  const measured = delta.filter((t) => t.status === 'changed' && judgeable(t) && t.effectiveRank != null);
  const top = measured.reduce((best, t) => (best && best.effectiveRank >= t.effectiveRank ? best : t), null);

  // The yardstick rank.py judged a tensor by: the median over the changed tensors of its canonical kind with
  // usable stats, effective rank and (for a tensor without one) sv_001 kept apart, and no median below MIN_GROUP.
  // Only expected kinds are judged this way, and a changed tensor of any other kind is the one ranked T2.
  const groups = new Map();
  for (const t of ranking) {
    if (t.status !== 'changed' || t.tier === 'T2' || !hasRankStats(t) || !judgeable(t)) continue;
    const stat = t.effectiveRank != null ? 'effectiveRank' : 'sv1';
    const key = `${stat} ${canonicalKind(t.kind)}`;
    if (!groups.has(key)) groups.set(key, { stat, kind: canonicalKind(t.kind), members: [] });
    groups.get(key).members.push(t);
  }
  for (const { stat, kind, members } of groups.values()) {
    const yardstick = { stat, kind, count: members.length, median: members.length >= MIN_GROUP ? median(members.map((t) => t[stat])) : null };
    for (const t of members) t.yardstick = yardstick;
  }

  return {
    base: report.base ?? '', suspect: report.suspect ?? '',
    baseName: shortName(report.base) || 'base', suspectName: shortName(report.suspect) || 'suspect',
    tensors, inventory, ranking, delta,
    hasDelta: delta.length > 0, deltaSource, hasRankingStats,
    tierCounts, statusCounts, statuses,
    kinds: [...kinds.values()].sort((a, b) => changedOf(b) - changedOf(a) || naturalCompare(a.kind, b.kind)),
    totalParams, identicalParams,
    measured,
    deltaSummary: {
      scanned: delta.length,
      measured: measured.length,
      medianRank: median(measured.map((t) => t.effectiveRank)),
      top,
      medianGap: median(measured.map((t) => t.gap).filter(usable)),
    },
  };
}

function matches(t, query) {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  const haystack = `${t.name} ${t.kind} ${statusLabel(t.status)} ${t.tier ?? ''} ${t.reason}`.toLowerCase();
  return terms.every((term) => haystack.includes(term));
}

/** `tiers`, `statuses`: a Set, empty = no restriction. `kind`: '' = any. */
export function filterTensors(rows, { tiers, statuses, kind = '', query = '' } = {}) {
  return rows.filter((t) =>
    (!tiers?.size || tiers.has(t.tier))
    && (!statuses?.size || statuses.has(t.status))
    && (!kind || t.kind === kind)
    && matches(t, query));
}

/** Stable sort by a tensor field; null, NaN and '' always sink to the bottom. */
export function sortTensors(rows, key, direction = 'asc') {
  const sign = direction === 'desc' ? -1 : 1;
  const missing = (v) => v == null || v === '' || Number.isNaN(v);
  return [...rows].sort((a, b) => {
    const x = a[key], y = b[key];
    if (missing(x) || missing(y)) return missing(x) - missing(y);
    return sign * (typeof x === 'string' ? naturalCompare(x, y) : x - y);
  });
}

export function pageOf(rows, page, size) {
  const pages = Math.max(1, Math.ceil(rows.length / size));
  const current = Math.min(Math.max(page, 0), pages - 1);
  const start = current * size;
  return { rows: rows.slice(start, start + size), page: current, pages, start, end: Math.min(start + size, rows.length) };
}

/** Four significant digits (1.00026 prints as 1.000, not as an exact 1); blank cells print as an en dash. */
export function formatNumber(value) {
  if (value == null) return '–';
  if (Number.isNaN(value)) return 'NaN';
  if (!Number.isFinite(value)) return value > 0 ? '∞' : '−∞';
  if (value === 0) return '0';
  const abs = Math.abs(value);
  if (abs >= 1e6 || abs < 1e-3) return value.toExponential(2).replace('e+', 'e').replaceAll('-', '−');
  if (Number.isInteger(value) || abs >= 1000) return Math.round(value).toLocaleString('en-US').replace('-', '−');
  return value.toPrecision(4).replace('-', '−');
}

export function formatCount(value) {
  return value == null ? '–' : value.toLocaleString('en-US');
}

/** `1 tensor`, `2 tensors`. */
export function plural(count, noun) {
  return `${formatCount(count)} ${noun}${count === 1 ? '' : 's'}`;
}

export function formatPercent(part, whole) {
  return whole > 0 ? `${(100 * part / whole).toFixed(2)}%` : '–';
}

/** Elements in a `128x32` shape; a scalar's shape is blank. */
export function shapeCount(shape) {
  return shape.split('x').filter(Boolean).reduce((count, size) => count * Number(size), 1);
}

export function formatShape(shape) {
  return shape ? shape.replaceAll('x', ' × ') : '–';
}

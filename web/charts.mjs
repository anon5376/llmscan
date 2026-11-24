// SVG charts in the palette of the CLI's PNG figures (src/llmscan/plots.py).
import { append, svg } from './dom.mjs';
import { formatNumber, naturalCompare } from './model.mjs';

export const PALETTE = {
  surface: '#fcfcfb', ink: '#0b0b0b', ink2: '#52514e', muted: '#898781', grid: '#e1e0d9', axis: '#c3c2b7',
};
// fixed categorical slots: a kind keeps its colour across every report, as in the PNGs
const SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
const KIND_ORDER = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens'];
// sequential blue: early layers light, late layers dark
const LAYER_STOPS = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281', '#0d366b'];

export function kindOrder(kinds) {
  const slot = (k) => (KIND_ORDER.includes(k) ? KIND_ORDER.indexOf(k) : KIND_ORDER.length);
  return [...new Set(kinds)].sort((a, b) => slot(a) - slot(b) || naturalCompare(a, b));
}

/** Known kinds take their fixed slot; unknown kinds fill the remaining slots, then gray. */
export function kindColors(kinds) {
  const colors = new Map(KIND_ORDER.filter((k) => kinds.includes(k)).map((k) => [k, SERIES[KIND_ORDER.indexOf(k)]]));
  const free = SERIES.filter((c) => ![...colors.values()].includes(c));
  for (const k of kindOrder(kinds)) if (!colors.has(k)) colors.set(k, free.shift() ?? PALETTE.muted);
  return colors;
}

/** Colour for a position t in [0, 1] along the layer ramp. */
export function layerColor(t) {
  const x = Math.min(Math.max(t, 0), 1) * (LAYER_STOPS.length - 1);
  const i = Math.min(Math.floor(x), LAYER_STOPS.length - 2);
  const channel = (stop, at) => parseInt(stop.slice(at, at + 2), 16);
  const mix = (at) => Math.round(channel(LAYER_STOPS[i], at) + (channel(LAYER_STOPS[i + 1], at) - channel(LAYER_STOPS[i], at)) * (x - i));
  return `rgb(${mix(1)}, ${mix(3)}, ${mix(5)})`;
}

export const layerGradient = `linear-gradient(to right, ${LAYER_STOPS.join(', ')})`;

/** About `count` round ticks covering [lo, hi]. */
export function linearTicks(lo, hi, count = 5) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw);
  const ticks = [];
  for (let i = Math.ceil(lo / step - 1e-9); i * step <= hi + step * 1e-9; i++) ticks.push(Number((i * step).toPrecision(12)));
  return ticks;
}

/** Powers of ten inside [lo, hi] (both positive). */
export function logTicks(lo, hi) {
  const ticks = [];
  for (let e = Math.ceil(Math.log10(lo) - 1e-9); e <= Math.floor(Math.log10(hi) + 1e-9); e++) ticks.push(Number(`1e${e}`)); // 10 ** -4 is 0.00009999999999999999
  return ticks;
}

/** Value -> pixel; log scales clamp non-positive values to the bottom of the domain. */
export function scale([d0, d1], [r0, r1], log = false) {
  const f = log ? (v) => Math.log10(Math.max(v, d0)) : (v) => v;
  const span = f(d1) - f(d0) || 1;
  return (v) => r0 + (f(v) - f(d0)) / span * (r1 - r0);
}

function padded(lo, hi, log) {
  if (log) return [lo / 1.6, hi * 1.6];
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.08;
  return [lo - pad, hi + pad];
}

const tickLabel = (v, log) => (log ? `1e${Math.round(Math.log10(v))}` : formatNumber(v));

/** Plot frame: gridlines, tick labels and axis titles. Returns the scales and the plot group. */
function frame({ width, height, x, y, xLabel, yLabel, xTicks, label }) {
  const m = { top: 12, right: 14, bottom: 40, left: 58 };
  const sx = scale(x.domain, [m.left, width - m.right]);
  const sy = scale(y.domain, [height - m.bottom, m.top], y.log);
  const yTicks = (y.log ? logTicks(...y.domain) : linearTicks(...y.domain)).filter((t) => t >= y.domain[0] && t <= y.domain[1]);
  const text = (props, value) => svg('text', { fill: PALETTE.ink2, 'font-size': 11, ...props }, value);
  const root = svg('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': label, class: 'chart' },
    yTicks.map((t) => [
      svg('line', { x1: m.left, x2: width - m.right, y1: sy(t), y2: sy(t), stroke: PALETTE.grid }),
      text({ x: m.left - 8, y: sy(t) + 4, 'text-anchor': 'end' }, tickLabel(t, y.log)),
    ]),
    xTicks.map((t) => [
      svg('line', { x1: sx(t), x2: sx(t), y1: m.top, y2: height - m.bottom, stroke: PALETTE.grid }),
      text({ x: sx(t), y: height - m.bottom + 16, 'text-anchor': 'middle' }, String(t)),
    ]),
    svg('line', { x1: m.left, x2: width - m.right, y1: height - m.bottom, y2: height - m.bottom, stroke: PALETTE.axis }),
    svg('line', { x1: m.left, x2: m.left, y1: m.top, y2: height - m.bottom, stroke: PALETTE.axis }),
    text({ x: (m.left + width - m.right) / 2, y: height - 6, 'text-anchor': 'middle' }, xLabel),
    text({ x: 14, y: (m.top + height - m.bottom) / 2, 'text-anchor': 'middle', transform: `rotate(-90 14 ${(m.top + height - m.bottom) / 2})` }, yLabel),
  );
  return { root, sx, sy };
}

const integerTicks = (lo, hi, count) => [...new Set(linearTicks(lo, hi, count).map(Math.round))].filter((t) => t >= lo && t <= hi);

function pickable(node, tensor, onPick) {
  node.classList.add('pick');
  node.addEventListener('click', () => onPick(tensor.name));
  return node;
}

/** Effective rank of ΔW against layer, one line per matrix kind; a point opens its tensor. */
export function rankByLayerChart(tensors, { selected, onPick }) {
  const points = tensors.filter((t) => t.layer != null);
  const kinds = kindOrder(points.map((t) => t.kind));
  const colors = kindColors(kinds);
  const layers = points.map((t) => t.layer), ranks = points.map((t) => t.effectiveRank);
  const log = Math.max(...ranks) / Math.min(...ranks) > 30;
  const x = { domain: padded(Math.min(...layers), Math.max(...layers)) };
  const y = { domain: padded(Math.min(...ranks), Math.max(...ranks), log), log };
  const { root, sx, sy } = frame({
    width: 720, height: 300, x, y, xLabel: 'layer', yLabel: 'effective rank of ΔW',
    xTicks: integerTicks(Math.min(...layers), Math.max(...layers), 12),
    label: `Effective rank of ΔW by layer for ${points.length} changed tensors of ${kinds.length} kinds. The table below lists the same values.`,
  });
  // kinds sit side by side inside a layer, so two kinds with the same rank do not hide each other
  const step = Math.min(0.16, 0.6 / kinds.length);
  for (const [i, kind] of kinds.entries()) {
    const series = points.filter((t) => t.kind === kind).sort((a, b) => a.layer - b.layer);
    const at = (t) => sx(t.layer + (i - (kinds.length - 1) / 2) * step);
    append(root,
      svg('polyline', { points: series.map((t) => `${at(t)},${sy(t.effectiveRank)}`).join(' '), fill: 'none', stroke: colors.get(kind), 'stroke-width': 1.5 }),
      series.map((t) => pickable(svg('circle', {
        cx: at(t), cy: sy(t.effectiveRank), r: t.name === selected ? 6.5 : 4.5,
        fill: colors.get(kind), stroke: t.name === selected ? PALETTE.ink : PALETTE.surface, 'stroke-width': t.name === selected ? 2 : 1.5,
      }, svg('title', null, `${t.name}\neffective rank ${formatNumber(t.effectiveRank)}`)), t, onPick)),
    );
  }
  return { chart: root, legend: kinds.map((kind) => ({ kind, color: colors.get(kind) })) };
}

function spectrumPath(t, sx, sy) {
  return t.spectrum.map((v, i) => `${sx(i + 1)},${sy(v)}`).join(' ');
}

function spectrumDomain(tensors) {
  const values = tensors.flatMap((t) => t.spectrum).filter((v) => Number.isFinite(v) && v > 0);
  const longest = Math.max(2, ...tensors.map((t) => t.spectrum.length));
  return {
    x: { domain: [1, longest] },
    y: { domain: values.length ? padded(Math.min(...values), Math.max(...values), true) : [0.1, 10], log: true },
    xTicks: integerTicks(1, longest, 6),
  };
}

/** Singular values of every changed tensor of one kind; lines run light (early layer) to dark (late). */
export function kindSpectraChart(kind, tensors, { layerRange: [lo, hi], selected, onPick }) {
  const { x, y, xTicks } = spectrumDomain(tensors);
  const { root, sx, sy } = frame({
    width: 360, height: 250, x, y, xTicks, xLabel: 'singular value index i', yLabel: 'σᵢ(ΔW)',
    label: `Singular values of ΔW for ${tensors.length} changed ${kind} tensors, log scale.`,
  });
  const ordered = [...tensors].sort((a, b) => (a.name === selected) - (b.name === selected) || (a.layer ?? -1) - (b.layer ?? -1));
  for (const t of ordered) {
    const color = t.layer == null ? PALETTE.ink2 : layerColor((t.layer - lo) / Math.max(hi - lo, 1));
    append(root, pickable(svg('polyline', {
      points: spectrumPath(t, sx, sy), fill: 'none', 'stroke-linejoin': 'round',
      stroke: t.name === selected ? PALETTE.ink : color, 'stroke-width': t.name === selected ? 2.5 : 1.3,
    }, svg('title', null, `${t.name}\neffective rank ${formatNumber(t.effectiveRank)}`)), t, onPick));
  }
  return root;
}

/** One tensor's spectrum in `color`, drawn over the other changed tensors of its kind in gray. */
export function tensorSpectrumChart(tensor, peers, color) {
  const { x, y, xTicks } = spectrumDomain([tensor, ...peers]);
  const { root, sx, sy } = frame({
    width: 360, height: 220, x, y, xTicks, xLabel: 'singular value index i', yLabel: 'σᵢ(ΔW)',
    label: `Singular values of ΔW for ${tensor.name}, log scale`
      + (peers.length ? `, over ${peers.length} other changed ${tensor.kind} tensors in gray.` : '.'),
  });
  append(root,
    peers.map((t) => svg('polyline', { points: spectrumPath(t, sx, sy), fill: 'none', stroke: PALETTE.axis, 'stroke-width': 1 })),
    svg('polyline', { points: spectrumPath(tensor, sx, sy), fill: 'none', stroke: color, 'stroke-width': 2.2, 'stroke-linejoin': 'round' }),
    tensor.spectrum.length <= 12 && tensor.spectrum.map((v, i) => svg('circle', { cx: sx(i + 1), cy: sy(v), r: 3, fill: color })),
  );
  return root;
}

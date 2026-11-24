// The workbench's view-model and chart maths (web/model.mjs, web/charts.mjs): node --test tests/web/workbench.test.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { readReport } from "../../web/report.mjs";
import {
  distanceToPath, extent, kindColors, kindOffset, kindOrder, layerColor, linearTicks, logTicks, scale, spectrumDomain, spectrumRuns,
} from "../../web/charts.mjs";
import {
  buildWorkbench, canonicalKind, filterTensors, formatNumber, formatPercent, median, naturalCompare, pageOf, parseNumber,
  plural, shortName, sortTensors, tierInfo,
} from "../../web/model.mjs";

const WEB = new URL("../../web/", import.meta.url);

// What readReport resolves to: records keyed by the CLI's CSV headers, every cell a string.
const inventoryRow = (name, layer, kind, status, shape = "8x8", suspectShape = shape) => ({
  name, layer, kind, status, base_dtype: "BF16", suspect_dtype: "BF16",
  base_shape: status === "only_suspect" ? "" : shape, suspect_shape: status === "only_base" ? "" : suspectShape, numel: "64",
});
const deltaRow = (name, layer, kind, status, rank, values = []) => ({
  name, layer, kind, status, shape: "8x8", effective_rank: rank, seconds: rank ? "0.01" : "",
  sv_001: values[0] ?? "", sv_002: values[1] ?? "", sv_003: values[2] ?? "",
});
const REPORT = {
  base: "/models/Tiny", suspect: "C:\\models\\Tiny-abliterated\\",
  inventory: [
    inventoryRow("lm_head.weight", "", "lm_head", "shape_mismatch", "32x8", "34x8"),
    inventoryRow("model.layers.0.mlp.down_proj.weight", "0", "down_proj", "changed"),
    inventoryRow("model.layers.1.mlp.down_proj.weight", "1", "down_proj", "changed"),
    inventoryRow("model.layers.2.mlp.down_proj.weight", "2", "down_proj", "changed"),
    inventoryRow("model.layers.10.mlp.down_proj.weight", "10", "down_proj", "changed"),
    inventoryRow("model.layers.2.self_attn.q_proj.weight", "2", "q_proj", "changed"),
    inventoryRow("model.layers.2.self_attn.k_proj.weight", "2", "k_proj", "identical"),
    inventoryRow("model.norm.weight", "", "norm", "identical", "8"),
  ],
  ranking: [
    { rank: "1", tier: "T1", name: "lm_head.weight", layer: "", kind: "lm_head", status: "shape_mismatch", effective_rank: "", sv_001: "", reason: "shape differs: base 32x8 vs suspect 34x8" },
    { rank: "2", tier: "T2", name: "model.layers.2.self_attn.q_proj.weight", layer: "2", kind: "q_proj", status: "changed", effective_rank: "2.5", sv_001: "0.4", reason: "changed q_proj: outside expected kinds (down_proj, o_proj)" },
    { rank: "3", tier: "T3", name: "model.layers.10.mlp.down_proj.weight", layer: "10", kind: "down_proj", status: "changed", effective_rank: "6", sv_001: "0.9", reason: "effective_rank 6 is 6.0x the median 1 of changed down_proj tensors (threshold 3x)" },
    { rank: "4", tier: "T4", name: "model.layers.0.mlp.down_proj.weight", layer: "0", kind: "down_proj", status: "changed", effective_rank: "1.01", sv_001: "0.5", reason: "changed down_proj, consistent with abliteration" },
    { rank: "5", tier: "T4", name: "model.layers.1.mlp.down_proj.weight", layer: "1", kind: "down_proj", status: "changed", effective_rank: "1", sv_001: "0.4", reason: "changed down_proj, consistent with abliteration" },
    { rank: "6", tier: "T4", name: "model.layers.2.mlp.down_proj.weight", layer: "2", kind: "down_proj", status: "changed", effective_rank: "1", sv_001: "0.3", reason: "changed down_proj, consistent with abliteration" },
  ],
  delta: [
    deltaRow("model.layers.0.mlp.down_proj.weight", "0", "down_proj", "changed", "1.01", ["0.5", "0.005", "0.004"]),
    deltaRow("model.layers.1.mlp.down_proj.weight", "1", "down_proj", "changed", "1", ["0.4", "0.004", "0.003"]),
    deltaRow("model.layers.2.mlp.down_proj.weight", "2", "down_proj", "changed", "1", ["0.3", "0.003"]),
    deltaRow("model.layers.10.mlp.down_proj.weight", "10", "down_proj", "changed", "6", ["0.9", "0.8", "0.7"]),
    deltaRow("model.layers.2.self_attn.q_proj.weight", "2", "q_proj", "changed", "2.5", ["0.4", "0.3", "0.01"]),
    deltaRow("model.layers.2.self_attn.k_proj.weight", "2", "k_proj", "identical", ""),
  ],
};

test("joins inventory, ranking and delta into one record per tensor", () => {
  const model = buildWorkbench(REPORT);
  assert.equal(model.tensors.size, 8);
  assert.deepEqual(model.ranking.map((t) => t.rank), [1, 2, 3, 4, 5, 6]);
  const outlier = model.tensors.get("model.layers.10.mlp.down_proj.weight");
  assert.equal(model.ranking[2], outlier);
  assert.deepEqual(
    { tier: outlier.tier, layer: outlier.layer, status: outlier.status, numel: outlier.numel, effectiveRank: outlier.effectiveRank, spectrum: outlier.spectrum },
    { tier: "T3", layer: 10, status: "changed", numel: 64, effectiveRank: 6, spectrum: [0.9, 0.8, 0.7] },
  );
  assert.match(outlier.reason, /6\.0x the median/);
  assert.ok(Math.abs(model.tensors.get("model.layers.0.mlp.down_proj.weight").gap - 100) < 1e-9);
  const unranked = model.tensors.get("model.layers.2.self_attn.k_proj.weight");
  assert.deepEqual([unranked.tier, unranked.rank, unranked.scanned, unranked.effectiveRank], [null, null, true, null]);
  assert.equal(model.tensors.get("model.norm.weight").scanned, false);
});

test("summarises tiers, statuses, kinds and the per-kind yardstick", () => {
  const model = buildWorkbench(REPORT);
  assert.deepEqual([...model.tierCounts], [["T1", 1], ["T2", 1], ["T3", 1], ["T4", 3]]);
  assert.deepEqual(model.statuses, ["identical", "changed", "shape_mismatch"]);
  assert.deepEqual([...model.statusCounts].sort(), [["changed", 5], ["identical", 2], ["shape_mismatch", 1]]);
  assert.deepEqual(model.kinds.map((k) => [k.kind, k.total, k.counts.get("changed") ?? 0]),
    [["down_proj", 4, 4], ["q_proj", 1, 1], ["k_proj", 1, 0], ["lm_head", 1, 0], ["norm", 1, 0]]);
  assert.equal(formatPercent(model.identicalParams, model.totalParams), "25.00%");
  assert.deepEqual(model.tensors.get("model.layers.10.mlp.down_proj.weight").yardstick, { stat: "effectiveRank", kind: "down_proj", count: 4, median: 1.005 });
  assert.equal(model.tensors.get("model.layers.2.self_attn.q_proj.weight").yardstick, null); // T2: not judged by a median
  assert.equal(model.deltaSummary.top.name, "model.layers.10.mlp.down_proj.weight");
  assert.deepEqual([model.deltaSummary.scanned, model.deltaSummary.measured], [6, 5]);
  assert.deepEqual([model.baseName, model.suspectName], ["Tiny", "Tiny-abliterated"]);
});

// Changed tensors as [name, kind, tier, effective_rank, sv_001], in the three files the CLI writes for them.
function changedReport(rows) {
  const layer = (name) => /\.(\d+)\./.exec(name)?.[1] ?? "";
  return {
    base: "b", suspect: "s",
    inventory: rows.map(([name, kind]) => inventoryRow(name, layer(name), kind, "changed")),
    ranking: rows.map(([name, kind, tier, rank, sv = "0.5"], i) => ({ rank: String(i + 1), tier, name, layer: layer(name), kind, status: "changed", effective_rank: rank, sv_001: sv, reason: "" })),
    delta: rows.map(([name, kind, , rank, sv = "0.5"]) => deltaRow(name, layer(name), kind, "changed", rank, [sv, "0.01"])),
  };
}

test("a tensor's yardstick is the median the CLI ranked it by: canonical kind, usable stats, three tensors up", () => {
  assert.deepEqual(["experts.down_proj", "ffn_down_exps", "ffn_down_shexp", "down_proj.bias", "o_proj"].map(canonicalKind),
    ["down_proj", "ffn_down", "ffn_down", "down_proj.bias", "o_proj"]);

  // GGUF MoE: the CLI wrote "effective_rank 5.448 is 5.4x the median 1 of changed ffn_down tensors" for blk.4
  const gguf = buildWorkbench(changedReport([
    ["blk.4.ffn_down_shexp.weight", "ffn_down_shexp", "T3", "5.448"],
    ["blk.0.attn_q.weight", "attn_q", "T2", "1"],
    ...[0, 1, 2, 3].map((i) => [`blk.${i}.ffn_down_exps.weight`, "ffn_down_exps", "T4", "1"]),
    ["blk.2.ffn_down_shexp.weight", "ffn_down_shexp", "T4", "1"],
    ["blk.0.attn_output.weight", "attn_output", "T4", "1"], ["blk.1.attn_output.weight", "attn_output", "T4", "1"],
  ]));
  assert.deepEqual(gguf.tensors.get("blk.4.ffn_down_shexp.weight").yardstick, { stat: "effectiveRank", kind: "ffn_down", count: 6, median: 1 });
  assert.equal(gguf.tensors.get("blk.0.ffn_down_exps.weight").yardstick, gguf.tensors.get("blk.4.ffn_down_shexp.weight").yardstick);
  assert.equal(gguf.tensors.get("blk.0.attn_q.weight").yardstick, null); // T2: an unexpected kind has no median
  assert.deepEqual(gguf.tensors.get("blk.0.attn_output.weight").yardstick, { stat: "effectiveRank", kind: "attn_output", count: 2, median: null }); // under three

  // HF MoE: the shared expert's down_proj is judged with the fused experts.down_proj tensors
  const shared = "model.layers.3.mlp.shared_expert.down_proj.weight";
  const hf = buildWorkbench(changedReport([
    [shared, "down_proj", "T3", "4.635"],
    ...[0, 1, 2, 3, 4, 5].map((i) => [`model.layers.${i}.mlp.experts.down_proj`, "experts.down_proj", "T4", "1"]),
  ]));
  assert.deepEqual(hf.tensors.get(shared).yardstick, { stat: "effectiveRank", kind: "down_proj", count: 7, median: 1 });

  // a row the CLI calls unusable stays out of every median and the charts; sv_001 is the yardstick without an effective rank
  const mixed = buildWorkbench(changedReport([
    ["l.0.o_proj.weight", "o_proj", "T3", "nan"], ["l.1.o_proj.weight", "o_proj", "T3", "2", "0"],
    ...[2, 3, 4].map((i) => [`l.${i}.o_proj.weight`, "o_proj", "T4", "1"]),
    ...[5, 6, 7].map((i) => [`l.${i}.o_proj.weight`, "o_proj", "T4", "", String(i)]),
  ]));
  assert.equal(mixed.tensors.get("l.0.o_proj.weight").yardstick, null);
  assert.equal(mixed.tensors.get("l.1.o_proj.weight").yardstick, null);
  assert.deepEqual(mixed.tensors.get("l.2.o_proj.weight").yardstick, { stat: "effectiveRank", kind: "o_proj", count: 3, median: 1 });
  assert.deepEqual(mixed.tensors.get("l.6.o_proj.weight").yardstick, { stat: "sv1", kind: "o_proj", count: 3, median: 6 });
  assert.deepEqual(mixed.measured.map((t) => t.layer), [2, 3, 4]);
});

test("an inventory-only report keeps its ranking and has no ΔW stats", () => {
  const ranking = REPORT.ranking.map((r) => ({ ...r, effective_rank: "", sv_001: "" }));
  const model = buildWorkbench({ ...REPORT, ranking, delta: [] });
  assert.equal(model.hasDelta, false);
  assert.equal(model.ranking.length, 6);
  assert.deepEqual(model.measured, []);
  assert.deepEqual([model.ranking[2].effectiveRank, model.ranking[2].sv1, model.ranking[2].scanned], [null, null, false]);
  assert.deepEqual([model.deltaSummary.medianRank, model.deltaSummary.top], [null, null]);
});

test("a delta.csv left behind by an earlier run is flagged, not presented as this ranking's stats", () => {
  const withoutStats = REPORT.ranking.map((r) => ({ ...r, effective_rank: "", sv_001: "" }));
  assert.equal(buildWorkbench(REPORT).deltaUnranked, false);
  assert.equal(buildWorkbench({ ...REPORT, ranking: withoutStats }).deltaUnranked, true); // --inventory-only rerun
  assert.equal(buildWorkbench({ ...REPORT, ranking: withoutStats, delta: [] }).deltaUnranked, false);
  // the suspect was replaced before the rerun: what the old scan measured is identical now, so no longer ranked
  const moved = REPORT.ranking.slice(0, 1); // only a T1 row: it never has stats
  assert.equal(buildWorkbench({ ...REPORT, ranking: moved }).deltaUnranked, true);
  // the earlier scan compared identical weights: no stats to miss, but its rows call changed tensors identical
  const identical = REPORT.delta.map((r) => ({ ...r, status: "identical", effective_rank: "", sv_001: "" }));
  assert.equal(buildWorkbench({ ...REPORT, ranking: withoutStats, delta: identical }).deltaUnranked, true);
  assert.equal(buildWorkbench({ ...REPORT, ranking: withoutStats, delta: identical.slice(-1) }).deltaUnranked, false);
});

test("a scan that had nothing to analyse keeps its T3 rows and is not called stale", () => {
  // what the CLI writes for changed conv-shaped down_proj weights: a header-only delta.csv and "not scanned" T3 rows
  const name = "model.layers.0.mlp.down_proj.weight";
  const reason = "stats unusable: not scanned (shape 32x64x1 is not 2-D); cannot be judged consistent with abliteration";
  const model = buildWorkbench({
    base: "b", suspect: "s", delta: [],
    inventory: [{ name, layer: "0", kind: "down_proj", status: "changed", base_dtype: "F32", suspect_dtype: "F32", base_shape: "32x64x1", suspect_shape: "32x64x1", numel: "2048" }],
    ranking: [{ rank: "1", tier: "T3", name, layer: "0", kind: "down_proj", status: "changed", effective_rank: "", sv_001: "", reason }],
  });
  assert.deepEqual([model.hasDelta, model.deltaUnranked, model.tierCounts.get("T3")], [false, false, 1]);
  assert.deepEqual([model.ranking[0].tier, model.ranking[0].reason, model.ranking[0].scanned], ["T3", reason, false]);
});

test("an empty report and an unknown tier or status are shown, not dropped", () => {
  const empty = buildWorkbench({ base: "a", suspect: "b", inventory: [], ranking: [], delta: [] });
  assert.deepEqual([empty.inventory.length, empty.ranking.length, empty.totalParams, empty.hasDelta], [0, 0, 0, false]);
  assert.equal(formatPercent(empty.identicalParams, empty.totalParams), "–");

  const odd = buildWorkbench({
    base: "a", suspect: "b",
    inventory: [{ ...inventoryRow("x.weight", "", "x", "quantized") }],
    ranking: [{ rank: "1", tier: "T9", name: "x.weight", layer: "", kind: "x", status: "quantized", effective_rank: "nan", sv_001: "inf", reason: "new tier" }],
    delta: [],
  });
  assert.equal(odd.tierCounts.get("T9"), 1);
  assert.deepEqual(odd.statuses, ["quantized"]);
  assert.deepEqual(tierInfo("T9"), { id: "T9", label: "T9", meaning: "" });
  assert.ok(Number.isNaN(odd.ranking[0].effectiveRank));
  assert.equal(odd.ranking[0].sv1, Infinity);
});

test("filters by tier, status, kind and every search term", () => {
  const model = buildWorkbench(REPORT);
  const names = (rows) => rows.map((t) => t.name);
  assert.equal(filterTensors(model.ranking, { tiers: new Set(["T4"]) }).length, 3);
  assert.deepEqual(names(filterTensors(model.ranking, { tiers: new Set(["T1", "T3"]) })), ["lm_head.weight", "model.layers.10.mlp.down_proj.weight"]);
  assert.deepEqual(names(filterTensors(model.ranking, { query: "  MEDIAN layers.10 " })), ["model.layers.10.mlp.down_proj.weight"]);
  assert.deepEqual(names(filterTensors(model.inventory, { statuses: new Set(["identical"]), kind: "norm" })), ["model.norm.weight"]);
  assert.deepEqual(names(filterTensors(model.inventory, { query: "shape mismatch" })), ["lm_head.weight"]);
  assert.equal(filterTensors(model.inventory, { tiers: new Set(), statuses: new Set(), kind: "", query: "" }).length, 8);
});

test("sorts naturally and keeps blank cells at the bottom in both directions", () => {
  const model = buildWorkbench(REPORT);
  const layers = (rows) => rows.map((t) => t.layer);
  assert.deepEqual(layers(sortTensors(model.inventory, "layer", "asc")), [0, 1, 2, 2, 2, 10, null, null]);
  assert.deepEqual(layers(sortTensors(model.inventory, "layer", "desc")), [10, 2, 2, 2, 1, 0, null, null]);
  assert.deepEqual(sortTensors(model.ranking, "effectiveRank", "desc").map((t) => t.effectiveRank), [6, 2.5, 1.01, 1, 1, null]);
  assert.deepEqual(["layers.10.w", "layers.2.w", "layers.2.b", "embed"].sort(naturalCompare), ["embed", "layers.2.b", "layers.2.w", "layers.10.w"]);
  const nan = sortTensors([{ v: NaN }, { v: 2 }, { v: null }, { v: 1 }], "v", "desc").map((t) => t.v);
  assert.deepEqual(nan.slice(0, 2), [2, 1]);
});

test("pages clamp to the available rows", () => {
  const rows = Array.from({ length: 250 }, (_, i) => i);
  assert.deepEqual(pageOf(rows, 2, 100), { rows: rows.slice(200), page: 2, pages: 3, start: 200, end: 250 });
  assert.equal(pageOf(rows, 9, 100).page, 2);
  assert.deepEqual(pageOf([], 0, 100), { rows: [], page: 0, pages: 1, start: 0, end: 0 });
});

test("parses and prints the numbers the CLI writes", () => {
  assert.deepEqual(["", " ", "abc", "1.00002", "5e-07", "nan", "inf", "-inf"].map(parseNumber).map(String),
    ["null", "null", "null", "1.00002", "5e-7", "NaN", "Infinity", "-Infinity"]);
  assert.deepEqual([null, NaN, Infinity, 0, 1.00002, 4.97312, 0.458873, 1427.3, 7.55e6, 0.000334269].map(formatNumber),
    ["–", "NaN", "∞", "0", "1.000", "4.973", "0.4589", "1,427", "7.55e6", "3.34e−4"]);
  assert.deepEqual([1, 6144, -2.5].map(formatNumber), ["1", "6,144", "−2.500"]);
  assert.equal(median([3, 1, 2, 10]), 2.5);
  assert.equal(median([]), null);
  assert.deepEqual(["org/Model-v2", "/a/b/", "C:\\x\\y", ""].map(shortName), ["Model-v2", "b", "y", ""]);
});

test("chart colours follow the CLI figures and scales stay inside the plot", () => {
  const colors = kindColors(["down_proj", "o_proj", "attn_output", "zeta"]);
  assert.deepEqual([colors.get("o_proj"), colors.get("down_proj")], ["#eda100", "#6250d6"]); // the PNG slots
  assert.equal(new Set(colors.values()).size, 4);
  assert.deepEqual(kindOrder(["zeta", "down_proj", "q_proj", "attn_output", "q_proj"]), ["q_proj", "down_proj", "attn_output", "zeta"]);
  assert.equal(kindColors(Array.from({ length: 12 }, (_, i) => `kind${i}`)).get("kind9"), "#898781");
  assert.deepEqual([layerColor(0), layerColor(1), layerColor(-3)], ["rgb(134, 182, 239)", "rgb(13, 54, 107)", "rgb(134, 182, 239)"]);
  assert.deepEqual(linearTicks(0.92, 5.3), [1, 2, 3, 4, 5]);
  assert.deepEqual(linearTicks(0, 27, 12), [0, 5, 10, 15, 20, 25]);
  assert.deepEqual(linearTicks(3, 3), [3]);
  assert.deepEqual(logTicks(2e-5, 0.7), [1e-4, 1e-3, 1e-2, 1e-1]);
  // 64 tensors with 2,048 singular values each: more values than Math.min(...values) accepts as arguments
  assert.deepEqual(extent(Array.from({ length: 64 * 2048 }, (_, i) => 1 + (i * 7919) % 1000)), [1, 1000]);
  assert.deepEqual(extent([]), [Infinity, -Infinity]);
  // every kind of a layer is drawn inside the half-layer margin, even when layer 0 is the only one
  for (const count of [1, 2, 3, 8, 40]) {
    const offsets = Array.from({ length: count }, (_, i) => kindOffset(i, count));
    assert.ok(Math.max(...offsets.map(Math.abs)) <= 0.3, `${count} kinds`);
    assert.equal(new Set(offsets).size, count);
  }
  // zero and non-finite singular values are gaps, never coordinates
  assert.deepEqual(spectrumRuns([2, 1, 0, 0.5, NaN, 0.2, 0.1, Infinity]), [[[1, 2], [2, 1]], [[4, 0.5]], [[6, 0.2], [7, 0.1]]]);
  assert.deepEqual(spectrumRuns([NaN, 0, -1]), []);
  assert.deepEqual(spectrumRuns([3]), [[[1, 3]]]);
  const y = scale([1e-4, 1], [200, 0], true);
  assert.deepEqual([y(1e-4), y(1e-2), y(1), y(0)].map(Math.round), [200, 100, 0, 200]);
});

test("a pointer picks the nearest mark, and every panel of a report shares its axes", () => {
  const line = [[0, 0], [10, 0], [10, 10]];
  assert.deepEqual([[5, 3], [14, 5], [-3, -4], [10, 12]].map((p) => distanceToPath(p, line)), [3, 4, 5, 2]);
  assert.equal(distanceToPath([3, 4], [[0, 0]]), 5); // a single dot
  const domain = spectrumDomain([{ spectrum: [0.5, 0.01] }, { spectrum: [8, 0, NaN, 2e-6] }]);
  assert.deepEqual(domain.x.domain, [1, 4]);
  assert.deepEqual(domain.y.domain, [2e-6 / 1.6, 8 * 1.6]);
  assert.deepEqual([plural(1, "tensor"), plural(0, "kind"), plural(1200, "changed o_proj tensor")], ["1 tensor", "0 kinds", "1,200 changed o_proj tensors"]);
});

test("the bundled demo loads through readReport into a complete workbench", async () => {
  const files = ["pair.txt", "inventory.csv", "ranking.csv", "delta.csv"]
    .map((name) => new File([readFileSync(new URL(`demo/${name}`, WEB))], name));
  const model = buildWorkbench(await readReport(files));
  assert.deepEqual([model.baseName, model.suspectName], ["tiny-llama", "tiny-llama-abliterated-tampered"]);
  assert.deepEqual([...model.tierCounts], [["T1", 3], ["T2", 2], ["T3", 1], ["T4", 21]]);
  assert.equal(model.inventory.length, 112);
  for (const t of model.ranking) {
    assert.ok(t.reason && t.numel > 0 && t.status !== "identical", t.name); // every ranked row found its inventory row
  }
  assert.ok(model.hasDelta);
  assert.equal(model.measured.length, 24);
  assert.ok(model.measured.every((t) => t.spectrum.length === 32 && t.gap > 1));
  assert.equal(model.deltaSummary.top, model.ranking.find((t) => t.tier === "T3"));
});

test("the page is local-first: same-origin only, and report text never becomes markup", () => {
  const sources = Object.fromEntries(["index.html", "workbench.css", "app.mjs", "charts.mjs", "dom.mjs", "model.mjs"]
    .map((name) => [name, readFileSync(new URL(name, WEB), "utf-8")]));
  for (const [name, text] of Object.entries(sources)) {
    const remote = text.match(/(?:https?:)?\/\/[\w.-]+\.[a-z]{2,}[^\s"'`)]*/g) ?? [];
    assert.deepEqual(remote.filter((url) => url !== "http://www.w3.org/2000/svg"), [], `${name} points off this origin`);
    assert.doesNotMatch(text, /innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\(|new Function/, name);
  }
  const csp = /http-equiv="Content-Security-Policy" content="([^"]+)"/.exec(sources["index.html"])[1];
  for (const rule of ["default-src 'none'", "script-src 'self'", "style-src 'self'", "connect-src 'self'", "form-action 'none'"]) {
    assert.ok(csp.split("; ").includes(rule), `CSP lacks ${rule}`);
  }
  assert.doesNotMatch(csp, /unsafe|\*|https?:/);
});

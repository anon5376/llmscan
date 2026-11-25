// Run with: node --test tests/web/report.test.mjs   (Node 20 or newer, no packages)
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { readReport } from "../../web/report.mjs";

const DEMO = new URL("../../web/demo/", import.meta.url);
const INVENTORY_HEADER = "name,layer,kind,status,base_dtype,suspect_dtype,base_shape,suspect_shape,numel";
const RANKING_HEADER = "rank,tier,name,layer,kind,status,effective_rank,sv_001,reason";
const DELTA_HEADER = "name,layer,kind,status,shape,effective_rank,seconds,sv_001";

/** A File as a folder picker hands it over: `folder` becomes its webkitRelativePath. */
function file(name, content, folder) {
  const f = new File([content], name);
  if (folder !== undefined) Object.defineProperty(f, "webkitRelativePath", { value: `${folder}/${name}` });
  return f;
}

function demo(name, folder) {
  return file(name, readFileSync(new URL(name, DEMO)), folder);
}

/** A small valid output folder; `parts` replaces a file's content, or drops the file with null. */
function report(parts = {}, folder) {
  const contents = {
    "pair.txt": "base: /models/base\nsuspect: /models/suspect\n",
    "inventory.csv": `${INVENTORY_HEADER}\r\na.weight,,a,changed,F32,F32,2x2,2x2,4\r\nb.weight,,b,identical,F32,F32,2x2,2x2,4\r\n`,
    "ranking.csv": `${RANKING_HEADER}\r\n1,T2,a.weight,,a,changed,1.5,0.25,changed a\r\n`,
    "delta.csv": `${DELTA_HEADER}\r\na.weight,,a,changed,2x2,1.5,0.00,0.25\r\nb.weight,,b,identical,2x2,,,\r\n`,
    ...parts,
  };
  return Object.entries(contents).filter(([, content]) => content !== null).map(([name, content]) => file(name, content, folder));
}

const rejects = (files, message) => assert.rejects(readReport(files), (error) => {
  assert.ok(error instanceof Error);
  assert.match(error.message, message);
  return true;
});

test("reads the demo, a full llmscan run", async () => {
  const r = await readReport(["pair.txt", "inventory.csv", "ranking.csv", "delta.csv"].map((name) => demo(name)));
  assert.equal(r.base, "synthetic/tiny-llama");
  assert.equal(r.suspect, "synthetic/tiny-llama-abliterated-tampered");
  assert.equal(r.inventory.length, 112);
  assert.equal(r.ranking.length, 27);
  assert.equal(r.delta.length, 84);
  assert.deepEqual(Object.keys(r.inventory[0]), INVENTORY_HEADER.split(","));
  assert.deepEqual(Object.keys(r.ranking[0]), RANKING_HEADER.split(","));
  assert.ok("sv_032" in r.delta[0]);
  assert.deepEqual(r.ranking[5], {
    rank: "6", tier: "T3", name: "model.layers.8.mlp.down_proj.weight", layer: "8", kind: "down_proj", status: "changed",
    effective_rank: "5.91867", sv_001: "0.730039",
    reason: "effective_rank 5.919 is 5.9x the median 1 of changed down_proj tensors (threshold 3x)",
  });
  assert.match(r.ranking[3].reason, /^changed q_proj: outside expected kinds \(attn_output, down_proj, /);  // quoted in the file
  for (const row of [...r.inventory, ...r.ranking, ...r.delta]) {
    for (const cell of Object.values(row)) assert.equal(typeof cell, "string");
  }
});

test("an inventory-only run has no delta.csv: delta is empty", async () => {
  const r = await readReport(["pair.txt", "inventory.csv", "ranking.csv"].map((name) => demo(name)));
  assert.deepEqual(r.delta, []);
  assert.equal(r.ranking.length, 27);
});

test("reads a picked folder and never opens the other files in it", async () => {
  const unread = (name) => Object.assign(file(name, "", "out/run"), {
    arrayBuffer: () => assert.fail(`${name} was read`),
  });
  const files = [unread("model.safetensors"), ...report({}, "out/run"), unread("inventory.txt"), unread("spectra.png")];
  const r = await readReport(new Set(files));
  assert.deepEqual([r.base, r.suspect], ["/models/base", "/models/suspect"]);
  assert.equal(r.inventory.length, 2);
});

test("ignores what llmscan left in its private .llmscan-* directory", async () => {
  const r = await readReport([...report({}, "run"), file("delta.csv", "half written", "run/.llmscan-x1")]);
  assert.equal(r.delta.length, 2);
});

test("parses quoted cells: commas, doubled quotes, line breaks, and keeps markup as text", async () => {
  const names = ['a,b', 'say "hi"', "line\r\nbreak\nagain", "<img src=x onerror=alert(1)>", "=1+1", "ünï → 模型", '"'];
  const quoted = ['"a,b"', '"say ""hi"""', '"line\r\nbreak\nagain"', "<img src=x onerror=alert(1)>", "=1+1", "ünï → 模型", '""""'];
  const r = await readReport(report({
    "inventory.csv": `${INVENTORY_HEADER}\r\n${quoted.map((q) => `${q},,k,changed,F32,F32,2x2,2x2,4`).join("\r\n")}\r\n`,
    "ranking.csv": `${RANKING_HEADER}\r\n${quoted.map((q, i) => `${i + 1},T2,${q},,k,changed,,,"why, ""exactly"""`).join("\r\n")}\r\n`,
    "delta.csv": null,
  }));
  assert.deepEqual(r.inventory.map((row) => row.name), names);
  assert.deepEqual(r.ranking.map((row) => row.name), names);
  assert.equal(r.ranking[0].reason, 'why, "exactly"');
  assert.deepEqual([r.ranking[0].layer, r.ranking[0].effective_rank], ["", ""]);
});

test("accepts LF or CR line ends, a byte order mark, no final line end, blank lines and extra columns", async () => {
  const r = await readReport(report({
    "pair.txt": "base: C:\\models\\base\r\nsuspect: C:\\models\\suspect: v2\r\n",
    "inventory.csv": `\uFEFF${INVENTORY_HEADER},__proto__\na.weight,,a,changed,F32,F32,2x2,2x2,4,extra\n\n`,
    "ranking.csv": `${RANKING_HEADER}\r1,T2,a.weight,,a,changed,1.5,0.25,`,
    "delta.csv": `${DELTA_HEADER}\r\n`,
  }));
  assert.deepEqual([r.base, r.suspect], ["C:\\models\\base", "C:\\models\\suspect: v2"]);
  assert.equal(r.inventory.length, 1);
  assert.equal(Object.keys(r.inventory[0])[0], "name");
  assert.equal(Object.getOwnPropertyDescriptor(r.inventory[0], "__proto__").value, "extra");
  assert.equal(r.ranking[0].reason, "");
  assert.deepEqual(r.delta, []);
});

test("rejects a selection without the required files", async () => {
  await rejects(null, /No files were selected/);
  await rejects([], /No llmscan output found/);
  await rejects([file("model.safetensors", "")], /No llmscan output found/);
  for (const name of ["pair.txt", "inventory.csv", "ranking.csv"]) {
    await rejects(report({ [name]: null }), new RegExp(`^${name.replace(".", "\\.")} is missing`));
  }
});

test("rejects more than one comparison", async () => {
  await rejects([...report({}, "out/a-vs-b"), ...report({}, "out/c-vs-b")],
    /llmscan output in 2 folders \("out\/a-vs-b", "out\/c-vs-b"\)\. Choose the output folder of one comparison/);
  await rejects([...report(), file("ranking.csv", "")], /Two files named ranking\.csv/);
  await rejects(report({ "pair.txt": "base: a\nsuspect: b\nbase: c\nsuspect: d\n" }), /pair\.txt does not name exactly one base and one suspect/);
  await rejects(report({ "pair.txt": "base: a\n" }), /pair\.txt does not name exactly one base and one suspect/);
  await rejects(report({ "ranking.csv": `${RANKING_HEADER}\r\n1,T1,other.weight,,o,only_suspect,,,why\r\n` }),
    /ranking\.csv lists the tensor "other\.weight", which inventory\.csv does not have/);
  await rejects(report({ "ranking.csv": `${RANKING_HEADER}\r\n1,T1,a.weight,,a,only_suspect,,,why\r\n` }),
    /ranking\.csv and inventory\.csv disagree on the status of "a\.weight"/);
});

test("loads the required files when delta.csv contains a tensor from an earlier run", async () => {
  const r = await readReport(report({ "delta.csv": `${DELTA_HEADER}\r\nlm_head.weight,,lm_head,identical,2x2,,,\r\n` }));
  assert.deepEqual(r.ranking.map((row) => row.name), ["a.weight"]);
  assert.deepEqual(r.inventory.map((row) => row.name), ["a.weight", "b.weight"]);
  assert.deepEqual(r.delta.map((row) => row.name), ["lm_head.weight"]);
});

test("rejects files that are not llmscan CSV", async () => {
  const row = "a.weight,,a,changed,F32,F32,2x2,2x2,4";
  await rejects(report({ "inventory.csv": "" }), /inventory\.csv is empty/);
  await rejects(report({ "inventory.csv": "name,layer,kind,status\r\n" }),
    /inventory\.csv has no base_dtype, suspect_dtype, base_shape, suspect_shape, numel columns/);
  await rejects(report({ "ranking.csv": RANKING_HEADER.replace(",tier", "") + "\r\n" }), /ranking\.csv has no tier column,/);
  await rejects(report({ "delta.csv": "<html></html>" }), /delta\.csv has no name, layer, kind, status, shape, effective_rank columns/);
  await rejects(report({ "inventory.csv": `${INVENTORY_HEADER},name\r\n` }), /inventory\.csv has two columns named "name"/);
  await rejects(report({ "inventory.csv": `${INVENTORY_HEADER}\r\n${row}\r\nb.weight,,b\r\n` }),
    /inventory\.csv: row 2 has 3 cells, but the header has 9/);
  await rejects(report({ "inventory.csv": `${INVENTORY_HEADER}\r\n${row}\r\n"b.weight,,b,changed,F32,F32,2x2,2x2,4\r\n` }),
    /inventory\.csv: a quoted cell in row 2 is never closed/);
  await rejects(report({ "inventory.csv": `${INVENTORY_HEADER}\r\n"a".weight,,a,changed,F32,F32,2x2,2x2,4\r\n` }),
    /inventory\.csv: text follows the closing quote of a cell in row 1/);
  await rejects(report({ "inventory.csv": `${INVENTORY_HEADER}\r\n${row}\r\n${row}\r\n` }), /inventory\.csv lists the tensor "a\.weight" twice/);
  await rejects(report({ "ranking.csv": new Uint8Array([0xff, 0xfe, 0x72, 0x00]) }), /ranking\.csv is not UTF-8 text/);
});

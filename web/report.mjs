// Reads one llmscan output folder that the user picked in the browser.
// Nothing here touches the network, and only the four report files below are ever opened,
// so model weights that sit in the same selection stay unread. Every value is returned as
// the plain text llmscan wrote; render it as text, never as HTML.

const PAIR = "pair.txt";
const INVENTORY = "inventory.csv";
const RANKING = "ranking.csv";
const DELTA = "delta.csv";

const REQUIRED_HEADERS = {
  [INVENTORY]: ["name", "layer", "kind", "status", "base_dtype", "suspect_dtype", "base_shape", "suspect_shape", "numel"],
  [RANKING]: ["rank", "tier", "name", "layer", "kind", "status", "effective_rank", "sv_001", "reason"],
  [DELTA]: ["name", "layer", "kind", "status", "shape", "effective_rank"],
};

const REPORT_FILES = new Set([PAIR, INVENTORY, RANKING, DELTA]);

/**
 * @typedef {Record<string, string>} Row one CSV record, keyed by the file's own header
 * @typedef {{base: string, suspect: string, inventory: Row[], ranking: Row[], delta: Row[]}} Report
 */

/**
 * Read one comparison written by the llmscan command line.
 *
 * @param {Iterable<File>} files the files of one output folder, e.g. the FileList of an
 *   `<input type="file" webkitdirectory>`; unrelated files are ignored without being read.
 * @returns {Promise<Report>} `delta` is empty when the folder has no delta.csv (`--inventory-only`).
 *   Rejects with an Error whose message can be shown to the user as is.
 */
export async function readReport(files) {
  const found = locate(files);
  for (const name of [PAIR, INVENTORY, RANKING]) {
    if (!found.has(name)) {
      throw new Error(`${name} is missing. Choose a folder written by llmscan: it holds ${PAIR}, ${INVENTORY} and ${RANKING}.`);
    }
  }
  const pair = parsePair(await text(found.get(PAIR), PAIR));
  const inventory = await table(found.get(INVENTORY), INVENTORY);
  const ranking = await table(found.get(RANKING), RANKING);
  const delta = found.has(DELTA) ? await table(found.get(DELTA), DELTA) : [];
  const known = tensorNames(inventory, INVENTORY);
  for (const [rows, name] of [[ranking, RANKING], [delta, DELTA]]) {
    const stray = [...tensorNames(rows, name)].find((tensor) => !known.has(tensor));
    if (stray !== undefined) {
      throw new Error(`${name} lists the tensor ${quote(stray)}, which ${INVENTORY} does not have, so these files come from different comparisons.`);
    }
  }
  return { ...pair, inventory, ranking, delta };
}

/** The report files in the selection, by name; they must all sit in one folder. */
function locate(files) {
  if (files == null || typeof files[Symbol.iterator] !== "function") {
    throw new Error("No files were selected.");
  }
  const folders = new Map();
  for (const file of files) {
    const path = String(file.webkitRelativePath || file.name).split("/");
    const name = path.pop();
    // .llmscan-* is the private directory llmscan writes through before renaming a file into place
    if (!REPORT_FILES.has(name) || path.some((part) => part.startsWith(".llmscan-"))) continue;
    const folder = path.join("/");
    if (!folders.has(folder)) folders.set(folder, new Map());
    if (folders.get(folder).has(name)) {
      throw new Error(`Two files named ${name} were selected. Choose the output folder of one comparison.`);
    }
    folders.get(folder).set(name, file);
  }
  if (folders.size === 0) {
    throw new Error(`No llmscan output found. Choose a folder written by llmscan: it holds ${PAIR}, ${INVENTORY} and ${RANKING}.`);
  }
  if (folders.size > 1) {
    const names = [...folders.keys()].map((folder) => quote(folder || "(top level)"));
    const shown = names.slice(0, 3).join(", ") + (names.length > 3 ? ` and ${names.length - 3} more` : "");
    throw new Error(`The selection holds llmscan output in ${names.length} folders (${shown}). Choose the output folder of one comparison.`);
  }
  return folders.values().next().value;
}

async function text(file, name) {
  let bytes;
  try {
    bytes = await file.arrayBuffer();
  } catch (cause) {
    throw new Error(`${name} could not be read.`, { cause });
  }
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (cause) {
    throw new Error(`${name} is not UTF-8 text.`, { cause });
  }
}

function parsePair(content) {
  const match = /^base: ([^\r\n]+)\r?\nsuspect: ([^\r\n]+)(?:\r?\n)?$/.exec(content);
  if (!match) {
    throw new Error(`${PAIR} does not name exactly one base and one suspect model, the way llmscan writes it.`);
  }
  return { base: match[1], suspect: match[2] };
}

async function table(file, name) {
  const [header, ...records] = parseCsv(await text(file, name), name);
  if (!header) throw new Error(`${name} is empty.`);
  const repeated = header.find((column, i) => header.indexOf(column) !== i);
  if (repeated !== undefined) throw new Error(`${name} has two columns named ${quote(repeated)}.`);
  const missing = REQUIRED_HEADERS[name].filter((column) => !header.includes(column));
  if (missing.length) {
    throw new Error(`${name} has no ${missing.join(", ")} column${missing.length > 1 ? "s" : ""}, so it is not an llmscan ${name}.`);
  }
  return records.map((cells, i) => {
    if (cells.length !== header.length) {
      throw new Error(`${name}: row ${i + 1} has ${cells.length} cells, but the header has ${header.length}.`);
    }
    return Object.fromEntries(header.map((column, j) => [column, cells[j]]));  // own keys, even for "__proto__"
  });
}

function tensorNames(rows, name) {
  const names = new Set();
  for (const row of rows) {
    if (names.has(row.name)) throw new Error(`${name} lists the tensor ${quote(row.name)} twice.`);
    names.add(row.name);
  }
  return names;
}

/**
 * CSV as Python's csv module writes it (RFC 4180): cells holding a comma, a quote or a line
 * break are quoted, and a quote inside them is doubled. Records end with CRLF, LF or CR.
 * Empty lines are skipped. Returns one array of cells per record.
 */
function parseCsv(content, name) {
  const records = [];
  const unquotedEnd = /[,\r\n]/g;
  let cells = [];
  let at = 0;
  const where = () => (records.length ? `row ${records.length}` : "the header");
  while (at < content.length) {
    let cell;
    if (content[at] === '"') {
      cell = "";
      for (at += 1; ; at += 1) {  // each pass copies up to the next quote; a doubled quote continues the cell
        const close = content.indexOf('"', at);
        if (close < 0) throw new Error(`${name}: a quoted cell in ${where()} is never closed.`);
        cell += content.slice(at, close);
        at = close + 1;
        if (content[at] !== '"') break;
        cell += '"';
      }
      if (at < content.length && !",\r\n".includes(content[at])) {
        throw new Error(`${name}: text follows the closing quote of a cell in ${where()}.`);
      }
    } else {
      unquotedEnd.lastIndex = at;
      const end = unquotedEnd.exec(content)?.index ?? content.length;
      cell = content.slice(at, end);
      at = end;
    }
    cells.push(cell);
    if (content[at] === ",") {
      at += 1;
      if (at === content.length) cells.push("");
      else continue;
    }
    at += content[at] === "\r" && content[at + 1] === "\n" ? 2 : 1;
    if (cells.length > 1 || cells[0] !== "") records.push(cells);
    cells = [];
  }
  return records;
}

/** A value from a file, shortened and quoted for an error message. */
function quote(value) {
  return JSON.stringify(value.length > 80 ? value.slice(0, 80) + "…" : value);
}

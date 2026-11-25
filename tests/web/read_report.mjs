// Print readReport's result for an llmscan output folder as JSON: node tests/web/read_report.mjs <folder>
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

import { readReport } from "../../web/report.mjs";

const folder = process.argv[2];
const files = readdirSync(folder)
  .filter((name) => statSync(join(folder, name)).isFile())
  .map((name) => new File([readFileSync(join(folder, name))], name));
process.stdout.write(JSON.stringify(await readReport(files)));

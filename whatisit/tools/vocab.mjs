// Which words does the robot judge know?   node whatisit/tools/vocab.mjs laser pointer "drop bear"
// Prints each word with "ok" or "MISSING" (the robot judge can only match key words it knows).
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const J = createRequire(import.meta.url)("../src/judge.js");
const buf = readFileSync(join(ROOT, "site/judge-vectors.bin"));
const space = new J.Space(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
for (const w of process.argv.slice(2)) console.log(space.lookup(w.toLowerCase()) ? "ok      " : "MISSING ", w);

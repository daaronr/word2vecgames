// Compare judges against hand-scored guesses.
//
//   node whatisit/eval/score_judges.mjs
//
// Reads whatisit/eval/judge-eval.json (gold scores, written before any judge ran), scores every
// guess with the robot judge, and adds any other judge whose scores sit in whatisit/eval/results/
// (<name>.json mapping round id -> {scores:[{n, score}]}; files <name>-1.json, <name>-2.json ...
// are merged). Prints agreement with the gold scores.
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const require = createRequire(import.meta.url);
const J = require(join(ROOT, "src", "judge.js"));

const items = new Map();
for (const f of readdirSync(join(ROOT, "content"))) for (const it of JSON.parse(readFileSync(join(ROOT, "content", f), "utf8"))) items.set(it.id, it);
const evalSet = JSON.parse(readFileSync(join(HERE, "judge-eval.json"), "utf8")).rounds;

// ---------- judges ----------
const judges = {};
const buf = readFileSync(join(ROOT, "site", "judge-vectors.bin"));
const space = new J.Space(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
judges.robot = Object.fromEntries(evalSet.map((r) => [r.id, r.guesses.map((g) => J.scoreGuess(space, g.text, items.get(r.id)).score)]));

const resDir = join(HERE, "results");
if (existsSync(resDir)) {
  const groups = {};
  for (const f of readdirSync(resDir).filter((f) => f.endsWith(".json"))) {
    const name = f.replace(/(-\d+)?\.json$/, "");
    groups[name] = Object.assign(groups[name] || {}, JSON.parse(readFileSync(join(resDir, f), "utf8")));
  }
  for (const [name, byRound] of Object.entries(groups)) {
    judges[name] = {};
    for (const r of evalSet) {
      const reply = byRound[r.id];
      if (!reply) continue;
      if (Array.isArray(reply)) { judges[name][r.id] = reply; continue; } // plain score arrays
      const parsed = J.parseJudgeReply(reply, r.guesses.length);
      judges[name][r.id] = parsed.scores.map((s) => (s ? s.score : null));
    }
  }
}

// ---------- metrics ----------
function ranks(a) {
  const idx = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]);
  const r = new Array(a.length);
  for (let i = 0; i < idx.length; ) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2;
    i = j + 1;
  }
  return r;
}
function pearson(x, y) {
  const n = x.length;
  const mx = x.reduce((s, v) => s + v, 0) / n;
  const my = y.reduce((s, v) => s + v, 0) / n;
  let sxy = 0, sx = 0, sy = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sx += (x[i] - mx) ** 2; sy += (y[i] - my) ** 2; }
  return sx && sy ? sxy / Math.sqrt(sx * sy) : 0;
}
const spearman = (x, y) => pearson(ranks(x), ranks(y));

function report(name, byRound) {
  const gold = [], pred = [], rows = [];
  let rounds = 0, winners = 0, spSum = 0, trapsWarm = 0, traps = 0, missed = 0, strong = 0, agree = 0, n = 0, abs = 0;
  for (const r of evalSet) {
    const p = byRound[r.id];
    if (!p || p.some((v) => v == null)) continue;
    const g = r.guesses.map((x) => x.gold);
    rounds++;
    spSum += spearman(g, p);
    // Winner: the judge's top guess is one of the gold-best guesses (within 5 points of the best).
    const top = p.indexOf(Math.max(...p));
    if (g[top] >= Math.max(...g) - 5) winners++;
    r.guesses.forEach((x, i) => {
      gold.push(x.gold); pred.push(p[i]); n++; abs += Math.abs(x.gold - p[i]);
      if ((x.gold >= 60) === (p[i] >= 60)) agree++;
      if (x.gold <= 10) { traps++; if (p[i] >= 40) trapsWarm++; }
      if (x.gold >= 85) { strong++; if (p[i] < 50) missed++; }
      rows.push({ id: r.id, text: x.text, gold: x.gold, pred: p[i], err: p[i] - x.gold });
    });
  }
  const pct = (a, b) => (b ? Math.round((100 * a) / b) + "%" : "-");
  console.log(`\n== ${name} (${rounds} rounds, ${n} guesses)`);
  console.log(`  picks the right winner      ${pct(winners, rounds)}`);
  console.log(`  rank agreement per round    ${(spSum / rounds).toFixed(2)} (Spearman, 1 = same order)`);
  console.log(`  overall correlation         ${pearson(gold, pred).toFixed(2)} (Pearson)`);
  console.log(`  close / not close (>= 60)   ${pct(agree, n)} agree`);
  console.log(`  mean absolute error         ${(abs / n).toFixed(0)} points`);
  console.log(`  wrong guesses scored warm   ${trapsWarm} of ${traps} (gold <= 10, judge >= 40)`);
  console.log(`  right guesses scored cold   ${missed} of ${strong} (gold >= 85, judge < 50)`);
  rows.sort((a, b) => Math.abs(b.err) - Math.abs(a.err));
  console.log("  biggest disagreements:");
  for (const x of rows.slice(0, 8)) console.log(`    ${String(x.pred).padStart(3)} vs gold ${String(x.gold).padStart(3)}  ${x.id}: "${x.text}"`);
}

for (const [name, byRound] of Object.entries(judges)) report(name, byRound);

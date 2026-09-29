// Run: node tests/engine.test.js   (checks the in-browser engine against the bundled vectors)
const fs = require("fs");
const path = require("path");
const assert = require("assert");
const B = require("../web/engine.js");

const D = path.join(__dirname, "..", "web", "data");
const vocab = fs.readFileSync(path.join(D, "vocab.txt"), "utf8").split("\n");
const pools = JSON.parse(fs.readFileSync(path.join(D, "pools.json"), "utf8"));
const puzzles = JSON.parse(fs.readFileSync(path.join(D, "puzzles.json"), "utf8"));
const space = new B.Space(vocab, new Int8Array(fs.readFileSync(path.join(D, "vectors.bin"))), pools.dim);

// Classic analogy: king - man + woman lands on queen.
const r = space.score("king", "queen", [{ word: "man", sign: -1 }, { word: "woman", sign: 1 }]);
assert.strictEqual(r.rank, 1, `king-man+woman: queen rank ${r.rank}`);
assert.strictEqual(r.near[0], "queen");

// Ball vectors are unit length.
const n = Math.hypot(...r.vec);
assert(Math.abs(n - 1) < 1e-5, "ball not normalised");

// Deals are deterministic per seed, well-formed, and beat the start word.
const a = B.deal(space, pools, "daily-2026-09-28");
const b = B.deal(space, pools, "daily-2026-09-28");
assert.deepStrictEqual(a, b);
for (let i = 0; i < 25; i++) {
  const d = B.deal(space, pools, "t" + i);
  assert.strictEqual(new Set(d.hand).size, B.HAND_SIZE);
  // no filler: every tile is linked to the jack or the start word
  for (const w of d.hand) {
    assert(Math.max(space.sim(w, d.start), space.sim(w, d.target)) >= 0.2, `unlinked tile ${w} in ${d.start}->${d.target}`);
  }
  assert(!d.hand.includes(d.start) && !d.hand.includes(d.target));
  const best = space.allThrows(d.start, d.target, d.hand)[0];
  assert(best.sim > space.sim(d.start, d.target) + 0.15, `weak deal ${d.start}->${d.target}`);
}

// All throws: 9 tiles, up to 3 → 9·2 + 36·4 + 84·8 = 834.
assert.strictEqual(space.allThrows(a.start, a.target, a.hand).length, 834);

// The tutorial (web/app.js TUT) promises: boat → plane starts well back, "+ sky" gets closer,
// and "+ sky − water" makes plane the nearest word. Rebuilding the vectors must keep this true.
const tutStart = space.survey(space.row("boat"), "plane", ["boat"], 1).rank;
const tut1 = space.score("boat", "plane", [{ word: "sky", sign: 1 }]);
const tut2 = space.score("boat", "plane", [{ word: "sky", sign: 1 }, { word: "water", sign: -1 }]);
assert(tutStart > 10 && tut1.rank < tutStart && tut2.rank === 1, `tutorial ranks ${tutStart} → ${tut1.rank} → ${tut2.rank}`);
// Where a ball "stopped" never names a near-copy of a thrown word (boat + sky should not land "near boats").
assert(!tut1.near.some((w) => w.startsWith("boat")), `near words ${tut1.near}`);

// Every puzzle is playable with the bundled vocabulary.
const unplayable = puzzles.filter((p) => !space.has(p.start_word) || !space.has(p.target_word));
assert.deepStrictEqual(unplayable.map((p) => p.id), []);

// Every active puzzle is worth playing against these vectors: the jack doesn't start among the
// start word's 8 nearest words, and the best throw the hand allows gets it inside the top 30.
// (Puzzles that fail are marked "retired" in puzzles.json rather than deleted.)
const closer = (x, y) => x.rank < y.rank || (x.rank === y.rank && x.sim > y.sim);
const weak = [];
for (const p of puzzles.filter((q) => !q.retired)) {
  const hand = p.allowed_cards.filter((w) => w !== "WILDCARD" && space.has(w) && w !== p.start_word && w !== p.target_word);
  const r0 = space.survey(space.row(p.start_word), p.target_word, [p.start_word], 1).rank;
  const par = space.allThrows(p.start_word, p.target_word, hand).slice(0, 50)
    .map((t) => ({ ...t, rank: space.score(p.start_word, p.target_word, t.tiles).rank })).reduce((m, t) => (m && !closer(t, m) ? m : t), null);
  if (r0 <= 8 || par.rank > 30) weak.push(`#${p.id} ${p.start_word}→${p.target_word} (start ${r0}, par ${par.rank})`);
}
assert.deepStrictEqual(weak, [], "weak puzzles: retire or fix them");

// Court: the start ball sits at distance 1, angle 0.
const basis = B.courtBasis(space, a.start, a.target, "x");
const s = basis.place(space.row(a.start));
assert(Math.abs(s.dist - 1) < 1e-4 && Math.abs(s.theta) < 1e-4);

for (let i = 0; i < 6; i++) {
  const d = B.deal(space, pools, "sample" + i);
  const best = space.allThrows(d.start, d.target, d.hand)[0];
  const sc = space.score(d.start, d.target, best.tiles);
  console.log(`${d.start} → ${d.target}  [${d.hand.join(", ")}]  par ${best.tiles.map(t => (t.sign > 0 ? "+" : "−") + t.word).join(" ")} = ${best.sim.toFixed(2)} (#${sc.rank}, near ${sc.near[0]})`);
}

// ---------- the common-sense word set (web/data-sense, ConceptNet Numberbatch) ----------
const DS = path.join(__dirname, "..", "web", "data-sense");
const sensePools = JSON.parse(fs.readFileSync(path.join(DS, "pools.json"), "utf8"));
const sense = new B.Space(fs.readFileSync(path.join(DS, "vocab.txt"), "utf8").split("\n"),
  new Int8Array(fs.readFileSync(path.join(DS, "vectors.bin"))), sensePools.dim);
const EXPLAIN = 0.3; // WORD_SETS.sense.explain in web/app.js

// Its tutorial (WORD_SETS.sense.tut): hat → shoe; "+ foot" gets closer; "+ foot − head" makes shoe the nearest word.
const h0 = sense.survey(sense.row("hat"), "shoe", ["hat"], 1).rank;
const h1 = sense.score("hat", "shoe", [{ word: "foot", sign: 1 }]).rank;
const h2 = sense.score("hat", "shoe", [{ word: "foot", sign: 1 }, { word: "head", sign: -1 }]).rank;
assert(h0 > 20 && h1 < h0 && h2 === 1, `sense tutorial ranks ${h0} → ${h1} → ${h2}`);
for (const w of ["hand", "foot", "sock", "head", "walk", "cap"]) assert(sense.has(w), `sense tutorial word ${w}`);

// Deals are deterministic, full, and their strongest throws include one a person could explain.
assert.deepStrictEqual(B.deal(sense, sensePools, "x", { explain: EXPLAIN }), B.deal(sense, sensePools, "x", { explain: EXPLAIN }));
let explained = 0;
for (let i = 0; i < 15; i++) {
  const d = B.deal(sense, sensePools, "s" + i, { explain: EXPLAIN });
  assert.strictEqual(new Set(d.hand).size, B.HAND_SIZE);
  if (sense.allThrows(d.start, d.target, d.hand).slice(0, 12).some((t) => B.explainable(sense, d.start, d.target, t.tiles, EXPLAIN))) explained++;
}
assert(explained >= 14, `explainable sense deals ${explained}/15`);

// Active puzzles meet the same bar with these words (retired ones carry "retired_sense").
const weakSense = [];
for (const p of puzzles.filter((q) => !q.retired_sense)) {
  const hand = p.allowed_cards.filter((w) => w !== "WILDCARD" && sense.has(w) && w !== p.start_word && w !== p.target_word);
  assert(sense.has(p.start_word) && sense.has(p.target_word), `puzzle #${p.id} words missing from the sense set`);
  const r0 = sense.survey(sense.row(p.start_word), p.target_word, [p.start_word], 1).rank;
  const cands = sense.allThrows(p.start_word, p.target_word, hand).slice(0, 50);
  const ok = cands.filter((t) => B.explainable(sense, p.start_word, p.target_word, t.tiles, EXPLAIN));
  const par = (ok.length ? ok : cands).map((t) => ({ ...t, rank: sense.score(p.start_word, p.target_word, t.tiles).rank }))
    .reduce((m, t) => (m && !closer(t, m) ? m : t), null);
  if (r0 <= 8 || par.rank > 30 || !ok.length) weakSense.push(`#${p.id} ${p.start_word}→${p.target_word} (start ${r0}, par ${par.rank})`);
}
assert.deepStrictEqual(weakSense, [], "weak puzzles with common-sense words: mark them retired_sense");

console.log("engine ok");

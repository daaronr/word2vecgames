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
  assert.strictEqual(new Set(d.hand).size, 7);
  assert(!d.hand.includes(d.start) && !d.hand.includes(d.target));
  const best = space.allThrows(d.start, d.target, d.hand)[0];
  assert(best.sim > space.sim(d.start, d.target) + 0.15, `weak deal ${d.start}->${d.target}`);
}

// All throws: 7 tiles, up to 3 → 7·2 + 21·4 + 35·8 = 378.
assert.strictEqual(space.allThrows(a.start, a.target, a.hand).length, 378);

// Every puzzle is playable with the bundled vocabulary.
const unplayable = puzzles.filter((p) => !space.has(p.start_word) || !space.has(p.target_word));
assert.deepStrictEqual(unplayable.map((p) => p.id), []);

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
console.log("engine ok");

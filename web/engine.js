/*
 * Word Bocce engine — runs entirely in the browser (or Node, for tests).
 *
 * A throw is plain vector arithmetic on unit word vectors:
 *     ball = unit( v(start) + s1·v(tile1) + s2·v(tile2) + ... ),  s = ±1
 * Its score is cos(ball, v(jack)). "Rank" is how many vocabulary words sit
 * closer to the ball than the jack does (+1): rank 1 means the jack is the
 * nearest word to where the ball landed — a "bacio" (kiss) in bocce terms.
 */
(function (root) {
  "use strict";

  const MAX_TILES = 3;
  const HAND_SIZE = 9;

  // ---------- seeded randomness ----------
  function hashSeed(str) {
    let h = 1779033703 ^ str.length;
    for (let i = 0; i < str.length; i++) {
      h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
      h = (h << 13) | (h >>> 19);
    }
    return (h >>> 0) || 1;
  }
  function rng(seed) {
    let a = typeof seed === "string" ? hashSeed(seed) : seed >>> 0;
    const next = () => {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    next.int = (n) => Math.floor(next() * n);
    next.pick = (arr) => arr[next.int(arr.length)];
    next.shuffle = (arr) => {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = next.int(i + 1);
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
      return arr;
    };
    return next;
  }

  // ---------- vector space ----------
  class Space {
    constructor(vocab, int8, dim) {
      this.vocab = vocab;
      this.dim = dim;
      this.n = vocab.length;
      this.index = new Map(vocab.map((w, i) => [w, i]));
      const M = new Float32Array(this.n * dim);
      for (let r = 0; r < this.n; r++) {
        let ss = 0;
        const o = r * dim;
        for (let k = 0; k < dim; k++) ss += int8[o + k] * int8[o + k];
        const inv = ss > 0 ? 1 / Math.sqrt(ss) : 0;
        for (let k = 0; k < dim; k++) M[o + k] = int8[o + k] * inv;
      }
      this.M = M;
    }
    has(w) { return this.index.has(w); }
    row(w) {
      const i = this.index.get(w);
      if (i === undefined) throw new Error(`"${w}" is not in the vocabulary`);
      return this.M.subarray(i * this.dim, (i + 1) * this.dim);
    }
    cos(a, b) {
      let s = 0;
      for (let k = 0; k < this.dim; k++) s += a[k] * b[k];
      return s;
    }
    sim(w1, w2) { return this.cos(this.row(w1), this.row(w2)); }

    /** tiles: [{word, sign}] with sign = +1 | -1 */
    ball(start, tiles) {
      const v = Float32Array.from(this.row(start));
      for (const t of tiles) {
        const r = this.row(t.word);
        for (let k = 0; k < this.dim; k++) v[k] += t.sign * r[k];
      }
      let ss = 0;
      for (let k = 0; k < this.dim; k++) ss += v[k] * v[k];
      const inv = ss > 0 ? 1 / Math.sqrt(ss) : 0;
      for (let k = 0; k < this.dim; k++) v[k] *= inv;
      return v;
    }

    /** Rank of `target` around vector v, plus the nearest words (inputs excluded). */
    survey(v, target, exclude, k = 3) {
      const skip = new Set(exclude);
      const tsim = this.cos(v, this.row(target));
      let rank = 1;
      const top = []; // [sim, word] kept sorted descending, length ≤ k
      const { M, dim, n, vocab } = this;
      for (let r = 0; r < n; r++) {
        const w = vocab[r];
        if (skip.has(w)) continue;
        let s = 0;
        const o = r * dim;
        for (let j = 0; j < dim; j++) s += v[j] * M[o + j];
        if (s > tsim && w !== target) rank++;
        if (top.length < k || s > top[top.length - 1][0]) {
          top.push([s, w]);
          top.sort((a, b) => b[0] - a[0]);
          if (top.length > k) top.pop();
        }
      }
      return { sim: tsim, rank, near: top.map((t) => t[1]) };
    }

    /**
     * Full scoring of one throw. `near` names where the ball stopped; it skips near-copies of the
     * words thrown ("boats" after throwing "boat"), which say nothing new. The rank still counts them.
     */
    score(start, target, tiles) {
      const v = this.ball(start, tiles);
      const inputs = [start, ...tiles.map((t) => t.word)];
      const s = this.survey(v, target, inputs, 8);
      const fresh = s.near.filter((w) => w === target || !inputs.some((x) => related(w, x)));
      return { ...s, near: (fresh.length ? fresh : s.near).slice(0, 3), vec: v };
    }

    /** Every legal throw from a hand (1..maxTiles distinct tiles, each ±), best first. */
    allThrows(start, target, hand, maxTiles = MAX_TILES) {
      const words = hand.filter((w) => this.has(w));
      const tv = this.row(target);
      const out = [];
      const combo = (from, picked) => {
        if (picked.length) {
          const m = picked.length;
          for (let mask = 0; mask < 1 << m; mask++) {
            const tiles = picked.map((w, i) => ({ word: w, sign: mask & (1 << i) ? -1 : 1 }));
            out.push({ tiles, sim: this.cos(this.ball(start, tiles), tv) });
          }
        }
        if (picked.length === maxTiles) return;
        for (let i = from; i < words.length; i++) combo(i + 1, [...picked, words[i]]);
      };
      combo(0, []);
      out.sort((a, b) => b.sim - a.sim);
      return out;
    }
  }

  // ---------- dealing an end ----------
  const related = (a, b) => a.slice(0, 4) === b.slice(0, 4) || a.includes(b) || b.includes(a);

  /**
   * Deal a start word, a jack (target) and a hand of HAND_SIZE tiles, every one
   * linked to the jack or the start word (no random filler):
   *   2 pulls      — point at the jack and not at the start
   *   2 sheds      — carry the start word's flavour; worth subtracting
   *   2 lures      — linked to both, so adding them drags the start along
   *   3 near-misses — look like they point at the jack, but more weakly than the pulls
   */
  function deal(space, pools, seed) {
    const R = rng(seed);
    const targets = pools.targets.filter((w) => space.has(w));
    const cards = pools.cards.filter((w) => space.has(w));
    let start, jack;
    for (let tries = 0; tries < 400 && !start; tries++) {
      jack = R.pick(targets);
      for (let j = 0; j < 60; j++) {
        const s = R.pick(targets);
        const c = space.sim(s, jack);
        if (c > 0.12 && c < 0.35 && !related(s, jack)) { start = s; break; }
      }
    }
    const S = space.row(start), T = space.row(jack);
    const scored = [];
    for (const w of cards) {
      if (w === start || w === jack || related(w, jack) || related(w, start)) continue;
      const r = space.row(w);
      const cs = space.cos(r, S), ct = space.cos(r, T);
      if (ct > 0.6) continue; // near-synonyms of the jack make it trivial
      scored.push({ w, cs, ct });
    }
    const topBy = (f, k) => [...scored].sort((a, b) => f(b) - f(a)).slice(0, k).map((x) => x.w);
    const pull = topBy((x) => x.ct - 0.6 * Math.max(0, x.cs), 12);
    const shed = topBy((x) => x.cs - x.ct, 10);
    const lure = topBy((x) => x.cs + x.ct, 40);
    const miss = scored.filter((x) => x.ct >= 0.28 && x.ct < 0.45 && x.cs < 0.3)
      .sort((a, b) => b.ct - a.ct).slice(0, 60).map((x) => x.w);

    const hand = [];
    const take = (list, k) => {
      const pool = R.shuffle(list.filter((w) => !hand.includes(w)));
      hand.push(...pool.slice(0, k));
    };
    take(pull, 2);
    take(shed, 2);
    take(lure, 2);
    take(miss, 3);
    // Rare thin decks: top up from the linked lists, never from random words.
    for (const list of [lure, pull, shed]) take(list, HAND_SIZE - hand.length);
    R.shuffle(hand);
    return { start, target: jack, hand };
  }

  /** Where a ball sits on the court. Returns polar coords relative to the jack. */
  function courtBasis(space, start, target, seed) {
    const T = space.row(target);
    const perp = (v) => {
      const c = space.cos(v, T);
      const p = new Float32Array(space.dim);
      for (let k = 0; k < space.dim; k++) p[k] = v[k] - c * T[k];
      return p;
    };
    const unit = (p) => {
      let ss = 0;
      for (let k = 0; k < p.length; k++) ss += p[k] * p[k];
      const inv = ss > 0 ? 1 / Math.sqrt(ss) : 0;
      for (let k = 0; k < p.length; k++) p[k] *= inv;
      return p;
    };
    const b1 = unit(perp(space.row(start)));
    // a fixed pseudo-random second axis, orthogonal to jack and b1
    const R = rng(seed + "|axis");
    const b2 = perp(Float32Array.from({ length: space.dim }, () => R() - 0.5));
    const d = space.cos(b2, b1);
    for (let k = 0; k < space.dim; k++) b2[k] -= d * b1[k];
    unit(b2);
    const startAngle = Math.acos(Math.max(-1, Math.min(1, space.cos(space.row(start), T))));
    return {
      startAngle,
      place(v) {
        const angle = Math.acos(Math.max(-1, Math.min(1, space.cos(v, T))));
        const p = perp(v);
        const theta = Math.atan2(space.cos(p, b2), space.cos(p, b1));
        return { dist: angle / startAngle, theta };
      },
    };
  }

  function tier(rank) {
    if (rank === 1) return { key: "bacio", label: "Bacio!" };      // the ball kisses the jack
    if (rank <= 10) return { key: "close", label: "Close" };
    if (rank <= 100) return { key: "hunt", label: "In the hunt" };
    if (rank <= 1000) return { key: "wide", label: "Wide" };
    return { key: "lost", label: "Long way off" };
  }

  async function load(base, vectorsFile = "vectors.bin") {
    const ok = (r) => { if (!r.ok) throw new Error(`${r.url}: HTTP ${r.status}`); return r; };
    const [vocabTxt, buf, pools, puzzles] = await Promise.all([
      fetch(base + "vocab.txt").then(ok).then((r) => r.text()),
      fetch(base + vectorsFile).then(ok).then((r) => r.arrayBuffer()),
      fetch(base + "pools.json").then(ok).then((r) => r.json()),
      fetch(base + "puzzles.json").then(ok).then((r) => r.json()),
    ]);
    const vocab = vocabTxt.split("\n");
    const space = new Space(vocab, new Int8Array(buf), pools.dim);
    return { space, pools, puzzles };
  }

  const api = { MAX_TILES, HAND_SIZE, related, rng, hashSeed, Space, deal, courtBasis, tier, load };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Bocce = api;
})(typeof self !== "undefined" ? self : this);

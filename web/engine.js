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

    /**
     * Break a throw's similarity to the jack into one share per word. Because
     *   cos(ball, jack) = Σ sign·cos(word, jack) / |Σ sign·v(word)|
     * the shares ("push") add up exactly to the ball's similarity. The start word counts as a +1 word.
     * This is exact arithmetic; *why* two words sit close is not something the numbers tell us.
     */
    explainThrow(start, target, tiles) {
      const words = [{ word: start, sign: 1, isStart: true }, ...tiles];
      const v = new Float32Array(this.dim);
      for (const w of words) {
        const r = this.row(w.word);
        for (let k = 0; k < this.dim; k++) v[k] += w.sign * r[k];
      }
      let ss = 0;
      for (let k = 0; k < this.dim; k++) ss += v[k] * v[k];
      const norm = Math.sqrt(ss) || 1;
      const parts = words.map((w) => {
        const toJack = this.sim(w.word, target), toStart = w.isStart ? 1 : this.sim(w.word, start);
        return { ...w, toJack, toStart, push: (w.sign * toJack) / norm };
      });
      return { parts, norm, sim: parts.reduce((a, p) => a + p.push, 0) };
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
  // Near-copies: "boats" for "boat". On the token map also "Boat" and "␣boat" ("␣" marks a leading space).
  const bare = (w) => (w.charCodeAt(0) === 0x2423 ? w.slice(1) : w).toLowerCase();
  const related = (x, y) => {
    const a = bare(x), b = bare(y);
    return a.slice(0, 4) === b.slice(0, 4) || a.includes(b) || b.includes(a);
  };

  /**
   * Deal a start word, a jack (target) and a hand of HAND_SIZE tiles, every one
   * linked to the jack or the start word (no random filler):
   *   2 pulls      — point at the jack and not at the start
   *   2 sheds      — carry the start word's flavour; worth subtracting
   *   2 lures      — linked to both, so adding them drags the start along
   *   3 near-misses — look like they point at the jack, but more weakly than the pulls
   * On the token map (pools.pieces), two of the near-misses become pieces of words linked to the
   * jack ("rimp" for fisherman, "resso" for coffee), so the hand looks like what GPT-2 reads.
   */
  function deal(space, pools, seed, opts = {}) {
    if (!opts.explain) return dealOnce(space, pools, seed);
    // Re-deal (deterministically) until one of the strongest throws can be explained word by word.
    let d = null;
    for (let attempt = 0; attempt < 25; attempt++) {
      d = dealOnce(space, pools, attempt ? `${seed}#${attempt}` : seed);
      const top = space.allThrows(d.start, d.target, d.hand).slice(0, 12);
      if (top.some((t) => explainable(space, d.start, d.target, t.tiles, opts.explain))) return d;
    }
    return d;
  }

  /**
   * A throw a person could explain: every added word is clearly related to the jack, and every
   * subtracted word is clearly related to the start word (more than to the jack). `threshold` is a
   * cosine; it depends on the vectors (0.3 suits the common-sense set).
   */
  function explainable(space, start, target, tiles, threshold) {
    return tiles.every((t) => (t.sign > 0
      ? space.sim(t.word, target) >= threshold
      : space.sim(t.word, start) >= threshold && space.sim(t.word, start) > space.sim(t.word, target)));
  }

  function dealOnce(space, pools, seed) {
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
    const nPieces = pools.pieces ? 2 : 0;
    take(pull, 2);
    take(shed, 2);
    take(lure, 2);
    take(miss, 3 - nPieces);
    if (nPieces) {
      // Prefer pieces that point at this jack rather than at everything ("ner", "ists"): score each by
      // its similarity to the jack minus its average similarity to a fixed sample of jacks.
      const sample = targets.filter((_, i) => i % 25 === 0);
      const generic = (w) => sample.reduce((a, t) => a + space.sim(w, t), 0) / sample.length;
      const pieces = pools.pieces.filter((w) => space.has(w) && !related(w, jack) && !related(w, start))
        .map((w) => ({ w, ct: space.sim(w, jack) })).filter((x) => x.ct < 0.6)
        .sort((a, b) => b.ct - a.ct).slice(0, 60)
        .map((x) => ({ ...x, s: x.ct - generic(x.w) })).sort((a, b) => b.s - a.s).slice(0, 12).map((x) => x.w);
      take(pieces, nPieces);
    }
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

  /** Load a word bundle from `base`; puzzles are shared between bundles and may live elsewhere. */
  async function load(base, vectorsFile = "vectors.bin", puzzlesURL = base + "puzzles.json") {
    const ok = (r) => { if (!r.ok) throw new Error(`${r.url}: HTTP ${r.status}`); return r; };
    const [vocabTxt, buf, pools, puzzles] = await Promise.all([
      fetch(base + "vocab.txt").then(ok).then((r) => r.text()),
      fetch(base + vectorsFile).then(ok).then((r) => r.arrayBuffer()),
      fetch(base + "pools.json").then(ok).then((r) => r.json()),
      fetch(puzzlesURL).then(ok).then((r) => r.json()),
    ]);
    const vocab = vocabTxt.split("\n");
    const space = new Space(vocab, new Int8Array(buf), pools.dim);
    return { space, pools, puzzles };
  }

  // ---------- chase ("cat and mouse") on a terrain: a small, themed set of words ----------
  /**
   * Players stand on words of the terrain. A move hops: from word W, card C with sign s lands on the
   * terrain word nearest to unit(v(W) + s·v(C)), other than W and C. The cat catches the mouse by
   * landing on the mouse's word or close to it: the mouse's word is among the `catchK` terrain words
   * nearest the cat. The cat explodes if it lands among the `trapK` words nearest a trap.
   * Keeping everyone on a small themed terrain keeps the chase catchable: with cards from the whole
   * vocabulary, one card throws the mouse anywhere and the cat almost never has a card to follow.
   */
  class Terrain {
    constructor(space, words, opts = {}) {
      this.space = space;
      this.words = [...new Set(words)].filter((w) => space.has(w));
      this.catchK = opts.catchK || 5;
      this.trapK = opts.trapK || 2;
    }
    nearest(v, exclude = []) {
      let best = null, bs = -Infinity;
      for (const w of this.words) {
        if (exclude.includes(w)) continue;
        const c = this.space.cos(v, this.space.row(w));
        if (c > bs) { bs = c; best = w; }
      }
      return best;
    }
    hop(from, card, sign) { return this.nearest(this.space.ball(from, [{ word: card, sign }]), [from, card]); }
    /** 0 if the same word, else 1 + how many terrain words sit closer to `from` than `to` does. */
    rank(from, to) {
      if (from === to) return 0;
      const v = this.space.row(from), t = this.space.cos(v, this.space.row(to));
      let r = 1;
      for (const w of this.words) if (w !== from && w !== to && this.space.cos(v, this.space.row(w)) > t) r++;
      return r;
    }
    caught(cat, mouse) { return this.rank(cat, mouse) <= this.catchK; }
    trapped(cat, traps) { return traps.find((t) => this.rank(t, cat) <= this.trapK) || null; }
    /** Every move from `from` with these cards: [{ card, sign, to }]. */
    moves(from, cards) {
      const out = [];
      for (const card of cards) for (const sign of [1, -1]) {
        if (card === from) continue; // "muppets − muppets" goes nowhere
        const to = this.hop(from, card, sign);
        if (to) out.push({ card, sign, to });
      }
      return out;
    }
    /**
     * A bot's move. The cat wants the mouse's word near its landing spot and avoids traps; the mouse
     * wants to land far from the cat. `reach` > 1 picks at random among the best few (an easier bot).
     */
    botMove(role, me, other, cards, traps, R, reach = 1) {
      const scored = this.moves(me, cards).map((m) => ({ ...m,
        score: role === "cat" ? -this.rank(m.to, other) - (this.trapped(m.to, traps) ? 1000 : 0) : this.rank(other, m.to) }));
      scored.sort((a, b) => b.score - a.score);
      return scored.length ? scored[Math.floor(R() * Math.min(reach, scored.length))] : null;
    }
    /** A start word, two traps away from it, and the cards: a shared face-up row and two private hands. */
    deal(seed, { shared = 16, hand = 3 } = {}) {
      const R = rng(seed);
      const pool = R.shuffle(this.words.slice());
      const start = pool.shift();
      const traps = [];
      for (let i = 0; i < pool.length && traps.length < 2; i++) {
        if (this.rank(start, pool[i]) > this.catchK * 3) traps.push(pool.splice(i--, 1)[0]);
      }
      return { start, traps, shared: pool.slice(0, shared), hands: [pool.slice(shared, shared + hand), pool.slice(shared + hand, shared + 2 * hand)] };
    }
    /** Each word's place on a 2-D map: its two strongest directions across the terrain (PCA), scaled to 0..1. */
    map() {
      if (this._map) return this._map;
      const { space } = this, n = this.words.length, d = space.dim;
      const X = this.words.map((w) => Float64Array.from(space.row(w)));
      const mean = new Float64Array(d);
      for (const x of X) for (let k = 0; k < d; k++) mean[k] += x[k] / n;
      for (const x of X) for (let k = 0; k < d; k++) x[k] -= mean[k];
      const R = rng("terrain-map"), comps = [];
      for (let c = 0; c < 2; c++) {
        let v = Float64Array.from({ length: d }, () => R() - 0.5);
        for (let it = 0; it < 60; it++) {
          const next = new Float64Array(d);
          for (const x of X) {
            let p = 0;
            for (let k = 0; k < d; k++) p += x[k] * v[k];
            for (let k = 0; k < d; k++) next[k] += p * x[k];
          }
          for (const u of comps) { let p = 0; for (let k = 0; k < d; k++) p += next[k] * u[k]; for (let k = 0; k < d; k++) next[k] -= p * u[k]; }
          let ss = 0;
          for (let k = 0; k < d; k++) ss += next[k] * next[k];
          v = next.map((z) => z / (Math.sqrt(ss) || 1));
        }
        comps.push(v);
      }
      const pts = X.map((x) => comps.map((u) => x.reduce((a, z, k) => a + z * u[k], 0)));
      const lo = [0, 1].map((i) => Math.min(...pts.map((p) => p[i]))), hi = [0, 1].map((i) => Math.max(...pts.map((p) => p[i])));
      this._map = new Map(this.words.map((w, j) => [w, pts[j].map((z, i) => (z - lo[i]) / ((hi[i] - lo[i]) || 1))]));
      return this._map;
    }
  }

  const api = { MAX_TILES, HAND_SIZE, related, explainable, rng, hashSeed, Space, Terrain, deal, courtBasis, tier, load };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Bocce = api;
})(typeof self !== "undefined" ? self : this);

/* Word Bocce — UI. Depends on engine.js (window.Bocce). */
(function () {
  "use strict";
  const B = window.Bocce;
  // Two ways of placing words on the court; players can switch (footer and help dialog).
  //   sense: ConceptNet Numberbatch, text statistics blended with a knowledge base of everyday facts.
  //          Courts are only dealt if their best throw can be explained word by word (explain = cosine bar).
  //   text:  GloVe, learned purely from which words appear together in Wikipedia and news text.
  // Each set has its own Daily seed and puzzle retirement field (both audited against that set).
  const WORD_SETS = {
    sense: {
      dir: "data-sense/", name: "Common sense", short: "common-sense", explain: 0.3,
      dailySeed: "daily-sense-", dailyKey: "daily-sense:", retired: "retired_sense",
      credit: "Words: ConceptNet Numberbatch 19.08 (CC BY-SA 4.0), everyday words only.",
      // hat → shoe: 249th → "+ foot" 5th → "+ foot − head" 1st
      tut: { start: "hat", target: "shoe", add: "foot", sub: "head", hand: ["hand", "foot", "sock", "head", "walk", "cap"],
        first: "A shoe is a bit like a hat that you wear on your foot. Tap “foot” to add it to your throw.",
        second: "You don't wear shoes on your head, so let's take head away." },
    },
    text: {
      dir: window.WORD_BOCCE_DATA || "data/", name: "Raw text", short: "raw-text", explain: 0,
      dailySeed: "daily-", dailyKey: "daily:", retired: "retired", vectorsFile: window.WORD_BOCCE_VECTORS,
      credit: "Words: GloVe 6B, 100 dimensions (public domain).",
      // boat → plane: 30th → "+ sky" 7th → "+ sky − water" 1st
      tut: { start: "boat", target: "plane", add: "sky", sub: "water", hand: ["road", "sky", "fish", "water", "island", "engine"],
        first: "A plane is a bit like a boat that travels through the sky. Tap “sky” to add it to your throw.",
        second: "Planes don't float on water, so let's take water away." },
    },
  };
  const PUZZLES_URL = (window.WORD_BOCCE_DATA || "data/") + "puzzles.json";
  const SOLO_BALLS = 4;
  const VS_BALLS = 3;
  const VS_TARGET = 5;
  const LAUNCH = Date.UTC(2026, 8, 28); // Daily No. 1
  const SHARE_URL = "https://wordbocce.davidreinstein.org/"; // last line of a copied Daily result
  const reduceMotion = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- tiny helpers ----------
  const $ = (s, el = document) => el.querySelector(s);
  const SVGNS = "http://www.w3.org/2000/svg";
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (k === "html") el.innerHTML = v;
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid);
    return el;
  }
  function s(tag, attrs, ...kids) {
    const el = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v != null) el.setAttribute(k, v);
    for (const kid of kids.flat()) if (kid != null) el.append(kid);
    return el;
  }
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const store = {
    get(k, d) { try { const v = localStorage.getItem("wordbocce:" + k); return v ? JSON.parse(v) : d; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem("wordbocce:" + k, JSON.stringify(v)); } catch (e) { /* storage unavailable */ } },
  };
  // ---------- sound: synthesised with Web Audio, no audio files ----------
  const sfx = (() => {
    let ctx = null, noiseBuf = null, on = store.get("sound", true);
    const ac = () => {
      if (!on) return null;
      if (!ctx) {
        const C = window.AudioContext || window.webkitAudioContext;
        if (!C) return null;
        ctx = new C();
      }
      if (ctx.state === "suspended") ctx.resume();
      return ctx;
    };
    function tone(c, freq, t0, dur, { type = "sine", gain = 0.2, to = null } = {}) {
      const o = c.createOscillator(), g = c.createGain();
      o.type = type;
      o.frequency.setValueAtTime(freq, t0);
      if (to) o.frequency.exponentialRampToValueAtTime(to, t0 + dur);
      g.gain.setValueAtTime(0.0001, t0);
      g.gain.exponentialRampToValueAtTime(gain, t0 + 0.006);
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      o.connect(g).connect(c.destination);
      o.start(t0);
      o.stop(t0 + dur + 0.05);
    }
    function gravel(c, t0, dur, gain) {
      if (!noiseBuf) {
        noiseBuf = c.createBuffer(1, c.sampleRate, c.sampleRate);
        const d = noiseBuf.getChannelData(0);
        for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
      }
      const src = c.createBufferSource(), bp = c.createBiquadFilter(), g = c.createGain();
      src.buffer = noiseBuf;
      src.loop = true;
      bp.type = "bandpass";
      bp.frequency.setValueAtTime(1400, t0);
      bp.frequency.exponentialRampToValueAtTime(500, t0 + dur);
      bp.Q.value = 0.7;
      g.gain.setValueAtTime(0.0001, t0);
      g.gain.exponentialRampToValueAtTime(gain, t0 + 0.03);
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      src.connect(bp).connect(g).connect(c.destination);
      src.start(t0);
      src.stop(t0 + dur + 0.05);
    }
    const play = (fn) => { try { const c = ac(); if (c) fn(c, c.currentTime); } catch (e) { /* audio unavailable */ } };
    return {
      get on() { return on; },
      set(v) { on = v; store.set("sound", v); },
      tile: (sign) => play((c, t) => tone(c, sign > 0 ? 900 : 640, t, 0.08, { type: "triangle", gain: 0.12 })),
      untile: () => play((c, t) => tone(c, 420, t, 0.07, { type: "triangle", gain: 0.08 })),
      nope: () => play((c, t) => tone(c, 150, t, 0.14, { type: "square", gain: 0.04 })),
      /** A throw: the ball lands at `landAt` seconds from now, then rolls on gravel until `stopAt`. */
      roll: (landAt, stopAt) => play((c, t) => {
        tone(c, 160, t + landAt, 0.2, { gain: 0.35, to: 55 });
        gravel(c, t + landAt, stopAt - landAt + 0.08, 0.22);
      }),
      clack: () => play((c, t) => {
        tone(c, 2300, t, 0.05, { type: "triangle", gain: 0.16 });
        tone(c, 3400, t + 0.004, 0.04, { gain: 0.09 });
      }),
      bacio: () => play((c, t) => {
        tone(c, 1318.5, t, 0.9, { gain: 0.14 });
        tone(c, 1975.5, t + 0.13, 1.1, { gain: 0.11 });
      }),
      win: () => play((c, t) => [523.3, 659.3, 784, 1046.5].forEach((f, i) => tone(c, f, t + i * 0.1, 0.35, { type: "triangle", gain: 0.12 }))),
      lose: () => play((c, t) => [392, 293.7].forEach((f, i) => tone(c, f, t + i * 0.18, 0.4, { type: "triangle", gain: 0.1 }))),
    };
  })();

  const fmt = (x) => x.toFixed(2).replace(/^(-?)0\./, "$1.");
  const signChar = (n) => (n > 0 ? "+" : "−");
  const eqText = (start, tiles) => [start, ...tiles.map((t) => `${signChar(t.sign)} ${t.word}`)].join(" ");
  const tileKey = (tiles) => tiles.map((t) => signChar(t.sign) + t.word).sort().join(" ");
  function todayISO() {
    const d = new Date();
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  }
  const ordinal = (n) => {
    const v = n % 100, suf = v >= 11 && v <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[n % 10] || "th");
    return n.toLocaleString() + suf;
  };
  const q = (w) => `“${w}”`;

  // ---------- explanations: tap (touch) or hover/focus (desktop) ----------
  const TIPS = {
    rank: "Rank is how the game measures closeness. Take the spot where your ball is, and list all 40,000 words from closest in meaning to farthest. The jack's place in that list is its rank. #1 means the jack is the closest word of all: a perfect throw.",
    similarity: "Similarity (cosine similarity) compares two word vectors: 1 means pointing the same way, 0 means unrelated, below 0 means opposite. It moves in small steps, so it shows progress even when the rank barely changes.",
    par: "Par is the best throw this hand allows. The game tries every combination of up to three tiles, each added or subtracted (834 throws for a nine-tile hand), then ranks the 50 most promising and keeps the one that puts the jack nearest the top of the list.",
    rings: "Each ring is a rank boundary. Inside the ‘top 10’ ring, the jack is among the 10 words closest to your ball; inside ‘top 100’, among the closest 100; and so on. Each ring inward is ten times harder to reach.",
    near: "The word closest in meaning to where your ball stopped (not counting the words you threw). It shows what your throw ‘means’.",
    party: "Everyone plays the same court at once, three balls each. When every ball is thrown, the player with the ball nearest the jack wins the round and scores a point for each of their balls that beats everyone else's best.",
    versus: "Whoever is farther from the jack throws next. When both sides are out of balls, the side with the closest ball wins the round and scores one point for every ball closer than the other side's best.",
    tiers: "Bacio (Italian for ‘kiss’, a ball touching the jack): rank #1. Close: top 10. In the hunt: top 100. Wide: top 1,000. Long way off: further than that.",
  };
  function tip(key, label) {
    return h("button", { class: "tip", type: "button", "data-tip": TIPS[key], "aria-label": "Explain: " + (label || key) }, "?");
  }
  const tipBox = document.createElement("div");
  tipBox.className = "tipbox";
  tipBox.setAttribute("role", "tooltip");
  tipBox.hidden = true;
  let tipFor = null;
  function showTip(el) {
    tipFor = el;
    tipBox.textContent = el.dataset.tip;
    tipBox.hidden = false;
    const r = el.getBoundingClientRect();
    const w = tipBox.offsetWidth, ht = tipBox.offsetHeight;
    tipBox.style.left = Math.max(8, Math.min(innerWidth - w - 8, r.left + r.width / 2 - w / 2)) + "px";
    tipBox.style.top = (r.bottom + 8 + ht > innerHeight ? r.top - ht - 8 : r.bottom + 8) + "px";
  }
  function hideTip() { tipBox.hidden = true; tipFor = null; }
  document.addEventListener("pointerover", (e) => {
    const el = e.target.closest && e.target.closest("[data-tip]");
    if (el && e.pointerType === "mouse") showTip(el);
  });
  document.addEventListener("pointerout", (e) => {
    const el = e.target.closest && e.target.closest("[data-tip]");
    if (el && e.pointerType === "mouse" && !el.contains(e.relatedTarget)) hideTip();
  });
  document.addEventListener("click", (e) => {
    const el = e.target.closest && e.target.closest("[data-tip]");
    if (el) { e.preventDefault(); if (tipFor === el && !tipBox.hidden) hideTip(); else showTip(el); }
    else if (!tipBox.contains(e.target)) hideTip();
  });
  document.addEventListener("focusin", (e) => { if (e.target.dataset && e.target.dataset.tip) showTip(e.target); });
  document.addEventListener("focusout", (e) => { if (e.target === tipFor) hideTip(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideTip(); });
  window.addEventListener("scroll", hideTip, { passive: true });

  const dailyNo = (iso) => Math.floor((Date.parse(iso + "T00:00:00Z") - LAUNCH) / 864e5) + 1;

  // ---------- game state ----------
  let G = null;          // the current word set's bundle: { space, pools, puzzles }
  let wordSet = "sense"; // key into WORD_SETS
  const bundles = {};    // loaded word sets, by key
  const SET = () => WORD_SETS[wordSet];
  let mode = "daily";
  const ends = {};       // per-mode current end
  let vs = null;         // versus match
  let busy = false;      // a ball is rolling / bot is thinking
  let statusMsg = { text: "", warn: false };

  /** Load a word set (once) and make it current. */
  async function loadWordSet(key) {
    if (!bundles[key]) {
      const ws = WORD_SETS[key];
      const b = await B.load(ws.dir, ws.vectorsFile, PUZZLES_URL);
      // Puzzles that failed the audit against this word set are marked in puzzles.json; skip them.
      b.puzzles = b.puzzles.filter((p) => !p[ws.retired]);
      bundles[key] = b;
    }
    wordSet = key;
    G = bundles[key];
    return G;
  }
  const dealFor = (seed) => B.deal(G.space, G.pools, seed, { explain: SET().explain });

  function makeEnd(kind, seed, start, target, hand, extra) {
    const space = G.space;
    const explain = SET().explain;
    const throws = space.allThrows(start, target, hand);
    const end = {
      kind, seed, start, target, hand, set: wordSet,
      startSim: space.sim(start, target),
      startRank: space.survey(space.row(start), target, [start], 1).rank,
      basis: B.courtBasis(space, start, target, seed),
      throws,
      // Par is judged by rank, like the balls, and only shown once the round is over, so it is worked
      // out lazily. Ranking all 800+ throws is too slow on a phone; the rank-best throw sits among the
      // 50 most similar in nearly every deal (the most similar alone is rank-best only ~1 time in 4).
      // With common-sense words, par must also be a throw a person could explain word by word.
      get par() {
        if (this._par === undefined) {
          let cands = this.throws.slice(0, 50);
          if (explain) {
            const ok = cands.filter((t) => B.explainable(space, start, target, t.tiles, explain));
            if (ok.length) cands = ok;
          }
          this._par = cands.map((t) => ({ ...t, rank: space.score(start, target, t.tiles).rank }))
            .reduce((m, t) => (m && !closer(t, m) ? m : t), null);
        }
        return this._par;
      },
      balls: [],
      rack: [],
      done: false,
      ...extra,
    };
    return end;
  }

  function dealDaily() {
    const iso = todayISO();
    const seed = SET().dailySeed + iso;
    const d = dealFor(seed);
    const end = makeEnd("daily", seed, d.start, d.target, d.hand, { iso, no: dailyNo(iso), sides: ["red"], perSide: SOLO_BALLS });
    for (const tiles of store.get(SET().dailyKey + iso, [])) placeBall(end, "red", tiles);
    if (end.balls.length >= SOLO_BALLS) end.done = true;
    return end;
  }
  // ---------- tutorial: a guided first game, one instruction at a time ----------
  // Each word set has its own example (WORD_SETS[..].tut), checked in tests/engine.test.js.
  function dealTutorial() {
    const T = SET().tut;
    return makeEnd("tutorial", "tutorial", T.start, T.target, T.hand.filter((w) => G.space.has(w)),
      { sides: ["red"], perSide: 2, intro: true, tut: T });
  }
  /** What the tutorial asks for next: { text, tile } (tap this word) or { text, throw: true }. */
  function tutorialStep(end) {
    const T = end.tut, A = T.add, S = T.sub;
    const inRack = (w) => end.rack.find((t) => t.word === w);
    const fixSign = (w) => `That subtracted “${w}” (the − sign). Tap it once more to take it off, then again to add it.`;
    const landed = end.balls.filter((b) => !b.pending);
    if (landed.length === 0) {
      if (!inRack(A)) return { text: T.first, tile: A };
      if (inRack(A).sign < 0) return { text: fixSign(A), tile: A };
      return { text: `Your throw is “${T.start} + ${A}”. Tap Throw to roll the ball.`, throw: true };
    }
    const b = landed[0];
    const lead = b.rank < end.startRank
      ? `Closer! Your ball stopped near “${b.near[0]}”. “${T.target}” went from the ${ordinal(end.startRank)} nearest word to the ${ordinal(b.rank)}. A perfect throw makes it 1st. `
      : `Your ball stopped near “${b.near[0]}”. `;
    if (!inRack(A)) return { text: lead + `${T.second} First tap “${A}” again.`, tile: A };
    if (inRack(A).sign < 0) return { text: fixSign(A), tile: A };
    if (!inRack(S)) return { text: `Now tap “${S}” twice. One tap adds a word (+); a second tap subtracts it (−).`, tile: S };
    if (inRack(S).sign > 0) return { text: `Right, that added it. Tap “${S}” once more to subtract it.`, tile: S };
    return { text: `Your throw is “${T.start} + ${A} − ${S}”. Throw!`, throw: true };
  }

  function dealPractice() {
    const seed = "practice-" + Date.now() + "-" + Math.random().toString(36).slice(2, 7);
    const d = dealFor(seed);
    return makeEnd("practice", seed, d.start, d.target, d.hand, { sides: ["red"], perSide: SOLO_BALLS });
  }
  function dealPuzzle(p) {
    const space = G.space;
    const wild = p.allowed_cards.includes("WILDCARD");
    const hand = p.allowed_cards.filter((w) => w !== "WILDCARD" && space.has(w) && w !== p.start_word && w !== p.target_word
      && space.sim(w, p.target_word) <= 0.85);
    return makeEnd("puzzle", "puzzle-" + p.id, p.start_word, p.target_word, hand,
      { puzzle: p, wild, wildWord: null, sides: ["red"], perSide: SOLO_BALLS, showHint: false });
  }
  function dealVersus() {
    const seed = "vs-" + Date.now() + "-" + Math.random().toString(36).slice(2, 7);
    const d = dealFor(seed);
    return makeEnd("versus", seed, d.start, d.target, d.hand, { sides: ["red", "blue"], perSide: VS_BALLS });
  }

  /** Score a throw and put the ball in the end (no animation). */
  function placeBall(end, side, tiles) {
    const sc = G.space.score(end.start, end.target, tiles);
    const p = end.basis.place(sc.vec);
    end.balls.push({ side, tiles, sim: sc.sim, rank: sc.rank, near: sc.near, place: p, n: end.balls.length + 1 });
    return end.balls[end.balls.length - 1];
  }

  const ballsOf = (end, side) => end.balls.filter((b) => b.side === side);
  // Closeness is judged the way players see it: by the jack's rank, with similarity only breaking ties.
  // (A ball can make the jack 1st while having slightly lower similarity than one that left it 7th.)
  const closer = (a, b) => a.rank < b.rank || (a.rank === b.rank && a.sim > b.sim);
  const bestOf = (end, side) => ballsOf(end, side).reduce((m, b) => (m && !closer(b, m) ? m : b), null);
  const bestBall = (end) => end.balls.reduce((m, b) => (m && !closer(b, m) ? m : b), null);

  /** Bocce order: the side farther from the jack throws next, while it has balls. */
  function nextSide(end) {
    if (end.kind === "party") {
      // Online room: everyone throws at once; you may throw while you have balls left.
      if (end.done || !party || !partyPlayer(party.me)) return null;
      return end.balls.filter((b) => b.pid === party.me).length < end.perSide ? myColor() : null;
    }
    if (end.sides.length === 1) return end.balls.length < end.perSide ? "red" : null;
    const left = (sd) => end.perSide - ballsOf(end, sd).length;
    if (!left("red") && !left("blue")) return null;
    if (!left("red")) return "blue";
    if (!left("blue")) return "red";
    if (!end.balls.length) return vs.first;
    const r = bestOf(end, "red"), b = bestOf(end, "blue");
    if (!r) return "red";
    if (!b) return "blue";
    return closer(b, r) ? "red" : "blue";
  }

  // ---------- court geometry ----------
  // Distance from the jack is drawn on a log-rank scale: a ball whose nearest
  // word is the jack touches it; rings mark the top 10 / 100 / 1,000 / 10,000.
  const W = 400, H = 560, JX = 200, JY = 150, SY = 492, RS = SY - JY, KISS = 21;
  const rankR = (rank) => (rank <= 1 ? KISS : KISS + ((RS - KISS) * Math.log10(rank)) / Math.log10(G.space.n));
  function toXY(place, rank) {
    // Sideways angle is exaggerated so balls fan out; start (theta 0) stays straight down.
    const phi = Math.max(-2.5, Math.min(2.5, place.theta * 3));
    const r = rankR(rank);
    let x = JX + r * Math.sin(phi);
    let y = JY + r * Math.cos(phi);
    x = Math.max(26, Math.min(W - 26, x));
    y = Math.max(26, Math.min(H - 30, y));
    return { x, y };
  }

  // ---------- court rendering ----------
  let courtEls = null;
  // Court text is drawn in court units, so it shrinks with the court. On small screens words are
  // scaled up (CSS var --lb) to stay readable; the layout code spaces labels by the same factor.
  let labelBoost = 1;
  function buildCourt(end) {
    const svg = s("svg", { class: "court", viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": `Bocce court. Jack: ${end.target}. Start: ${end.start}.` });
    svg.append(s("defs", {},
      s("clipPath", { id: "courtClip" }, s("rect", { x: 10, y: 10, width: W - 20, height: H - 20, rx: 8 })),
      s("filter", { id: "rake", x: 0, y: 0, width: "100%", height: "100%" },
        s("feTurbulence", { type: "fractalNoise", baseFrequency: "0.018 0.55", numOctaves: 2, seed: 4, result: "noise" }),
        s("feColorMatrix", { in: "noise", type: "matrix", values: "0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1.3 -0.45", result: "mask" }),
        s("feFlood", { style: "flood-color:var(--gravel-shade)", result: "tint" }),
        s("feComposite", { in: "tint", in2: "mask", operator: "in" })),
      s("filter", { id: "grit", x: 0, y: 0, width: "100%", height: "100%" },
        s("feTurbulence", { type: "fractalNoise", baseFrequency: "0.9", numOctaves: 1, seed: 9 }),
        s("feColorMatrix", { type: "matrix", values: "0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 0.9 -0.45" })),
      s("radialGradient", { id: "shine", cx: "35%", cy: "30%", r: "65%" },
        s("stop", { offset: "0", style: "stop-color:#fff;stop-opacity:.7" }),
        s("stop", { offset: ".45", style: "stop-color:#fff;stop-opacity:0" }),
        s("stop", { offset: "1", style: "stop-color:#000;stop-opacity:.28" }))));
    svg.append(s("rect", { x: 10, y: 10, width: W - 20, height: H - 20, rx: 8, style: "fill:var(--gravel)" }));
    svg.append(s("rect", { x: 10, y: 10, width: W - 20, height: H - 20, rx: 8, filter: "url(#rake)", "clip-path": "url(#courtClip)" }));
    svg.append(s("rect", { x: 10, y: 10, width: W - 20, height: H - 20, rx: 8, filter: "url(#grit)", opacity: 0.18 }));

    // rank rings: how many words sit closer to the ball than the jack does
    const rings = s("g", { "clip-path": "url(#courtClip)" });
    for (const [rank, label] of [[10, "top 10"], [100, "top 100"], [1000, "top 1,000"], [10000, "top 10,000"]]) {
      const r = rankR(rank);
      rings.append(s("circle", { class: "ring", cx: JX, cy: JY, r: r.toFixed(1) }));
      const a = Math.PI * 0.2;
      rings.append(s("text", { class: "ring-label", x: (JX + r * Math.cos(a) + 3).toFixed(1), y: (JY + r * Math.sin(a) + 11).toFixed(1) }, label));
    }
    svg.append(rings);
    svg.append(s("line", { class: "foul", x1: 18, x2: W - 18, y1: SY + 26, y2: SY + 26 }));
    svg.append(s("text", { class: "foul-label", x: 20, y: SY + 42 }, "foul line"));

    const measure = s("g", {});
    const ghosts = s("g", {});
    const balls = s("g", {});
    const labels = s("g", {});
    const mapLayer = s("g", { class: "map" });
    svg.append(mapLayer, measure, ghosts);

    // start ball (ghost) and jack
    const st = toXY({ theta: 0 }, end.startRank);
    svg.append(s("circle", { cx: st.x, cy: st.y, r: 12, style: "fill:none;stroke:var(--ink);stroke-width:1.5;stroke-dasharray:3 3" }));
    svg.append(s("text", { class: "start-label", x: st.x + 18, y: st.y + 5 }, end.start));
    svg.append(balls);
    svg.append(s("ellipse", { cx: JX + 2, cy: JY + 7, rx: 8, ry: 3, style: "fill:#000;opacity:.18" }));
    svg.append(s("circle", { cx: JX, cy: JY, r: 8.5, style: "fill:var(--jack);stroke:rgba(0,0,0,.35);stroke-width:1" }));
    svg.append(s("circle", { cx: JX, cy: JY, r: 8.5, fill: "url(#shine)" }));
    svg.append(s("text", { class: "jack-label", x: JX, y: JY - 18 * labelBoost, "text-anchor": "middle" }, end.target));
    svg.append(labels);

    // Everything drawn claims a box, so later words can avoid it (see placeText).
    const boxes = [textBox(JX, JY - 18 * labelBoost, end.target, 24 * labelBoost, "middle", 0.66), circleBox(JX, JY, 10),
      textBox(st.x + 18, st.y + 5, end.start, 14 * labelBoost, "start"), circleBox(st.x, st.y, 13)];
    for (const el of rings.querySelectorAll(".ring-label")) boxes.push(textBox(+el.getAttribute("x"), +el.getAttribute("y"), el.textContent, 9.5, "start", 0.62));
    courtEls = { svg, measure, ghosts, balls, labels, end, boxes, labelled: [] };
    const shown = end.balls.filter((b) => !b.pending);
    for (const b of shown) drawBall(b);
    // Label the newest balls first; the very newest always gets its word.
    [...shown].reverse().forEach((b, i) => drawLabel(b, i === 0));
    drawMeasure(end);
    if (end.done && end.par) drawGhost(end);
    if (end.whatIf) drawWhatIf(end);
    if (mapShown(end)) drawMap(end, mapLayer);
    return svg;
  }

  // ---------- court label layout ----------
  function textBox(x, y, text, size, anchor = "middle", k = 0.58) {
    const w = String(text).length * size * k;
    const x1 = anchor === "middle" ? x - w / 2 : x;
    return { x1, x2: x1 + w, y1: y - size * 0.82, y2: y + size * 0.28 };
  }
  const circleBox = (x, y, r) => ({ x1: x - r, x2: x + r, y1: y - r, y2: y + r });
  const overlaps = (a, b) => a.x1 < b.x2 && b.x1 < a.x2 && a.y1 < b.y2 && b.y1 < a.y2;
  const onCourt = (b) => b.x1 > 12 && b.x2 < W - 12 && b.y1 > 12 && b.y2 < H - 12;
  /**
   * Put a word next to a point: try below, above, right, left; take the first spot that is on the
   * court and covers nothing already drawn. Returns the text node, or null if there was no room
   * (unless `force`, which takes the first spot anyway).
   */
  function placeText(layer, x, y, word, size, cls, force, gap = 13) {
    const w = word.length * size * 0.58;
    const spots = [[x, y + gap + size * 0.9, "middle"], [x, y - gap - size * 0.3, "middle"],
      [x + gap + 3, y + size * 0.32, "start"], [x - gap - 3 - w, y + size * 0.32, "start"]];
    let pick = null;
    for (const [sx, sy, anchor] of spots) {
      const box = textBox(sx, sy, word, size, anchor);
      if (onCourt(box) && !courtEls.boxes.some((b) => overlaps(b, box))) { pick = { sx, sy, anchor, box }; break; }
    }
    if (!pick && force) { const [sx, sy, anchor] = spots[0]; pick = { sx, sy, anchor, box: textBox(sx, sy, word, size, anchor) }; }
    if (!pick) return null;
    courtEls.boxes.push(pick.box);
    const el = s("text", { class: cls, x: pick.sx.toFixed(1), y: pick.sy.toFixed(1), "text-anchor": pick.anchor }, word);
    layer.append(el);
    return el;
  }

  // ---------- mapping the space: a few real words placed on the court ----------
  // The jack's nearest neighbours, plus "bridge" words that sit between the start and the jack,
  // each drawn where a ball would stop if it landed exactly on that word. Few and faint, so the
  // court stays readable. Shown after a round (and on request in Practice and the tutorial).
  function mapWords(end) {
    if (end.map) return end.map;
    const sp = G.space, T = sp.row(end.target), S = sp.row(end.start);
    const fresh = (w, taken) => !taken.includes(w) && ![end.start, end.target, ...taken].some((x) => B.related(w, x));
    const pick = (vec, k, taken) => sp.survey(vec, end.target, [end.start, end.target], k + 10).near
      .filter((w) => fresh(w, taken)).slice(0, k);
    const near = pick(T, 5, []);
    const mid = Float32Array.from(S, (v, i) => v + T[i]);
    const bridge = pick(mid, 3, near);
    end.map = [...near.map((w) => ({ word: w, kind: "near" })), ...bridge.map((w) => ({ word: w, kind: "bridge" }))]
      .map((m) => ({ ...m, place: end.basis.place(sp.row(m.word)), rank: sp.survey(sp.row(m.word), end.target, [m.word], 1).rank }));
    return end.map;
  }
  const mapAllowed = (end) => end.done || end.kind === "practice" || end.kind === "tutorial";
  const mapShown = (end) => mapAllowed(end) && (end.showMap !== undefined ? end.showMap : end.done);
  function drawMap(end, layer) {
    const shownWords = new Set(courtEls.labelled.map((l) => l.word));
    for (const m of mapWords(end)) {
      if (shownWords.has(m.word)) continue; // already on the court as where a ball landed
      const { x, y } = toXY(m.place, m.rank);
      const dot = circleBox(x, y, 3);
      if (courtEls.boxes.some((b) => overlaps(b, dot))) continue;
      if (!placeText(layer, x, y, m.word, 12.5 * labelBoost, `map-word ${m.kind}`, false, 5)) continue;
      courtEls.boxes.push(dot);
      layer.append(s("circle", { class: `map-dot ${m.kind}`, cx: x.toFixed(1), cy: y.toFixed(1), r: 2.6 }));
    }
  }

  function ballNode(b) {
    const g = s("g", { class: "ball" });
    g.append(s("ellipse", { class: "shadow", cx: 2, cy: 10, rx: 12, ry: 4, style: "fill:#000;opacity:.22" }));
    g.append(s("circle", { r: 12.5, style: `fill:var(--${b.side});stroke:rgba(0,0,0,.35);stroke-width:1` }));
    g.append(s("circle", { r: 12.5, fill: "url(#shine)" }));
    g.append(s("text", { class: "ball-num", "text-anchor": "middle", y: 4 }, String(b.n)));
    return g;
  }
  function drawBall(b) {
    const { x, y } = toXY(b.place, b.rank);
    const g = ballNode(b);
    g.setAttribute("transform", `translate(${x.toFixed(1)} ${y.toFixed(1)})`);
    courtEls.balls.append(g);
    courtEls.boxes.push(circleBox(x, y, 13));
    return g;
  }
  /** The word where a ball stopped. One label per word per area; `force` for the ball just thrown. */
  function drawLabel(b, force) {
    const { x, y } = toXY(b.place, b.rank);
    const word = b.near[0];
    if (courtEls.labelled.some((l) => l.word === word && Math.hypot(l.x - x, l.y - y) < 90)) return;
    if (placeText(courtEls.labels, x, y, word, 15.5 * labelBoost, "ball-label", force)) courtEls.labelled.push({ word, x, y });
  }
  function drawMeasure(end) {
    const m = courtEls.measure;
    m.replaceChildren();
    const best = bestBall(end);
    if (!best) return;
    const { x, y } = toXY(best.place, best.rank);
    m.append(s("line", { class: "measure", x1: JX, y1: JY, x2: x, y2: y }));
  }
  function drawGhost(end) {
    const sc = G.space.score(end.start, end.target, end.par.tiles);
    const { x, y } = toXY(end.basis.place(sc.vec), sc.rank);
    courtEls.ghosts.replaceChildren(s("circle", { cx: x, cy: y, r: 12.5, style: "fill:none;stroke:var(--chalk);stroke-width:2;stroke-dasharray:4 3" }));
    placeText(courtEls.ghosts, x, y, "par", 11 * labelBoost, "ghost-label", true);
  }
  /** The "try your own words" ball: dashed, with where it stopped. */
  function drawWhatIf(end) {
    const w = end.whatIf;
    const { x, y } = toXY(end.basis.place(w.vec), w.rank);
    courtEls.ghosts.append(s("circle", { class: "whatif", cx: x, cy: y, r: 12.5 }));
    courtEls.boxes.push(circleBox(x, y, 13));
    placeText(courtEls.ghosts, x, y, "your idea: " + w.near[0], 13 * labelBoost, "whatif-label", true);
  }

  function animateBall(b) {
    return new Promise((resolve) => {
      const to = toXY(b.place, b.rank);
      const from = toXY({ theta: 0 }, courtEls.end.startRank);
      const SX = from.x, SY = from.y;
      const g = ballNode(b);
      courtEls.balls.append(g);
      const shadow = g.querySelector(".shadow");
      // bocce contact: the ball comes to rest touching another ball
      const touches = courtEls.end.balls.some((o) => o !== b && !o.pending &&
        Math.hypot(toXY(o.place, o.rank).x - to.x, toXY(o.place, o.rank).y - to.y) < 27);
      courtEls.boxes.push(circleBox(to.x, to.y, 13));
      if (reduceMotion) {
        sfx.roll(0, 0.1);
        g.setAttribute("transform", `translate(${to.x} ${to.y})`);
        drawLabel(b, true);
        return resolve();
      }
      const dur = 950, t0 = performance.now();
      sfx.roll(dur * 0.5 / 1000, dur / 1000);
      const bend = (to.x - SX) * 0.18;
      const step = (now) => {
        const t = Math.min(1, (now - t0) / dur);
        const e = 1 - Math.pow(1 - t, 3);
        const x = SX + (to.x - SX) * e + bend * Math.sin(Math.PI * e);
        const y = SY + (to.y - SY) * e;
        const air = t < 0.5 ? Math.sin((Math.PI * t) / 0.5) : 0;
        g.setAttribute("transform", `translate(${x.toFixed(1)} ${(y - air * 26).toFixed(1)}) scale(${(1 + air * 0.28).toFixed(3)})`);
        shadow.setAttribute("cy", String(10 + air * 26));
        shadow.setAttribute("opacity", String(1 - air * 0.6));
        if (t < 1) requestAnimationFrame(step);
        else {
          const puff = s("circle", { class: "puff", cx: to.x, cy: to.y, r: 13, style: "transform-box:fill-box;transform-origin:center" });
          courtEls.labels.append(puff);
          puff.animate([{ transform: "scale(1)", opacity: 0.9 }, { transform: "scale(2.3)", opacity: 0 }],
            { duration: 450, easing: "ease-out" }).onfinish = () => puff.remove();
          if (touches) sfx.clack();
          drawLabel(b, true);
          resolve();
        }
      };
      requestAnimationFrame(step);
    });
  }

  // ---------- rendering the page ----------
  const main = () => $("#main");

  function render() {
    hideTip();
    for (const b of document.querySelectorAll(".modes button")) {
      b.setAttribute("aria-selected", String(b.dataset.mode === mode || (mode === "party" && b.dataset.mode === "versus")));
    }
    const focusKey = document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.key : null;
    const m = main();
    if (mode === "puzzles" && !ends.puzzle) m.replaceChildren(renderPuzzleList());
    else if (mode === "versus" && !vs) m.replaceChildren(renderVersusSetup());
    else if (mode === "party" && !partyInRound()) m.replaceChildren(renderPartyLobby());
    else {
      const end = current();
      m.replaceChildren(h("div", { class: "table" },
        renderMatchup(end),
        h("div", { class: "court-col" }, renderScorebar(end), buildCourt(end),
          h("p", { class: "legend" },
            h("span", { class: "legend-text" }, "Dotted rings: the jack is in the ball's top 10, 100, 1,000 or 10,000 nearest words. ", tip("rings", "the rings"), " "),
            mapAllowed(end) && h("button", { class: "linkish", type: "button", "data-key": "map",
              onclick: () => { end.showMap = !mapShown(end); render(); } },
              mapShown(end) ? "Hide nearby words" : "Show nearby words on the court"))),
        renderBench(end)));
    }
    if (focusKey) { const el = m.querySelector(`[data-key="${CSS.escape(focusKey)}"]`); if (el) el.focus(); }
    fitDrawer();
  }
  // On phones the play panel is pinned to the bottom of the screen: leave room for it below the page,
  // and shrink the court to fit the space above it so the whole game is visible without scrolling.
  function fitDrawer() {
    const p = main().querySelector(".play");
    const pinned = !!p && getComputedStyle(p).position === "fixed";
    document.body.style.paddingBottom = pinned ? p.offsetHeight + 12 + "px" : "";
    const svg = main().querySelector("svg.court");
    if (!svg) return;
    svg.style.width = "";
    if (pinned) {
      const top = svg.getBoundingClientRect().top + scrollY;
      const room = Math.max(250, innerHeight - p.offsetHeight - top - 10);
      svg.style.width = Math.min(svg.parentElement.clientWidth, (room * 400) / 560) + "px";
    }
    const boost = Math.round(Math.max(1, Math.min(2, 340 / (svg.getBoundingClientRect().width || 400))) * 20) / 20;
    svg.style.setProperty("--lb", boost);
    if (boost !== labelBoost) {
      labelBoost = boost;
      if (!busy) requestAnimationFrame(render); // re-space the labels for the new size
    }
  }
  window.addEventListener("resize", () => { if (G) fitDrawer(); });
  function current() { return ends[mode === "puzzles" ? "puzzle" : mode]; }

  function renderScorebar(end) {
    if (end.kind === "party") {
      const st = party.state;
      const chips = st.players.filter((p) => !p.left || end.balls.some((b) => b.pid === p.id)).map((p) => {
        const theirs = end.balls.filter((b) => b.pid === p.id);
        const best = theirs.filter((b) => !b.pending).reduce((m, b) => (m && !closer(b, m) ? m : b), null);
        const left = end.perSide - theirs.length;
        return h("span", { class: `pchip ${p.left ? "gone" : ""}` },
          h("span", { class: `pip ${p.color} full`, "aria-hidden": "true" }),
          h("b", {}, p.id === party.me ? "You" : p.name), h("span", { class: "pts" }, String(st.scores[p.id] || 0)),
          best ? h("small", {}, ordinal(best.rank)) : null,
          h("span", { class: "pips", "aria-label": `${left} balls left` },
            Array.from({ length: end.perSide }, (_, i) => h("span", { class: `pip ${p.color} ${i < left ? "full" : ""}` }))));
      });
      return h("div", { class: "scorebar party" }, h("div", { class: "pchips" }, chips),
        h("span", {}, `Room ${party.code} · round ${st.round} · points in bold `, tip("party", "how room scoring works")));
    }
    if (end.kind === "versus") {
      const side = (sd) => h("span", { class: `side ${sd}` }, h("b", {}, String(vs.score[sd])), vs.names[sd],
        h("span", { class: "pips", "aria-label": `${end.perSide - ballsOf(end, sd).length} balls left` },
          Array.from({ length: end.perSide }, (_, i) => h("span", { class: `pip ${sd} ${i < end.perSide - ballsOf(end, sd).length ? "full" : ""}` }))));
      return h("div", { class: "scorebar" },
        h("div", { class: "versus-score" }, side("red"), side("blue")),
        h("span", {}, `Round ${vs.endNo} · first to ${VS_TARGET} points `, tip("versus", "how versus scoring works")));
    }
    const left = end.perSide - end.balls.length;
    const label = end.kind === "daily" ? h("span", {}, h("strong", {}, `Daily No. ${end.no}`), ` · ${end.iso}`)
      : end.kind === "puzzle" ? h("span", {}, h("strong", {}, `Puzzle ${end.puzzle.id}`), ` · ${end.puzzle.difficulty}`)
      : end.kind === "tutorial" ? h("span", {}, h("strong", {}, "Tutorial"), " · about 30 seconds")
      : h("span", {}, h("strong", {}, "Practice"), " · endless deals");
    return h("div", { class: "scorebar" }, label,
      h("span", {}, `${left} of ${end.perSide} balls left `,
        h("span", { class: "pips", "aria-hidden": "true" }, Array.from({ length: end.perSide }, (_, i) => h("span", { class: `pip red ${i < left ? "full" : ""}` })))));
  }

  function renderMatchup(end) {
    return h("div", { class: "matchup" },
      h("span", { class: "from" }, end.start), h("span", { class: "arrow", "aria-hidden": "true" }, "→"),
      h("span", { class: "to", title: "the jack (target word)" }, end.target),
      h("span", { class: "caption" },
        `Roll your ball from ${q(end.start)} to the yellow jack, ${q(end.target)}. Tap words to add (+) or subtract (−) their meaning, then Throw.`),
      h("span", { class: "caption small" },
        `Right now ${q(end.target)} is the ${ordinal(end.startRank)} nearest word to your ball. Get it to 1st. `, tip("rank", "rank")));
  }

  function renderBench(end) {
    const side = nextSide(end);
    const botTurn = end.kind === "versus" && vs.opponent === "bot" && side === "blue";
    const bench = h("section", { class: "bench", "aria-label": "Your throw" });

    if (end.kind === "puzzle") bench.append(h("p", { class: "caption" }, `${end.puzzle.name}: ${end.puzzle.description}`));

    if (end.kind === "versus" && !end.done && side) {
      bench.append(h("div", { class: "turn-banner" }, h("span", { class: `pip ${side} full` }),
        botTurn ? "The bot is lining up a throw…" : `${vs.names[side]} to throw` +
          (end.balls.length && vs.opponent === "friend" ? " (pass the device)" : "") +
          (end.balls.length ? ", because they're farther from the jack." : ".")));
    }

    if (end.kind === "party" && !end.done) {
      const mineLeft = end.perSide - end.balls.filter((b) => b.pid === party.me).length;
      const waiting = partyActive().filter((p) => p.id !== party.me && end.balls.filter((b) => b.pid === p.id).length < end.perSide).map((p) => p.name);
      bench.append(h("div", { class: "turn-banner" }, h("span", { class: `pip ${myColor()} full` }),
        mineLeft ? `Everyone throws at once. You have ${mineLeft} of ${end.perSide} balls left. Other players' words stay hidden until the round ends.`
          : `All your balls are thrown. Waiting for ${listNames(waiting)}…`));
    }

    if (end.kind === "tutorial" && end.intro) {
      bench.append(h("div", { class: "play" }, h("div", { class: "coach intro" },
        h("p", {}, h("b", {}, "Word Bocce is bocce played with meanings.")),
        h("p", {}, "Every word has a spot on the court. Words with similar meanings sit close together."),
        h("p", {}, `Your ball starts on ${q(end.start)} (the dashed circle). The yellow ball, the jack, is ${q(end.target)}. You move your ball by adding and subtracting words. Get it as close to the jack as you can.`),
        h("div", { class: "row" },
          h("button", { class: "primary", type: "button", "data-key": "tut-go", onclick: () => { end.intro = false; render(); } }, "Show me"),
          h("button", { class: "ghost", type: "button", onclick: () => switchMode("daily") }, "Skip")))));
      return bench;
    }

    if (!end.done && !(end.kind === "party" && !side && !busy)) {
      const tut = end.kind === "tutorial" && !busy ? tutorialStep(end) : null;
      const play = h("div", { class: "play" });
      if (tut) play.append(h("p", { class: "coach", role: "status" }, tut.text));

      // rack
      const rack = h("div", { class: "rack", "aria-live": "polite" }, h("span", { class: "base" }, end.start));
      if (!end.rack.length) rack.append(h("span", { class: "hint" }, "your throw: tap words below"));
      for (const t of end.rack) {
        rack.append(h("button", { class: `chip ${t.sign > 0 ? "plus" : "minus"}`, type: "button", "data-key": "chip-" + t.word,
          title: "Tap to switch between adding and subtracting", disabled: busy || botTurn || end.kind === "tutorial",
          onclick: () => { t.sign = -t.sign; sfx.tile(t.sign); render(); } }, `${signChar(t.sign)} ${t.word}`));
      }
      play.append(rack);

      const throwBtn = h("button", { class: `throw ${side && side !== "red" ? side : ""} ${tut && tut.throw ? "pulse" : ""}`, type: "button", "data-key": "throw",
        disabled: busy || botTurn || !end.rack.length || (end.kind === "tutorial" && !(tut && tut.throw)),
        onclick: () => playerThrow(end, side) }, "Throw");
      play.append(h("div", { class: "actions" }, throwBtn,
        end.kind !== "tutorial" && h("button", { class: "ghost", type: "button", "data-key": "clear", disabled: busy || !end.rack.length,
          onclick: () => { end.rack = []; setStatus(""); render(); } }, "Clear"),
        end.kind === "practice" && h("button", { class: "ghost", type: "button", "data-key": "redeal",
          disabled: busy, onclick: () => { ends.practice = dealPractice(); setStatus(""); render(); } }, "Deal a new court"),
        end.kind === "puzzle" && h("button", { class: "ghost", type: "button", "data-key": "hint",
          onclick: () => { end.showHint = true; render(); } }, "Hint"),
        end.kind === "puzzle" && h("button", { class: "ghost", type: "button", "data-key": "back",
          onclick: () => { ends.puzzle = null; render(); } }, "All puzzles")));
      if (!tut) {
        const why = statusMsg.why && end.balls.includes(statusMsg.why) ? statusMsg.why : null;
        play.append(h("div", { class: `status ${statusMsg.warn ? "warn" : ""}`, role: "status" },
          end.kind === "puzzle" && end.showHint && !statusMsg.text ? "Hint: " + end.puzzle.hint : statusMsg.text,
          why && " ", why && whyButton(end, why, readThrow(end, why).surprising ? "That one's surprising. Why?" : "Why?")));
      }

      // hand
      const hand = h("div", { class: "hand", role: "group", "aria-label": "Your tiles" });
      const words = end.wildWord ? [...end.hand, end.wildWord] : end.hand;
      for (const w of words) {
        const inRack = end.rack.find((t) => t.word === w);
        const cls = inRack ? (inRack.sign > 0 ? "plus" : "minus") : "";
        const wanted = tut && tut.tile === w;
        hand.append(h("button", { class: `tile ${cls} ${w === end.wildWord ? "joker" : ""} ${wanted ? "pulse" : ""}`, type: "button", "data-key": "tile-" + w,
          "aria-pressed": inRack ? "true" : "false", disabled: busy || botTurn || (end.kind === "tutorial" && !wanted),
          "aria-label": inRack ? `${w}, ${inRack.sign > 0 ? "added" : "subtracted"}` : w,
          onclick: () => cycleTile(end, w) },
          w, h("span", { class: "sign", "aria-hidden": "true" }, inRack ? signChar(inRack.sign) : "")));
      }
      if (end.wild) {
        hand.append(h("button", { class: "tile joker", type: "button", "data-key": "wild", disabled: busy,
          onclick: () => { end.wildOpen = true; render(); setTimeout(() => { const i = $("#wildInput"); if (i) i.focus(); }); } },
          end.wildWord ? "change wild word" : "any word…", h("span", { class: "sign" }, "✱")));
      }
      play.append(hand);
      if (end.kind !== "tutorial") play.append(h("p", { class: "hand-help" }, "Tap a word once to add it (+), twice to subtract it (−), three times to take it back. Up to three words per throw."));
      if (end.wild && end.wildOpen) play.append(renderWildForm(end));
      bench.append(play);
    }

    if (end.done) {
      bench.append(end.kind === "versus" ? renderVersusEnd(end) : end.kind === "tutorial" ? renderTutorialEnd(end)
        : end.kind === "party" ? renderPartyResults(end) : renderSoloEnd(end));
      if (end.kind !== "tutorial") bench.append(renderWhatIf(end));
    }
    bench.append(renderLog(end));
    return bench;
  }

  function renderWildForm(end) {
    const form = h("form", { class: "joker-form", onsubmit: (ev) => {
      ev.preventDefault();
      const w = $("#wildInput").value.trim().toLowerCase();
      if (!G.space.has(w)) return setStatus(`"${w}" isn't in the 40,000-word vocabulary. Try a more common word.`, true);
      if (w === end.start || w === end.target) return setStatus("The wild tile can't be the start word or the jack.", true);
      if (end.hand.includes(w)) return setStatus(`"${w}" is already in your hand.`, true);
      end.rack = end.rack.filter((t) => t.word !== end.wildWord);
      end.wildWord = w;
      end.wildOpen = false;
      setStatus(`Wild tile set to "${w}".`);
      render();
    } },
    h("label", { class: "sr", for: "wildInput" }, "Wild word"),
    h("input", { id: "wildInput", autocomplete: "off", autocapitalize: "none", spellcheck: "false", placeholder: "type any word", value: end.wildWord || "" }),
    h("button", { class: "ghost", type: "submit" }, "Use it"));
    return form;
  }

  function renderLog(end) {
    const best = bestBall(end);
    const ol = h("ol", { class: "log", "aria-label": "Throws this round" });
    if (end.balls.some((b) => !b.pending)) {
      ol.append(h("li", { class: "head", "aria-hidden": "true" }, h("span", {}),
        h("span", {}, end.kind === "party" ? "Throw → where it landed " : "Your throw → where it landed ", tip("near", "where it landed")),
        h("span", { class: "score" }, "Jack's rank ", tip("rank", "rank"))));
    }
    for (const b of [...end.balls].reverse()) {
      if (b.pending) continue;
      const t = B.tier(b.rank);
      const who = end.kind === "party" ? (b.pid === party.me ? "You: " : partyName(b.pid) + ": ") : "";
      const hidden = end.kind === "party" && !end.done && b.pid !== party.me;
      ol.append(h("li", { class: `${b === best ? "best" : ""} tier-${t.key}` },
        h("span", { class: `ballmark ${b.side}`, "aria-hidden": "true" }, String(b.n)),
        h("span", { class: "eq" }, who && h("b", { class: "who" }, who),
          hidden ? h("span", { class: "near" }, "words hidden until the round ends ") : eqText(end.start, b.tiles) + " ",
          h("span", { class: "near" }, "→ by ", h("b", {}, b.near[0])),
          !hidden && " ", !hidden && whyButton(end, b, "why?")),
        h("span", { class: "score" }, h("b", {}, `#${b.rank.toLocaleString()}`),
          h("small", { class: "tierlabel", "data-tip": TIPS.tiers, tabindex: "0" }, t.label),
          h("small", { class: "sim", "data-tip": TIPS.similarity, tabindex: "0" }, `similarity ${fmt(b.sim)}`))));
    }
    return ol;
  }

  // ---------- "why did that happen?" ----------
  // A reading of one throw. The per-word shares are exact arithmetic (Space.explainThrow); the
  // plain-language reasons are guesses from those numbers, and the dialog says so. Notes flag the
  // things that tend to confuse people; "strong" ones make the throw count as surprising.
  function readThrow(end, ball) {
    if (ball._read) return ball._read;
    const sp = G.space, S = end.start, J = end.target;
    const ex = sp.explainThrow(S, J, ball.tiles);
    const notes = [];
    const lines = ex.parts.map((p) => {
      const nm = p.isStart ? q(p.word) : `${signChar(p.sign)} ${p.word}`, tj = fmt(p.toJack);
      if (p.isStart) {
        return `Your start word ${q(S)} already shares ${p.toJack >= 0.5 ? "a lot" : p.toJack >= 0.25 ? "something" : "little"} with ${q(J)} (similarity ${tj}).`;
      }
      if (p.sign > 0) {
        if (p.toJack >= 0.35) return `${nm} pulled toward ${q(J)}: the two are close on this map (${tj}).`;
        if (p.toJack >= 0.15) return `${nm} pulled only a little toward ${q(J)}: they're loosely linked (${tj}).`;
        notes.push({ text: `${q(p.word)} has almost nothing to do with ${q(J)} on this map (${tj}), so adding it mostly dragged the ball toward ${q(p.word)}'s own neighbourhood.` });
        return `${nm} hardly points at ${q(J)} (${tj}); it mostly pulled the ball toward its own neighbourhood.`;
      }
      // Subtracting a word takes away everything it shares with the jack too, so what matters is its
      // link to the jack, not whether it "feels" like a feature of the start word.
      if (p.toJack >= 0.3 || (p.toJack >= p.toStart && p.toJack > 0.15)) {
        notes.push({ strong: true, text: `Taking away ${q(p.word)} also took away a lot of ${q(J)}: on this map ${q(p.word)} is close to ${q(J)} (${tj}) as well as to ${q(S)} (${fmt(p.toStart)}). It's natural to picture subtraction as removing one feature, but it removes everything the two words have in common, including the link you wanted to keep.` });
        return `${nm} pushed you away from ${q(J)}: ${p.word} is itself close to ${J} (${tj}), so subtracting it removed much of what ${S} and ${J} share.`;
      }
      if (p.toJack <= 0.1) return `${nm} cost almost nothing (${p.word} and ${J} are barely linked, ${tj}) and moved the ball away from ${p.word}-related words.`;
      return `${nm} moved the ball away from the ${p.word}-related side of ${q(S)} (${fmt(p.toStart)}), at a small cost: ${p.word} is somewhat linked to ${J} (${tj}).`;
    });
    if (ball.rank > end.startRank && ball.sim > end.startSim + 0.01) {
      notes.push({ strong: true, text: `Similarity to ${q(J)} went up (${fmt(end.startSim)} → ${fmt(ball.sim)}), yet its rank got worse: other words (${ball.near.slice(0, 2).map(q).join(", ")}) got even closer. Rank counts how many words beat the jack, not just how close it is.` });
    }
    const landing = ball.near[0];
    const links = [S, J, ...ball.tiles.map((t) => t.word)].filter((w) => sp.has(w)).map((w) => sp.sim(landing, w));
    if (landing !== J && Math.max(...links) < 0.2) {
      notes.push({ strong: true, text: `The ball landed near ${q(landing)}, which isn't close to any of your words or to the jack. When words partly cancel each other out, what's left can be a weaker, stranger meaning. (On the raw-text map, "bacon" minus "meat" leaves Bacon the surname.)` });
    }
    if (!end._startNear) end._startNear = sp.survey(sp.row(S), J, [S], 4).near.filter((w) => w !== J).slice(0, 3);
    ball._read = { ex, lines, notes, surprising: notes.some((n) => n.strong) || ball.rank > end.startRank, startNear: end._startNear };
    return ball._read;
  }
  function whyButton(end, ball, label) {
    return h("button", { class: "linkish why", type: "button", onclick: (e) => { e.stopPropagation(); openWhy(end, ball); } }, label);
  }
  function openWhy(end, ball) {
    const r = readThrow(end, ball), J = end.target;
    const maxPush = Math.max(...r.ex.parts.map((p) => Math.abs(p.push)), 0.01);
    const body = $("#whyBody");
    body.replaceChildren(...[
      h("p", { class: "why-eq" }, h("b", {}, eqText(end.start, ball.tiles)), ` → stopped near ${q(ball.near[0])}`),
      h("p", {}, ball.rank === 1 ? `${q(J)} is the nearest word to where the ball stopped: a bacio.`
        : `${q(J)} is the ${ordinal(ball.rank)} nearest word to where it stopped. It was ${ordinal(end.startRank)} at the start.`),
      h("h3", {}, "What each word did"),
      h("ul", { class: "why-parts" }, r.ex.parts.map((p, i) => h("li", {},
        h("span", { class: "why-bar", "aria-hidden": "true" },
          h("i", { class: p.push >= 0 ? "pos" : "neg", style: `width:${Math.round((Math.abs(p.push) / maxPush) * 100)}%` })),
        h("span", {}, r.lines[i])))),
      h("p", { class: "hint" }, `The bars are each word's share of the ball's similarity to ${q(J)} (${fmt(r.ex.sim)}); they add up exactly. That part is plain arithmetic.`),
      h("h3", {}, "Who's crowding the jack"),
      h("p", {}, `Nearest words to your start word: ${r.startNear.map(q).join(", ")}. Nearest to where the ball stopped: ${ball.near.map(q).join(", ")}.`),
      r.notes.length ? h("h3", {}, "What might be surprising") : null,
      r.notes.length ? h("ul", { class: "why-notes" }, r.notes.map((n) => h("li", {}, n.text))) : null,
      h("div", { class: "why-caveat" },
        h("p", {}, h("b", {}, "This is our reading of the numbers, not the model's actual reasons. "),
          "The game can measure exactly how close two words are. Why they ended up close is a guess: nobody can fully explain what a model like this has learned."),
        h("button", { class: "linkish", type: "button", onclick: () => { $("#why").close(); $("#interp").showModal(); } },
          "Why can't it explain itself?"))].filter(Boolean));
    $("#why").showModal();
  }

  function renderTutorialEnd(end) {
    const best = bestBall(end);
    store.set("tutorialDone", true);
    return h("div", { class: "endcard", role: "region", "aria-label": "Tutorial done" },
      h("h2", {}, best.rank === 1 ? "Bacio! A perfect throw." : "Nice throw."),
      h("p", {}, `${eqText(end.start, best.tiles)} landed ${best.rank === 1 ? "right by" : "near"} ${q(end.target)}. `,
        best.rank === 1 ? `It's now the nearest word to your ball. (Bacio is Italian for "kiss": a ball touching the jack.)` : ""),
      h("p", {}, "That's the whole game. Each round you get a start word, a jack and nine words to throw with:"),
      h("ul", {},
        h("li", {}, "Add words that point toward the jack."),
        h("li", {}, "Subtract words that drag you back toward the start."),
        h("li", {}, "You get four balls. Your best one counts.")),
      h("p", {}, "Some words are traps: they look related but pull the wrong way. At the end you'll see par, the best throw your words allowed."),
      h("div", { class: "row" },
        h("button", { type: "button", "data-key": "tut-daily", onclick: () => switchMode("daily") }, "Play today's court"),
        h("button", { class: "alt", type: "button", onclick: () => switchMode("practice") }, "Practice")));
  }

  function renderSoloEnd(end) {
    const best = bestBall(end);
    const par = end.par;
    const span = par ? par.sim - end.startSim : 1;
    const pct = Math.max(0, Math.min(1, (best.sim - end.startSim) / (span || 1)));
    // Stars from whichever is kinder: share of the par distance, or the rank reached.
    const byPct = pct >= 0.9 ? 3 : pct >= 0.7 ? 2 : pct >= 0.4 ? 1 : 0;
    const byRank = best.rank === 1 ? 3 : best.rank <= 3 ? 2 : best.rank <= 10 ? 1 : 0;
    const stars = Math.max(byPct, byRank);
    const head = ["A rough round", "A decent round", "A fine round", "A perfect round"][stars];
    if (end.kind === "puzzle") {
      const k = "puzzle:" + end.puzzle.id;
      if (stars > store.get(k, 0)) store.set(k, stars);
    }
    const parScore = par ? G.space.score(end.start, end.target, par.tiles) : null;
    const beatPar = parScore && closer(best, par);
    const around = G.space.survey(G.space.row(end.target), end.target, [end.target], 8).near;
    const card = h("div", { class: "endcard", role: "region", "aria-label": "Round summary" },
      h("h2", {}, head),
      h("div", { class: "stars-row", "aria-label": `${stars} of 3 stars` },
        [0, 1, 2].map((i) => h("span", { class: i < stars ? "" : "off" }, "●"))),
      h("p", {}, best.rank === 1
        ? `Your best ball made ${q(end.target)} the nearest word of all: 1st. `
        : `With your best ball, ${q(end.target)} was the ${ordinal(best.rank)} nearest word. `,
        `It started ${ordinal(end.startRank)}. `, tip("rank", "rank")),
      parScore && h("p", {}, beatPar
        ? "Your wild word beat par, the best throw from the fixed tiles."
        : `You closed ${Math.round(pct * 100)}% of the gap between your start and par. `, tip("par", "par")),
      h("div", { class: "meter", "aria-hidden": "true" }, h("i", { style: `width:${Math.round(pct * 100)}%` })),
      h("div", { class: "meter-ends", "aria-hidden": "true" }, h("span", {}, "start"), h("span", {}, "par")),
      parScore && h("p", { class: "par" }, `Par: ${eqText(end.start, par.tiles)} → ${ordinal(parScore.rank)}`),
      h("p", { class: "neighbours" }, `The words closest in meaning to ${q(end.target)}: ${around.join(", ")}.`));
    const row = h("div", { class: "row" });
    if (end.kind === "daily") {
      row.append(h("button", { type: "button", "data-key": "share", onclick: () => share(end, best, pct) }, "Copy result"),
        h("button", { class: "alt", type: "button", onclick: () => switchMode("practice") }, "Keep practising"));
    } else if (end.kind === "practice") {
      row.append(h("button", { type: "button", "data-key": "again", onclick: () => { ends.practice = dealPractice(); render(); } }, "Deal a new court"));
    } else {
      const i = G.puzzles.indexOf(end.puzzle);
      const nextP = G.puzzles[i + 1];
      if (nextP) row.append(h("button", { type: "button", "data-key": "nextpz", onclick: () => { ends.puzzle = dealPuzzle(nextP); render(); } }, "Next puzzle"));
      row.append(h("button", { class: "alt", type: "button", onclick: () => { ends.puzzle = dealPuzzle(end.puzzle); render(); } }, "Replay"),
        h("button", { class: "alt", type: "button", onclick: () => { ends.puzzle = null; render(); } }, "All puzzles"));
    }
    card.append(row);
    return card;
  }

  async function share(end, best, pct) {
    const dots = { bacio: "🟡", close: "🟢", hunt: "🟤", wide: "⚪", lost: "⚫" };
    const text = [`Word Bocce · Daily No. ${end.no}`, `${end.start} → ${end.target}`,
      end.balls.map((b) => dots[B.tier(b.rank).key]).join("") + `  best #${best.rank} · ${Math.round(pct * 100)}% to par`,
      SHARE_URL].join("\n");
    try {
      await navigator.clipboard.writeText(text);
      setStatus("Result copied. Paste it anywhere.");
    } catch (e) {
      setStatus("Copy isn't allowed here. Your result: " + text.replace(/\n/g, " · "));
    }
    render();
  }

  // ---------- "try your own words": explore after a round, and suggest better tiles ----------
  // Any in-vocabulary words, not just the hand. Suggestions are saved with the court they belong
  // to, so the deal logic and the puzzles can be tuned from what players wished they'd had.
  const SUGGEST_KEY = "suggestions";
  // Where suggestions are sent: the game server's /api/suggestions. When the page is served by that
  // server (the Linode, port 8000) it's the same origin; from GitHub Pages it must be an HTTPS address.
  // Until one answers, suggestions wait in this browser and are retried on later visits.
  const FEEDBACK_ORIGINS = {
    "wordbocce.davidreinstein.org": "https://45-79-160-157.sslip.io",
    "daaronr.github.io": "https://45-79-160-157.sslip.io",
  };
  const feedbackURL = () => {
    if (window.WORD_BOCCE_API !== undefined) return window.WORD_BOCCE_API && window.WORD_BOCCE_API + "/api/suggestions";
    if (location.port === "8000" || location.hostname.endsWith("sslip.io")) return "/api/suggestions";
    return FEEDBACK_ORIGINS[location.hostname] ? FEEDBACK_ORIGINS[location.hostname] + "/api/suggestions" : null;
  };
  function clientId() {
    let id = store.get("clientId", "");
    if (!id) { id = Math.random().toString(36).slice(2, 12); store.set("clientId", id); }
    return id;
  }
  let flushing = false;
  /** Send any suggestions not yet delivered. Returns true if all are delivered. */
  async function flushSuggestions() {
    const url = feedbackURL();
    if (!url || flushing) return false;
    flushing = true;
    try {
      const list = store.get(SUGGEST_KEY, []);
      for (const sg of list) {
        if (sg.sent) continue;
        const ctl = new AbortController();
        const timer = setTimeout(() => ctl.abort(), 8000);
        try {
          const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ...sg, client: clientId() }), signal: ctl.signal });
          if (r.ok || r.status === 422) sg.sent = true; // 422: malformed, never going to succeed
          else break;
        } catch (e) { break; } finally { clearTimeout(timer); }
      }
      store.set(SUGGEST_KEY, list);
      return list.every((sg) => sg.sent);
    } finally { flushing = false; }
  }
  function parseThrow(text) {
    const tiles = [];
    const re = /([+\-−–])?\s*([a-z][a-z'-]*)/gi;
    let m;
    while ((m = re.exec(text)) && tiles.length < 4) {
      const word = m[2].toLowerCase();
      if (!tiles.some((t) => t.word === word)) tiles.push({ word, sign: m[1] && m[1] !== "+" ? -1 : 1 });
    }
    return tiles;
  }
  function renderWhatIf(end) {
    const w = end.whatIf;
    const best = end.kind === "party" ? end.balls.filter((b) => b.pid === party.me).reduce((m, b) => (m && !closer(b, m) ? m : b), null) : bestBall(end);
    const box = h("section", { class: "whatif-box", "aria-label": "Try your own words" },
      h("h3", {}, "Try your own words"),
      h("p", { class: "hint" }, `Wish you'd had other words? Type any throw, like “+ ears − hear”, to see where it would have landed.`));
    const form = h("form", { class: "joker-form", onsubmit: (ev) => {
      ev.preventDefault();
      const text = form.querySelector("input").value;
      const tiles = parseThrow(text);
      const missing = tiles.filter((t) => !G.space.has(t.word)).map((t) => t.word);
      if (!tiles.length) { end.whatIfError = "Type one to four words, each with + or − in front."; return render(); }
      if (missing.length) { end.whatIfError = `Not in the 40,000-word vocabulary: ${missing.join(", ")}. Try a more common word.`; return render(); }
      if (tiles.some((t) => t.word === end.start || t.word === end.target)) { end.whatIfError = "Leave out the start word and the jack."; return render(); }
      const sc = G.space.score(end.start, end.target, tiles);
      end.whatIf = { text, tiles, rank: sc.rank, sim: sc.sim, near: sc.near, vec: sc.vec };
      end.whatIfError = "";
      end.whatIfSaved = false;
      render();
    } },
    h("label", { class: "sr", for: "whatIfInput" }, "Your throw"),
    h("input", { id: "whatIfInput", "data-key": "whatif-input", autocomplete: "off", autocapitalize: "none", spellcheck: "false",
      placeholder: "+ ears − hear", value: w ? w.text : "" }),
    h("button", { class: "ghost", type: "submit", "data-key": "whatif-go" }, "Try it"));
    box.append(form);
    if (end.whatIfError) box.append(h("p", { class: "status warn" }, end.whatIfError));
    if (w) {
      const vsBest = best ? (w.rank < best.rank ? " Better than your best ball" : w.rank === best.rank ? " Level with your best ball" : " Not as close as your best ball") + ` (${ordinal(best.rank)}).` : "";
      box.append(h("p", { class: "whatif-result" }, h("b", {}, eqText(end.start, w.tiles)), ` lands near ${q(w.near[0])}. `,
        `${q(end.target)} would be the ${ordinal(w.rank)} nearest word.${vsBest} It's the dashed purple ball on the court.`));
      const outside = w.tiles.filter((t) => !end.hand.includes(t.word)).map((t) => t.word);
      if (outside.length && !end.whatIfSaved) {
        box.append(h("p", {}, `Would having ${outside.map(q).join(" and ")} among the words have made this court more fun?`),
          h("label", { class: "sr", for: "whatIfNote" }, "Note (optional)"),
          h("input", { id: "whatIfNote", class: "note", "data-key": "whatif-note", maxlength: 300, autocomplete: "off",
            placeholder: "Anything else? (optional)" }),
          h("div", { class: "row" },
            h("button", { class: "ghost", type: "button", "data-key": "fun-yes", onclick: () => saveSuggestion(end, outside, "more fun") }, "Yes, more fun"),
            h("button", { class: "ghost", type: "button", "data-key": "fun-same", onclick: () => saveSuggestion(end, outside, "about the same") }, "About the same"),
            h("button", { class: "ghost", type: "button", "data-key": "fun-no", onclick: () => saveSuggestion(end, outside, "too easy or odd") }, "No: too easy, or odd")));
      }
      if (end.whatIfSaved) {
        box.append(h("p", { class: "hint" }, end.whatIfSaved === "sent"
          ? "Thanks, sent. Suggestions like this are used to improve the word lists."
          : end.whatIfSaved === "sending" ? "Thanks, sending…"
          : "Thanks, noted. It's saved on this device and will be sent next time the game can reach its server."));
      }
    }
    return box;
  }
  async function saveSuggestion(end, words, verdict) {
    const w = end.whatIf;
    const note = (($("#whatIfNote") || {}).value || "").trim().slice(0, 300);
    const list = store.get(SUGGEST_KEY, []);
    list.push({ at: new Date().toISOString(), kind: end.kind, seed: end.seed, start: end.start, target: end.target, hand: end.hand,
      throw: w.tiles, rank: w.rank, bestRank: (bestBall(end) || {}).rank || null, words, verdict, note, sent: false });
    // Keep at most 200, dropping delivered ones first.
    while (list.length > 200) list.splice(Math.max(0, list.findIndex((sg) => sg.sent)), 1);
    store.set(SUGGEST_KEY, list);
    end.whatIfSaved = feedbackURL() ? "sending" : "kept";
    render();
    if (end.whatIfSaved === "sending") { end.whatIfSaved = (await flushSuggestions()) ? "sent" : "kept"; render(); }
  }

  // ---------- actions ----------
  function setStatus(text, warn) { statusMsg = { text, warn: !!warn }; if (text) render(); }

  function cycleTile(end, w) {
    if (busy || end.done) return;
    const i = end.rack.findIndex((t) => t.word === w);
    if (i < 0) {
      if (end.rack.length >= B.MAX_TILES) {
        sfx.nope();
        setStatus("Three tiles per throw. Tap a lit tile to change it.", true);
        const again = main().querySelector(`[data-key="${CSS.escape("tile-" + w)}"]`);
        if (again) again.classList.add("shake");
        return;
      }
      end.rack.push({ word: w, sign: 1 });
      sfx.tile(1);
    } else if (end.rack[i].sign > 0) { end.rack[i].sign = -1; sfx.tile(-1); }
    else { end.rack.splice(i, 1); sfx.untile(); }
    statusMsg = { text: "", warn: false };
    render();
  }

  async function playerThrow(end, side) {
    if (busy || !end.rack.length || !side) return;
    await throwBall(end, side, end.rack.map((t) => ({ ...t })));
  }

  async function throwBall(end, side, tiles) {
    const key = tileKey(tiles);
    // In an online room players can't see each other's words, so only your own throws must differ.
    const taken = end.kind === "party" ? end.balls.filter((b) => b.pid === party.me) : end.balls;
    if (taken.some((b) => tileKey(b.tiles) === key)) {
      sfx.nope();
      return setStatus("That exact throw is already on the court. Change a tile or a sign.", true);
    }
    busy = true;
    statusMsg = { text: "", warn: false };
    const mine = ballsOf(end, side);
    const before = mine.length ? Math.min(...mine.map((b) => b.rank)) : end.startRank;
    const ball = placeBall(end, side, tiles);
    ball.pending = true;
    if (end.kind === "party") {
      // Show it rolling straight away; the host's echo of this throw is matched by key, not re-added.
      ball.pid = party.me;
      ball.key = party.me + ":" + mine.length;
      partySendThrow(tiles);
    }
    end.rack = [];
    render();
    await animateBall(ball);
    ball.pending = false;
    busy = false;
    if (end.kind === "daily") store.set(WORD_SETS[end.set].dailyKey + end.iso, end.balls.map((b) => b.tiles));
    if (ball.rank === 1) sfx.bacio();
    const verdict = ball.rank < before ? "Closer!" : ball.rank === before ? "About the same." : "Further away.";
    statusMsg = { text: ball.rank === 1
      ? `Bacio! Ball ${ball.n} stopped right by ${q(end.target)}: it's the nearest word to the ball.`
      : `${verdict} Ball ${ball.n} stopped near ${q(ball.near[0])}. ${q(end.target)} is the ${ordinal(ball.rank)} nearest word to it` +
        (mine.length ? ` (${end.kind === "versus" ? vs.names[side] + "'s" : "your"} best so far was ${ordinal(before)}).` : ` (it was ${ordinal(end.startRank)} at the start).`), warn: false, why: ball };
    if (end.kind === "tutorial") statusMsg = { text: "", warn: false };
    if (end.kind === "party") {
      // Other players' throws that arrived while this ball was rolling.
      if (party && party.queued) return applyPartyState(party.state);
      return render();
    }
    if (!nextSide(end)) finishEnd(end);
    render();
    // On a phone the result card sits below the court, out of sight: bring it up.
    if (end.done) { const card = main().querySelector(".endcard"); if (card) card.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "nearest" }); }
    maybeBot();
  }

  function finishEnd(end) {
    end.done = true;
    end.showMap = undefined; // nearby words come up at the end, whatever was chosen during play
    if (end.kind === "versus") {
      const r = bestOf(end, "red"), b = bestOf(end, "blue");
      const winner = closer(b, r) ? "blue" : "red";
      const loserBest = winner === "red" ? b : r;
      const pts = ballsOf(end, winner).filter((x) => closer(x, loserBest)).length;
      vs.score[winner] += pts;
      vs.first = winner;
      end.result = { winner, pts };
      if (vs.score[winner] >= VS_TARGET) vs.over = true;
      // Against the bot, blue winning is a loss for you; two humans on one device always get the fanfare.
      setTimeout(vs.opponent === "bot" && winner === "blue" ? sfx.lose : sfx.win, 700);
    } else {
      setTimeout(bestBall(end).rank <= 10 ? sfx.win : sfx.lose, 700);
    }
  }

  // ---------- versus ----------
  function renderVersusSetup() {
    const start = (opponent, level) => {
      vs = { opponent, level, score: { red: 0, blue: 0 }, endNo: 1, first: "red", over: false,
        names: opponent === "bot" ? { red: "You", blue: "Bot" } : { red: "Red", blue: "Blue" } };
      ends.versus = dealVersus();
      render();
    };
    return h("div", { class: "setup" },
      h("h2", {}, "Versus"),
      h("p", {}, `Red and Blue share one hand of tiles and get ${VS_BALLS} balls each per round. Whoever is farther from the jack throws next. When the balls are gone, the closer side scores one point for every ball that beats the other side's best. First to ${VS_TARGET} points wins.`),
      h("div", { class: "row" },
        h("button", { class: "primary", type: "button", onclick: () => start("bot", "club") }, "Play the bot"),
        h("button", { type: "button", onclick: () => start("bot", "pro") }, "Play the bot (pro)"),
        h("button", { type: "button", onclick: () => start("friend") }, "Two players, one device")),
      h("h2", {}, "Online"),
      h("p", {}, "Play friends on their own phones or computers: everyone throws at the same court at once."),
      h("div", { class: "row" },
        h("button", { class: "primary", type: "button", "data-key": "go-party", onclick: () => switchMode("party") },
          party ? `Back to room ${party.code || ""}` : "Play online with friends")));
  }

  function renderVersusEnd(end) {
    const { winner, pts } = end.result;
    const par = end.par;
    const card = h("div", { class: "endcard", role: "region", "aria-label": "Round result" },
      h("h2", {}, vs.over ? `${vs.names[winner]} ${vs.names[winner] === "You" ? "win" : "wins"} the match` : `${vs.names[winner]} ${vs.names[winner] === "You" ? "win" : "wins"} the round`),
      h("p", {}, `${pts} point${pts === 1 ? "" : "s"}: one for each ${vs.names[winner] === "You" ? "of your balls" : "ball"} closer than the other side's best. `,
        `Score: ${vs.names.red} ${vs.score.red}, ${vs.names.blue} ${vs.score.blue}.`),
      par && h("p", { class: "par" }, `Par (the best throw this hand allowed): ${eqText(end.start, par.tiles)} `, tip("par", "par")));
    const row = h("div", { class: "row" });
    if (!vs.over) row.append(h("button", { type: "button", "data-key": "nextend", onclick: () => { vs.endNo++; ends.versus = dealVersus(); render(); maybeBot(); } }, "Next round"));
    row.append(h("button", { class: vs.over ? "" : "alt", type: "button", onclick: () => { vs = null; ends.versus = null; render(); } }, "New match"));
    card.append(row);
    return card;
  }

  async function maybeBot() {
    const end = ends.versus;
    if (mode !== "versus" || !vs || vs.opponent !== "bot" || !end || end.done || busy) return;
    if (nextSide(end) !== "blue") return;
    busy = true;
    render();
    await sleep(700);
    const used = new Set(end.balls.map((b) => tileKey(b.tiles)));
    const options = end.throws.filter((t) => !used.has(tileKey(t.tiles)));
    const reach = vs.level === "pro" ? 6 : 45;
    const pick = options[Math.floor(Math.pow(Math.random(), vs.level === "pro" ? 2 : 1.4) * Math.min(reach, options.length))];
    end.rack = pick.tiles.map((t) => ({ ...t }));
    busy = false;
    render();
    busy = true;
    await sleep(900);
    busy = false;
    if (ends.versus === end) await throwBall(end, "blue", pick.tiles);
  }

  // ---------- party: an online room where everyone plays the same court at once ----------
  // The host's browser keeps the room state and rebroadcasts it (see net.js). Deals and scores are
  // deterministic, so only the round's seed and each player's tile choices travel over the wire;
  // every browser scores the throws itself.
  const PARTY_BALLS = 3;
  const PARTY_COLORS = ["red", "blue", "green", "purple", "orange", "teal"];
  let party = null;          // { role: "host"|"guest", code, me, net, state, error, joining, closed, queued }
  let partyJoinCode = "";    // from a #room=CODE link, before joining

  const partyPlayer = (pid) => (party ? party.state.players.find((p) => p.id === pid) : null);
  const partyName = (pid) => (partyPlayer(pid) || { name: "Someone" }).name;
  const partyActive = () => (party ? party.state.players.filter((p) => !p.left) : []);
  const myColor = () => (partyPlayer(party && party.me) || { color: "red" }).color;
  const partyInRound = () => !!(party && party.state.round && ends.party && ends.party.round === party.state.round);
  const partyLink = (code) => location.href.split("#")[0] + "#room=" + code;
  const cleanName = (s) => String(s || "").replace(/\s+/g, " ").trim().slice(0, 18) || "Player";
  const listNames = (names) => (names.length <= 1 ? names.join("") : names.slice(0, -1).join(", ") + " and " + names[names.length - 1]);
  function setHash(hash) { try { history.replaceState(null, "", "#" + hash); } catch (e) { /* sandboxed */ } }

  // Scores are memoised per court and throw: the host and the results card both need them.
  const scoreMemo = new Map();
  function scoreOf(end, tiles) {
    const k = end.seed + "|" + tileKey(tiles);
    if (!scoreMemo.has(k)) { const sc = G.space.score(end.start, end.target, tiles); scoreMemo.set(k, { rank: sc.rank, sim: sc.sim }); }
    return scoreMemo.get(k);
  }
  /** Players ordered by their best ball; the winner scores a point per ball beating everyone else's best. */
  function partyStandings(st, end) {
    const byPid = new Map();
    for (const t of st.throws) {
      const sc = { ...scoreOf(end, t.tiles), tiles: t.tiles };
      const cur = byPid.get(t.pid);
      if (!cur || closer(sc, cur)) byPid.set(t.pid, sc);
    }
    const rows = [...byPid].map(([pid, best]) => ({ pid, best })).sort((a, b) => (closer(a.best, b.best) ? -1 : closer(b.best, a.best) ? 1 : 0));
    if (!rows.length) return { rows, winner: null, pts: 0 };
    const winner = rows[0].pid;
    const pts = rows.length > 1 ? st.throws.filter((t) => t.pid === winner && closer(scoreOf(end, t.tiles), rows[1].best)).length : 0;
    return { rows, winner, pts };
  }

  // host
  async function hostRoom(name) {
    store.set("name", name);
    party = { role: "host", me: "host", code: null, net: null, error: "",
      state: { players: [{ id: "host", name, color: PARTY_COLORS[0] }], round: 0, seed: null, vectors: wordSet, throws: [], scores: { host: 0 }, over: false, result: null } };
    render();
    try {
      party.net = await BocceNet.host({
        onOpen: (code) => { party.code = code; if (mode === "party") setHash("room=" + code); render(); },
        onJoin: () => {},
        onMessage: (pid, msg) => hostHandle(pid, msg),
        onLeave: (pid) => { const p = partyPlayer(pid); if (p) { p.left = true; hostCheckOver(); hostBroadcast(); } },
        onError: (m) => { if (party) { party.error = m; render(); } },
      });
    } catch (e) { party.error = e.message; render(); }
  }
  function hostHandle(pid, msg) {
    if (!party || !msg || typeof msg !== "object") return;
    const st = party.state;
    if (msg.t === "hello") {
      const name = cleanName(msg.name);
      let p = partyPlayer(pid);
      if (!p) {
        const used = new Set(partyActive().map((x) => x.color));
        const color = PARTY_COLORS.find((c) => !used.has(c));
        if (!color) return party.net.send(pid, { t: "full" });
        p = { id: pid, name, color };
        st.players.push(p);
        st.scores[pid] = st.scores[pid] || 0;
      } else { p.name = name; p.left = false; }
      hostBroadcast();
    } else if (msg.t === "throw") hostThrow(pid, msg);
  }
  function hostThrow(pid, msg) {
    const st = party.state;
    if (st.round !== msg.round || st.over || !partyPlayer(pid)) return;
    if (st.throws.filter((t) => t.pid === pid).length >= PARTY_BALLS) return;
    const tiles = (Array.isArray(msg.tiles) ? msg.tiles : []).slice(0, B.MAX_TILES)
      .filter((t) => t && G.space.has(String(t.word))).map((t) => ({ word: String(t.word), sign: t.sign < 0 ? -1 : 1 }));
    if (!tiles.length) return;
    st.throws.push({ pid, tiles });
    hostCheckOver();
    hostBroadcast();
  }
  function hostCheckOver() {
    const st = party.state;
    if (!st.round || st.over || !ends.party) return;
    const active = partyActive();
    if (!active.length || active.some((p) => st.throws.filter((t) => t.pid === p.id).length < PARTY_BALLS)) return;
    st.over = true;
    const { winner, pts } = partyStandings(st, ends.party);
    st.result = { winner, pts };
    if (winner) st.scores[winner] = (st.scores[winner] || 0) + pts;
  }
  function hostStartRound() {
    const st = party.state;
    st.round += 1;
    st.vectors = wordSet; // the host's current word set applies to the whole room
    st.seed = `party-${party.code}-${st.round}-${Math.random().toString(36).slice(2, 7)}`;
    st.throws = [];
    st.over = false;
    st.result = null;
    hostBroadcast();
  }
  function hostBroadcast() {
    if (!party) return;
    if (party.net) party.net.broadcast({ t: "state", state: party.state });
    applyPartyState(party.state);
  }

  // guest
  async function joinRoom(code, name) {
    store.set("name", name);
    party = { role: "guest", code: BocceNet.cleanCode(code), me: null, net: null, error: "", joining: true,
      state: { players: [], round: 0, seed: null, throws: [], scores: {}, over: false, result: null } };
    setHash("room=" + party.code);
    render();
    try {
      party.net = await BocceNet.join(party.code, {
        onOpen: (id) => { party.me = id; party.joining = false; party.net.send({ t: "hello", name }); render(); },
        onMessage: (msg) => {
          if (msg && msg.t === "state") applyPartyState(msg.state);
          else if (msg && msg.t === "full") { party.error = "That room is full (six players)."; render(); }
        },
        onClose: () => { if (party) { party.error = "The host closed the room."; party.closed = true; render(); } },
        onError: (m) => { if (party) { party.error = m; party.joining = false; render(); } },
      });
    } catch (e) { party.error = e.message; party.joining = false; render(); }
  }

  function partySendThrow(tiles) {
    const msg = { t: "throw", round: party.state.round, tiles };
    if (party.role === "host") hostThrow("host", msg);
    else party.net.send(msg);
  }

  /** Bring the court up to date with the room state: new round, new balls, round over. */
  function applyPartyState(st) {
    if (!party) return;
    party.state = st;
    if (busy) { party.queued = true; return; } // our own ball is rolling; catch up when it stops
    party.queued = false;
    const roomSet = st.vectors || "text";
    if (roomSet !== wordSet) {
      // The room plays with the host's word set: switch to it (loading it if needed), then catch up.
      if (!party.switching) {
        party.switching = true;
        loadWordSet(roomSet).then(() => { resetEnds(); party.switching = false; applyPartyState(party.state); showWordSet(); })
          .catch(() => { party.switching = false; party.error = "Couldn't load the room's word set."; render(); });
      }
      return;
    }
    let end = ends.party;
    if (st.round && (!end || end.round !== st.round || end.set !== roomSet)) {
      const d = dealFor(st.seed);
      end = ends.party = makeEnd("party", st.seed, d.start, d.target, d.hand, { sides: [], perSide: PARTY_BALLS, round: st.round });
      statusMsg = { text: "", warn: false };
    }
    let finished = false;
    if (end && end.round === st.round) {
      const count = {};
      let landed = 0;
      for (const t of st.throws) {
        const i = (count[t.pid] = (count[t.pid] || 0) + 1) - 1;
        const key = t.pid + ":" + i;
        if (end.balls.some((b) => b.key === key)) continue;
        const b = placeBall(end, (partyPlayer(t.pid) || { color: "red" }).color, t.tiles);
        b.key = key;
        b.pid = t.pid;
        landed++;
      }
      if (landed) sfx.roll(0, 0.15);
      finished = st.over && !end.done;
      end.done = !!st.over;
      if (finished) end.showMap = undefined;
      if (finished) setTimeout(st.result && st.result.winner === party.me ? sfx.win : sfx.lose, 500);
    }
    if (mode === "party") {
      render();
      if (finished) { const card = main().querySelector(".endcard"); if (card) card.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "nearest" }); }
    }
  }

  function leaveParty() {
    if (party && party.net) party.net.close();
    party = null;
    ends.party = null;
    partyJoinCode = "";
    setHash("party");
    render();
  }

  function renderPartyLobby() {
    const wrap = h("div", { class: "setup party-setup" }, h("h2", {}, "Play online with friends"));
    const nameInput = () => h("input", { id: "partyName", class: "text", "data-key": "party-name", autocomplete: "nickname", maxlength: 18,
      placeholder: "your name", value: store.get("name", "") });
    const nameVal = () => cleanName(($("#partyName") || {}).value);
    const players = () => h("ul", { class: "players" }, party.state.players.filter((p) => !p.left).map((p) =>
      h("li", {}, h("span", { class: `pip ${p.color} full`, "aria-hidden": "true" }), p.id === party.me ? `${p.name} (you)` : p.name,
        p.id === "host" ? h("small", {}, " host") : null)));

    if (!party) {
      wrap.append(
        h("p", {}, "Everyone plays the same court at the same time, each on their own phone or computer. Three balls each. You see each other's balls land; the words stay hidden until the round ends. The ball nearest the jack wins the round."),
        h("label", { for: "partyName" }, "Your name"), nameInput());
      if (partyJoinCode) {
        wrap.append(h("div", { class: "row" },
          h("button", { class: "primary", type: "button", "data-key": "party-join", onclick: () => joinRoom(partyJoinCode, nameVal()) }, `Join room ${partyJoinCode}`)),
          h("p", { class: "hint" }, "New to Word Bocce? ", h("button", { class: "linkish", type: "button", onclick: () => switchMode("tutorial") }, "Play the 30-second tutorial first"), ", then come back to this link."));
      } else {
        const codeIn = h("input", { id: "partyCode", class: "text code", "data-key": "party-code", autocomplete: "off", autocapitalize: "characters",
          maxlength: 8, placeholder: "ROOM CODE" });
        wrap.append(
          h("div", { class: "row" }, h("button", { class: "primary", type: "button", "data-key": "party-host", onclick: () => hostRoom(nameVal()) }, "Open a room")),
          h("p", { class: "or" }, "or join one:"),
          h("div", { class: "row" }, codeIn, h("button", { type: "button", "data-key": "party-join-code",
            onclick: () => { const c = BocceNet.cleanCode(codeIn.value); if (c) joinRoom(c, nameVal()); } }, "Join")));
      }
      wrap.append(h("p", { class: "hint" }, "Rooms connect browsers directly, introduced by the free PeerJS service. Nothing is stored, and the room ends when the host closes their tab."));
      return wrap;
    }

    if (party.error) {
      wrap.append(h("p", { class: "status warn" }, party.error),
        h("div", { class: "row" }, h("button", { type: "button", onclick: leaveParty }, "Back")));
      return wrap;
    }
    if (party.role === "host") {
      if (!party.code) { wrap.append(h("p", {}, "Opening a room…")); return wrap; }
      const link = partyLink(party.code);
      const linkIn = h("input", { class: "text", readonly: true, value: link, "aria-label": "Room link", onclick: (e) => e.target.select() });
      wrap.append(
        h("p", { class: "roomcode" }, "Room ", h("b", {}, party.code)),
        h("p", {}, "Send your friends this link. When everyone's in, start the round."),
        linkIn,
        h("div", { class: "row" },
          h("button", { type: "button", "data-key": "party-copy", onclick: async () => {
            try { await navigator.clipboard.writeText(link); party.copied = true; } catch (e) { linkIn.select(); }
            render();
          } }, party.copied ? "Link copied" : "Copy link"),
          navigator.share && h("button", { type: "button", onclick: () => navigator.share({ title: "Word Bocce", text: "Play Word Bocce with me:", url: link }).catch(() => {}) }, "Share…")),
        h("h3", {}, "Players"), players(),
        h("div", { class: "row" },
          h("button", { class: "primary", type: "button", "data-key": "party-start", onclick: hostStartRound },
            partyActive().length > 1 ? "Start the round" : "Start (just me for now)"),
          h("button", { type: "button", onclick: leaveParty }, "Close the room")));
    } else {
      wrap.append(party.joining ? h("p", {}, `Joining room ${party.code}…`)
        : h("p", {}, "You're in room ", h("b", {}, party.code), ". Waiting for the host to start the round."),
        h("h3", {}, "Players"), players(),
        h("div", { class: "row" }, h("button", { type: "button", onclick: leaveParty }, "Leave the room")));
    }
    return wrap;
  }

  function renderPartyResults(end) {
    const st = party.state;
    const { rows, winner, pts } = partyStandings(st, end);
    const winName = winner === party.me ? "You win" : `${partyName(winner)} wins`;
    const card = h("div", { class: "endcard", role: "region", "aria-label": "Round result" },
      h("h2", {}, rows.length ? `${winName} round ${st.round}` : "Round over"),
      rows.length > 1 && h("p", {}, `${pts} point${pts === 1 ? "" : "s"}: one for each ball closer than everyone else's best. `, tip("party", "room scoring")),
      rows.length > 1 && rows[0].best.rank === rows[1].best.rank && h("p", { class: "hint tie" },
        `Tied at ${ordinal(rows[0].best.rank)}, so the tie goes to the ball nearer in meaning: similarity ${fmt(rows[0].best.sim)} against ${fmt(rows[1].best.sim)}.`),
      h("ol", { class: "standings" }, rows.map((r) => h("li", {},
        h("span", { class: `pip ${(partyPlayer(r.pid) || { color: "red" }).color} full`, "aria-hidden": "true" }),
        h("b", {}, r.pid === party.me ? "You" : partyName(r.pid)), ` ${ordinal(r.best.rank)} · `,
        h("span", { class: "eqs" }, eqText(end.start, r.best.tiles)),
        h("small", {}, ` · ${st.scores[r.pid] || 0} pts`)))),
      end.par && h("p", { class: "par" }, `Par (the best throw these words allowed): ${eqText(end.start, end.par.tiles)} → ${ordinal(end.par.rank)} `, tip("par", "par")));
    const row = h("div", { class: "row" });
    if (party.role === "host") row.append(h("button", { type: "button", "data-key": "party-next", onclick: hostStartRound }, "Next round"));
    else row.append(h("span", { class: "hint" }, party.closed ? "The host has left." : "Waiting for the host to start the next round."));
    row.append(h("button", { class: "alt", type: "button", onclick: leaveParty }, party.role === "host" ? "Close the room" : "Leave"));
    card.append(row);
    return card;
  }

  window.addEventListener("beforeunload", (e) => {
    // Closing the host's tab ends the room for everyone: ask first.
    if (party && party.role === "host" && partyActive().length > 1) { e.preventDefault(); e.returnValue = ""; }
  });

  // ---------- puzzles ----------
  function renderPuzzleList() {
    const wrap = h("div", { class: "puzzles" });
    const done = G.puzzles.reduce((a, p) => a + store.get("puzzle:" + p.id, 0), 0);
    wrap.append(h("p", { class: "status" }, `${G.puzzles.length} hand-made courts. ${done} of ${G.puzzles.length * 3} stars earned.`));
    for (const diff of ["easy", "medium", "hard"]) {
      const list = G.puzzles.filter((p) => p.difficulty === diff);
      wrap.append(h("section", {}, h("h2", {}, diff[0].toUpperCase() + diff.slice(1)),
        h("div", { class: "grid" }, list.map((p) => {
          const st = store.get("puzzle:" + p.id, 0);
          return h("button", { class: "pz", type: "button", "data-key": "pz-" + p.id,
            onclick: () => { ends.puzzle = dealPuzzle(p); statusMsg = { text: "", warn: false }; render(); } },
            h("span", { class: "name" }, p.name),
            h("span", { class: "route" }, `${p.start_word} → ${p.target_word}`),
            h("span", { class: "stars", "aria-label": `${st} of 3 stars` }, h("b", {}, "●".repeat(st)), "○".repeat(3 - st)));
        }))));
    }
    return wrap;
  }

  // ---------- word sets: which map of meaning the court uses ----------
  function resetEnds() {
    for (const k of Object.keys(ends)) delete ends[k];
    vs = null;
  }
  /** Show the current word set in the header, footer and the words dialog. */
  function showWordSet(note) {
    const ws = SET();
    $("#wordsName").textContent = ws.name;
    $("#helpExample").textContent = `${ws.tut.start} + ${ws.tut.add} − ${ws.tut.sub} → ${ws.tut.target}`;
    $("#wordsCredit").textContent = ws.credit;
    if (G) $("#tagline").textContent = `played on a court of ${G.space.n.toLocaleString()} ${ws.short} words`;
    for (const el of document.querySelectorAll("[data-wordset]")) {
      const on = el.dataset.wordset === wordSet;
      el.classList.toggle("current", on);
      const btn = el.querySelector("button");
      btn.disabled = on;
      btn.textContent = on ? "In use" : `Use ${WORD_SETS[el.dataset.wordset].name.toLowerCase()} words`;
    }
    $("#wordsNote").textContent = note || "";
  }
  async function switchWordSet(key) {
    if (busy || key === wordSet) return;
    if (party && party.role === "guest") return showWordSet("In a room, the host chooses the words. Leave the room to switch.");
    if (party && party.state.round && !party.state.over) return showWordSet("Finish this round first; new words start with the next round.");
    showWordSet("Loading…");
    try { await loadWordSet(key); } catch (e) { return showWordSet("Couldn't load those words. Check your connection and try again."); }
    store.set("wordSet", key);
    resetEnds();
    showWordSet(party ? "Done. The room uses these words from the next round." : "Done. New courts use these words.");
    switchMode(mode);
  }

  // ---------- mode switching ----------
  function switchMode(m) {
    if (busy) return;
    mode = m;
    statusMsg = { text: "", warn: false };
    if (m === "daily" && !ends.daily) ends.daily = dealDaily();
    if (m === "practice" && !ends.practice) ends.practice = dealPractice();
    if (m === "tutorial") ends.tutorial = dealTutorial();
    setHash(m === "party" && (party && party.code || partyJoinCode) ? "room=" + (party && party.code || partyJoinCode) : m);
    render();
    maybeBot();
  }

  async function boot() {
    document.body.append(tipBox);
    for (const b of document.querySelectorAll(".modes button")) b.addEventListener("click", () => switchMode(b.dataset.mode));
    const soundBtn = $("#soundBtn");
    const showSound = () => {
      soundBtn.textContent = sfx.on ? "Sound on" : "Sound off";
      soundBtn.setAttribute("aria-pressed", String(sfx.on));
    };
    soundBtn.addEventListener("click", () => { sfx.set(!sfx.on); showSound(); sfx.tile(1); });
    showSound();
    const dlg = $("#help");
    $("#helpBtn").addEventListener("click", () => dlg.showModal());
    $("#helpClose").addEventListener("click", () => dlg.close());
    $("#helpTutorial").addEventListener("click", () => { dlg.close(); switchMode("tutorial"); });
    const wordsDlg = $("#words");
    const openWords = () => { showWordSet(); wordsDlg.showModal(); };
    $("#wordsBtn").addEventListener("click", openWords);
    $("#helpWords").addEventListener("click", () => { dlg.close(); openWords(); });
    $("#wordsClose").addEventListener("click", () => wordsDlg.close());
    $("#whyClose").addEventListener("click", () => $("#why").close());
    $("#interpClose").addEventListener("click", () => $("#interp").close());
    for (const el of document.querySelectorAll("[data-open-interp]")) {
      el.addEventListener("click", () => { for (const d of document.querySelectorAll("dialog[open]")) d.close(); $("#interp").showModal(); });
    }
    for (const el of document.querySelectorAll("[data-wordset] button")) {
      el.addEventListener("click", () => switchWordSet(el.closest("[data-wordset]").dataset.wordset));
    }
    const wanted = store.get("wordSet", "sense");
    try {
      await loadWordSet(WORD_SETS[wanted] ? wanted : "sense");
    } catch (e) {
      main().replaceChildren(h("div", { class: "loading" }, h("b", {}, "The court didn't load."),
        "The word vectors couldn't be fetched. If you opened index.html straight from disk, serve the folder instead (python -m http.server) and reload."));
      return;
    }
    showWordSet();
    flushSuggestions(); // deliver any suggestions left over from earlier visits
    const want = (location.hash || "").slice(1);
    // A room link (#room=CODE) goes straight to the join screen.
    const room = want.match(/^room=([A-Za-z0-9]+)/);
    if (room) {
      partyJoinCode = BocceNet.cleanCode(room[1]);
      return switchMode("party");
    }
    // First visit from a plain link: a guided game teaches faster than a page of rules.
    if (!want && !store.get("tutorialOffered", false)) {
      store.set("tutorialOffered", true);
      return switchMode("tutorial");
    }
    switchMode(["daily", "practice", "puzzles", "versus", "tutorial", "party"].includes(want) ? want : "daily");
  }
  boot();
})();

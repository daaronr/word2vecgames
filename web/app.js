/* Word Bocce — UI. Depends on engine.js (window.Bocce). */
(function () {
  "use strict";
  const B = window.Bocce;
  const DATA = (window.WORD_BOCCE_DATA || "data/");
  const SOLO_BALLS = 4;
  const VS_BALLS = 3;
  const VS_TARGET = 5;
  const LAUNCH = Date.UTC(2026, 8, 28); // Daily No. 1
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
    par: "Par is the best throw this hand allows. The game tries every combination of up to three tiles, each added or subtracted (378 throws in all), and keeps the one that gets the jack's rank lowest.",
    rings: "Each ring is a rank boundary. Inside the ‘top 10’ ring, the jack is among the 10 words closest to your ball; inside ‘top 100’, among the closest 100; and so on. Each ring inward is ten times harder to reach.",
    near: "The word closest in meaning to where your ball stopped (not counting the words you threw). It shows what your throw ‘means’.",
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
  let G = null;          // { space, pools, puzzles }
  let mode = "daily";
  const ends = {};       // per-mode current end
  let vs = null;         // versus match
  let busy = false;      // a ball is rolling / bot is thinking
  let statusMsg = { text: "", warn: false };

  function makeEnd(kind, seed, start, target, hand, extra) {
    const space = G.space;
    const throws = space.allThrows(start, target, hand);
    const end = {
      kind, seed, start, target, hand,
      startSim: space.sim(start, target),
      startRank: space.survey(space.row(start), target, [start], 1).rank,
      basis: B.courtBasis(space, start, target, seed),
      throws,
      par: throws[0] || null,
      balls: [],
      rack: [],
      done: false,
      ...extra,
    };
    return end;
  }

  function dealDaily() {
    const iso = todayISO();
    const seed = "daily-" + iso;
    const d = B.deal(G.space, G.pools, seed);
    const end = makeEnd("daily", seed, d.start, d.target, d.hand, { iso, no: dailyNo(iso), sides: ["red"], perSide: SOLO_BALLS });
    for (const tiles of store.get("daily:" + iso, [])) placeBall(end, "red", tiles);
    if (end.balls.length >= SOLO_BALLS) end.done = true;
    return end;
  }
  function dealPractice() {
    const seed = "practice-" + Date.now() + "-" + Math.random().toString(36).slice(2, 7);
    const d = B.deal(G.space, G.pools, seed);
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
    const d = B.deal(G.space, G.pools, seed);
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
  const bestOf = (end, side) => ballsOf(end, side).reduce((m, b) => (b && m && m.sim >= b.sim ? m : b), null);
  const bestBall = (end) => end.balls.reduce((m, b) => (m && m.sim >= b.sim ? m : b), null);

  /** Bocce order: the side farther from the jack throws next, while it has balls. */
  function nextSide(end) {
    if (end.sides.length === 1) return end.balls.length < end.perSide ? "red" : null;
    const left = (sd) => end.perSide - ballsOf(end, sd).length;
    if (!left("red") && !left("blue")) return null;
    if (!left("red")) return "blue";
    if (!left("blue")) return "red";
    if (!end.balls.length) return vs.first;
    const r = bestOf(end, "red"), b = bestOf(end, "blue");
    if (!r) return "red";
    if (!b) return "blue";
    return r.sim < b.sim ? "red" : "blue";
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
    svg.append(measure, ghosts);

    // start ball (ghost) and jack
    const st = toXY({ theta: 0 }, end.startRank);
    svg.append(s("circle", { cx: st.x, cy: st.y, r: 12, style: "fill:none;stroke:var(--ink);stroke-width:1.5;stroke-dasharray:3 3" }));
    svg.append(s("text", { class: "start-label", x: st.x + 18, y: st.y + 5 }, end.start));
    svg.append(balls);
    svg.append(s("ellipse", { cx: JX + 2, cy: JY + 7, rx: 8, ry: 3, style: "fill:#000;opacity:.18" }));
    svg.append(s("circle", { cx: JX, cy: JY, r: 8.5, style: "fill:var(--jack);stroke:rgba(0,0,0,.35);stroke-width:1" }));
    svg.append(s("circle", { cx: JX, cy: JY, r: 8.5, fill: "url(#shine)" }));
    svg.append(s("text", { class: "jack-label", x: JX, y: JY - 18, "text-anchor": "middle" }, end.target));
    svg.append(labels);

    courtEls = { svg, measure, ghosts, balls, labels, end };
    for (const b of end.balls) if (!b.pending) drawBall(b, 1);
    drawMeasure(end);
    if (end.done && end.par) drawGhost(end);
    return svg;
  }

  function ballNode(b) {
    const g = s("g", { class: "ball" });
    g.append(s("ellipse", { class: "shadow", cx: 2, cy: 10, rx: 12, ry: 4, style: "fill:#000;opacity:.22" }));
    g.append(s("circle", { r: 12.5, style: `fill:var(--${b.side});stroke:rgba(0,0,0,.35);stroke-width:1` }));
    g.append(s("circle", { r: 12.5, fill: "url(#shine)" }));
    g.append(s("text", { class: "ball-num", "text-anchor": "middle", y: 4 }, String(b.n)));
    return g;
  }
  function drawBall(b, t) {
    const { x, y } = toXY(b.place, b.rank);
    const g = ballNode(b);
    g.setAttribute("transform", `translate(${x.toFixed(1)} ${y.toFixed(1)})`);
    courtEls.balls.append(g);
    if (t === 1) drawLabel(b);
    return g;
  }
  function drawLabel(b) {
    const { x, y } = toXY(b.place, b.rank);
    const taken = [...courtEls.labels.querySelectorAll("text")].map((el) => ({
      x: +el.getAttribute("x"), y: +el.getAttribute("y") }));
    let ly = y + 27;
    if (taken.some((p) => Math.abs(p.x - x) < 60 && Math.abs(p.y - ly) < 13) || ly > H - 16) ly = y - 18;
    if (Math.abs(x - JX) < 50 && Math.abs(ly - (JY - 18)) < 16) ly = y + 27;
    courtEls.labels.append(s("text", { class: "ball-label", x: x.toFixed(1), y: ly.toFixed(1), "text-anchor": "middle" }, b.near[0]));
  }
  function drawMeasure(end) {
    const m = courtEls.measure;
    m.replaceChildren();
    const best = bestBall(end);
    if (!best) return;
    const { x, y } = toXY(best.place, best.rank);
    m.append(s("line", { class: "measure", x1: JX, y1: JY, x2: x, y2: y }));
    const mx = (JX + x) / 2, my = (JY + y) / 2;
    m.append(s("text", { class: "ring-label", x: mx + 6, y: my, style: "fill:var(--ink)" }, "similarity " + fmt(best.sim)));
  }
  function drawGhost(end) {
    const sc = G.space.score(end.start, end.target, end.par.tiles);
    const { x, y } = toXY(end.basis.place(sc.vec), sc.rank);
    courtEls.ghosts.replaceChildren(
      s("circle", { cx: x, cy: y, r: 12.5, style: "fill:none;stroke:var(--chalk);stroke-width:2;stroke-dasharray:4 3" }),
      s("text", { class: "ring-label", x: x + 16, y: y + 4, style: "fill:var(--ink)" }, "par"));
  }

  function animateBall(b) {
    return new Promise((resolve) => {
      const to = toXY(b.place, b.rank);
      const from = toXY({ theta: 0 }, courtEls.end.startRank);
      const SX = from.x, SY = from.y;
      const g = ballNode(b);
      courtEls.balls.append(g);
      const shadow = g.querySelector(".shadow");
      if (reduceMotion) {
        g.setAttribute("transform", `translate(${to.x} ${to.y})`);
        drawLabel(b);
        return resolve();
      }
      const dur = 950, t0 = performance.now();
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
          drawLabel(b);
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
    for (const b of document.querySelectorAll(".modes button")) b.setAttribute("aria-selected", String(b.dataset.mode === mode));
    const focusKey = document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.key : null;
    const m = main();
    if (mode === "puzzles" && !ends.puzzle) m.replaceChildren(renderPuzzleList());
    else if (mode === "versus" && !vs) m.replaceChildren(renderVersusSetup());
    else {
      const end = current();
      m.replaceChildren(h("div", { class: "table" },
        h("div", { class: "court-col" }, renderMatchup(end), renderScorebar(end), buildCourt(end),
          h("p", { class: "legend" }, "Dotted rings: the jack is in the ball's top 10, 100, 1,000 or 10,000 closest words. ", tip("rings", "the rings"))),
        renderBench(end)));
    }
    if (focusKey) { const el = m.querySelector(`[data-key="${CSS.escape(focusKey)}"]`); if (el) el.focus(); }
  }
  function current() { return ends[mode === "puzzles" ? "puzzle" : mode]; }

  function renderScorebar(end) {
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
        `Your ball starts as ${q(end.start)}. Right now ${q(end.target)} is only the ${ordinal(end.startRank)} closest word to it (rank #${end.startRank.toLocaleString()}) `,
        tip("rank", "rank"),
        `. Add and subtract word tiles to move the ball until ${q(end.target)} is the closest word of all: rank #1.`));
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

    if (!end.done) {
      // rack
      const rack = h("div", { class: "rack", "aria-live": "polite" }, h("span", { class: "base" }, end.start));
      if (!end.rack.length) rack.append(h("span", { class: "hint" }, "tap tiles below to add or subtract them"));
      for (const t of end.rack) {
        rack.append(h("button", { class: `chip ${t.sign > 0 ? "plus" : "minus"}`, type: "button", "data-key": "chip-" + t.word,
          title: "Tap to switch between adding and subtracting", disabled: busy || botTurn,
          onclick: () => { t.sign = -t.sign; render(); } }, `${signChar(t.sign)} ${t.word}`));
      }
      bench.append(rack);

      const throwBtn = h("button", { class: `throw ${side === "blue" ? "blue" : ""}`, type: "button", "data-key": "throw",
        disabled: busy || botTurn || !end.rack.length, onclick: () => playerThrow(end, side) }, "Throw");
      bench.append(h("div", { class: "actions" }, throwBtn,
        h("button", { class: "ghost", type: "button", "data-key": "clear", disabled: busy || !end.rack.length,
          onclick: () => { end.rack = []; setStatus(""); render(); } }, "Clear"),
        end.kind === "practice" && h("button", { class: "ghost", type: "button", "data-key": "redeal",
          disabled: busy, onclick: () => { ends.practice = dealPractice(); setStatus(""); render(); } }, "Deal a new court"),
        end.kind === "puzzle" && h("button", { class: "ghost", type: "button", "data-key": "hint",
          onclick: () => { end.showHint = true; render(); } }, "Hint"),
        end.kind === "puzzle" && h("button", { class: "ghost", type: "button", "data-key": "back",
          onclick: () => { ends.puzzle = null; render(); } }, "All puzzles")));
      bench.append(h("div", { class: `status ${statusMsg.warn ? "warn" : ""}`, role: "status" },
        end.kind === "puzzle" && end.showHint && !statusMsg.text ? "Hint: " + end.puzzle.hint : statusMsg.text));

      // hand
      const hand = h("div", { class: "hand", role: "group", "aria-label": "Your tiles" });
      const words = end.wildWord ? [...end.hand, end.wildWord] : end.hand;
      for (const w of words) {
        const inRack = end.rack.find((t) => t.word === w);
        const cls = inRack ? (inRack.sign > 0 ? "plus" : "minus") : "";
        hand.append(h("button", { class: `tile ${cls} ${w === end.wildWord ? "joker" : ""}`, type: "button", "data-key": "tile-" + w,
          "aria-pressed": inRack ? "true" : "false", disabled: busy || botTurn,
          "aria-label": inRack ? `${w}, ${inRack.sign > 0 ? "added" : "subtracted"}` : w,
          onclick: () => cycleTile(end, w) },
          w, h("span", { class: "sign", "aria-hidden": "true" }, inRack ? signChar(inRack.sign) : "")));
      }
      if (end.wild) {
        hand.append(h("button", { class: "tile joker", type: "button", "data-key": "wild", disabled: busy,
          onclick: () => { end.wildOpen = true; render(); setTimeout(() => { const i = $("#wildInput"); if (i) i.focus(); }); } },
          end.wildWord ? "change wild word" : "any word…", h("span", { class: "sign" }, "✱")));
      }
      bench.append(hand);
      bench.append(h("p", { class: "hand-help" }, "Tap: add. Tap again: subtract. Third tap: remove. Up to three tiles."));
      if (end.wild && end.wildOpen) bench.append(renderWildForm(end));
    }

    if (end.done) bench.append(end.kind === "versus" ? renderVersusEnd(end) : renderSoloEnd(end));
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
        h("span", {}, "Your throw → where it landed ", tip("near", "where it landed")),
        h("span", { class: "score" }, "Jack's rank ", tip("rank", "rank"))));
    }
    for (const b of [...end.balls].reverse()) {
      if (b.pending) continue;
      const t = B.tier(b.rank);
      ol.append(h("li", { class: `${b === best ? "best" : ""} tier-${t.key}` },
        h("span", { class: `ballmark ${b.side}`, "aria-hidden": "true" }, String(b.n)),
        h("span", { class: "eq" }, eqText(end.start, b.tiles), " ",
          h("span", { class: "near" }, "→ by ", h("b", {}, b.near[0]))),
        h("span", { class: "score" }, h("b", {}, `#${b.rank.toLocaleString()}`),
          h("small", { class: "tierlabel", "data-tip": TIPS.tiers, tabindex: "0" }, t.label),
          h("small", { class: "sim", "data-tip": TIPS.similarity, tabindex: "0" }, `similarity ${fmt(b.sim)}`))));
    }
    return ol;
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
    const beatPar = parScore && (best.sim > par.sim + 1e-6);
    const around = G.space.survey(G.space.row(end.target), end.target, [end.target], 8).near;
    const card = h("div", { class: "endcard", role: "region", "aria-label": "Round summary" },
      h("h2", {}, head),
      h("div", { class: "stars-row", "aria-label": `${stars} of 3 stars` },
        [0, 1, 2].map((i) => h("span", { class: i < stars ? "" : "off" }, "●"))),
      h("p", {}, best.rank === 1
        ? `Your best ball made ${q(end.target)} the closest word: rank #1. `
        : `With your best ball, ${q(end.target)} was the ${ordinal(best.rank)} closest word (rank #${best.rank.toLocaleString()}). `,
        `You started at #${end.startRank.toLocaleString()}. `, tip("rank", "rank")),
      parScore && h("p", {}, beatPar
        ? "Your wild word beat par, the best throw from the fixed tiles."
        : `You closed ${Math.round(pct * 100)}% of the gap between your start and par. `, tip("par", "par")),
      h("div", { class: "meter", "aria-hidden": "true" }, h("i", { style: `width:${Math.round(pct * 100)}%` })),
      h("div", { class: "meter-ends", "aria-hidden": "true" }, h("span", {}, "start"), h("span", {}, "par")),
      parScore && h("p", { class: "par" }, `Par: ${eqText(end.start, par.tiles)} → rank #${parScore.rank.toLocaleString()}`),
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
      end.balls.map((b) => dots[B.tier(b.rank).key]).join("") + `  best #${best.rank} · ${Math.round(pct * 100)}% to par`].join("\n");
    try {
      await navigator.clipboard.writeText(text);
      setStatus("Result copied. Paste it anywhere.");
    } catch (e) {
      setStatus("Copy isn't allowed here. Your result: " + text.replace(/\n/g, " · "));
    }
    render();
  }

  // ---------- actions ----------
  function setStatus(text, warn) { statusMsg = { text, warn: !!warn }; if (text) render(); }

  function cycleTile(end, w) {
    if (busy || end.done) return;
    const i = end.rack.findIndex((t) => t.word === w);
    if (i < 0) {
      if (end.rack.length >= B.MAX_TILES) {
        setStatus("Three tiles per throw. Tap a lit tile to change it.", true);
        const again = main().querySelector(`[data-key="${CSS.escape("tile-" + w)}"]`);
        if (again) again.classList.add("shake");
        return;
      }
      end.rack.push({ word: w, sign: 1 });
    } else if (end.rack[i].sign > 0) end.rack[i].sign = -1;
    else end.rack.splice(i, 1);
    statusMsg = { text: "", warn: false };
    render();
  }

  async function playerThrow(end, side) {
    if (busy || !end.rack.length || !side) return;
    await throwBall(end, side, end.rack.map((t) => ({ ...t })));
  }

  async function throwBall(end, side, tiles) {
    const key = tileKey(tiles);
    if (end.balls.some((b) => tileKey(b.tiles) === key)) {
      return setStatus("That exact throw is already on the court. Change a tile or a sign.", true);
    }
    busy = true;
    statusMsg = { text: "", warn: false };
    const ball = placeBall(end, side, tiles);
    ball.pending = true;
    end.rack = [];
    render();
    await animateBall(ball);
    ball.pending = false;
    busy = false;
    if (end.kind === "daily") store.set("daily:" + end.iso, end.balls.map((b) => b.tiles));
    const t = B.tier(ball.rank);
    statusMsg = { text: ball.rank === 1
      ? `Ball ${ball.n} landed right by ${q(end.target)}: it's the closest word to the ball. Bacio!`
      : `Ball ${ball.n} landed by ${q(ball.near[0])}. From there, ${q(end.target)} is the ${ordinal(ball.rank)} closest word (${t.label.toLowerCase()}). Rank #1 is the goal.`, warn: false };
    if (!nextSide(end)) finishEnd(end);
    render();
    maybeBot();
  }

  function finishEnd(end) {
    end.done = true;
    if (end.kind === "versus") {
      const r = bestOf(end, "red"), b = bestOf(end, "blue");
      const winner = r.sim >= b.sim ? "red" : "blue";
      const loserBest = winner === "red" ? b.sim : r.sim;
      const pts = ballsOf(end, winner).filter((x) => x.sim > loserBest).length;
      vs.score[winner] += pts;
      vs.first = winner;
      end.result = { winner, pts };
      if (vs.score[winner] >= VS_TARGET) vs.over = true;
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
        h("button", { type: "button", onclick: () => start("friend") }, "Two players, one device")));
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

  // ---------- mode switching ----------
  function switchMode(m) {
    if (busy) return;
    mode = m;
    statusMsg = { text: "", warn: false };
    if (m === "daily" && !ends.daily) ends.daily = dealDaily();
    if (m === "practice" && !ends.practice) ends.practice = dealPractice();
    try { history.replaceState(null, "", "#" + m); } catch (e) { /* sandboxed */ }
    render();
    maybeBot();
  }

  async function boot() {
    document.body.append(tipBox);
    for (const b of document.querySelectorAll(".modes button")) b.addEventListener("click", () => switchMode(b.dataset.mode));
    const dlg = $("#help");
    $("#helpBtn").addEventListener("click", () => dlg.showModal());
    $("#helpClose").addEventListener("click", () => dlg.close());
    try {
      G = await B.load(DATA, window.WORD_BOCCE_VECTORS);
    } catch (e) {
      main().replaceChildren(h("div", { class: "loading" }, h("b", {}, "The court didn't load."),
        "The word vectors couldn't be fetched. If you opened index.html straight from disk, serve the folder instead (python -m http.server) and reload."));
      return;
    }
    $("#tagline").textContent = `played on a court of ${G.space.n.toLocaleString()} words`;
    const want = (location.hash || "").slice(1);
    if (!store.get("seenHelp", false)) { store.set("seenHelp", true); try { dlg.showModal(); } catch (e) { /* no dialog */ } }
    switchMode(["daily", "practice", "puzzles", "versus"].includes(want) ? want : "daily");
  }
  boot();
})();

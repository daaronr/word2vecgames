/*
 * Seeded randomness and the Daily's item choice. Kept apart from the UI so Node tests can check
 * that a date always gives the same five mysteries (for a given content bundle).
 * Browser: window.WhatPick. Node: module.exports.
 */
(function (root) {
  "use strict";

  function hashSeed(str) {
    let x = 1779033703 ^ str.length;
    for (let i = 0; i < str.length; i++) {
      x = Math.imul(x ^ str.charCodeAt(i), 3432918353);
      x = (x << 13) | (x >>> 19);
    }
    return (x >>> 0) || 1;
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
    next.shuffle = (arr) => {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = next.int(i + 1);
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
      return arr;
    };
    return next;
  }

  const DAILY_EPOCH = Date.UTC(2026, 9, 1); // 1 October 2026 is No. 1
  // Every category but Double take (PG-13) can appear; each day shows five of them.
  const DAILY_CATS = ["web", "brand", "plate", "patent", "search", "zoom", "paper", "lyric", "latenight", "headline"];

  function dayNumber(dateStr) {
    const [y, m, d] = dateStr.split("-").map(Number);
    return Math.round((Date.UTC(y, m - 1, d) - DAILY_EPOCH) / 86400000) + 1;
  }

  function dailyCats(content) {
    return DAILY_CATS.filter((c) => content.some((it) => it.cat === c && it.rating !== "PG-13"));
  }
  // Day d's five categories: a fixed shuffle seeded by the day number.
  const catsOn = (cats, d) => rng("daily-cats-" + d).shuffle(cats.slice()).slice(0, Math.min(5, cats.length));

  /** Five items, one per category, the same for everyone on a date. Never PG-13; fun 3+ when possible. */
  function dailyItems(content, dateStr) {
    const n = dayNumber(dateStr);
    const cats = dailyCats(content);
    const today = catsOn(cats, n);
    // Each category's pool is shuffled once and walked one step per day the category appears, so
    // nothing repeats until the pool runs out.
    const shown = Object.fromEntries(today.map((c) => [c, 0]));
    for (let d = 1; d < n; d++) for (const c of catsOn(cats, d)) if (c in shown) shown[c]++;
    const picks = today.map((cat) => ({ cat, shown: shown[cat] }));
    const mod = (a, m) => ((a % m) + m) % m;
    picks.sort((a, b) => DAILY_CATS.indexOf(a.cat) - DAILY_CATS.indexOf(b.cat));
    const used = new Set();
    const clean = content.filter((it) => it.rating !== "PG-13");
    return picks.map(({ cat, shown }) => {
      let p = clean.filter((it) => it.cat === cat && (it.fun || 3) >= 3);
      if (!p.length) p = clean.filter((it) => it.cat === cat);
      const perm = rng("daily-" + cat).shuffle(p.slice().sort((a, b) => (a.id < b.id ? -1 : 1)));
      let k = mod(shown, perm.length);
      for (let t = 0; t < perm.length && used.has(perm[k].id); t++) k = (k + 1) % perm.length;
      used.add(perm[k].id);
      return perm[k];
    });
  }

  const api = { hashSeed, rng, dayNumber, dailyItems, DAILY_CATS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.WhatPick = api;
})(typeof window !== "undefined" ? window : globalThis);

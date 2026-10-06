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
  const DAILY_CATS = ["web", "brand", "plate", "patent", "search"];

  function dayNumber(dateStr) {
    const [y, m, d] = dateStr.split("-").map(Number);
    return Math.round((Date.UTC(y, m - 1, d) - DAILY_EPOCH) / 86400000) + 1;
  }

  /**
   * One item per category, the same for everyone on a date. Each category's pool is shuffled once
   * with a fixed seed and walked one step a day, so nothing repeats until the pool runs out.
   * Never PG-13, and only items rated at least 3 for fun when the pool allows.
   */
  function dailyItems(content, dateStr) {
    const n = dayNumber(dateStr);
    const used = new Set();
    return DAILY_CATS.map((cat) => {
      const clean = content.filter((it) => it.rating !== "PG-13");
      let p = clean.filter((it) => it.cat === cat && (it.fun || 3) >= 3);
      if (!p.length) p = clean.filter((it) => it.cat === cat);
      if (!p.length) p = clean;
      const perm = rng("daily-" + cat).shuffle(p.slice().sort((a, b) => (a.id < b.id ? -1 : 1)));
      let k = (((n - 1) % perm.length) + perm.length) % perm.length;
      for (let t = 0; t < perm.length && used.has(perm[k].id); t++) k = (k + 1) % perm.length;
      used.add(perm[k].id);
      return perm[k];
    });
  }

  const api = { hashSeed, rng, dayNumber, dailyItems, DAILY_CATS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.WhatPick = api;
})(typeof window !== "undefined" ? window : globalThis);

/* Shared pieces for every trial: content, storage, judges, and the UI parts they reuse.
   tools/build.mjs concatenates core.js, daily.js, party.js, byo.js, bluff.js and app.js into one
   script, so these top-level names are shared. CONFIG is written in by the build. */
"use strict";

const J = window.WhatJudge;
const VERSION = "trial-1";

// ---------- small helpers ----------
function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "value") el.value = v;
    else if (typeof v === "boolean") el[k] = v;
    else if (k === "style" && typeof v === "object") {
      for (const [sk, sv] of Object.entries(v)) {
        if (sk.startsWith("--")) el.style.setProperty(sk, String(sv));
        else el.style[sk] = sv;
      }
    }
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
const $ = (sel, root = document) => root.querySelector(sel);
function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }
function svgIcon(kind) {
  const paths = {
    lock: '<rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-4.3-4.3"/>',
  };
  const span = document.createElement("span");
  span.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[kind]}</svg>`;
  return span.firstChild;
}

const { rng } = window.WhatPick;
const RAND = rng(Date.now() ^ Math.floor(Math.random() * 1e9));

function load(key, fallback) {
  try {
    const v = localStorage.getItem("wii:" + key);
    return v == null ? fallback : JSON.parse(v);
  } catch (e) {
    return fallback;
  }
}
function save(key, value) {
  try { localStorage.setItem("wii:" + key, JSON.stringify(value)); } catch (e) { /* storage blocked */ }
}

let toastTimer = null;
function toast(msg) {
  document.querySelectorAll(".toast").forEach((t) => t.remove());
  const t = h("div", { class: "toast", role: "status" }, msg);
  document.body.append(t);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.remove(), 2600);
}
async function copyText(text, okMsg) {
  try {
    await navigator.clipboard.writeText(text);
    toast(okMsg || "Copied");
    return true;
  } catch (e) {
    modal((close) => [
      h("h3", {}, "Copy this"),
      h("p", { class: "muted small" }, "Your browser blocked the copy button. Select the text and copy it yourself."),
      h("textarea", { class: "input code", rows: 8, readonly: true, value: text }),
      h("div", { class: "row" }, h("button", { class: "btn", onclick: close }, "Done")),
    ]);
    return false;
  }
}
function modal(build) {
  const back = h("div", { class: "modal-back", role: "dialog", "aria-modal": "true" });
  const close = () => back.remove();
  const box = h("div", { class: "modal" }, build(close));
  back.append(box);
  back.addEventListener("click", (e) => { if (e.target === back) close(); });
  document.body.append(back);
  const first = box.querySelector("textarea, input, button");
  if (first) first.focus();
  return close;
}

// ---------- content ----------
const CATS = {
  web: { label: "Dot-com mysteries", short: "Dot-com", kind: "domain" },
  brand: { label: "What are they selling?", short: "Selling", kind: "brand" },
  plate: { label: "Vanity plates", short: "Plates", kind: "plate" },
  patent: { label: "Patent office", short: "Patents", kind: "patent" },
  search: { label: "Top result", short: "Top result", kind: "phrase" },
  double: { label: "Double take", short: "Double take", kind: "domain", pg13: true },
  zoom: { label: "Up close", short: "Up close", kind: "image" },
  paper: { label: "Paper titles", short: "Papers", kind: "paper" },
  lyric: { label: "Odd lyrics", short: "Lyrics", kind: "lyric" },
  latenight: { label: "What's the deal with…?", short: "The deal", kind: "news" },
  headline: { label: "Headline, what?", short: "Headlines", kind: "headline" },
};
const CAT_ORDER = ["web", "brand", "plate", "patent", "search", "zoom", "paper", "lyric", "latenight", "headline", "double"];

const CONTENT = (() => {
  const all = JSON.parse(document.getElementById("wii-content").textContent);
  for (const it of all) {
    it.kind = it.kind || (CATS[it.cat] && CATS[it.cat].kind) || "phrase";
    it.key = (it.key || []).map((k) => String(k).toLowerCase());
  }
  return all;
})();
const SETTINGS = Object.assign({ pg13: false, aiSource: "auto", apiKey: "", model: "claude-opus-5-5", players: [], pass: "" }, load("settings", {}));
function saveSettings() { save("settings", SETTINGS); }

function allowed(item) { return SETTINGS.pg13 || item.rating !== "PG-13"; }
function pool(cat, opts = {}) {
  return CONTENT.filter((it) => it.cat === cat && (opts.anyRating || allowed(it)));
}
function catsAvailable() {
  return CAT_ORDER.filter((c) => pool(c).length > 0);
}
function kindLabel(item) {
  const label = J.KIND_LABEL[item.kind] || "";
  return item.song ? `${label} ("${item.song}")` : label;
}
function looksLikeDomain(text) {
  return /^(https?:\/\/)?([a-z0-9-]+\.)+[a-z]{2,}(\/\S*)?$/i.test(String(text).trim());
}

// ---------- storage of ratings and suggestions ----------
const FORM_NAMES = { item: "wii-rating", trial: "wii-trial", suggestion: "wii-suggestion", judge: "wii-judge" };
const DB_COLLECTIONS = { item: "ratings", trial: "trials", suggestion: "suggestions", judge: "judgeComparisons" };
const RT = { sample: null, db: null, user: null }; // claude.ai runtime capabilities, when present

const Store = {
  local() { return load("outbox", []); },
  /** Save a record. With a key, a later send replaces the earlier one (same rid). */
  async send(type, data, key) {
    const rid = key || "r" + Date.now().toString(36) + RAND.int(1e9).toString(36);
    const rec = Object.assign({ type, rid, at: new Date().toISOString(), env: CONFIG.env, version: VERSION }, data);
    const box = Store.local().filter((r) => r.rid !== rid);
    box.push(rec);
    save("outbox", box.slice(-500));
    if (RT.db) {
      try {
        await RT.db.collection(DB_COLLECTIONS[type]).doc(rid).set(rec);
        return "shared";
      } catch (e) { /* fall through to the next option */ }
    }
    if (CONFIG.env === "netlify" && location.protocol.startsWith("http")) {
      try {
        const body = new URLSearchParams({
          "form-name": FORM_NAMES[type],
          kind: type,
          rid,
          id: String(rec.id || rec.trial || rec.prompt || ""),
          stars: String(rec.stars || rec.fun || ""),
          text: String(rec.note || rec.change || rec.truth || "").slice(0, 2000),
          data: JSON.stringify(rec).slice(0, 8000),
        });
        const res = await fetch("/", { method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body: body.toString() });
        if (res.ok) return "sent";
      } catch (e) { /* offline, or forms are not switched on */ }
    }
    return "local";
  },
};
function savedMsg(where) {
  return where === "shared" ? "Saved for the team" : where === "sent" ? "Sent. Thank you." : "Saved on this device";
}

// ---------- robot judge ----------
let spacePromise = null;
let SPACE = null;
function loadRobot() {
  if (!spacePromise) {
    spacePromise = J.loadSpace(CONFIG.vectorUrls)
      .then((s) => { SPACE = s; return s; })
      .catch((e) => { console.warn("Robot judge unavailable:", e); spacePromise = null; return null; });
  }
  return spacePromise;
}
/** The free judge: the item's answer key, then word-vector closeness. */
function robotScore(guess, item) {
  return J.scoreItem(SPACE, guess, item);
}

// ---------- AI judge ----------
const MODELS = [
  { id: "claude-opus-5-5", label: "Claude Opus 5.5 (about 1¢ a round)" },
  { id: "claude-sonnet-5-5", label: "Claude Sonnet 5.5 (about ½¢ a round)" },
  { id: "claude-haiku-4-5", label: "Claude Haiku 4.5 (about 0.15¢ a round)" },
];
// The site's own AI judge (netlify/functions/judge.mjs): free for players, paid by the site.
// null = not tried yet, true = working, false = unavailable (not deployed, or today's cap is used up).
let SITE_OK = null;
let SITE_NOTE = "";
async function siteJudge(body) {
  if (!CONFIG.judgeApi || SITE_OK === false) throw new Error("The site's AI judge isn't available here.");
  let res;
  try {
    const headers = { "Content-Type": "application/json" };
    if (SETTINGS.pass) headers["X-Judge-Pass"] = SETTINGS.pass; // tester pass, from a ?pass= link
    res = await fetch(CONFIG.judgeApi, { method: "POST", headers, body: JSON.stringify(body) });
  } catch (e) {
    throw new Error("Couldn't reach the AI judge.");
  }
  const out = await res.json().catch(() => ({}));
  if (!res.ok) {
    // 503: switched off, out of credit or over the site's cap. Stop asking for this visit.
    if ([404, 405, 501, 503].includes(res.status)) {
      SITE_OK = false;
      SITE_NOTE = out.error || "";
    }
    if ((res.status === 429 && out.reason === "visitor") || (res.status === 403 && out.reason === "pass")) {
      SITE_OK = false;
      SITE_NOTE = out.error || "";
    }
    throw new Error(out.error || "The AI judge couldn't answer (" + res.status + ").");
  }
  SITE_OK = true;
  return out;
}
function siteJudgeUsable() { return !!CONFIG.judgeApi && SITE_OK !== false; }

function aiSources() {
  const out = [];
  if (siteJudgeUsable()) out.push("site");
  if (RT.sample) out.push("claude");
  if (CONFIG.env !== "artifact" && SETTINGS.apiKey) out.push("key");
  out.push("paste");
  return out;
}
function aiSource() {
  const avail = aiSources();
  return avail.includes(SETTINGS.aiSource) ? SETTINGS.aiSource : avail[0];
}
function aiSourceLabel(src) {
  return { site: "the game's AI judge", claude: "Claude, on your own plan", key: "your API key", paste: "a chatbot you paste into" }[src];
}

let sdkPromise = null;
function loadSdk() {
  if (!sdkPromise) {
    const el = document.getElementById("anthropic-sdk-src");
    if (!el) return Promise.reject(new Error("The API-key judge is not part of this build."));
    const url = URL.createObjectURL(new Blob([el.textContent], { type: "text/javascript" }));
    sdkPromise = import(url);
  }
  return sdkPromise;
}

/** round: {prompt, kindLabel, ask, truth, more, results}; guesses: [{name, text}] */
async function aiJudge(round, guesses) {
  const prompt = J.judgePrompt(round, guesses);
  const src = aiSource();
  if (src === "site") {
    const body = round.id
      ? { id: round.id, guesses }
      : { custom: { prompt: round.prompt, truth: round.truth, kind: /web address/.test(round.kindLabel || "") ? "domain" : "phrase", results: round.results }, guesses };
    const out = await siteJudge(body);
    return { scores: out.scores, funniest: out.funniest, comment: out.comment, source: "site" };
  }
  if (src === "claude") {
    try {
      const out = await RT.sample.json(prompt, { modelTier: "quick" });
      return Object.assign(J.parseJudgeReply(out, guesses.length), { source: "claude" });
    } catch (e) {
      if (e && (e.code === "not_granted" || e.code === "sampling_disabled")) RT.sample = null;
      throw new Error(sampleErrorText(e));
    }
  }
  if (src === "key") {
    const mod = await loadSdk();
    const client = new mod.Anthropic({ apiKey: SETTINGS.apiKey, dangerouslyAllowBrowser: true });
    const model = SETTINGS.model || "claude-opus-5-5";
    const params = {
      model,
      max_tokens: 4000,
      messages: [{ role: "user", content: prompt }],
      output_config: { format: mod.jsonSchemaOutputFormat(J.JUDGE_SCHEMA) },
    };
    if (model !== "claude-haiku-4-5") {
      params.output_config.effort = "low";
      params.betas = ["server-side-fallback-2026-07-01"];
      params.fallbacks = "default";
    }
    let msg;
    try {
      msg = await client.beta.messages.parse(params);
    } catch (e) {
      throw new Error(apiErrorText(e));
    }
    if (msg.stop_reason === "refusal") throw new Error("The model declined to judge this round.");
    const text = (msg.content || []).filter((b) => b.type === "text").map((b) => b.text).join("");
    return Object.assign(J.parseJudgeReply(msg.parsed_output || text, guesses.length), { source: "key", model });
  }
  return pasteJudge(prompt, guesses.length);
}
function sampleErrorText(e) {
  const code = e && e.code;
  if (code === "not_granted") return "Claude wasn't allowed to judge on this page. You can still paste the prompt into a chatbot.";
  if (code === "rate_limited") return "Claude is busy or your usage limit was reached. Try again in a bit.";
  if (code === "invalid_json") return "Claude's verdict didn't come back in the right shape. Try again.";
  return "Claude couldn't judge this one" + (e && e.message ? ": " + e.message : ".");
}
function apiErrorText(e) {
  const status = e && e.status;
  if (status === 401) return "That API key was rejected. Check it in Settings.";
  if (status === 429) return "Rate limited by the API. Wait a moment and try again.";
  if (status === 400) return "The API rejected the request: " + (e.message || "bad request");
  return "The API call failed" + (e && e.message ? ": " + e.message : ".");
}
function pasteJudge(prompt, n) {
  return new Promise((resolve, reject) => {
    let done = false;
    const close = modal((closeFn) => {
      const reply = h("textarea", { class: "input", id: "paste-reply", rows: 6, placeholder: "Paste the chatbot's whole reply here" });
      const err = h("p", { class: "small", style: { color: "var(--bad)" } });
      return [
        h("h3", {}, "Ask your own AI to judge"),
        h("ol", { class: "small", style: { margin: 0, paddingLeft: "1.2em" } },
          h("li", {}, "Copy the prompt below."),
          h("li", {}, "Paste it into ChatGPT, Claude, Gemini or any chatbot you use."),
          h("li", {}, "Paste the reply back here.")),
        h("div", { class: "code" }, prompt),
        h("div", { class: "row" }, h("button", { class: "btn small", onclick: () => copyText(prompt, "Prompt copied") }, "Copy prompt")),
        h("div", { class: "field" }, h("label", { for: "paste-reply" }, "The reply"), reply),
        err,
        h("div", { class: "row" },
          h("button", { class: "btn primary", onclick: () => {
            try {
              const out = J.parseJudgeReply(reply.value, n);
              done = true;
              closeFn();
              resolve(Object.assign(out, { source: "paste" }));
            } catch (e) {
              err.textContent = "Couldn't read a verdict in that reply. Paste all of it, including the part in curly brackets.";
            }
          } }, "Use this verdict"),
          h("button", { class: "btn ghost", onclick: () => { closeFn(); if (!done) reject(new Error("cancelled")); } }, "Cancel")),
      ];
    });
    void close;
  });
}

// ---------- UI parts ----------
function catLine(item, extra) {
  return h("div", { class: "catline" },
    h("span", { class: "eyebrow" }, (CATS[item.cat] || { label: "Your mystery" }).label),
    item.rating === "PG-13" ? h("span", { class: "chip" }, "PG-13") : null,
    extra || null);
}

/** Magnifier view of an image item: [size %, position x %, position y %] for a zoom level. */
function lensView(item, level) {
  const zooms = item.zooms || [10, 5, 2.5, 1];
  const z = zooms[Math.min(level || 0, zooms.length - 1)];
  const [fx, fy] = item.focus || [0.5, 0.5];
  const pos = (f) => {
    if (z <= 1) return 50;
    const left = Math.max(0, Math.min(1 - 1 / z, f - 0.5 / z));
    return (100 * left * z) / (z - 1);
  };
  return { size: z * 100 + "%", x: pos(fx) + "%", y: pos(fy) + "%", z };
}
/** Point every magnifier inside `root` at a new zoom level (after a clue). */
function zoomTo(root, item, level) {
  root.querySelectorAll(".lens").forEach((el) => {
    const v = lensView(item, level);
    el.style.backgroundSize = v.size;
    el.style.backgroundPosition = `${v.x} ${v.y}`;
    el.dataset.z = v.z;
  });
}

function habitat(item, level = 0) {
  const p = item.prompt;
  if (item.kind === "domain") {
    const m = String(p).match(/^(https?:\/\/)?(.*)$/i);
    return h("div", { class: "bar", "aria-label": "Web address: " + p },
      h("div", { class: "dots" }, h("i"), h("i"), h("i")),
      h("div", { class: "field" }, svgIcon("lock"), h("span", { class: "url" }, h("span", { class: "proto" }, "https://"), m[2])));
  }
  if (item.kind === "plate") {
    return h("div", { class: "plate-wrap" },
      h("div", { class: "plate", role: "img", "aria-label": "California licence plate reading " + p },
        h("span", { class: "bolt l" }), h("span", { class: "bolt r" }),
        h("div", { class: "state" }, "California"),
        h("div", { class: "chars" }, p),
        h("div", { class: "foot" }, "Application, 2015–16")));
  }
  if (item.kind === "patent") {
    return h("div", { class: "patent" },
      h("div", { class: "hd" },
        h("span", {}, h("b", {}, "United States Patent"), " [19]"),
        h("span", { class: "tabular" }, "[11] ", item.number || "")),
      h("div", { class: "inid" }, "[45] Date of patent: ", item.year || ""),
      h("div", { class: "inid" }, "[54]"),
      h("div", { class: "title" }, p));
  }
  if (item.kind === "brand") {
    return h("div", { class: "sign" },
      h("div", { class: "awning" }),
      h("div", { class: "board" }, h("div", { class: "name" }, p), h("div", { class: "est" }, "Open for business")));
  }
  if (item.kind === "image") {
    const v = lensView(item, level);
    return h("div", { class: "lens-wrap" },
      h("div", { class: "lens", role: "img", "aria-label": "A magnified detail of a picture", "data-z": v.z,
        style: { backgroundImage: `url("${item.src || ""}")`, backgroundSize: v.size, backgroundPosition: `${v.x} ${v.y}` } }),
      h("span", { class: "handle", "aria-hidden": "true" }));
  }
  if (item.kind === "paper") {
    return h("div", { class: "paper" },
      h("div", { class: "jhd" }, h("span", {}, "Journal of ", h("span", { class: "redact" }, "Something")), h("span", { class: "tabular" }, "Vol. ", item.year ? String(item.year).slice(0, 2) + "··" : "··")),
      h("div", { class: "ptitle" }, p, h("span", { class: "rest", "aria-label": "rest of the title hidden" }, ": …")),
      h("div", { class: "authors" }, h("span", { class: "redact" }, "A. Author"), ", ", h("span", { class: "redact" }, "B. Author")),
      h("div", { class: "lines", "aria-hidden": "true" }, h("i"), h("i"), h("i"), h("i")));
  }
  if (item.kind === "lyric") {
    return h("div", { class: "lyric" }, h("div", { class: "staff", "aria-hidden": "true" }), h("p", { class: "line" }, "“", p, "”"));
  }
  if (item.kind === "news") {
    return h("div", { class: "tv" },
      h("div", { class: "screen" },
        h("div", { class: "desk", "aria-hidden": "true" }),
        h("div", { class: "chyron" }, h("span", { class: "tag" }, "Tonight"), h("span", { class: "story" }, p))));
  }
  if (item.kind === "headline") {
    return h("div", { class: "news" },
      h("div", { class: "mast" }, "The Daily Ledger"),
      h("div", { class: "dateline" }, h("span", {}, "Late edition"), h("span", {}, "Price 25¢")),
      h("div", { class: "head" }, p),
      h("div", { class: "cols", "aria-hidden": "true" }, h("i"), h("i"), h("i"), h("i"), h("i"), h("i")));
  }
  return h("div", { class: "searchbox" }, svgIcon("search"), h("span", { class: "q" }, p, h("span", { class: "caret" })));
}

/** The mystery as players see it. opts.level: clues shown so far (zooms an image out). */
function mysteryCard(item, extra, opts = {}) {
  return h("div", { class: "mystery" }, catLine(item, extra), habitat(item, opts.level || 0), h("p", { class: "ask" }, item.ask || "What is it?"));
}

const REVEAL_HEAD = { phrase: "The top result", paper: "The paper", lyric: "What it means", news: "The joke", headline: "What it meant", image: "What it is" };
const LINK_TEXT = { plate: "The DMV dataset", patent: "Read the patent", phrase: "Search it yourself", paper: "Read the paper", lyric: "About the song", news: "See a host's take", headline: "Where it's documented", image: "The picture's source" };

function revealCard(item, opts = {}) {
  const parts = [h("span", { class: "eyebrow" }, REVEAL_HEAD[item.kind] || "What it is")];
  if (item.kind === "image" && item.src) parts.push(h("img", { class: "reveal-img", src: item.src, alt: item.truth }));
  parts.push(h("p", { class: "truth" }, item.truth));
  if (item.kind === "plate" && item.dmv) {
    parts.push(h("span", { class: "stamp " + (item.dmv === "approved" ? "ok" : "no") }, item.dmv === "approved" ? "Approved" : "Denied"));
  }
  if (item.kind === "paper" && item.full) {
    parts.push(h("p", { class: "cite" }, h("i", {}, item.full), ". ", [item.authors, item.journal, item.year].filter(Boolean).join(", "), "."));
  }
  if (item.kind === "lyric" && item.song) parts.push(h("p", { class: "cite" }, "From ", h("i", {}, item.song), item.year ? ` (${item.year})` : "", "."));
  if (item.kind === "news" && item.hosts && item.hosts.length) parts.push(h("p", { class: "cite" }, (item.when ? item.when + ". " : "") + "Bits from " + item.hosts.join(", ") + "."));
  if (item.kind === "headline" && item.where) parts.push(h("p", { class: "cite" }, item.where));
  if (item.more) parts.push(h("p", { class: "more" }, item.more));
  if (item.top3 && item.top3.length) {
    parts.push(h("div", { class: "results" }, item.top3.map((r, i) =>
      h("div", { class: "r" }, h("div", { class: "small muted" }, "#" + (i + 1)), h("div", {}, r.title), h("div", { class: "d" }, r.domain)))));
  }
  const links = [];
  if (item.url) links.push(h("a", { href: item.url, target: "_blank", rel: "noopener noreferrer" }, LINK_TEXT[item.kind] || "See it for yourself"));
  if (item.credit) links.push(h("span", { class: "small muted" }, item.credit));
  if (item.status === "unverified") links.push(h("span", { class: "chip" }, "Draft: not yet re-checked online"));
  else if (item.checked) links.push(h("span", { class: "small muted" }, "Checked " + item.checked + (item.number ? " \u00b7 " + item.number : "")));
  if (links.length) parts.push(h("div", { class: "row small" }, links));
  if (opts.extra) parts.push(opts.extra);
  return h("div", { class: "reveal" }, parts);
}

function warmthMeter(score) {
  const w = J.warmth(score);
  return h("div", { class: "warmth", "aria-label": `${w.label}, ${score} out of 100` },
    h("div", { class: "lab" }, h("span", { class: "word lv" + w.level }, w.label), h("span", { class: "num" }, score, "/100")),
    h("div", { class: "track" }, h("span", { class: "pin", style: { left: Math.max(2, Math.min(98, score)) + "%" } })));
}

function matchChips(res) {
  const chips = res.matches.filter((m) => m.word && m.sim >= 0.35).map((m) =>
    h("span", { class: "chip" }, h("b", {}, m.word), " → ", m.key, " ", h("span", { class: "muted" }, m.sim >= 0.999 ? "same word" : m.sim.toFixed(2))));
  if (!chips.length) chips.push(h("span", { class: "chip" }, "No word landed near the answer"));
  if (res.unknown && res.unknown.length) chips.push(h("span", { class: "chip muted" }, "Didn't know: " + res.unknown.join(", ")));
  return h("div", { class: "chips" }, chips);
}

function stars(onPick, current = 0, label = "Rating") {
  const wrap = h("div", { class: "stars", role: "group", "aria-label": label });
  const paint = (n) => wrap.querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-pressed", String(i < n)));
  for (let i = 1; i <= 5; i++) {
    wrap.append(h("button", { type: "button", "aria-label": i + " of 5", "aria-pressed": String(i <= current), onclick: () => { paint(i); onPick(i); } }, "★"));
  }
  return wrap;
}

function seg(options, current, onPick) {
  const wrap = h("div", { class: "seg", role: "group" });
  for (const [value, label] of options) {
    wrap.append(h("button", { type: "button", "aria-pressed": String(value === current), onclick: (e) => {
      wrap.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false"));
      e.currentTarget.setAttribute("aria-pressed", "true");
      onPick(value);
    } }, label));
  }
  return wrap;
}

/** Toggle buttons; any number can be on. onChange gets the list of values that are on. */
function multiSeg(options, onChange) {
  const on = new Set();
  const wrap = h("div", { class: "seg", role: "group" });
  for (const [value, label] of options) {
    wrap.append(h("button", { type: "button", "aria-pressed": "false", onclick: (e) => {
      on.has(value) ? on.delete(value) : on.add(value);
      e.currentTarget.setAttribute("aria-pressed", String(on.has(value)));
      onChange([...on]);
    } }, label));
  }
  return wrap;
}

// Why a mystery was (or wasn't) fun. Difficulty is asked separately: something can be just right
// and still dull, or hard and still a delight.
const FUN_TAGS = [
  ["surprise", "Surprising reveal"], ["laugh", "Made me laugh"], ["learned", "Learned something"],
  ["name", "Clever name"], ["clues", "Good clues"], ["argue", "Fun to argue about"],
  ["flat", "Answer fell flat"], ["obscure", "Too obscure"], ["unfair", "Felt unfair"], ["confusing", "Confusing question"],
];

/** "How fun was that one?" for an item. The element has .flush() to save any unsaved rating. */
function funRating(item, mode, extra = {}) {
  const key = "r" + Date.now().toString(36) + RAND.int(1e9).toString(36);
  const rec = Object.assign({ id: item.id || null, prompt: item.prompt, cat: item.cat || "byo", mode, fun: 0, tags: [], feel: "", note: "" }, extra);
  let lastSent = "";
  let timer = null;
  const status = h("span", { class: "small muted" });
  const flush = async () => {
    clearTimeout(timer);
    if (!rec.fun && !rec.tags.length && !rec.feel && !rec.note) return;
    const sig = JSON.stringify(rec);
    if (sig === lastSent) return;
    lastSent = sig;
    status.textContent = savedMsg(await Store.send("item", rec, key));
  };
  const later = () => { clearTimeout(timer); timer = setTimeout(flush, 1500); };
  const note = h("input", { class: "input", id: "fun-note-" + key, placeholder: "Anything to add? (optional)", maxlength: 500, oninput: (e) => { rec.note = e.target.value; later(); } });
  const el = h("div", { class: "rate" },
    h("div", { class: "spread" }, h("span", { class: "label" }, "How fun was that one?"), status),
    stars((n) => { rec.fun = n; later(); }, 0, "How fun"),
    h("span", { class: "small muted" }, "What made it that way? Pick any."),
    multiSeg(FUN_TAGS, (v) => { rec.tags = v; later(); }),
    h("span", { class: "small muted" }, "Difficulty"),
    seg([["easy", "Too easy"], ["right", "About right"], ["hard", "Too hard"]], "", (v) => { rec.feel = v; later(); }),
    note,
    item.id ? h("button", { class: "btn ghost small", style: { justifySelf: "start" }, onclick: () => { flush(); suggestScreen({ clueFor: item }); } }, "Suggest a better clue") : null);
  el.flush = flush;
  return el;
}

/** Rate a whole trial (A-D). */
function trialRating(trial, title) {
  const rec = { trial, stars: 0, liked: "", change: "" };
  const status = h("span", { class: "small muted" });
  return h("div", { class: "panel stack" },
    h("div", { class: "spread" }, h("h3", {}, title || "Rate this version"), status),
    stars((n) => { rec.stars = n; }, 0, "Rate this version"),
    h("div", { class: "field" }, h("label", { for: "liked-" + trial }, "What worked?"),
      h("textarea", { class: "input", id: "liked-" + trial, rows: 2, oninput: (e) => { rec.liked = e.target.value.slice(0, 1500); } })),
    h("div", { class: "field" }, h("label", { for: "change-" + trial }, "What would make it more fun?"),
      h("textarea", { class: "input", id: "change-" + trial, rows: 2, oninput: (e) => { rec.change = e.target.value.slice(0, 1500); } })),
    h("div", { class: "row" }, h("button", { class: "btn primary", onclick: async () => {
      if (!rec.stars && !rec.liked && !rec.change) { toast("Pick some stars or write a line first"); return; }
      status.textContent = savedMsg(await Store.send("trial", rec));
    } }, "Send feedback")));
}

/** Players or teams editor. Returns {el, get()} */
function playersEditor(initial, { min = 2, max = 8, noun = "Player" } = {}) {
  const names = initial.slice(0, max);
  while (names.length < min) names.push(noun + " " + (names.length + 1));
  const list = h("div", { class: "stack-sm" });
  const paint = () => {
    clear(list);
    names.forEach((n, i) => {
      list.append(h("div", { class: "guessrow" },
        h("input", { class: "input", id: `pname-${noun}-${i}`, "aria-label": noun + " " + (i + 1), value: n, maxlength: 24, oninput: (e) => { names[i] = e.target.value; } }),
        names.length > min ? h("button", { class: "btn ghost small", "aria-label": "Remove", onclick: () => { names.splice(i, 1); paint(); } }, "Remove") : null));
    });
    if (names.length < max) list.append(h("button", { class: "btn small", style: { justifySelf: "start" }, onclick: () => { names.push(noun + " " + (names.length + 1)); paint(); } }, "Add " + noun.toLowerCase()));
  };
  paint();
  return { el: list, get: () => names.map((n, i) => (n || "").trim() || noun + " " + (i + 1)) };
}

/**
 * Pass-and-play: each player in turn types privately. Renders into `host`; resolves with texts.
 * opts: {item, prompt (label), placeholder, check(text, i) -> error string or ""}
 */
function collectPrivately(host, players, opts) {
  return new Promise((resolve) => {
    const out = new Array(players.length).fill("");
    let i = 0;
    const step = (phase) => {
      clear(host);
      if (i >= players.length) { resolve(out); return; }
      const who = players[i];
      if (phase === "pass") {
        host.append(h("div", { class: "panel passcard" },
          h("span", { class: "eyebrow" }, `${i + 1} of ${players.length}`),
          h("div", {}, "Pass the device to"),
          h("div", { class: "who" }, who),
          h("button", { class: "btn primary", onclick: () => step("type") }, "I'm " + who)));
        return;
      }
      const input = h("input", { class: "input", id: "private-" + i, autocomplete: "off", placeholder: opts.placeholder || "Your guess", maxlength: 160 });
      const err = h("p", { class: "small", style: { color: "var(--bad)" } });
      const submit = () => {
        const text = input.value.trim();
        if (!text) { err.textContent = "Write something first."; return; }
        const problem = opts.check ? opts.check(text, i) : "";
        if (problem) { err.textContent = problem; return; }
        out[i] = text;
        i++;
        step("pass");
      };
      input.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });
      host.append(h("div", { class: "stack" },
        opts.item ? mysteryCard(opts.item, null, { level: opts.level }) : null,
        h("div", { class: "field" }, h("label", { for: "private-" + i }, `${who}: ${opts.prompt || "your guess"}`), input),
        err,
        h("div", { class: "row" },
          h("button", { class: "btn primary", onclick: submit }, "Done, hide it"),
          opts.skippable ? h("button", { class: "btn ghost", onclick: () => { out[i] = ""; i++; step("pass"); } }, "Skip") : null)));
      input.focus();
    };
    step("pass");
  });
}

/**
 * Judge a set of guesses. Shows controls inside `host`; calls onAward(winners, funniest, details).
 * opts.threshold: everyone scoring at least this is marked right (Daily Double, Final); otherwise the
 * top score wins. opts.penalty: also ask who was way off (they lose points; details.penalized).
 */
function judgePanel(host, { round, item, players, guesses, mode, defaultJudge, onAward, threshold, penalty, pickLabel }) {
  const typed = guesses && guesses.some((g) => g);
  const picked = new Set();
  const off = new Set();
  let funniest = null;
  let details = { judge: "room" };
  const box = h("div", { class: "stack" });
  const pickRow = h("div", { class: "pick", role: "group", "aria-label": "Winner" });
  const offRow = h("div", { class: "pick off", role: "group", "aria-label": "Way off" });
  const funRow = h("div", { class: "pick", role: "group", "aria-label": "Funniest guess" });
  const paintPicks = () => {
    pickRow.querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-pressed", String(picked.has(i))));
    offRow.querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-pressed", String(off.has(i))));
    funRow.querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-pressed", String(funniest === i)));
  };
  players.forEach((p, i) => {
    pickRow.append(h("button", { type: "button", onclick: () => { picked.has(i) ? picked.delete(i) : (picked.add(i), off.delete(i)); paintPicks(); } }, p));
    offRow.append(h("button", { type: "button", onclick: () => { off.has(i) ? off.delete(i) : (off.add(i), picked.delete(i)); paintPicks(); } }, p));
    funRow.append(h("button", { type: "button", onclick: () => { funniest = funniest === i ? null : i; paintPicks(); } }, p));
  });
  const results = h("div", { class: "guesslog" });
  const showScores = (scores, label) => {
    clear(results);
    const order = players.map((_, i) => i).filter((i) => guesses[i]).sort((a, b) => (scores[b] ? scores[b].score : -1) - (scores[a] ? scores[a].score : -1));
    for (const i of order) {
      const s = scores[i];
      const w = J.warmth(s ? s.score : 0);
      results.append(h("div", { class: "g" },
        h("span", { class: "t" }, players[i], ": “", guesses[i], "”"),
        h("span", { class: "s lv" + w.level }, s ? s.score : "–"),
        s && s.why ? h("span", { class: "why small muted" }, s.why) : null));
    }
    results.prepend(h("p", { class: "small muted" }, label));
    picked.clear();
    off.clear();
    const top = order.length ? (scores[order[0]] ? scores[order[0]].score : 0) : 0;
    order.forEach((i) => {
      const sc = scores[i] ? scores[i].score : 0;
      if (threshold != null ? sc >= threshold : sc === top && top > 0) picked.add(i);
      if (penalty && sc < 25) off.add(i);
    });
    paintPicks();
  };
  // The free judge: answer key first, then word vectors. Returns how many guesses the key covered.
  const freeScores = async () => {
    const sp = await loadRobot();
    if (!sp) return null;
    const scores = guesses.map((g) => {
      if (!g) return null;
      const r = round.custom ? J.scoreAgainst(SPACE, g, round.target) : robotScore(g, item);
      return { score: r.score, why: r.by === "key" ? r.hint : "", byKey: r.by === "key" };
    });
    return scores;
  };
  const robotBtn = h("button", { class: "btn small", onclick: async () => {
    robotBtn.disabled = true;
    const scores = await freeScores();
    robotBtn.disabled = false;
    if (!scores) { toast("The robot judge couldn't load its word list."); return; }
    details = { judge: "robot", scores: scores.map((s) => (s ? s.score : null)) };
    showScores(scores, "Free judge: the mystery's answer key where a guess matches it, word-vector closeness otherwise.");
  } }, "Free judge");
  let auto = false; // set while the AI judge runs by default rather than by a tap
  const aiBtn = h("button", { class: "btn small", onclick: async () => {
    const wasAuto = auto;
    auto = false;
    aiBtn.disabled = true;
    const old = aiBtn.textContent;
    aiBtn.textContent = "Judging…";
    try {
      const list = players.map((p, i) => ({ name: p, text: guesses[i] })).filter((g) => g.text);
      const idx = players.map((_, i) => i).filter((i) => guesses[i]);
      const out = await aiJudge(round, list);
      const scores = new Array(players.length).fill(null);
      out.scores.forEach((s, k) => { scores[idx[k]] = s; });
      details = { judge: "ai", source: out.source, scores: scores.map((s) => (s ? s.score : null)), comment: out.comment };
      showScores(scores, `AI judge (${aiSourceLabel(out.source)}). ${out.comment || ""}`);
      if (out.funniest != null) { funniest = idx[out.funniest]; paintPicks(); }
    } catch (e) {
      if (e.message !== "cancelled") toast(e.message);
      if (wasAuto) robotBtn.click();
    } finally {
      aiBtn.disabled = false;
      aiBtn.textContent = old;
    }
  } }, "AI judge");

  const ask = pickLabel || (threshold != null ? `Who got it? (${threshold} or more counts)` : "Who was closest?");
  box.append(h("p", { class: "label" }, typed ? ask : ask + " Read your guesses out, then tap names."));
  if (typed) box.append(h("div", { class: "row" }, robotBtn, aiBtn, h("span", { class: "small muted" }, "or just tap a name")));
  box.append(results, pickRow);
  if (penalty) box.append(h("p", { class: "label" }, "Way off? (they lose the value)"), offRow);
  if (threshold == null) box.append(h("p", { class: "label" }, "Funniest guess (bonus, optional)"), funRow);
  box.append(h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => {
    details.winners = [...picked];
    details.penalized = [...off];
    details.funniest = funniest;
    if (details.judge !== "room" && typed) {
      Store.send("judge", { id: item && item.id, prompt: round.prompt, truth: round.truth, mode, guesses, judge: details.judge, scores: details.scores, winners: details.winners });
    }
    onAward([...picked], funniest, details);
  } }, "Award and continue")));
  host.append(box);
  if (!typed) return;
  // AI by default, unless the answer key already covers every guess (then it costs nothing).
  // Pasting into a chatbot is never started automatically: without a built-in AI, the free judge runs.
  const aiAuto = () => { if (aiSource() === "paste") robotBtn.click(); else { auto = true; aiBtn.click(); } };
  if (defaultJudge === "ai" && !round.custom) {
    freeScores().then((scores) => {
      if (scores && scores.every((s, i) => !guesses[i] || (s && s.byKey))) {
        details = { judge: "robot", scores: scores.map((s) => (s ? s.score : null)) };
        showScores(scores, "Answer key: every guess matched one scored in advance.");
      } else aiAuto();
    });
  } else if (defaultJudge === "ai") aiAuto();
  else if (defaultJudge === "robot") robotBtn.click();
}

function timerEl(seconds, onEnd) {
  if (!seconds) return { el: null, stop() {} };
  let left = seconds;
  const el = h("span", { class: "timer", "aria-live": "polite" }, left + "s");
  const id = setInterval(() => {
    left--;
    el.textContent = left > 0 ? left + "s" : "Time!";
    if (left <= 0) { clearInterval(id); if (onEnd) onEnd(); }
  }, 1000);
  return { el, stop() { clearInterval(id); } };
}

/* Screens that aren't a trial (hub, suggest, settings, notes for testers), the router, and boot. */

const SCREENS = {
  hub: () => hubScreen(),
  daily: () => dailyScreen(),
  practice: () => dailyScreen({ practice: true }),
  party: () => partyScreen(),
  byo: () => byoScreen(),
  bluff: () => bluffScreen(),
  suggest: () => suggestScreen({ fromGo: true }),
  settings: () => settingsScreen(),
  notes: () => notesScreen(),
};
let current = "hub";

function go(name, arg) {
  current = SCREENS[name] ? name : "hub";
  const main = clear($("#main"));
  main.append(arg ? arg() : SCREENS[current]());
  document.querySelectorAll(".topnav button").forEach((b) => b.setAttribute("aria-current", String(b.dataset.go === current)));
  try { history.replaceState(null, "", current === "hub" ? location.pathname + location.search : "#" + current); } catch (e) { /* sandboxed */ }
  window.scrollTo({ top: 0 });
}

function shell() {
  const app = $("#app");
  clear(app);
  const nav = h("nav", { class: "topnav", "aria-label": "Main" },
    [["hub", "Play"], ["suggest", "Suggest"], ["notes", "Notes"], ["settings", "Settings"]].map(([k, label]) =>
      h("button", { "data-go": k, onclick: () => go(k) }, label)));
  app.append(h("div", { class: "wrap" },
    h("header", { class: "topbar" }, h("button", { class: "wordmark", onclick: () => go("hub"), "aria-label": "What Is It? home" }, "What Is It", h("span", { class: "q" }, "?")), nav),
    h("main", { id: "main" })));
}

// ---------- hub ----------
function hubScreen() {
  const root = h("div", { class: "stack" });
  const sample = (() => {
    const p = CONTENT.filter((it) => allowed(it) && it.fun >= 4);
    return p.length ? p[RAND.int(p.length)] : CONTENT[0];
  })();
  const tryBox = h("div", { class: "stack" });
  const paintTry = (item) => {
    clear(tryBox);
    tryBox.append(mysteryCard(item), h("div", { class: "row" },
      h("button", { class: "btn primary", onclick: (e) => { e.currentTarget.parentElement.replaceWith(revealCard(item)); } }, "Show me what it is"),
      h("button", { class: "btn ghost", onclick: () => { const p = CONTENT.filter(allowed); paintTry(p[RAND.int(p.length)]); } }, "Another one")));
  };
  paintTry(sample);

  const trials = [
    ["A", "daily", "Daily five", "Solo, about five minutes. Five mysteries a day, one per category. Type a guess and the robot judge tells you how warm you are. Clues cost points.", "1 player"],
    ["B", "party", "Party board", "Pick a square from a board of categories; everyone guesses; the closest guess takes it. Points board or tic-tac-toe.", "2+ players or teams, one screen"],
    ["C", "byo", "Bring your own", "The original game. Someone types any word, phrase or web address; everyone guesses; you look it up together and judge.", "2+ players, a search engine"],
    ["D", "bluff", "Bluff", "Everyone writes a fake answer that sounds true. Find the real one, fool your friends.", "3+ players, pass the phone"],
  ];
  root.append(
    h("section", { class: "hero" },
      h("h1", {}, "Guess what it really is", h("span", { class: "q" }, ".")),
      h("p", { class: "lede" }, "Web addresses, brand names, licence plates and patent titles that aren't what they seem. Four early versions to try, rate and argue about.")),
    h("section", { class: "panel" }, h("p", { class: "eyebrow", style: { marginBottom: "10px" } }, "Quick one"), tryBox),
    h("section", { class: "stack-sm" }, h("h2", { style: { fontSize: "var(--step-1)" } }, "The trial versions"),
      h("div", { class: "trials" }, trials.map(([tag, screen, name, desc, who]) => h("article", { class: "trial" },
        h("span", { class: "tag" }, "Trial " + tag),
        h("h2", {}, name),
        h("p", {}, desc),
        h("div", { class: "foot" }, h("span", { class: "meta" }, who), h("button", { class: "btn primary small", onclick: () => go(screen) }, "Play")))))),
    h("section", { class: "panel stack-sm" },
      h("h2", { style: { fontSize: "var(--step-1)" } }, "Got a good one?"),
      h("p", { class: "muted" }, "Suggest a mystery or a better clue. If we use it, we'll credit you by the name you give."),
      h("div", { class: "row" }, h("button", { class: "btn", onclick: () => go("suggest") }, "Suggest a mystery"))),
    h("p", { class: "small muted" }, `${CONTENT.length} mysteries so far: `, catsSummary(), "."));
  return root;
}
function catsSummary() {
  return CAT_ORDER.map((c) => `${CONTENT.filter((it) => it.cat === c).length} ${CATS[c].short.toLowerCase()}`).join(", ");
}

// ---------- suggest ----------
function suggestScreen(opts = {}) {
  if (!opts.fromGo) { go("suggest", () => suggestScreen(Object.assign({}, opts, { fromGo: true }))); return null; }
  const root = h("div", { class: "stack" });
  const clueFor = opts.clueFor;
  const pre = opts.prefill || {};
  const f = {};
  const field = (id, label, el, hint) => h("div", { class: "field" }, h("label", { for: id }, label), el, hint ? h("span", { class: "small muted" }, hint) : null);
  f.prompt = h("input", { class: "input", id: "sg-prompt", maxlength: 120, value: clueFor ? clueFor.prompt : pre.prompt || "", readonly: !!clueFor });
  f.truth = h("textarea", { class: "input", id: "sg-truth", rows: 2, maxlength: 400, value: pre.truth || "" });
  f.url = h("input", { class: "input", id: "sg-url", maxlength: 300, placeholder: "https://" });
  f.c1 = h("input", { class: "input", id: "sg-c1", maxlength: 160, placeholder: "Vague" });
  f.c2 = h("input", { class: "input", id: "sg-c2", maxlength: 160, placeholder: "Warmer" });
  f.c3 = h("input", { class: "input", id: "sg-c3", maxlength: 160, placeholder: "Nearly gives it away" });
  f.decoy = h("input", { class: "input", id: "sg-decoy", maxlength: 160, placeholder: "The obvious-but-wrong reading" });
  f.credit = h("input", { class: "input", id: "sg-credit", maxlength: 60 });
  f.contact = h("input", { class: "input", id: "sg-contact", maxlength: 120 });
  let kind = clueFor ? clueFor.kind : pre.kind || "domain";
  let reward = "";
  const status = h("p", { class: "small muted" });
  root.append(
    h("div", { class: "stack-sm" }, h("span", { class: "eyebrow" }, "Suggest"),
      h("h2", {}, clueFor ? "Suggest a better clue" : "Suggest a mystery"),
      h("p", { class: "lede" }, clueFor ? "Clues work best in a ladder: a vague one, a warmer one, and one that nearly gives it away." : "The best ones pull your guess one way and turn out to be something else. Web addresses, brands, plates, patents, phrases: anything with a surprising answer.")),
    clueFor ? h("div", { class: "panel stack-sm" }, mysteryCard(clueFor), h("p", { class: "small muted" }, "Current clues: ", (clueFor.clues || []).join(" / "))) : null,
    h("div", { class: "panel stack" },
      field("sg-prompt", "The mystery", f.prompt),
      clueFor ? null : h("div", { class: "stack-sm" }, h("span", { class: "label" }, "What kind?"),
        seg([["domain", "Web address"], ["brand", "Brand"], ["plate", "Licence plate"], ["patent", "Patent"], ["phrase", "Phrase to search"]], kind, (v) => { kind = v; })),
      clueFor ? null : field("sg-truth", "What it really is", f.truth, "One or two plain sentences."),
      clueFor ? null : field("sg-url", "Where can we check it?", f.url),
      h("div", { class: "stack-sm" }, h("span", { class: "label" }, "Clues (optional)"), f.c1, f.c2, f.c3),
      clueFor ? null : field("sg-decoy", "A tempting wrong answer (optional)", f.decoy),
      field("sg-credit", "Credit me as (optional)", f.credit),
      field("sg-contact", "Email, if you'd like to hear about rewards (optional)", f.contact, "Only used to contact you about this suggestion."),
      h("div", { class: "stack-sm" }, h("span", { class: "label" }, "Which reward would make you suggest more?"),
        seg([["credit", "Credit by name"], ["plays", "Extra plays"], ["prize", "A monthly prize"], ["none", "None needed"]], "", (v) => { reward = v; })),
      h("div", { class: "row" }, h("button", { class: "btn primary", onclick: async () => {
        const rec = {
          prompt: f.prompt.value.trim(), kind, truth: f.truth.value.trim(), url: f.url.value.trim(),
          clues: [f.c1.value, f.c2.value, f.c3.value].map((c) => c.trim()).filter(Boolean), decoy: f.decoy.value.trim(),
          credit: f.credit.value.trim(), contact: f.contact.value.trim(), reward, clueFor: clueFor ? clueFor.id : null,
        };
        if (!rec.prompt || (!clueFor && !rec.truth) || (clueFor && !rec.clues.length)) {
          status.textContent = clueFor ? "Write at least one clue." : "Fill in the mystery and what it really is.";
          return;
        }
        status.textContent = savedMsg(await Store.send("suggestion", rec)) + ". Thanks for this one.";
        [f.c1, f.c2, f.c3, f.decoy].forEach((x) => { x.value = ""; });
        if (!clueFor) { f.prompt.value = ""; f.truth.value = ""; f.url.value = ""; }
      } }, "Send suggestion")),
      status));
  return root;
}

// ---------- settings ----------
function settingsScreen() {
  const root = h("div", { class: "stack" });
  const srcs = aiSources();
  const key = h("input", { class: "input", id: "set-key", type: "password", autocomplete: "off", placeholder: "sk-ant-…", value: SETTINGS.apiKey });
  const model = h("select", { class: "input", id: "set-model" }, MODELS.map((m) => h("option", { value: m.id, selected: m.id === SETTINGS.model }, m.label)));
  const pg = h("input", { type: "checkbox", id: "set-pg13", checked: SETTINGS.pg13, onchange: (e) => { SETTINGS.pg13 = e.target.checked; saveSettings(); } });
  const choices = [["auto", "Best available"]];
  if (CONFIG.judgeApi) choices.push(["site", "The game's AI judge"]);
  if (RT.sample) choices.push(["claude", "Claude (your plan)"]);
  if (CONFIG.env !== "artifact") choices.push(["key", "Your API key"]);
  choices.push(["paste", "Paste into a chatbot"]);
  root.append(
    h("div", { class: "stack-sm" }, h("span", { class: "eyebrow" }, "Settings"), h("h2", {}, "Judges and content")),
    h("section", { class: "panel stack" },
      h("h3", {}, "AI judge"),
      h("p", { class: "muted" }, "The AI judge reads every guess like a person would. On the website it is free for players; you can also run it on your own AI account. The word-vector robot is the offline fallback."),
      h("ul", { class: "small", style: { margin: 0, paddingLeft: "1.2em", display: "grid", gap: "4px" } },
        CONFIG.judgeApi ? h("li", {}, h("b", {}, "The game's AI judge: "), siteJudgeUsable() ? "free for players; a small model run by the site." : "not available right now.") : null,
        h("li", {}, h("b", {}, "Claude, on your own plan: "), RT.sample ? "available here." : "works when this page is opened inside claude.ai."),
        CONFIG.env !== "artifact" ? h("li", {}, h("b", {}, "Your Anthropic API key: "), SETTINGS.apiKey ? "saved in this browser." : "add one below. It stays in this browser and goes only to Anthropic.") : null,
        h("li", {}, h("b", {}, "Paste into a chatbot: "), "always works. Copy the prompt into ChatGPT, Claude or Gemini and paste the reply back.")),
      h("div", { class: "stack-sm" }, h("span", { class: "label" }, "Use"), seg(choices, SETTINGS.aiSource, (v) => { SETTINGS.aiSource = v; saveSettings(); })),
      h("p", { class: "small muted" }, "Right now the AI judge would use ", h("b", {}, aiSourceLabel(aiSource())), "."),
      CONFIG.env !== "artifact" ? h("div", { class: "stack-sm" },
        h("div", { class: "field" }, h("label", { for: "set-key" }, "Anthropic API key (optional)"), key),
        h("div", { class: "field" }, h("label", { for: "set-model" }, "Model"), model),
        h("div", { class: "row" },
          h("button", { class: "btn small", onclick: () => { SETTINGS.apiKey = key.value.trim(); SETTINGS.model = model.value; saveSettings(); toast(SETTINGS.apiKey ? "Key saved in this browser" : "Key cleared"); go("settings"); } }, "Save"),
          h("button", { class: "btn ghost small", onclick: () => { SETTINGS.apiKey = ""; saveSettings(); go("settings"); } }, "Forget key")),
        h("p", { class: "small muted" }, "Costs are rough, per judged round, on your account. Anyone who can run code in this browser could read a saved key, so use a key with a low spending limit.")) : null),
    h("section", { class: "panel stack-sm" },
      h("h3", {}, "Content"),
      h("label", { class: "check", for: "set-pg13" }, pg, h("span", {}, "Include ", h("b", {}, "Double take"), ": web addresses and plates that read as something rude (PG-13). Never in the Daily."))),
    h("section", { class: "panel stack-sm" },
      h("h3", {}, "Your ratings and suggestions"),
      h("p", { class: "small muted" }, `${Store.local().length} saved on this device. `, CONFIG.env === "artifact" ? "On claude.ai they are also shared with the team when you have permission." : "On the Netlify site they are also sent to the team when the site's forms are switched on."),
      h("div", { class: "row" },
        h("button", { class: "btn small", onclick: () => copyText(JSON.stringify(Store.local(), null, 1), "Copied your ratings") }, "Copy them all"),
        h("button", { class: "btn ghost small", onclick: () => { save("outbox", []); toast("Cleared"); go("settings"); } }, "Clear"))));
  return root;
}

// ---------- notes for testers ----------
function notesScreen() {
  const root = h("div", { class: "stack" });
  root.append(
    h("div", { class: "stack-sm" }, h("span", { class: "eyebrow" }, "Notes for testers"), h("h2", {}, "What we're trying to learn")),
    h("section", { class: "panel prose" },
      h("p", {}, "Four versions of one idea: someone shows a cryptic string, everyone guesses what it really is, and the reveal settles it. Please play at least two, then rate each one."),
      h("ul", {},
        h("li", {}, h("b", {}, "A. Daily five: "), "is a solo, NYT-style daily fun on its own? Is the robot judge fair enough, and do clues and the side bets help?"),
        h("li", {}, h("b", {}, "B. Party board: "), "does a board of categories make a good evening? Points board or tic-tac-toe?"),
        h("li", {}, h("b", {}, "C. Bring your own: "), "the original. Is it more fun with your own mysteries and live searching?"),
        h("li", {}, h("b", {}, "D. Bluff: "), "is writing fakes more fun than guessing the truth?")),
      h("p", {}, "After each mystery you can rate how fun it was. Those ratings are how we'll pick which mysteries and categories to keep.")),
    h("section", { class: "panel prose" },
      h("h3", {}, "Judging"),
      h("p", {}, "The robot judge compares your words with each answer's key ideas using ConceptNet Numberbatch word vectors (the same word map as Word Bocce). It is free and instant but literal. The AI judge reads guesses like a person and also picks the funniest; it runs on your own AI account. Compare them in the Daily after each reveal.")),
    h("section", { class: "panel prose" },
      h("h3", {}, "Where the mysteries come from"),
      h("ul", {},
        h("li", {}, "Licence plates: real applications to the California DMV, 2015–16, with each owner's explanation and the reviewer's notes, from Noah Veltman's public-records dataset (github.com/veltman/ca-license-plates)."),
        h("li", {}, "Patents: checked against the patent text on Google Patents."),
        h("li", {}, "Web addresses, brands and top results: checked in October 2026. Sites change; tell us if one is out of date."),
        h("li", {}, "Some brand and “double take” items are marked as not yet re-checked online; treat them as drafts."))),
    h("section", { class: "panel prose small" },
      h("h3", {}, "Credits"),
      h("p", {}, "Word vectors: ConceptNet Numberbatch 19.08 by Robyn Speer, Joshua Chin and Catherine Havasi, CC BY-SA 4.0. API-key judge: Anthropic TypeScript SDK (MIT). Built in the word2vecgames repository alongside Word Bocce.")));
  return root;
}

// ---------- boot ----------
(function boot() {
  shell();
  const start = (location.hash || "").replace("#", "");
  go(SCREENS[start] ? start : "hub");
  window.addEventListener("hashchange", () => { const k = location.hash.replace("#", ""); if (SCREENS[k] && k !== current) go(k); });
  if (window.claude && typeof window.claude.use === "function") {
    window.claude.use("sample").then((s) => { RT.sample = s; if (current === "settings") go("settings"); }).catch(() => {});
    window.claude.use("db").then((d) => { RT.db = d; }).catch(() => {});
  }
})();

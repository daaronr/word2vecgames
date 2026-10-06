/* Trial C: Bring your own. The original game: someone types any word, phrase or web address,
   everyone guesses, then you look it up together and judge who was closest. Optional Family-Feud
   bonus for matching the 2nd and 3rd search results. */

const BYO_IDEAS = [
  "kafka.com", "Bring your own balls", "Spaghetti tree", "Radio garden", "Point Nemo", "Toad in the hole",
  "Head cheese", "Chocolate rain", "Jack in the Box", "Bored Panda", "The Onion", "Five Guys",
  "pointerpointer.com", "isitchristmas.com", "zombo.com", "hackertyper.net", "theuselessweb.com",
  "Dihydrogen monoxide", "Mechanical Turk", "Three Wolf Moon", "Pet rock", "Hush puppies", "Monkey's eyebrow",
  "Correct horse battery staple", "Truth or Consequences", "Cracker Barrel", "Up goer five", "Drop bears",
  "Kettle war", "Wilhelm scream",
];

function byoScreen() {
  const root = h("div", { class: "stack" });
  const S = { prompt: "", players: SETTINGS.players.length ? SETTINGS.players.slice() : ["Player 1", "Player 2", "Player 3"], entry: "typed", tally: {} };
  ask();
  return root;

  function itemFor() {
    return { id: null, cat: "byo", kind: looksLikeDomain(S.prompt) ? "domain" : "phrase", prompt: S.prompt.replace(/^https?:\/\//i, ""), ask: looksLikeDomain(S.prompt) ? "What is this website?" : "What comes up first when you search this?" };
  }

  function ask() {
    clear(root);
    const input = h("input", { class: "input", id: "byo-prompt", maxlength: 120, value: S.prompt, placeholder: "A word, phrase or web address" });
    const editor = playersEditor(S.players, { min: 1, max: 8, noun: "Player" });
    const next = () => {
      const p = input.value.trim();
      if (!p) { toast("Type a mystery first, or tap “Give me one”"); return; }
      S.prompt = p;
      S.players = editor.get();
      SETTINGS.players = S.players;
      saveSettings();
      guesses();
    };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") next(); });
    root.append(
      h("div", { class: "stack-sm" }, h("span", { class: "eyebrow" }, "Trial C"), h("h2", {}, "Bring your own"),
        h("p", { class: "lede" }, "The original version. One person types something mysterious. Everyone guesses what it is, or what comes up first when you search it. Then you look it up together.")),
      h("div", { class: "panel stack" },
        h("div", { class: "field" }, h("label", { for: "byo-prompt" }, "The mystery"), input),
        h("div", { class: "row" },
          h("button", { class: "btn small", onclick: () => { input.value = BYO_IDEAS[RAND.int(BYO_IDEAS.length)]; } }, "Give me one"),
          h("span", { class: "small muted" }, "Ideas we haven't looked up for you: that's the point.")),
        h("p", { class: "label" }, "Who's playing"), editor.el,
        h("div", { class: "stack-sm" }, h("span", { class: "label" }, "Guesses"),
          seg([["typed", "Type them here, passing the device"], ["paper", "Paper or out loud"]], S.entry, (v) => { S.entry = v; })),
        h("div", { class: "row" }, h("button", { class: "btn primary", onclick: next }, "Start guessing"))),
      Object.keys(S.tally).length ? tallyPanel() : null,
      trialRating("C", "Rate Trial C: Bring your own"));
  }

  function tallyPanel() {
    return h("div", { class: "panel stack-sm" }, h("span", { class: "label" }, "Running score"),
      h("div", { class: "scores" }, S.players.map((p) => h("div", { class: "sc" }, h("span", { class: "small muted" }, p), h("b", { class: "tabular" }, S.tally[p] || 0)))));
  }

  async function guesses() {
    clear(root);
    const item = itemFor();
    loadRobot();
    root.append(mysteryCard(item));
    const area = h("div", { class: "stack" });
    root.append(area);
    if (S.entry === "typed") {
      const got = await collectPrivately(area, S.players, { item, prompt: "what will it turn out to be?", placeholder: "Your guess", skippable: true });
      lookup(got);
    } else {
      area.append(h("p", { class: "muted" }, "Everyone write down a guess. Then look it up."),
        h("button", { class: "btn primary", onclick: () => lookup(S.players.map(() => "")) }, "Look it up"));
    }
  }

  function lookup(gs) {
    clear(root);
    const item = itemFor();
    const q = encodeURIComponent(item.prompt);
    const truth = h("textarea", { class: "input", id: "byo-truth", rows: 3, placeholder: "e.g. “A forum thread where pool players argue about bringing their own pool balls”" });
    const r2 = h("input", { class: "input", id: "byo-r2", placeholder: "Result 2 (optional)" });
    const r3 = h("input", { class: "input", id: "byo-r3", placeholder: "Result 3 (optional)" });
    const link = (href, label) => h("a", { class: "btn small", href, target: "_blank", rel: "noopener noreferrer" }, label);
    root.append(mysteryCard(item),
      h("div", { class: "panel stack" },
        h("span", { class: "label" }, "Look it up together"),
        h("div", { class: "row" },
          link("https://www.google.com/search?q=" + q, "Google"),
          link("https://duckduckgo.com/?q=" + q, "DuckDuckGo"),
          link("https://www.bing.com/search?q=" + q, "Bing"),
          link("https://en.wikipedia.org/w/index.php?search=" + q, "Wikipedia"),
          item.kind === "domain" ? link("https://" + item.prompt, "Open the address") : null),
        item.kind === "domain" ? h("p", { class: "small muted" }, "We haven't checked this address. Search results are safer to look at first.") : null,
        h("div", { class: "field" }, h("label", { for: "byo-truth" }, "What did you find? The top result, or what the site is, in a sentence"), truth),
        h("details", {}, h("summary", { class: "small" }, "Family Feud bonus: add results 2 and 3"),
          h("div", { class: "stack-sm", style: { marginTop: "8px" } }, r2, r3,
            h("p", { class: "small muted" }, "Guesses that match the top result score 3, the second 2, the third 1 (robot-judged)."))),
        h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => {
          if (!truth.value.trim()) { toast("Describe what you found first"); return; }
          judge(gs, truth.value.trim(), [r2.value.trim(), r3.value.trim()].filter(Boolean));
        } }, "Judge the guesses"))));
  }

  function judge(gs, truthText, extra) {
    clear(root);
    const base = itemFor();
    const item = Object.assign(base, { truth: truthText, decoys: [], key: J.keysFromText(truthText) });
    const round = {
      prompt: item.prompt, kindLabel: item.kind === "domain" ? "a web address" : "a search phrase", ask: item.ask, truth: truthText,
      results: extra.length ? [truthText, ...extra] : null, custom: true, target: { keys: item.key, text: truthText },
    };
    root.append(mysteryCard(item), revealCard(item));
    const area = h("div", { class: "stack" });
    root.append(area);
    const typed = gs.some((g) => g);
    if (typed && extra.length) area.append(feudPanel(gs, [truthText, ...extra]));
    const rating = funRating(item, "byo", { truth: truthText });
    judgePanel(area, {
      round, item, players: S.players, guesses: gs, mode: "byo", defaultJudge: typed ? "robot" : "room",
      onAward: (winners, funniest) => {
        rating.flush();
        winners.forEach((w) => { const n = S.players[w]; S.tally[n] = (S.tally[n] || 0) + 1; });
        if (funniest != null) { const n = S.players[funniest]; S.tally[n] = (S.tally[n] || 0) + 0.5; }
        after(item);
      },
    });
    area.append(rating);
  }

  function feudPanel(gs, results) {
    const box = h("div", { class: "panel stack-sm" }, h("span", { class: "label" }, "Family Feud bonus"), h("p", { class: "small muted" }, "Scoring…"));
    loadRobot().then((sp) => {
      clear(box).append(h("span", { class: "label" }, "Family Feud bonus"));
      if (!sp) { box.append(h("p", { class: "small muted" }, "The robot judge couldn't load, so no bonus scoring.")); return; }
      results.forEach((res, k) => {
        const hits = [];
        gs.forEach((g, i) => {
          if (!g) return;
          const s = J.scoreAgainst(sp, g, { keys: J.keysFromText(res), text: res }).score;
          if (s >= 45) hits.push(`${S.players[i]} (${s})`);
        });
        box.append(h("div", { class: "spread small" }, h("span", {}, h("b", {}, `#${k + 1} (${3 - k} pts) `), res), h("span", { class: "muted" }, hits.length ? hits.join(", ") : "nobody")));
      });
    });
    return box;
  }

  function after(item) {
    clear(root);
    root.append(h("span", { class: "eyebrow" }, "Trial C · Round done"), tallyPanel(),
      h("div", { class: "panel stack-sm" },
        h("p", {}, "Was ", h("b", {}, item.prompt), " a good one? Send it in and it may join the official game."),
        h("div", { class: "row" }, h("button", { class: "btn", onclick: () => suggestScreen({ prefill: { prompt: item.prompt, truth: item.truth, kind: item.kind } }) }, "Suggest it for the game"))),
      h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => { S.prompt = ""; ask(); } }, "Next mystery")));
  }
}

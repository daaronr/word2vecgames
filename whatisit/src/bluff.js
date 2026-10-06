/* Trial D: Bluff. Pass the device: everyone writes a believable fake answer, the real one is mixed
   in (plus, if you like, one of our tempting wrong answers), and everyone votes. Points for finding
   the truth and for fooling the others. */

function bluffScreen() {
  const root = h("div", { class: "stack" });
  let B = null;
  setup();
  return root;

  function setup() {
    clear(root);
    const editor = playersEditor(SETTINGS.players.length >= 3 ? SETTINGS.players : ["Player 1", "Player 2", "Player 3"], { min: 3, max: 8, noun: "Player" });
    const cfg = Object.assign({ rounds: 5, decoy: true }, load("bluff-cfg", {}));
    const decoy = h("input", { type: "checkbox", id: "bluff-decoy", checked: cfg.decoy });
    const pg = h("input", { type: "checkbox", id: "bluff-pg13", checked: SETTINGS.pg13 });
    root.append(
      h("div", { class: "stack-sm" }, h("span", { class: "eyebrow" }, "Trial D"), h("h2", {}, "Bluff"),
        h("p", { class: "lede" }, "Three or more players, one device. Everyone writes a fake answer that sounds true. Spot the real one; fool the others.")),
      h("div", { class: "panel stack" },
        h("p", { class: "label" }, "Players"), editor.el,
        h("div", { class: "stack-sm" }, h("span", { class: "label" }, "Rounds"), seg([[3, "3"], [5, "5"], [8, "8"]], cfg.rounds, (v) => { cfg.rounds = v; })),
        h("label", { class: "check", for: "bluff-decoy" }, decoy, h("span", {}, "Add one of our tempting wrong answers to each round (good with 3 or 4 players)")),
        h("label", { class: "check", for: "bluff-pg13" }, pg, h("span", {}, "Include ", h("b", {}, "Double take"), " (PG-13)")),
        h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => {
          cfg.decoy = decoy.checked;
          SETTINGS.pg13 = pg.checked;
          SETTINGS.players = editor.get();
          saveSettings();
          save("bluff-cfg", cfg);
          B = { cfg, players: SETTINGS.players.slice(), scores: SETTINGS.players.map(() => 0), n: 0, used: new Set() };
          loadRobot();
          roundStart();
        } }, "Start"))));
  }

  function nextItem() {
    const p = CONTENT.filter((it) => allowed(it) && !B.used.has(it.id) && (it.decoys || []).length);
    const it = p[RAND.int(p.length)];
    B.used.add(it.id);
    return it;
  }

  function roundStart() {
    B.n++;
    const item = nextItem();
    clear(root);
    root.append(h("span", { class: "eyebrow" }, `Trial D · Round ${B.n} of ${B.cfg.rounds}`), mysteryCard(item, null, { level: 2 }));
    const area = h("div", { class: "stack" });
    root.append(area);
    area.append(h("p", { class: "muted" }, "Everyone read it. Then each player, in private, writes a fake answer that could pass for the truth."),
      h("div", { class: "row" }, h("button", { class: "btn primary", onclick: async () => {
        const fakes = await collectPrivately(area, B.players, {
          item, prompt: "write a believable fake answer", placeholder: "Something that sounds true",
          check: (t) => {
            if (SPACE && J.scoreGuess(SPACE, t, item).score >= 85) return "That's very close to the real answer. Write something false.";
            return "";
          },
        });
        vote(item, fakes);
      } }, "Write fakes")));
  }

  function vote(item, fakes) {
    // Options: the truth, each player's fake, and maybe one of our decoys. Identical texts merge.
    const opts = [];
    const addOpt = (text, by) => {
      const k = text.trim().toLowerCase();
      const ex = opts.find((o) => o.key === k);
      if (ex) ex.by.push(by);
      else opts.push({ key: k, text: text.trim(), by: [by] });
    };
    addOpt(item.truth, "truth");
    fakes.forEach((f, i) => addOpt(f, i));
    if (B.cfg.decoy && item.decoys && item.decoys.length) addOpt(item.decoys[RAND.int(item.decoys.length)], "house");
    RAND.shuffle(opts);
    const votes = new Array(B.players.length).fill(null);
    let i = 0;
    const host = h("div", { class: "stack" });
    clear(root);
    root.append(h("span", { class: "eyebrow" }, `Trial D · Round ${B.n} · Vote`), host);
    const step = (phase) => {
      clear(host);
      if (i >= B.players.length) { reveal(item, opts, votes); return; }
      const who = B.players[i];
      if (phase === "pass") {
        host.append(h("div", { class: "panel passcard" }, h("span", { class: "eyebrow" }, `${i + 1} of ${B.players.length}`),
          h("div", {}, "Pass the device to"), h("div", { class: "who" }, who),
          h("button", { class: "btn primary", onclick: () => step("vote") }, "I'm " + who)));
        return;
      }
      host.append(mysteryCard(item, null, { level: 2 }), h("p", { class: "label" }, `${who}: which one is real?`),
        h("div", { class: "choices" }, opts.map((o, k) => o.by.includes(i)
          ? h("button", { class: "choice", disabled: true }, o.text, h("span", { class: "small muted" }, " (yours)"))
          : h("button", { class: "choice", onclick: () => { votes[i] = k; i++; step("pass"); } }, o.text))));
    };
    step("pass");
  }

  function reveal(item, opts, votes) {
    const gained = B.players.map(() => 0);
    votes.forEach((k, voter) => {
      if (k == null) return;
      const o = opts[k];
      if (o.by.includes("truth")) gained[voter] += 2;
      o.by.forEach((b) => { if (typeof b === "number" && b !== voter) gained[b] += 1; });
    });
    gained.forEach((g, i) => { B.scores[i] += g; });
    clear(root);
    root.append(h("span", { class: "eyebrow" }, `Trial D · Round ${B.n} · The truth`), mysteryCard(item, null, { level: 3 }), revealCard(item));
    root.append(h("div", { class: "votes" }, opts.map((o, k) => {
      const voters = votes.map((v, i) => (v === k ? B.players[i] : null)).filter(Boolean);
      const authors = o.by.map((b) => (b === "truth" ? "The real answer" : b === "house" ? "Our tempting wrong answer" : "Written by " + B.players[b]));
      return h("div", { class: "vote" + (o.by.includes("truth") ? " truth" : "") },
        h("div", {}, o.by.includes("truth") ? h("b", {}, o.text) : o.text),
        h("div", { class: "by" }, authors.join(" · "), " · ", voters.length ? "picked by " + voters.join(", ") : "nobody picked it"));
    })));
    root.append(h("div", { class: "scores" }, B.players.map((p, i) => h("div", { class: "sc" },
      h("span", { class: "small muted" }, p), h("b", { class: "tabular" }, B.scores[i]), gained[i] ? h("span", { class: "small" }, "+" + gained[i] + " this round") : null))));
    const rating = funRating(item, "bluff", { fooled: votes.filter((k) => k != null && !opts[k].by.includes("truth")).length });
    root.append(rating);
    root.append(h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => {
      rating.flush();
      if (B.n >= B.cfg.rounds) final();
      else roundStart();
    } }, B.n >= B.cfg.rounds ? "Final scores" : "Next round")));
  }

  function final() {
    clear(root);
    const top = Math.max(...B.scores);
    const winners = B.players.filter((_, i) => B.scores[i] === top);
    root.append(h("span", { class: "eyebrow" }, "Trial D · Final"), h("h2", { class: "scorebig" }, `${winners.join(" and ")} win${winners.length > 1 ? "" : "s"}`),
      h("div", { class: "scores" }, B.players.map((p, i) => h("div", { class: "sc" + (B.scores[i] === top ? " lead" : "") }, h("span", { class: "small muted" }, p), h("b", { class: "tabular" }, B.scores[i])))),
      h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => { B.n = 0; B.scores = B.players.map(() => 0); roundStart(); } }, "Play again"),
        h("button", { class: "btn", onclick: setup }, "Change players")),
      trialRating("D", "Rate Trial D: Bluff"));
  }
}

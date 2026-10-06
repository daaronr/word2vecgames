/* Trial B: Party board. Same room or on a call. Pick a square; everyone guesses; the closest guess
   takes it. Two layouts: a points board (categories x 100/200/300) or tic-tac-toe for two teams. */

const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
const CLUE_VALUE = [1, 0.75, 0.5, 0.25];

function partyScreen() {
  const root = h("div", { class: "stack" });
  let P = null;
  setup();
  return root;

  function setup() {
    clear(root);
    const cfg = Object.assign({ style: "points", entry: "typed", judge: siteJudgeUsable() ? "ai" : "robot", timer: 0 }, load("party-cfg", {}));
    let editor;
    const edHost = h("div");
    const paintEditor = () => {
      clear(edHost);
      const names = cfg.style === "ttt" ? (load("party-teams", ["Team X", "Team O"])) : (SETTINGS.players.length ? SETTINGS.players : ["Player 1", "Player 2", "Player 3"]);
      editor = cfg.style === "ttt"
        ? playersEditor(names.slice(0, 2), { min: 2, max: 2, noun: "Team" })
        : playersEditor(names, { min: 2, max: 8, noun: "Player" });
      edHost.append(h("p", { class: "label" }, cfg.style === "ttt" ? "Two teams" : "Players or teams"), editor.el);
    };
    paintEditor();
    const pg = h("input", { type: "checkbox", id: "party-pg13", checked: SETTINGS.pg13 });
    root.append(
      h("div", { class: "stack-sm" }, h("span", { class: "eyebrow" }, "Trial B"), h("h2", {}, "Party board"),
        h("p", { class: "lede" }, "For a room or a video call. Pick a square, everyone guesses what the mystery really is, and the closest guess takes the square.")),
      h("div", { class: "panel stack" },
        h("div", { class: "stack-sm" }, h("span", { class: "label" }, "Board"),
          seg([["points", "Points board"], ["ttt", "Tic-tac-toe, two teams"]], cfg.style, (v) => { cfg.style = v; paintEditor(); })),
        edHost,
        h("div", { class: "stack-sm" }, h("span", { class: "label" }, "Guesses"),
          seg([["typed", "Type them on this device, passing it round"], ["paper", "Paper or out loud"]], cfg.entry, (v) => { cfg.entry = v; })),
        h("div", { class: "stack-sm" }, h("span", { class: "label" }, "Judge"),
          seg([["robot", "Robot"], ["ai", "AI"], ["room", "The room decides"]], cfg.judge, (v) => { cfg.judge = v; }),
          h("p", { class: "small muted" }, "Robot and AI need typed guesses. You can always overrule them by tapping a name.")),
        h("div", { class: "stack-sm" }, h("span", { class: "label" }, "Timer"),
          seg([[0, "None"], [30, "30 seconds"], [60, "60 seconds"]], cfg.timer, (v) => { cfg.timer = v; })),
        h("label", { class: "check", for: "party-pg13" }, pg, h("span", {}, "Include ", h("b", {}, "Double take"), " (PG-13 misreadings)")),
        h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => {
          SETTINGS.pg13 = pg.checked;
          const names = editor.get();
          if (cfg.style === "ttt") save("party-teams", names);
          else { SETTINGS.players = names; }
          saveSettings();
          save("party-cfg", cfg);
          start(cfg, names);
        } }, "Deal the board"))));
  }

  function start(cfg, names) {
    loadRobot();
    const used = new Set();
    const take = (cat, diff) => {
      const p = pool(cat).filter((it) => !used.has(it.id));
      const exact = p.filter((it) => (it.difficulty || 2) === diff);
      const from = exact.length ? exact : p.length ? p : CONTENT.filter((it) => allowed(it) && !used.has(it.id));
      const it = from[RAND.int(from.length)];
      used.add(it.id);
      return it;
    };
    P = { cfg, players: names.map((n) => ({ name: n, score: 0 })), turn: 0, log: [] };
    if (cfg.style === "points") {
      const cats = catsAvailable().slice(0, 6);
      P.cols = cats.map((cat) => ({ cat, tiles: [1, 2, 3].map((d) => ({ value: d * 100, item: take(cat, d), done: false, by: [] })) }));
    } else {
      const cats = RAND.shuffle(catsAvailable().slice());
      P.cells = Array.from({ length: 9 }, (_, i) => {
        const cat = cats[i % cats.length];
        return { cat, item: take(cat, 1 + RAND.int(3)), mark: null, done: false };
      });
    }
    board();
  }

  function scoresBar() {
    const top = Math.max(...P.players.map((p) => p.score));
    return h("div", { class: "scores" }, P.players.map((p, i) => h("div", { class: "sc" + (p.score === top && top > 0 ? " lead" : "") },
      h("span", { class: "small muted" }, P.cfg.style === "ttt" ? (i === 0 ? "X · " : "O · ") + p.name : p.name),
      h("b", { class: "tabular" }, P.cfg.style === "ttt" ? P.cells.filter((c) => c.mark === i).length : p.score))));
  }

  function board() {
    clear(root);
    root.append(h("div", { class: "spread" }, h("span", { class: "eyebrow" }, "Trial B · Party board"),
      h("button", { class: "btn ghost small", onclick: () => { if (P) end(); } }, "End game")));
    root.append(scoresBar());
    if (P.cfg.style === "points") {
      const grid = h("div", { class: "board points", style: { "--cols": P.cols.length } });
      P.cols.forEach((c) => grid.append(h("div", { class: "head" }, CATS[c.cat].label)));
      for (let row = 0; row < 3; row++) {
        P.cols.forEach((c) => {
          const t = c.tiles[row];
          grid.append(h("button", { class: "tile", disabled: t.done, "aria-label": `${CATS[c.cat].label}, ${t.value}`, onclick: () => round({ kind: "points", tile: t, cat: c.cat }) },
            t.done ? (t.by.length ? P.players[t.by[0]].name.slice(0, 10) : "–") : t.value));
        });
      }
      root.append(grid);
      if (P.cols.every((c) => c.tiles.every((t) => t.done))) { end(); return; }
      root.append(h("p", { class: "small muted" }, "Take turns choosing. Bigger numbers are harder."));
    } else {
      const team = P.players[P.turn % 2];
      root.append(h("p", { class: "label" }, `${P.turn % 2 === 0 ? "X" : "O"} · ${team.name} picks a square. Both teams guess; the closer one claims it.`));
      const grid = h("div", { class: "board ttt" });
      P.cells.forEach((c) => grid.append(h("button", { class: "tile" + (c.mark === 0 ? " x" : c.mark === 1 ? " o" : ""), disabled: c.done, "aria-label": c.done ? "Taken" : CATS[c.cat].label, onclick: () => round({ kind: "ttt", cell: c, cat: c.cat }) },
        c.mark === 0 ? h("span", { class: "mark" }, "X") : c.mark === 1 ? h("span", { class: "mark" }, "O") : c.done ? "–" : CATS[c.cat].short)));
      root.append(grid);
    }
  }

  function round(sel) {
    const item = sel.kind === "points" ? sel.tile.item : sel.cell.item;
    const names = P.players.map((p) => p.name);
    let clues = 0;
    clear(root);
    const valueEl = h("span", { class: "chip" });
    const paintValue = () => {
      valueEl.textContent = sel.kind === "points" ? `Worth ${Math.round(sel.tile.value * CLUE_VALUE[clues])}` : "Claims the square";
    };
    paintValue();
    const t = timerEl(P.cfg.timer, () => toast("Time's up"));
    root.append(mysteryCard(item, valueEl));
    const clueBox = h("div", { class: "stack-sm" });
    const clueBtn = h("button", { class: "btn small", onclick: () => {
      if (clues >= (item.clues || []).length) return;
      clueBox.append(h("p", { class: "clue" }, h("b", {}, "Clue " + (clues + 1) + ": "), item.clues[clues]));
      clues++;
      paintValue();
      if (clues >= (item.clues || []).length) clueBtn.disabled = true;
    } }, "Give a clue");
    const area = h("div", { class: "stack" });
    root.append(h("div", { class: "row" }, clueBtn, t.el), clueBox, area);

    const reveal = (guesses) => {
      t.stop();
      clear(area);
      area.append(revealCard(item));
      const rating = funRating(item, "party");
      judgePanel(area, {
        round: { id: item.id, prompt: item.prompt, kindLabel: kindLabel(item), ask: item.ask, truth: item.truth, more: item.more },
        item, players: names, guesses, mode: "party", defaultJudge: P.cfg.entry === "typed" ? P.cfg.judge : "room",
        onAward: (winners, funniest) => {
          rating.flush();
          if (sel.kind === "points") {
            const pts = Math.round(sel.tile.value * CLUE_VALUE[clues]);
            winners.forEach((w) => { P.players[w].score += pts; });
            if (funniest != null) P.players[funniest].score += 50;
            sel.tile.done = true;
            sel.tile.by = winners;
          } else {
            sel.cell.done = true;
            sel.cell.mark = winners.length === 1 ? winners[0] : null;
            P.turn++;
            const winner = LINES.find((l) => l.every((k) => P.cells[k].mark === 0)) ? 0 : LINES.find((l) => l.every((k) => P.cells[k].mark === 1)) ? 1 : null;
            if (winner != null || P.cells.every((c) => c.done)) { end(winner); return; }
          }
          board();
        },
      });
      area.append(rating);
    };

    if (P.cfg.entry === "typed") {
      area.append(h("p", { class: "muted" }, "Everyone take a look. Then pass the device round so each player types a guess in private."),
        h("div", { class: "row" }, h("button", { class: "btn primary", onclick: async () => {
          const guesses = await collectPrivately(area, names, { item, prompt: "what is it?", placeholder: "Your guess, in a few words", skippable: true });
          clear(area);
          area.append(h("p", { class: "label" }, "All guesses are in."), h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => reveal(guesses) }, "Reveal the answer")));
        } }, "Collect guesses")));
    } else {
      area.append(h("p", { class: "muted" }, "Everyone write a guess on paper (or keep it in your head). When you're ready, reveal."),
        h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => reveal(names.map(() => "")) }, "Reveal the answer")));
    }
  }

  function end(tttWinner) {
    clear(root);
    let headline;
    if (P.cfg.style === "ttt") {
      headline = tttWinner != null ? `${P.players[tttWinner].name} got three in a row` : (() => {
        const x = P.cells.filter((c) => c.mark === 0).length;
        const o = P.cells.filter((c) => c.mark === 1).length;
        return x === o ? "A draw" : `${P.players[x > o ? 0 : 1].name} took more squares`;
      })();
    } else {
      const top = Math.max(...P.players.map((p) => p.score));
      const winners = P.players.filter((p) => p.score === top).map((p) => p.name);
      headline = top > 0 ? `${winners.join(" and ")} win${winners.length > 1 ? "" : "s"}` : "Nobody scored";
    }
    root.append(h("span", { class: "eyebrow" }, "Trial B · Final"), h("h2", { class: "scorebig" }, headline), scoresBar(),
      h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => start(P.cfg, P.players.map((p) => p.name)) }, "Play again"),
        h("button", { class: "btn", onclick: setup }, "Change setup")),
      trialRating("B", "Rate Trial B: Party board"));
  }
}

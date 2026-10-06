/* Trial B: Party board. Same room or on a call. Pick a square; everyone guesses; the closest guess
   takes it. Three boards:
   - Points, Jeopardy-style: categories x values. By default a column's lower values must go before its
     higher ones (on TV any square may be picked; that is an option). Whoever takes a square picks
     next. One hidden Daily Double: the picker alone answers, for a wager. Optionally, way-off guesses
     lose the value. A Final mystery with wagers ends the game.
   - Tic-tac-toe and Connect Four, two teams: the team that wins a round claims the square. */

const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
const CLUE_VALUE = [1, 0.75, 0.5, 0.25];
const C4_SIZE = { quick: { cols: 5, rows: 4 }, full: { cols: 7, rows: 6 } };
const RIGHT = 65; // a Daily Double or Final guess this close or closer counts as right

function partyScreen() {
  const root = h("div", { class: "stack" });
  let P = null;
  setup();
  return root;

  function teams(style) { return style === "ttt" || style === "c4"; }

  function setup() {
    clear(root);
    const cfg = Object.assign(
      { style: "points", entry: "typed", judge: siteJudgeUsable() ? "ai" : "robot", timer: 0, rows: 3, ladder: true, dd: true, final: true, penalty: false, c4: "quick", cats: null },
      load("party-cfg", {}));
    let editor;
    const edHost = h("div");
    const optHost = h("div", { class: "stack" });
    const catHost = h("div", { class: "stack-sm" });
    const paintEditor = () => {
      clear(edHost);
      const names = teams(cfg.style) ? load("party-teams", ["Team X", "Team O"]) : (SETTINGS.players.length ? SETTINGS.players : ["Player 1", "Player 2", "Player 3"]);
      editor = teams(cfg.style)
        ? playersEditor(names.slice(0, 2), { min: 2, max: 2, noun: "Team" })
        : playersEditor(names, { min: 2, max: 8, noun: "Player" });
      edHost.append(h("p", { class: "label" }, teams(cfg.style) ? "Two teams" : "Players or teams"), editor.el);
    };
    let catPick;
    const paintCats = () => {
      clear(catHost);
      const avail = catsAvailable().filter((c) => c !== "double" || SETTINGS.pg13 || pg.checked);
      const max = cfg.style === "points" ? 6 : cfg.style === "c4" ? C4_SIZE[cfg.c4].cols : 9;
      const saved = (cfg.cats || []).filter((c) => avail.includes(c));
      const start = saved.length ? saved.slice(0, max) : RAND.shuffle(avail.filter((c) => c !== "double")).slice(0, Math.min(max, 6));
      catPick = chipPicker(avail.map((c) => [c, CATS[c].label]), start, max);
      catHost.append(h("span", { class: "label" }, cfg.style === "c4" ? `Categories (one per column, up to ${max})` : `Categories (up to ${max})`), catPick.el);
    };
    const paintOpts = () => {
      clear(optHost);
      if (cfg.style === "points") {
        optHost.append(
          h("div", { class: "stack-sm" }, h("span", { class: "label" }, "Rows"),
            seg([[3, "3 rows: 100 to 300"], [5, "5 rows: 200 to 1,000"]], cfg.rows, (v) => { cfg.rows = v; })),
          h("div", { class: "stack-sm" }, h("span", { class: "label" }, "Picking squares"),
            seg([[true, "Lowest first in each column"], [false, "Any square (as on TV)"]], cfg.ladder, (v) => { cfg.ladder = v; })),
          check("party-dd", "One hidden Daily Double: the picker answers alone, for a wager", cfg.dd, (v) => { cfg.dd = v; }),
          check("party-penalty", "Way-off guesses lose the value", cfg.penalty, (v) => { cfg.penalty = v; }),
          check("party-final", "Final mystery with wagers", cfg.final, (v) => { cfg.final = v; }));
      } else if (cfg.style === "c4") {
        optHost.append(h("div", { class: "stack-sm" }, h("span", { class: "label" }, "Board size"),
          seg([["quick", "Quick: 5 columns, 4 rows"], ["full", "Full: 7 columns, 6 rows"]], cfg.c4, (v) => { cfg.c4 = v; paintCats(); })));
      }
    };
    const pg = h("input", { type: "checkbox", id: "party-pg13", checked: SETTINGS.pg13, onchange: () => paintCats() });
    paintEditor();
    paintCats();
    paintOpts();
    root.append(
      h("div", { class: "stack-sm" }, h("span", { class: "eyebrow" }, "Trial B"), h("h2", {}, "Party board"),
        h("p", { class: "lede" }, "For a room or a video call. Pick a square, everyone guesses what the mystery really is, and the closest guess takes the square.")),
      h("div", { class: "panel stack" },
        h("div", { class: "stack-sm" }, h("span", { class: "label" }, "Board"),
          seg([["points", "Points board"], ["ttt", "Tic-tac-toe"], ["c4", "Connect Four"]], cfg.style, (v) => {
            cfg.style = v;
            paintEditor();
            paintCats();
            paintOpts();
          })),
        edHost, catHost, optHost,
        h("div", { class: "stack-sm" }, h("span", { class: "label" }, "Guesses"),
          seg([["typed", "Type them on this device, passing it round"], ["paper", "Paper or out loud"]], cfg.entry, (v) => { cfg.entry = v; })),
        h("div", { class: "stack-sm" }, h("span", { class: "label" }, "Judge"),
          seg([["robot", "Free judge"], ["ai", "AI"], ["room", "The room decides"]], cfg.judge, (v) => { cfg.judge = v; }),
          h("p", { class: "small muted" }, "The free judge uses each mystery's answer key and word vectors. Judges need typed guesses; you can always overrule them by tapping a name.")),
        h("div", { class: "stack-sm" }, h("span", { class: "label" }, "Timer"),
          seg([[0, "None"], [30, "30 seconds"], [60, "60 seconds"]], cfg.timer, (v) => { cfg.timer = v; })),
        h("label", { class: "check", for: "party-pg13" }, pg, h("span", {}, "Include ", h("b", {}, "Double take"), " (PG-13 misreadings)")),
        h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => {
          SETTINGS.pg13 = pg.checked;
          const names = editor.get();
          if (teams(cfg.style)) save("party-teams", names);
          else SETTINGS.players = names;
          saveSettings();
          cfg.cats = catPick.get();
          if (!cfg.cats.length) { toast("Pick at least one category"); return; }
          save("party-cfg", cfg);
          start(cfg, names);
        } }, "Deal the board"))));
  }

  function check(id, label, on, onChange) {
    const box = h("input", { type: "checkbox", id, checked: on, onchange: (e) => onChange(e.target.checked) });
    return h("label", { class: "check", for: id }, box, h("span", {}, label));
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
    const cats = cfg.cats && cfg.cats.length ? cfg.cats : catsAvailable().slice(0, 6);
    P = { cfg, players: names.map((n) => ({ name: n, score: 0 })), turn: 0, control: 0, take };
    if (cfg.style === "points") {
      const rows = cfg.rows === 5 ? 5 : 3;
      const values = rows === 5 ? [200, 400, 600, 800, 1000] : [100, 200, 300];
      const diff = rows === 5 ? [1, 1, 2, 3, 3] : [1, 2, 3];
      P.cols = RAND.shuffle(cats.slice()).slice(0, 6).map((cat) => ({ cat, tiles: values.map((v, r) => ({ value: v, item: take(cat, diff[r]), done: false, by: [] })) }));
      if (cfg.dd) {
        const deep = P.cols.flatMap((c) => c.tiles.slice(1));
        deep[RAND.int(deep.length)].dd = true;
      }
      if (cfg.final) {
        const fc = cats[RAND.int(cats.length)];
        P.final = take(fc, 3);
      }
    } else if (cfg.style === "ttt") {
      const order = RAND.shuffle(cats.slice());
      P.cells = Array.from({ length: 9 }, (_, i) => {
        const cat = order[i % order.length];
        return { cat, item: take(cat, 1 + RAND.int(3)), mark: null, done: false };
      });
    } else {
      const { cols, rows } = C4_SIZE[cfg.c4] || C4_SIZE.quick;
      const order = RAND.shuffle(cats.slice());
      P.c4 = { cols, rows, grid: new Array(cols * rows).fill(null), colCats: Array.from({ length: cols }, (_, i) => order[i % order.length]) };
    }
    board();
  }

  function topValue() {
    return Math.max(...P.cols.flatMap((c) => c.tiles.map((t) => t.value)));
  }
  function open(col, t) {
    if (t.done) return false;
    if (!P.cfg.ladder) return true;
    return col.tiles.every((x) => x === t || x.value > t.value || x.done);
  }

  function scoresBar() {
    const top = Math.max(...P.players.map((p) => p.score));
    const two = P.cfg.style !== "points";
    const count = (i) => (P.cfg.style === "ttt" ? P.cells.filter((c) => c.mark === i).length : P.c4.grid.filter((m) => m === i).length);
    return h("div", { class: "scores" }, P.players.map((p, i) => h("div", { class: "sc" + (!two && p.score === top && top > 0 ? " lead" : "") + (two ? (i === 0 ? " team-x" : " team-o") : "") + (!two && P.control === i ? " control" : "") },
      h("span", { class: "small muted" }, two ? (i === 0 ? "X · " : "O · ") + p.name : p.name),
      h("b", { class: "tabular" }, two ? count(i) : p.score))));
  }

  function board() {
    clear(root);
    root.append(h("div", { class: "spread" }, h("span", { class: "eyebrow" }, "Trial B · Party board"),
      h("button", { class: "btn ghost small", onclick: () => { if (P) end(); } }, "End game")));
    root.append(scoresBar());
    if (P.cfg.style === "points") pointsBoard();
    else if (P.cfg.style === "ttt") tttBoard();
    else c4Board();
  }

  function pointsBoard() {
    if (P.cols.every((c) => c.tiles.every((t) => t.done))) {
      if (P.final && !P.finalDone) finalRound();
      else end();
      return;
    }
    root.append(h("p", { class: "label" }, `${P.players[P.control].name} has control: pick a square.`));
    const grid = h("div", { class: "board points", style: { "--cols": P.cols.length } });
    P.cols.forEach((c) => grid.append(h("div", { class: "head" }, CATS[c.cat].label)));
    const rows = P.cols[0].tiles.length;
    for (let row = 0; row < rows; row++) {
      P.cols.forEach((c) => {
        const t = c.tiles[row];
        const can = open(c, t);
        grid.append(h("button", { class: "tile" + (!t.done && !can ? " locked" : ""), disabled: !can, "aria-label": `${CATS[c.cat].label}, ${t.value}${!t.done && !can ? ", locked" : ""}`,
          onclick: () => (t.dd ? dailyDouble(c, t) : round({ kind: "points", tile: t, cat: c.cat, item: t.item })) },
          t.done ? (t.by.length ? P.players[t.by[0]].name.slice(0, 10) : "–") : t.value));
      });
    }
    root.append(grid);
    root.append(h("p", { class: "small muted" },
      P.cfg.ladder ? "Clear a column's lower values to unlock its higher ones. " : "Any square is open. ",
      "Whoever takes a square picks next. Bigger numbers are harder.",
      P.cfg.dd ? " One square hides a Daily Double." : ""));
  }

  function tttBoard() {
    const team = P.players[P.turn % 2];
    root.append(h("p", { class: "label" }, `${P.turn % 2 === 0 ? "X" : "O"} · ${team.name} picks a square. Both teams guess; the closer one claims it.`));
    const grid = h("div", { class: "board ttt" });
    P.cells.forEach((c) => grid.append(h("button", { class: "tile" + (c.mark === 0 ? " x" : c.mark === 1 ? " o" : ""), disabled: c.done, "aria-label": c.done ? "Taken" : CATS[c.cat].label, onclick: () => round({ kind: "ttt", cell: c, cat: c.cat, item: c.item }) },
      c.mark === 0 ? h("span", { class: "mark" }, "X") : c.mark === 1 ? h("span", { class: "mark" }, "O") : c.done ? "–" : CATS[c.cat].short)));
    root.append(grid);
  }

  function c4Board() {
    const { cols, rows, grid, colCats } = P.c4;
    const team = P.players[P.turn % 2];
    root.append(h("p", { class: "label" }, `${P.turn % 2 === 0 ? "X" : "O"} · ${team.name} picks a column. Both teams guess; the closer one drops a disc there.`));
    const el = h("div", { class: "board c4", style: { "--cols": cols } });
    for (let c = 0; c < cols; c++) {
      const full = grid[(rows - 1) * cols + c] !== null;
      el.append(h("button", { class: "drop", disabled: full, "aria-label": `Column ${c + 1}: ${CATS[colCats[c]].label}${full ? ", full" : ""}`,
        onclick: () => {
          const r = grid.findIndex((m, k) => k % cols === c && m === null);
          const height = Math.floor(r / cols);
          round({ kind: "c4", col: c, cat: colCats[c], item: P.take(colCats[c], Math.min(3, 1 + Math.floor((height * 3) / rows))) });
        } }, CATS[colCats[c]].short));
    }
    for (let r = rows - 1; r >= 0; r--) {
      for (let c = 0; c < cols; c++) {
        const m = grid[r * cols + c];
        el.append(h("span", { class: "slot" + (m === 0 ? " x" : m === 1 ? " o" : ""), "aria-label": m === 0 ? "X" : m === 1 ? "O" : "empty" }));
      }
    }
    root.append(el);
    root.append(h("p", { class: "small muted" }, "Four in a row across, up or diagonally wins. If neither team is closer, no disc drops."));
  }

  function c4Winner() {
    const { cols, rows, grid } = P.c4;
    const at = (c, r) => (c >= 0 && c < cols && r >= 0 && r < rows ? grid[r * cols + c] : null);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const m = at(c, r);
        if (m === null) continue;
        for (const [dc, dr] of [[1, 0], [0, 1], [1, 1], [1, -1]]) {
          if ([1, 2, 3].every((k) => at(c + dc * k, r + dr * k) === m)) return m;
        }
      }
    }
    return null;
  }

  /** A round: show the mystery, give clues, collect guesses, reveal, judge, award. */
  function round(sel) {
    const item = sel.item;
    const names = P.players.map((p) => p.name);
    let clues = 0;
    clear(root);
    const valueEl = h("span", { class: "chip" });
    const paintValue = () => {
      valueEl.textContent = sel.kind === "points" ? `Worth ${Math.round(sel.tile.value * CLUE_VALUE[clues])}` : sel.kind === "c4" ? "Wins the disc" : "Claims the square";
    };
    paintValue();
    const t = timerEl(P.cfg.timer, () => toast("Time's up"));
    const card = mysteryCard(item, valueEl);
    root.append(card);
    const clueBox = h("div", { class: "stack-sm" });
    const clueBtn = h("button", { class: "btn small", onclick: () => {
      if (clues >= (item.clues || []).length) return;
      clueBox.append(h("p", { class: "clue" }, h("b", {}, "Clue " + (clues + 1) + ": "), item.clues[clues]));
      clues++;
      zoomTo(card, item, clues);
      paintValue();
      if (clues >= (item.clues || []).length) clueBtn.disabled = true;
    } }, item.kind === "image" ? "Zoom out (clue)" : "Give a clue");
    const area = h("div", { class: "stack" });
    root.append(h("div", { class: "row" }, clueBtn, t.el), clueBox, area);

    const reveal = (guesses) => {
      t.stop();
      clear(area);
      zoomTo(card, item, 99);
      area.append(revealCard(item));
      const rating = funRating(item, "party");
      judgePanel(area, {
        round: { id: item.id, prompt: item.prompt, kindLabel: kindLabel(item), ask: item.ask, truth: item.truth, more: item.more },
        item, players: names, guesses, mode: "party", defaultJudge: P.cfg.entry === "typed" ? P.cfg.judge : "room",
        penalty: sel.kind === "points" && P.cfg.penalty,
        onAward: (winners, funniest, details) => {
          rating.flush();
          if (sel.kind === "points") {
            const pts = Math.round(sel.tile.value * CLUE_VALUE[clues]);
            winners.forEach((w) => { P.players[w].score += pts; });
            (details.penalized || []).forEach((w) => { P.players[w].score -= pts; });
            if (funniest != null) P.players[funniest].score += 50;
            sel.tile.done = true;
            sel.tile.by = winners;
            if (winners.length && !winners.includes(P.control)) P.control = winners[0];
          } else if (sel.kind === "ttt") {
            sel.cell.done = true;
            sel.cell.mark = winners.length === 1 ? winners[0] : null;
            P.turn++;
            const winner = LINES.find((l) => l.every((k) => P.cells[k].mark === 0)) ? 0 : LINES.find((l) => l.every((k) => P.cells[k].mark === 1)) ? 1 : null;
            if (winner != null || P.cells.every((c) => c.done)) { end(winner); return; }
          } else {
            const { cols, grid } = P.c4;
            if (winners.length === 1) {
              const k = grid.findIndex((m, i) => i % cols === sel.col && m === null);
              if (k >= 0) grid[k] = winners[0];
            }
            P.turn++;
            const winner = c4Winner();
            if (winner != null || grid.every((m) => m !== null)) { end(winner); return; }
          }
          board();
        },
      });
      area.append(rating);
    };
    collect(area, item, names, () => clues, reveal);
  }

  /** Guesses typed in private, passing the device round, or on paper. */
  function collect(area, item, names, level, then, label) {
    if (P.cfg.entry === "typed") {
      area.append(h("p", { class: "muted" }, label || "Everyone take a look. Then pass the device round so each player types a guess in private."),
        h("div", { class: "row" }, h("button", { class: "btn primary", onclick: async () => {
          const guesses = await collectPrivately(area, names, { item, level: level(), prompt: "what is it?", placeholder: "Your guess, in a few words", skippable: true });
          clear(area);
          area.append(h("p", { class: "label" }, "All guesses are in."), h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => then(guesses) }, "Reveal the answer")));
        } }, "Collect guesses")));
    } else {
      area.append(h("p", { class: "muted" }, "Everyone write a guess on paper (or keep it in your head). When you're ready, reveal."),
        h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => then(names.map(() => "")) }, "Reveal the answer")));
    }
  }

  function wagerInput(id, max, value) {
    return h("input", { class: "input", id, type: "number", inputmode: "numeric", min: 0, max, step: 50, value: String(value) });
  }

  function dailyDouble(col, tile) {
    const who = P.control;
    const me = P.players[who];
    const max = Math.max(me.score, topValue());
    clear(root);
    const input = wagerInput("dd-wager", max, Math.min(max, tile.value));
    const err = h("p", { class: "small", style: { color: "var(--bad)" } });
    root.append(h("div", { class: "panel stack dd" },
      h("span", { class: "eyebrow" }, CATS[col.cat].label),
      h("h2", { class: "scorebig" }, "Daily Double"),
      h("p", {}, `${me.name} answers alone. Wager up to ${max} (your score, or the board's top value if that's more). Clues cut what you can win, not what you can lose.`),
      h("div", { class: "field" }, h("label", { for: "dd-wager" }, `${me.name}'s wager`), input), err,
      h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => {
        const w = Math.round(Number(input.value));
        if (!(w >= 5 && w <= max)) { err.textContent = `Wager between 5 and ${max}.`; return; }
        play(w);
      } }, "Lock in the wager"))));
    input.focus();

    function play(wager) {
      const item = tile.item;
      let clues = 0;
      clear(root);
      const valueEl = h("span", { class: "chip" }, `Daily Double · ${wager}`);
      const card = mysteryCard(item, valueEl);
      const clueBox = h("div", { class: "stack-sm" });
      const clueBtn = h("button", { class: "btn small", onclick: () => {
        if (clues >= (item.clues || []).length) return;
        clueBox.append(h("p", { class: "clue" }, h("b", {}, "Clue " + (clues + 1) + ": "), item.clues[clues]));
        clues++;
        zoomTo(card, item, clues);
        valueEl.textContent = `Daily Double · win ${Math.round(wager * CLUE_VALUE[clues])}, lose ${wager}`;
        if (clues >= (item.clues || []).length) clueBtn.disabled = true;
      } }, item.kind === "image" ? "Zoom out (clue)" : "Give a clue");
      const area = h("div", { class: "stack" });
      const guessIn = h("input", { class: "input", id: "dd-guess", autocomplete: "off", maxlength: 160, placeholder: "Your guess, in a few words" });
      root.append(card, h("div", { class: "row" }, clueBtn), clueBox, area);
      const reveal = (guess) => {
        clear(area);
        zoomTo(card, item, 99);
        area.append(revealCard(item));
        const rating = funRating(item, "party", { dd: true });
        judgePanel(area, {
          round: { id: item.id, prompt: item.prompt, kindLabel: kindLabel(item), ask: item.ask, truth: item.truth, more: item.more },
          item, players: [me.name], guesses: [guess], mode: "party", defaultJudge: guess ? P.cfg.judge : "room", threshold: RIGHT,
          pickLabel: `Did ${me.name} get it? (${RIGHT} or more counts)`,
          onAward: (winners) => {
            rating.flush();
            me.score += winners.length ? Math.round(wager * CLUE_VALUE[clues]) : -wager;
            tile.done = true;
            tile.by = winners.length ? [who] : [];
            board();
          },
        });
        area.append(rating);
      };
      if (P.cfg.entry === "typed") {
        area.append(h("div", { class: "field" }, h("label", { for: "dd-guess" }, `${me.name}: what is it?`), guessIn),
          h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => reveal(guessIn.value.trim()) }, "Reveal the answer")));
        guessIn.focus();
      } else {
        area.append(h("p", { class: "muted" }, `${me.name}, say your answer out loud.`),
          h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => reveal("") }, "Reveal the answer")));
      }
    }
  }

  function finalRound() {
    const item = P.final;
    const names = P.players.map((p) => p.name);
    const maxes = P.players.map((p) => Math.max(0, p.score));
    clear(root);
    root.append(h("span", { class: "eyebrow" }, "Trial B · Final mystery"), scoresBar(),
      h("div", { class: "panel stack" },
        h("h2", {}, "Final mystery: ", CATS[item.cat].label),
        h("p", {}, "Everyone wagers up to their score (nothing if it's zero or less), knowing only the category. Then everyone guesses. ",
          `${RIGHT} or more counts as right: win your wager; otherwise lose it.`)));
    const area = h("div", { class: "stack" });
    root.append(area);
    const afterWagers = (wagers) => {
      clear(area);
      const card = mysteryCard(item);
      area.append(card);
      const guessArea = h("div", { class: "stack" });
      area.append(guessArea);
      collect(guessArea, item, names, () => 0, (guesses) => {
        clear(guessArea);
        zoomTo(card, item, 99);
        guessArea.append(revealCard(item));
        judgePanel(guessArea, {
          round: { id: item.id, prompt: item.prompt, kindLabel: kindLabel(item), ask: item.ask, truth: item.truth, more: item.more },
          item, players: names, guesses, mode: "party", defaultJudge: P.cfg.entry === "typed" ? P.cfg.judge : "room", threshold: RIGHT,
          pickLabel: `Who got it? (${RIGHT} or more counts)`,
          onAward: (winners) => {
            if (wagers) applyWagers(wagers, winners);
            else paperWagers(winners);
          },
        });
      }, "Now everyone sees the mystery. Pass the device round so each player types a guess in private.");
    };
    const applyWagers = (wagers, winners) => {
      P.players.forEach((p, i) => { p.score += winners.includes(i) ? wagers[i] : -wagers[i]; });
      P.finalDone = true;
      end();
    };
    // Paper games: wagers were written down; type them in after the reveal.
    const paperWagers = (winners) => {
      clear(area);
      const inputs = P.players.map((p, i) => wagerInput("final-w-" + i, maxes[i], 0));
      area.append(h("div", { class: "panel stack" }, h("p", { class: "label" }, "Enter the wagers people wrote down"),
        P.players.map((p, i) => h("div", { class: "field" }, h("label", { for: "final-w-" + i }, `${p.name} (up to ${maxes[i]})${winners.includes(i) ? ": right" : ": wrong"}`), inputs[i])),
        h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => {
          applyWagers(inputs.map((el, i) => Math.max(0, Math.min(maxes[i], Math.round(Number(el.value) || 0)))), winners);
        } }, "Settle up"))));
    };
    if (P.cfg.entry === "typed") {
      area.append(h("div", { class: "row" }, h("button", { class: "btn primary", onclick: async () => {
        const typed = await collectPrivately(area, names, {
          prompt: "your wager", placeholder: "A number",
          check: (text, i) => {
            const w = Number(text);
            return Number.isFinite(w) && w >= 0 && w <= maxes[i] ? "" : `Wager between 0 and ${maxes[i]}.`;
          },
        });
        afterWagers(typed.map((t) => Math.round(Number(t) || 0)));
      } }, "Collect wagers")));
    } else {
      area.append(h("p", { class: "muted" }, "Everyone write a wager on paper."),
        h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => afterWagers(null) }, "Show the mystery")));
    }
  }

  function end(teamWinner) {
    clear(root);
    let headline;
    if (P.cfg.style !== "points") {
      const count = (i) => (P.cfg.style === "ttt" ? P.cells.filter((c) => c.mark === i).length : P.c4.grid.filter((m) => m === i).length);
      const line = P.cfg.style === "ttt" ? "three in a row" : "four in a row";
      headline = teamWinner != null ? `${P.players[teamWinner].name} got ${line}` : (() => {
        const x = count(0);
        const o = count(1);
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

/** Toggle chips with a limit. Returns {el, get()}; get() keeps the options' order. */
function chipPicker(options, initial, max) {
  const on = new Set(initial);
  const wrap = h("div", { class: "seg", role: "group" });
  const paint = () => wrap.querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-pressed", String(on.has(options[i][0]))));
  for (const [value, label] of options) {
    wrap.append(h("button", { type: "button", onclick: () => {
      if (on.has(value)) on.delete(value);
      else if (on.size < max) on.add(value);
      else toast(`Up to ${max}`);
      paint();
    } }, label));
  }
  paint();
  return { el: wrap, get: () => options.map(([v]) => v).filter((v) => on.has(v)) };
}

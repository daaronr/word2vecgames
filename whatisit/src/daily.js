/* Trial A: Daily five. Solo; five mysteries a day, one per category. Type guesses and the robot
   judge says how warm each one is; clues and "pick from four" are there when stuck, at a cost. */

const CLUE_MULT = [1, 0.85, 0.7, 0.55];
const MAX_GUESSES = 4;
const { dayNumber } = window.WhatPick;

function todayStr() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}
function dailyItems(dateStr) {
  return window.WhatPick.dailyItems(CONTENT, dateStr);
}

function practiceItems() {
  const cats = catsAvailable().filter((c) => c !== "double" || SETTINGS.pg13);
  return RAND.shuffle(cats.slice()).slice(0, 5).map((c) => {
    const p = pool(c);
    return p[RAND.int(p.length)];
  });
}

function newRound() {
  return { guesses: [], clues: 0, mc: null, verdict: null, revealed: false, points: 0, ai: null };
}

function roundPoints(item, r) {
  const best = r.guesses.reduce((m, g) => Math.max(m, g.s), 0);
  let base = best;
  if (r.mc && r.mc.pick != null) base = r.mc.right ? Math.max(best, 50) : Math.round(best / 2);
  let pts = Math.round(base * CLUE_MULT[r.clues]);
  if (item.kind === "plate" && r.verdict && item.dmv && r.verdict === item.dmv) pts += 10;
  return pts;
}
function pointsWhy(item, r) {
  const best = r.guesses.reduce((m, g) => Math.max(m, g.s), 0);
  const bits = [];
  if (r.mc && r.mc.pick != null) bits.push(r.mc.right ? "picked the right one of four (50)" : "picked a wrong one of four (half your best guess)");
  else bits.push(r.guesses.length ? "best guess " + best : "no guess");
  if (r.clues) bits.push(`× ${CLUE_MULT[r.clues]} for ${r.clues} clue${r.clues > 1 ? "s" : ""}`);
  if (item.kind === "plate" && r.verdict) bits.push(r.verdict === item.dmv ? "+10 for calling the DMV verdict" : "DMV verdict missed");
  return bits.join(", ");
}
function emojiFor(points) {
  return points >= 70 ? "🟩" : points >= 45 ? "🟨" : points >= 20 ? "🟧" : "⬛";
}

function dailyScreen(opts = {}) {
  const practice = !!opts.practice;
  const date = todayStr();
  const items = practice ? practiceItems() : dailyItems(date);
  const key = "daily:" + date;
  const st = (!practice && load(key, null)) || { date, i: 0, rounds: items.map(newRound) };
  const persist = () => { if (!practice) save(key, st); };
  const root = h("div", { class: "stack" });
  let rating = null;
  let robot = "loading";
  loadRobot().then((s) => { robot = s ? "ready" : "failed"; if (!st.rounds[st.i] || !st.rounds[st.i].revealed) render(); });
  render();
  return root;

  function header() {
    return h("div", { class: "spread" },
      h("div", { class: "stack-sm" },
        h("span", { class: "eyebrow" }, practice ? "Trial A · Practice five" : `Trial A · Daily No. ${dayNumber(date)}`),
        h("div", { class: "pips", "aria-label": `Mystery ${Math.min(st.i + 1, 5)} of 5` },
          items.map((_, i) => h("i", { class: i < st.i ? "done" : i === st.i ? "now" : "" })))),
      h("span", { class: "tabular muted small" }, st.rounds.reduce((s, r) => s + (r.revealed ? r.points : 0), 0), " pts"));
  }

  function render() {
    clear(root);
    root.append(header());
    if (st.i >= items.length) { renderEnd(); return; }
    const item = items[st.i];
    const r = st.rounds[st.i];
    root.append(mysteryCard(item, h("span", { class: "num" }, `${st.i + 1}/5`)));
    if (r.revealed) renderReveal(item, r);
    else renderPlay(item, r);
    window.scrollTo({ top: 0 });
  }

  function guessLog(r, withChips) {
    if (!r.guesses.length) return null;
    return h("div", { class: "guesslog" }, r.guesses.map((g, i) => {
      const w = J.warmth(g.s);
      const last = i === r.guesses.length - 1;
      return h("div", { class: "g" },
        h("span", { class: "t" }, g.t),
        h("span", { class: "s lv" + w.level }, g.s),
        last && withChips && g.res ? h("div", { class: "why" }, warmthMeter(g.s)) : null,
        last && withChips && g.res ? h("div", { class: "why" }, matchChips(g.res)) : null,
        g.trap ? h("div", { class: "why trapnote small" }, "That's close to a tempting wrong answer: “" + g.trap + "”") : null);
    }));
  }

  function renderPlay(item, r) {
    const left = MAX_GUESSES - r.guesses.length;
    const input = h("input", { class: "input", id: "daily-guess", autocomplete: "off", maxlength: 160, placeholder: r.guesses.length ? "Another guess" : "Your guess, in a few words" });
    const go = async () => {
      const text = input.value.trim();
      if (!text) return;
      const sp = await loadRobot();
      if (!sp) { toast("The robot judge didn't load. Use “Pick from four” instead."); return; }
      const res = robotScore(text, item);
      r.guesses.push({ t: text, s: res.score, trap: res.trap ? res.trap.decoy : null, res: { matches: res.matches, unknown: res.unknown } });
      if (res.score >= 85 || r.guesses.length >= MAX_GUESSES) finish(item, r);
      persist();
      render();
      const again = $("#daily-guess");
      if (again) again.focus();
    };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });

    const log = guessLog(r, true);
    if (log) root.append(log);
    if (r.clues) root.append(h("div", { class: "stack-sm" }, item.clues.slice(0, r.clues).map((c, i) => h("p", { class: "clue" }, h("b", {}, "Clue " + (i + 1) + ": "), c))));

    if (r.mc) {
      root.append(h("p", { class: "label" }, "Which one is it?"),
        h("div", { class: "choices" }, r.mc.opts.map((o, i) => h("button", { class: "choice", onclick: () => {
          r.mc.pick = i;
          r.mc.right = o === item.truth;
          finish(item, r);
          persist();
          render();
        } }, o))));
    } else {
      root.append(h("div", { class: "guessrow" }, input, h("button", { class: "btn primary", onclick: go, disabled: left <= 0 }, "Guess")));
      root.append(h("p", { class: "small muted" },
        robot === "loading" ? "Warming up the robot judge… " : robot === "failed" ? "The robot judge couldn't load here; pick from four instead. " : "",
        `${left} guess${left === 1 ? "" : "es"} left. A score of 85 or more counts as spot on.`));
    }

    if (item.kind === "plate" && item.dmv) {
      root.append(h("div", { class: "stack-sm" },
        h("span", { class: "label" }, "Side bet (+10): did the DMV approve it?"),
        seg([["approved", "Approved"], ["denied", "Denied"]], r.verdict, (v) => { r.verdict = v; persist(); })));
    }

    const clueBtn = h("button", { class: "btn small", disabled: r.clues >= Math.min(3, (item.clues || []).length) || !!r.mc, onclick: () => { r.clues++; persist(); render(); } },
      r.clues >= (item.clues || []).length ? "No more clues" : `Clue (×${CLUE_MULT[r.clues + 1] || CLUE_MULT[3]})`);
    const mcBtn = h("button", { class: "btn small", disabled: !!r.mc || !(item.decoys && item.decoys.length), onclick: () => {
      const opts2 = rng("mc-" + item.id).shuffle([item.truth, ...item.decoys.slice(0, 3)]);
      r.mc = { opts: opts2, pick: null, right: null };
      persist();
      render();
    } }, "Pick from four");
    const giveUp = h("button", { class: "btn ghost small", onclick: () => { finish(item, r); persist(); render(); } }, "Show me");
    root.append(h("div", { class: "row" }, clueBtn, mcBtn, giveUp));
    root.append(howJudging());
    if (!r.mc) input.focus();
  }

  function finish(item, r) {
    r.revealed = true;
    r.points = roundPoints(item, r);
  }

  function renderReveal(item, r) {
    const extra = [];
    if (item.kind === "plate" && r.verdict) {
      extra.push(h("p", { class: "small" }, `You said ${r.verdict}: `, r.verdict === item.dmv ? h("b", {}, "right (+10)") : "not this time"));
    }
    if (r.mc && r.mc.pick != null) {
      extra.push(h("p", { class: "small" }, r.mc.right ? "You picked the real one." : `You picked: “${r.mc.opts[r.mc.pick]}”`));
    }
    root.append(revealCard(item, { extra: extra.length ? h("div", { class: "stack-sm" }, extra) : null }));
    root.append(h("div", { class: "spread" },
      h("span", { class: "scorebig tabular" }, "+" + r.points),
      h("span", { class: "small muted", style: { maxWidth: "40ch", textAlign: "right" } }, pointsWhy(item, r))));
    const log = guessLog(r, false);
    if (log) root.append(log);

    if (r.guesses.length) {
      const aiBox = h("div", { class: "stack-sm" });
      if (r.ai) aiBox.append(aiCompare(r));
      else {
        const btn = h("button", { class: "btn small", onclick: async () => {
          btn.disabled = true;
          btn.textContent = "Asking…";
          try {
            const out = await aiJudge(
              { prompt: item.prompt, kindLabel: kindLabel(item), ask: item.ask, truth: item.truth, more: item.more },
              r.guesses.map((g, i) => ({ name: "Guess " + (i + 1), text: g.t })));
            r.ai = { source: out.source, scores: out.scores.map((s) => (s ? s.score : null)), why: out.scores.map((s) => (s ? s.why : "")), comment: out.comment };
            persist();
            Store.send("judge", { id: item.id, prompt: item.prompt, truth: item.truth, mode: "daily", guesses: r.guesses.map((g) => g.t), robot: r.guesses.map((g) => g.s), ai: r.ai.scores, source: out.source });
            clear(aiBox).append(aiCompare(r));
          } catch (e) {
            btn.disabled = false;
            btn.textContent = "Ask the AI judge for a second opinion";
            if (e.message !== "cancelled") toast(e.message);
          }
        } }, "Ask the AI judge for a second opinion");
        aiBox.append(h("div", { class: "row" }, btn, h("span", { class: "small muted" }, "Uses " + aiSourceLabel(aiSource()) + ". Compare it with the robot.")));
      }
      root.append(aiBox);
    }

    rating = funRating(item, practice ? "practice" : "daily", { points: r.points, guesses: r.guesses.length, clues: r.clues, mc: !!r.mc });
    root.append(rating);
    root.append(h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => {
      if (rating) rating.flush();
      st.i++;
      persist();
      render();
    } }, st.i < items.length - 1 ? "Next mystery" : "See your score")));
  }

  function aiCompare(r) {
    return h("div", { class: "panel stack-sm" },
      h("span", { class: "label" }, "Robot vs AI judge (" + aiSourceLabel(r.ai.source) + ")"),
      r.guesses.map((g, i) => h("div", { class: "spread small" },
        h("span", {}, "“", g.t, "”"),
        h("span", { class: "tabular" }, "robot ", h("b", {}, g.s), " · AI ", h("b", {}, r.ai.scores[i] == null ? "–" : r.ai.scores[i])),
        r.ai.why[i] ? h("span", { class: "muted", style: { flexBasis: "100%" } }, r.ai.why[i]) : null)),
      r.ai.comment ? h("p", { class: "small muted" }, r.ai.comment) : null);
  }

  function renderEnd() {
    const total = st.rounds.reduce((s, r) => s + r.points, 0);
    const grid = st.rounds.map((r) => emojiFor(r.points)).join("");
    const share = `What Is It? ${practice ? "practice" : "No. " + dayNumber(date)}\n${grid} ${total}/500${CONFIG.shareUrl ? "\n" + CONFIG.shareUrl : ""}`;
    root.append(h("div", { class: "panel stack" },
      h("span", { class: "eyebrow" }, practice ? "Practice done" : "Today's five, done"),
      h("div", { class: "spread" }, h("span", { class: "scorebig tabular" }, total, " / 500"), h("span", { class: "grid-emoji", "aria-label": "Results grid" }, grid)),
      h("div", { class: "row" },
        h("button", { class: "btn primary", onclick: () => copyText(share, "Result copied") }, "Copy result"),
        h("button", { class: "btn", onclick: () => go("practice") }, "Practice five more"))));
    root.append(h("div", { class: "stack-sm" }, items.map((it, i) => h("div", { class: "panel", style: { padding: "12px 14px" } },
      h("div", { class: "spread" }, h("span", { class: "eyebrow" }, CATS[it.cat].short), h("span", { class: "tabular small" }, emojiFor(st.rounds[i].points), " ", st.rounds[i].points)),
      h("div", {}, h("b", {}, it.prompt), " — ", it.truth)))));
    root.append(trialRating("A", "Rate Trial A: Daily five"));
  }
}

function howJudging() {
  return h("details", { class: "how" },
    h("summary", {}, "How the robot judge scores"),
    h("div", { class: "prose small" },
      h("p", {}, "Each mystery has a few key ideas (for Liquid Death: water, canned, drink). The robot compares the words of your guess with those ideas using ConceptNet Numberbatch word vectors, the same “common sense” word map as Word Bocce, so near-synonyms count: “store” lands close to “shop”."),
      h("p", {}, "It reads your first eight words and ignores filler like “a website where”. 85 or more is spot on. It is quick and free, and sometimes too generous with close cousins (shoes for a sock company). After the reveal you can ask an AI judge for a second opinion and compare.")));
}

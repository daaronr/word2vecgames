// The AI judge behind /api/judge, kept free of Netlify and network code so tests can run it with a
// fake model. The function (netlify/functions/judge.mjs) supplies `complete(prompt) -> text`.
import J from "../../src/judge.js";
import ITEMS from "./items.mjs";

const MAX_GUESSES = 10;
const MAX_GUESS = 160;
const MAX_TRUTH = 500;

export class BadRequest extends Error {}

const clean = (s, n) => String(s == null ? "" : s).replace(/[\u0000-\u001f]/g, " ").trim().slice(0, n);

/** How a mystery is described to the judge, e.g. 'a line from a song ("Yankee Doodle")'. */
export function kindLabel(it) {
  const label = J.KIND_LABEL[it.kind] || "";
  return it.song ? `${label} ("${it.song}")` : label;
}

/**
 * body: {mode: "score"|"judge", id?, custom?: {prompt, truth, kind}, guesses: [{name, text}] | [text]}
 * Curated items are looked up by id, so the function only ever judges this game's answers.
 */
export function buildRequest(body) {
  if (!body || typeof body !== "object") throw new BadRequest("Send JSON");
  const mode = body.mode === "score" ? "score" : "judge";
  let round;
  let id = null;
  if (body.id) {
    const it = ITEMS[String(body.id)];
    if (!it) throw new BadRequest("Unknown mystery");
    id = String(body.id);
    round = { prompt: it.prompt, kindLabel: kindLabel(it), ask: it.ask, truth: it.truth, more: it.more, graded: it.graded || null };
  } else if (body.custom) {
    const c = body.custom;
    const truth = clean(c.truth, MAX_TRUTH);
    if (!truth) throw new BadRequest("Say what you found");
    round = {
      prompt: clean(c.prompt, 120),
      kindLabel: c.kind === "domain" ? "a web address" : "a search phrase",
      ask: c.kind === "domain" ? "What is this website?" : "What comes up first when you search this?",
      truth,
      results: Array.isArray(c.results) ? c.results.slice(0, 3).map((r) => clean(r, 200)).filter(Boolean) : null,
    };
  } else throw new BadRequest("Name a mystery");
  const guesses = (Array.isArray(body.guesses) ? body.guesses : [])
    .slice(0, MAX_GUESSES)
    .map((g, i) => (typeof g === "string" ? { name: "Player " + (i + 1), text: g } : { name: clean(g && g.name, 30) || "Player " + (i + 1), text: g && g.text }))
    .map((g) => ({ name: g.name, text: clean(g.text, MAX_GUESS) }))
    .filter((g) => g.text);
  if (!guesses.length) throw new BadRequest("No guesses");
  if (mode === "score" && guesses.length !== 1) throw new BadRequest("Score one guess at a time");
  const prompt = mode === "score" ? J.scorePrompt(round, guesses[0].text) : J.judgePrompt(round, guesses);
  return { mode, id, round, guesses, prompt };
}

/** A single guess that restates one in the item's answer key gets that score, free. */
export function answerKey(req) {
  if (req.mode !== "score" || !req.round.graded) return null;
  const n = J.normGuess(req.guesses[0].text);
  const hit = n && req.round.graded.find((e) => J.normGuess(e.g) === n);
  return hit ? { score: hit.s, hint: hit.h || "", by: "key" } : null;
}

/** Cache key text: one guess at a curated item is cached by its normalized wording. */
export function cacheText(req) {
  if (req.mode === "score" && req.id) return `score\n${req.id}\n${J.normGuess(req.guesses[0].text)}`;
  return req.prompt;
}

export async function judge(body, complete) {
  const req = buildRequest(body);
  const free = answerKey(req);
  if (free) return free;
  const text = await complete(req.prompt, req.mode);
  if (req.mode === "score") return J.parseScoreReply(text);
  const out = J.parseJudgeReply(text, req.guesses.length);
  return { scores: out.scores, funniest: out.funniest, comment: out.comment };
}

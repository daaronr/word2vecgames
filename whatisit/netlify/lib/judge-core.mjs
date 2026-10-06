// The AI judge behind /api/judge, kept free of Netlify and network code so tests can run it with a
// fake model. The function (netlify/functions/judge.mjs) supplies `complete(prompt) -> text`.
import J from "../../src/judge.js";
import ITEMS from "./items.mjs";

const KIND = { web: "a web address", double: "a web address", brand: "a brand name", plate: "a California vanity plate", patent: "a US patent title", search: "a search phrase" };
const MAX_GUESSES = 10;
const MAX_GUESS = 160;
const MAX_TRUTH = 500;

export class BadRequest extends Error {}

const clean = (s, n) => String(s == null ? "" : s).replace(/[\u0000-\u001f]/g, " ").trim().slice(0, n);

/**
 * body: {mode: "score"|"judge", id?, custom?: {prompt, truth, kind}, guesses: [{name, text}] | [text]}
 * Curated items are looked up by id, so the function only ever judges this game's answers.
 */
export function buildRequest(body) {
  if (!body || typeof body !== "object") throw new BadRequest("Send JSON");
  const mode = body.mode === "score" ? "score" : "judge";
  let round;
  if (body.id) {
    const it = ITEMS[String(body.id)];
    if (!it) throw new BadRequest("Unknown mystery");
    round = { prompt: it.prompt, kindLabel: KIND[it.cat] || "", ask: it.ask, truth: it.truth, more: it.more };
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
  return { mode, round, guesses, prompt };
}

export async function judge(body, complete) {
  const req = buildRequest(body);
  const text = await complete(req.prompt, req.mode);
  if (req.mode === "score") return J.parseScoreReply(text);
  const out = J.parseJudgeReply(text, req.guesses.length);
  return { scores: out.scores, funniest: out.funniest, comment: out.comment };
}

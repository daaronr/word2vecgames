// Run real models on the hand-scored guesses, through Netlify AI Gateway (billed to the team's
// Netlify credits), and write their scores to eval/results/ for score_judges.mjs.
//
//   node --dns-result-order=ipv4first whatisit/eval/run_models.mjs [label ...] [--smoke]
//
// (IPv4 because the gateway token is tied to the IP that asked for it.) --smoke: one guess per model.
// Needs a Netlify CLI login with access to the whatisit-game site. Each model is run two ways, as
// the game uses it: "score" (one guess at a time, as in the Daily; answer key skipped so the model
// is tested on every guess) and "judge" (a whole round at once, as in party games). Prints tokens and
// Netlify credits used.
import { readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { buildRequest } from "../netlify/lib/judge-core.mjs";
import { complete, credits } from "../netlify/lib/providers.mjs";
import J from "../src/judge.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const SITE = "4750cbe3-06c3-4f50-9a57-63bc192f1454"; // whatisit-game
// File names must not end in -<digits> (score_judges.mjs merges name-1.json, name-2.json ...).
const MODELS = {
  gem25lite: ["gemini", "gemini-2.5-flash-lite"],
  gem31lite: ["gemini", "gemini-3.1-flash-lite"],
  gpt5nano: ["openai", "gpt-5-nano"],
  haiku45: ["anthropic", "claude-haiku-4-5"],
};
const ENV = { gemini: ["GEMINI_API_KEY", "GOOGLE_GEMINI_BASE_URL"], openai: ["OPENAI_API_KEY", "OPENAI_BASE_URL"], anthropic: ["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"] };

async function gateway() {
  const cfg = JSON.parse(readFileSync(join(homedir(), "Library/Preferences/netlify/config.json"), "utf8"));
  const token = cfg.users[cfg.userId].auth.token;
  const res = await fetch(`https://api.netlify.com/api/v1/sites/${SITE}/ai-gateway/token`, { headers: { Authorization: `Bearer ${token}` } });
  if (!res.ok) throw new Error(`AI Gateway token: HTTP ${res.status}`);
  return res.json(); // {token, url}
}

async function pool(tasks, n) {
  const out = new Array(tasks.length);
  let next = 0;
  await Promise.all(Array.from({ length: n }, async () => {
    while (next < tasks.length) { const i = next++; out[i] = await tasks[i](); }
  }));
  return out;
}

const SMOKE = process.argv.includes("--smoke");
const args = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const labels = args.length ? args : Object.keys(MODELS);
let rounds = JSON.parse(readFileSync(join(HERE, "judge-eval.json"), "utf8")).rounds;
if (SMOKE) rounds = [{ ...rounds[0], guesses: rounds[0].guesses.slice(0, 2) }];
const gw = await gateway();

for (const label of labels) {
  const [provider, model] = MODELS[label];
  // Point the SDK at the gateway, the way Netlify does inside a function. Base URLs per provider.
  const [keyVar, urlVar] = ENV[provider];
  process.env[keyVar] = gw.token;
  // The gateway routes by the provider's own API paths (OpenAI's SDK expects the /v1 in its base).
  process.env[urlVar] = gw.url.replace(/\/$/, "") + (provider === "openai" ? "/v1" : "");
  let input = 0, output = 0, fails = 0;
  const call = async (prompt, mode) => {
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const r = await complete(provider, model, prompt, mode);
        input += r.usage.input; output += r.usage.output;
        return r.text;
      } catch (e) {
        if (attempt === 2) { fails++; console.error(`  ${label} failed:`, (e && e.status) || "", String((e && e.message) || e).slice(0, 200)); return null; }
        await new Promise((s) => setTimeout(s, 1500 * (attempt + 1)));
      }
    }
  };
  const t0 = Date.now();
  // Score mode: one call per guess.
  const scoreTasks = [];
  for (const r of rounds) r.guesses.forEach((g, i) => scoreTasks.push(async () => {
    const text = await call(buildRequest({ mode: "score", id: r.id, guesses: [g.text] }).prompt, "score");
    let s = null;
    try { s = text == null ? null : J.parseScoreReply(text).score; } catch { fails++; }
    return [r.id, i, s];
  }));
  const scored = await pool(scoreTasks, 8);
  const byRound = {};
  for (const [id, i, s] of scored) (byRound[id] = byRound[id] || [])[i] = s;
  if (SMOKE) console.log("  sample:", JSON.stringify(byRound));
  else writeFileSync(join(HERE, "results", `api_${label}_score.json`), JSON.stringify(byRound, null, 1));
  const scoreCalls = scoreTasks.length;
  const scoreIn = input, scoreOut = output;
  // Judge mode: one call per round.
  const judged = await pool(rounds.map((r) => async () => [r.id, await call(buildRequest({ mode: "judge", id: r.id, guesses: r.guesses.map((g) => g.text) }).prompt, "judge")]), 6);
  if (SMOKE) console.log("  sample:", String(judged[0][1]).slice(0, 200));
  else writeFileSync(join(HERE, "results", `api_${label}_round.json`), JSON.stringify(Object.fromEntries(judged.filter(([, t]) => t != null)), null, 1));
  const c = (i, o) => credits(model, { input: i, output: o });
  console.log(`${label} (${model}): ${((Date.now() - t0) / 1000).toFixed(0)}s, ${fails} failures`);
  console.log(`  score mode: ${scoreCalls} calls, avg ${(scoreIn / scoreCalls).toFixed(0)} in / ${(scoreOut / scoreCalls).toFixed(0)} out tokens, ${(c(scoreIn, scoreOut) / scoreCalls).toFixed(3)} credits per call`);
  console.log(`  judge mode: ${rounds.length} calls, ${(c(input - scoreIn, output - scoreOut) / rounds.length).toFixed(3)} credits per call`);
  console.log(`  total ${c(input, output).toFixed(1)} credits`);
}

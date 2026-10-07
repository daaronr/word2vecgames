// Run: node whatisit/tests/judge-function.test.mjs   (the /api/judge logic, with a fake model)
import assert from "node:assert";
import { BadRequest, buildRequest, judge } from "../netlify/lib/judge-core.mjs";

// Curated mysteries are looked up by id; the truth never comes from the browser.
const r = buildRequest({ mode: "score", id: "kewpie", guesses: ["baby dolls"] });
assert.strictEqual(r.mode, "score");
assert(r.prompt.includes("Mayonnaise") && r.prompt.includes('"baby dolls"'));
assert.throws(() => buildRequest({ id: "no-such-item", guesses: ["x"] }), BadRequest);
assert.throws(() => buildRequest({ mode: "score", id: "kewpie", guesses: ["a", "b"] }), BadRequest);
assert.throws(() => buildRequest({ id: "kewpie", guesses: [] }), BadRequest);
assert.throws(() => buildRequest({ custom: { prompt: "x" }, guesses: ["y"] }), BadRequest);

// Long input is cut down; at most 10 guesses.
const big = buildRequest({ id: "kewpie", guesses: Array.from({ length: 30 }, () => "z".repeat(500)) });
assert.strictEqual(big.guesses.length, 10);
assert.strictEqual(big.guesses[0].text.length, 160);

// Bring-your-own rounds carry their own answer.
const byo = buildRequest({ custom: { prompt: "kafka.com", truth: "A PR agency in Orlando", kind: "domain", results: ["x", "y"] }, guesses: [{ name: "Ann", text: "a PR firm" }] });
assert(byo.prompt.includes("A PR agency in Orlando") && byo.prompt.includes("Ann"));

// Replies are parsed whatever wrapping the model adds.
const one = await judge({ mode: "score", id: "kewpie", guesses: ["mayo"] }, async () => 'Here: {"score": 97, "hint": "Spot on."}');
assert.deepStrictEqual(one, { score: 97, hint: "Spot on." });
const many = await judge({ id: "kewpie", guesses: ["mayo", "dolls"] }, async () =>
  '```json\n{"scores":[{"n":1,"score":95,"why":"yes"},{"n":2,"score":5,"why":"no"}],"funniest":2,"comment":"ok"}\n```');
assert.strictEqual(many.scores[1].score, 5);
assert.strictEqual(many.funniest, 1);
await assert.rejects(judge({ mode: "score", id: "kewpie", guesses: ["mayo"] }, async () => "no idea"));
// Answer keys: a guess that restates a graded one is scored without calling the model.
import ITEMS from "../netlify/lib/items.mjs";
import { answerKey, cacheText, kindLabel } from "../netlify/lib/judge-core.mjs";
const [kid, kit] = Object.entries(ITEMS).find(([, it]) => it.graded && it.graded.length) || [];
if (kid) {
  const e = kit.graded[0];
  const free = await judge({ mode: "score", id: kid, guesses: [" " + e.g.toUpperCase() + "."] }, async () => { throw new Error("the model should not be called"); });
  assert.deepStrictEqual(free, { score: e.s, hint: e.h, by: "key" });
  assert.strictEqual(answerKey(buildRequest({ mode: "score", id: kid, guesses: ["zzz qqq unlikely"] })), null);
}
// One guess at a curated item is cached by its wording, whatever the player's name or punctuation.
assert.strictEqual(cacheText(buildRequest({ mode: "score", id: "kewpie", guesses: ["Mayo!"] })), cacheText(buildRequest({ mode: "score", id: "kewpie", guesses: [{ name: "Bo", text: "mayo" }] })));
assert.strictEqual(kindLabel({ kind: "lyric", song: "Yankee Doodle" }), 'a line from a song ("Yankee Doodle")');

// The function itself: with no key of your own (or only Netlify's gateway key) the AI judge rests.
for (const k of ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "JUDGE_PROVIDER"]) delete process.env[k];
process.env.GEMINI_API_KEY = "netlify-gateway-key";
process.env.GOOGLE_GEMINI_BASE_URL = "https://example.netlify.app/.netlify/ai";
const fn = (await import("../netlify/functions/judge.mjs?gateway")).default;
const post = (body) => fn(new Request("http://x/api/judge", { method: "POST", body: JSON.stringify(body) }), { ip: "1.2.3.4" });
let res = await post({ mode: "score", id: "kewpie", guesses: ["something new and odd"] });
assert.strictEqual(res.status, 503);
assert.strictEqual((await res.json()).reason, "off");
if (kid) {
  res = await post({ mode: "score", id: kid, guesses: [kit.graded[0].g] });
  assert.strictEqual(res.status, 200, "answer-key hits work even with the AI switched off");
}
assert.strictEqual((await post({ id: "no-such-item", guesses: ["x"] })).status, 400);

// Tester passes: with JUDGE_PASSCODE set, a request without the pass is turned away before any cost.
delete process.env.GOOGLE_GEMINI_BASE_URL;
process.env.GEMINI_API_KEY = "own-key";
process.env.JUDGE_PASSCODE = "family, friends";
const fn2 = (await import("../netlify/functions/judge.mjs?pass")).default;
const post2 = (body, pass) => fn2(new Request("http://x/api/judge", { method: "POST", body: JSON.stringify(body), headers: pass ? { "X-Judge-Pass": pass } : {} }), { ip: "1.2.3.4" });
res = await post2({ mode: "score", id: "kewpie", guesses: ["something new and odd"] });
assert.strictEqual(res.status, 403);
assert.strictEqual((await res.json()).reason, "pass");
res = await post2({ mode: "score", id: "kewpie", guesses: ["something new and odd"] }, "wrong");
assert.strictEqual(res.status, 403);
if (kid) assert.strictEqual((await post2({ mode: "score", id: kid, guesses: [kit.graded[0].g] })).status, 200, "answer-key hits need no pass");
console.log("judge function: all checks passed");

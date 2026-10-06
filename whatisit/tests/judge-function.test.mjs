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
console.log("judge function: all checks passed");

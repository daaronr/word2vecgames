// Run: node whatisit/tests/whatisit.test.js   (content, robot judge and Daily checks for "What Is It?")
const fs = require("fs");
const path = require("path");
const assert = require("assert");
const J = require("../src/judge.js");
const Pick = require("../src/pick.js");

const ROOT = path.join(__dirname, "..");
const content = [];
for (const f of fs.readdirSync(path.join(ROOT, "content")).filter((f) => f.endsWith(".json"))) {
  for (const it of JSON.parse(fs.readFileSync(path.join(ROOT, "content", f), "utf8"))) content.push(Object.assign({ file: f }, it));
}
const buf = fs.readFileSync(path.join(ROOT, "site", "judge-vectors.bin"));
const space = new J.Space(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));

// ---------- content ----------
const CATS = new Set(["web", "brand", "plate", "patent", "search", "double"]);
const ids = new Set();
for (const it of content) {
  const where = `${it.file}: ${it.id}`;
  for (const k of ["id", "cat", "prompt", "ask", "truth", "key", "clues", "decoys", "rating"]) assert(it[k] !== undefined && it[k] !== "", `${where} lacks ${k}`);
  assert(!ids.has(it.id), `duplicate id ${it.id}`);
  ids.add(it.id);
  assert(CATS.has(it.cat), `${where}: unknown category ${it.cat}`);
  assert(["G", "PG", "PG-13"].includes(it.rating), `${where}: rating ${it.rating}`);
  assert(it.rating !== "PG-13" || it.cat === "double", `${where}: PG-13 items belong in Double take`);
  assert(it.key.length >= 2 && it.key.length <= 6, `${where}: 2-6 key words`);
  assert(it.clues.length >= 2, `${where}: needs a clue ladder`);
  assert(it.decoys.length >= 3, `${where}: "pick from four" needs three decoys`);
  assert(!it.decoys.includes(it.truth), `${where}: a decoy repeats the truth`);
  // The robot judge must know the main key idea, or nobody can score on it.
  assert(space.lookup(it.key[0]) || it.truth.toLowerCase().includes(it.key[0]), `${where}: first key "${it.key[0]}" is not in the word list`);
  if (it.cat === "plate") assert(["approved", "denied"].includes(it.dmv), `${where}: plate needs the DMV verdict`);
  if (it.cat === "patent") assert(/^US [\d,]+$/.test(it.number) && it.year > 1790, `${where}: patent number/year`);
}
const byCat = {};
for (const it of content) byCat[it.cat] = (byCat[it.cat] || 0) + 1;
for (const c of ["web", "brand", "plate", "patent", "search"]) assert(byCat[c] >= 10, `category ${c} has only ${byCat[c] || 0} items`);
const unknownKeys = content.flatMap((it) => it.key.filter((k) => !space.lookup(k)).map((k) => `${it.id}:${k}`));
// Secondary keys outside the word list only count when a guess repeats them exactly; keep that rare.
assert(unknownKeys.length <= content.length / 4, "too many key words the robot doesn't know: " + unknownKeys.join(", "));

// ---------- robot judge ----------
const item = (id) => content.find((it) => it.id === id);
const score = (guess, it) => J.scoreGuess(space, guess, it).score;

const eel = item("eelslap");
assert(score("slapping a guy with a fish", eel) >= 65, "a right guess is hot");
assert(score("online casino", eel) < 25, "an unrelated guess is cold");
assert(score("slapping a guy with a fish", eel) > score("eel recipes", eel), "closer guesses score higher");

const ld = item("liquid-death") || content.find((it) => it.prompt === "Liquid Death");
assert(score("canned water", ld) >= 85, "the main idea plus a detail is spot on");
assert(score("a drinks company", ld) >= 25 && score("a drinks company", ld) < 85, "the right area earns partial credit");
assert(score("water", ld) > score("beer", ld), "water beats beer for canned water");

// "Spot on" needs the main idea itself, not a close cousin.
const fake = { key: ["socks", "underwear", "clothing"], truth: "Socks, plus underwear.", decoys: [] };
assert(score("shoes", fake) < 85 && score("shoes", fake) >= 45, "shoes is warm, not spot on, for socks");
assert(score("socks", fake) >= 85, "socks is spot on");

// The trap: a guess that matches a decoy better than the truth is flagged.
const knock = item("plate-knku-out");
const trapped = J.scoreGuess(space, "a boxer's victory slogan", knock);
assert(trapped.trap && trapped.score < 30, "decoy guesses are flagged as traps");
assert(score("a doctor who puts you to sleep", knock) >= 85);

// Filler words don't count; empty guesses score 0.
assert.strictEqual(score("a website where", eel), 0);
assert.strictEqual(score("", eel), 0);
assert.deepStrictEqual(J.contentWords("A website where you SLAP the man's face"), ["slap", "man", "face"]);

// Bring-your-own answers become keys.
assert.deepStrictEqual(J.keysFromText("A forum thread where pool players argue about their own balls"), ["forum", "thread", "pool", "players", "argue", "balls"]);

// ---------- AI judge prompt and reply parsing ----------
const prompt = J.judgePrompt({ prompt: "kafka.com", kindLabel: "a web address", ask: "What is this website?", truth: "A PR agency in Orlando." }, [{ name: "Ann", text: "Franz Kafka fan site" }, { name: "Bo", text: "a marketing firm" }]);
assert(prompt.includes("A PR agency in Orlando.") && prompt.includes('2. Bo: "a marketing firm"'));
const reply = 'Sure!\n```json\n{"scores":[{"n":1,"score":10,"why":"wrong Kafka"},{"n":2,"score":75,"why":"close"}],"funniest":1,"comment":"Nice."}\n```';
const parsed = J.parseJudgeReply(reply, 2);
assert.strictEqual(parsed.scores[1].score, 75);
assert.strictEqual(parsed.funniest, 0);
assert.strictEqual(J.parseJudgeReply({ scores: [{ n: 1, score: 140, why: "" }], funniest: 0, comment: "" }, 1).scores[0].score, 100);
assert.strictEqual(J.parseJudgeReply({ scores: [], funniest: 0, comment: "" }, 2).funniest, null);
assert.throws(() => J.parseJudgeReply("no json here", 2));

// ---------- the Daily ----------
const day1 = Pick.dailyItems(content, "2026-10-06").map((it) => it.id);
assert.deepStrictEqual(Pick.dailyItems(content, "2026-10-06").map((it) => it.id), day1, "a date always gives the same five");
assert.strictEqual(new Set(day1).size, 5);
assert.notDeepStrictEqual(Pick.dailyItems(content, "2026-10-07").map((it) => it.id), day1, "the next day differs");
for (let d = 1; d <= 60; d++) {
  const date = new Date(Date.UTC(2026, 9, d)).toISOString().slice(0, 10);
  for (const it of Pick.dailyItems(content, date)) assert(it.rating !== "PG-13", "the Daily is never PG-13");
}
assert.strictEqual(Pick.dayNumber("2026-10-01"), 1);

console.log(`ok: ${content.length} items`, byCat, `${unknownKeys.length} secondary keys outside the word list`);

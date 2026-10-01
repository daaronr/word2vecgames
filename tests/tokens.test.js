// Run: node tests/tokens.test.js   (checks the GPT-2 tokenizer behind the Tokens tab)
const fs = require("fs");
const path = require("path");
const assert = require("assert");
const { Tokenizer } = require("../web/tokens.js");

const D = path.join(__dirname, "..", "web", "tokens");
const T = new Tokenizer(fs.readFileSync(path.join(D, "gpt2-merges.txt"), "utf8"));

// GPT-2's vocabulary: 256 bytes + 50,000 merges + <|endoftext|>.
assert.strictEqual(T.size, 50257);
assert.strictEqual(T.label(50256).text, "<|endoftext|>");

// Known GPT-2 token IDs (they match OpenAI's encoder.json).
assert.deepStrictEqual(T.encode("Hello world"), [15496, 995]);
assert.deepStrictEqual(T.encode(" SolidGoldMagikarp"), [43453]);
assert.deepStrictEqual(T.encode(" the cat sat on the mat"), [262, 3797, 3332, 319, 262, 2603]);
assert.deepStrictEqual(T.encode("\n\t"), [198, 197]);

// Any text survives the round trip, including pieces of characters (emoji, Japanese).
for (const s of ["Word Bocce is unbelievably fun!", "こんにちは", "🙂 ok", "naïve café", "  two  spaces\n", "It’s “quoted”"]) {
  assert.strictEqual(T.decode(T.encode(s)), s, `round trip ${JSON.stringify(s)}`);
}

// Labels show spaces, and the bytes of tokens that are only part of a character.
assert.deepStrictEqual(T.tokens("🙂").map((t) => [t.text, t.partial]), [["‹F0 9F›", true], ["‹99 82›", true]]);
assert.strictEqual(T.tokens(" shoe")[0].text, "␣shoe");

// The quiz (web/tokens/quiz.json) states each split in its lesson: keep them true.
const quiz = JSON.parse(fs.readFileSync(path.join(D, "quiz.json"), "utf8"));
assert(quiz.length >= 6);
for (const q of quiz) {
  assert.deepStrictEqual(T.tokens(q.text).map((t) => t.text), q.pieces, `quiz split for ${JSON.stringify(q.text)}`);
  // A lesson that spells out the pieces ("SH + OE") names the real ones.
  if (q.lesson.includes(" + ")) for (const p of q.pieces) assert(q.lesson.includes(p), `lesson for ${JSON.stringify(q.text)} omits ${p}`);
}

// The word view of the court tokenizes words as they appear mid-sentence, with a space in front.
// Most of the game's words are a single GPT-2 token that way.
for (const dir of ["data", "data-sense"]) {
  const pools = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "web", dir, "pools.json"), "utf8"));
  const single = pools.targets.filter((w) => T.encode(" " + w).length === 1).length;
  assert(single / pools.targets.length > 0.8, `${dir}: only ${single}/${pools.targets.length} jacks are one token`);
}

console.log("tokens: all checks passed");

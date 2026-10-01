# Learning how AI works with Word Bocce

Notes from 2026-10-01 on making the game more useful for people who want a
real feel for how AI language models work, and what was built toward it.

Word Bocce already teaches several core ideas by letting people play with
them: meanings stored as lists of numbers (embeddings), closeness as cosine
similarity, vector arithmetic, ranking a whole vocabulary against one vector,
and, through "Why?", the limits of explaining what a model has learned. It
leaves out the two things people most need to understand chatbots: **tokens**
(models don't read words) and **context** (a word's numbers change with the
words around it). The work below fills the first gap, and points at the
second.

Principles: use the real thing where we can (a real tokenizer, the court's
real vectors), say plainly when something is a toy, and keep the game
static-hostable with no new runtime services.

## Built in this change

- **Tokens tab** (`renderTokensPage` in `web/app.js`, tokenizer in
  `web/tokens.js`, data in `web/tokens/`):
  - *Split your own text*: GPT-2's real tokenizer (50,257 tokens), with
    token IDs, counts, and examples that show the quirks (spelling, numbers,
    Japanese, emoji, code).
  - *Guess the split*: eight quiz rounds, each answer explaining something
    that matters for chatbots: why counting letters is hard, why numbers and
    other languages cost more, why new words get split (frozen vocabulary),
    and the "SolidGoldMagikarp" glitch token. Splits are checked by
    `tests/tokens.test.js`, so the lessons can't drift from the tokenizer.
  - *From tokens to an answer*: the five steps a model takes (tokens,
    embedding lookup, context layers, scoring every token, repeat), each tied
    to what the court does or deliberately skips. Step 2 draws a real word's
    vector from the current map. Step 4 is a working toy: a throw's
    similarity to every word becomes chances under a **temperature** slider,
    and you can sample from it. Low temperature always gives "queen" for
    king − man + woman; high temperature gives nonsense.
  - *Shows / leaves out*: an honest list of what the game models and what it
    doesn't.
- **In-game token switch**: under the hand, "Show as AI tokens" relabels the
  tiles, the throw and the start → jack line as GPT-2 tokens with IDs (␣ marks
  the space a mid-sentence word carries). Players see that "jockey" is ␣j +
  ockey and "catholic" splits while "Catholic" doesn't. The court still uses
  whole words, and the note under the hand says so.
- **"See the numbers" in Why?**: each word of a throw drawn as a strip of its
  100 or 300 numbers, then the ball and the jack, so the arithmetic is
  visible: the ball is literally the start's strip plus and minus the others.
- **AI terms in the tips**: rank, similarity and "where it landed" now say
  what each is called in AI and where a chatbot does the same thing.

## Built: the "AI tokens" map (the token court)

A third map in the Words dialog, next to common sense and raw text, built
from a language model's own token table: GPT-2's input embeddings (50,257
tokens × 768 numbers). `tools/build_token_data.py` fetches just that tensor
from Hugging Face (`openai-community/gpt2`, about 150 MB by HTTP range
request), drops tokens that make bad labels (pieces of characters, control
characters, blocklisted words), mean-centres the rest (GPT-2's rows share one
large common direction that swamps cosine similarity), keeps 128 dimensions
(PCA) and quantises to int8: 49,745 tokens, 6.4 MB, the size of the
common-sense bundle. At 128 dimensions a token's 10 nearest neighbours are
about 71% the same as with all 768 (75% at 160, 83% at 256).

What it shows:

- Neighbours are tokens: ␣shoe sits with ␣shoes, ␣Shoes, ␣sneakers, ␣Nike and
  fragments like ␣sho; ␣bank with ␣Bank, bank, ␣banking. A model starts out
  with separate numbers for "shoe" and "Shoe" and has to learn they belong
  together. ␣SolidGoldMagikarp's neighbours are other glitch tokens.
- It's a better map than expected: king − man + woman lands on ␣queen (1st),
  and test deals reach rank 1 from a start around 2,000th.
- GPT-2 scores its next token against this same table (tied embeddings), so
  the court's rank is a simplified version of the model's last step: same
  table, cosine on cut-down rows, no layers in between.

Details: `WORD_SETS.tokens` in `app.js` (Daily seed `daily-tokens-`, tutorial
farmer + fish − farm → fisherman, `explain` 0.3, typed words map to their
mid-sentence token through `keyOf`), pools are the two word sets' familiar
words that are a single GPT-2 token after a space, and 25 of the 60 puzzles
carry `retired_tokens` (five use words GPT-2 splits, such as "rhinestone").
`engine.js`'s `related()` ignores the ␣ and case, so a ball doesn't "land
near" a copy of a thrown token. `tests/engine.test.js` checks the tutorial,
deals and puzzles as for the other maps. The "Show as AI tokens" switch is
hidden on this map, since its tiles are already tokens.

Rough edges: many court labels are case or spacing variants, some jacks are
abstract (the pools come from the word maps), and the game's wording still
says "word" in places where this map means "token".

## Further ideas, roughly in order of value for effort

1. **Context: the same word in two sentences.** The biggest thing the game
   leaves out. Precompute, offline, GPT-2's vectors for a word like "bank" in
   a dozen curated sentences, at the first, middle and last layer, and ship
   them as a small JSON file. Show where each lands among the token table's
   neighbours: identical at layer 0, pulled toward water or money later. No
   model runs in the browser.
2. **Compare maps.** In "Try your own words", show the same throw on both maps
   side by side: same words, different training text, different answer. It
   makes "a model is shaped by its data" concrete.
3. **Bias lens.** A few curated throws where the maps reproduce stereotypes
   (occupations and gender are the documented cases: Bolukbasi et al. 2016;
   Caliskan et al. 2017), framed as what the training text contained, with
   the caveat that some famous examples were overstated (the interp dialog
   already cites Nissim et al. 2020). Needs careful wording and probably a
   review by someone outside the project before it ships.
4. **The court is a shadow.** The court draws 100–300 dimensions in two. A
   toggle could redraw the same balls with a different projection, to show
   that distance on screen is a summary and rank is the real measure.
5. **A classroom pack.** A one-page worksheet and teacher notes mapping each
   screen to a concept (token, embedding, similarity, nearest neighbour,
   softmax and temperature, interpretability), with a 30-minute lesson plan.
   Cheap to make and probably the widest reach.
6. **Guess the jack** (idea 2 in `docs/game-modes.md`) doubles as an
   interpretability exercise: inferring a hidden meaning from where vectors
   land is what researchers try to do with model internals.
7. **A real small model in the browser** (transformers.js with distilgpt2,
   tens of MB) for live next-token probabilities on any text. Most
   realistic, but heavy, and a large runtime dependency; better as an
   optional lab page than part of the game.
8. **Measure it.** Record which quiz rounds people miss (through the existing
   suggestions endpoint, opt-in) to see which ideas land and which need a
   better explanation.

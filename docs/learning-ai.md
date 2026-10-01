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

## Next: an "AI tokens" map (the token court)

The fullest version of "Word Bocce with tokens" is a third map, next to
common sense and raw text, built from a language model's own token table:
GPT-2's input embeddings (50,257 tokens × 768 numbers, MIT licence).

Why it's worth it:

- The court's neighbours would be tokens: ␣shoe, ␣Shoe, Shoe, ␣shoes,
  fragments like "ville", and glitch tokens near the middle. That shows
  directly that a model starts out with separate numbers for "shoe" and
  "Shoe" and has to learn they're related.
- GPT-2 uses the *same* table to score its next token (tied embeddings), so
  the court's rank would be, quite literally, the model's last step without
  the layers in between. The Tokens tab's step 4 would stop being a toy.
- The Tokens tab's IDs would match the court's tokens exactly.

Plan:

1. `tools/build_token_data.py`: read `wte` from GPT-2 small
   (`openai-community/gpt2`, `model.safetensors`, on Hugging Face), mean-centre
   it (GPT-2's embeddings share one large common direction that swamps
   cosine similarity), reduce to about 128 dimensions with PCA, and quantise
   to int8: about 6.4 MB for all 50,257 rows, the size of the common-sense
   bundle. Keep rows in token-ID order; write labels with `Tokenizer.label`.
   Check how much the reduction changes each token's nearest neighbours
   before settling the dimension count.
2. Pools: the jacks and cards from `web/data/pools.json` that are a single
   GPT-2 token with a space in front (about 95% of jacks are; `tests/tokens.test.js`
   checks it stays above 80%).
3. A `WORD_SETS.tokens` entry in `app.js`: its own Daily seed, tutorial
   example (search for one the way the others were found), `explain`
   threshold tuned on the data, and a `retired_tokens` field in
   `puzzles.json` after auditing the puzzles (or hide Puzzles for this map).
4. Extend `tests/engine.test.js` to the third set, as for the other two.

Not done here because this session's network policy blocks Hugging Face.
Running step 1 on a machine with normal internet access (or allowing
`huggingface.co` for the cloud environment) unblocks the rest.

Expect a rougher map than the two word maps: a model's first layer isn't
trained to be a good map on its own, analogies work less often, and
neighbours are crowded with case and spacing variants. That roughness is part
of the lesson, but the deal thresholds will need tuning so courts stay
playable.

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

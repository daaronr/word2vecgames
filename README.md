# Word Bocce

A lawn game played in word-vector space. The yellow **jack** is a target word;
your ball starts as another word. Add and subtract word tiles to roll it closer:

```
king − man + woman  →  queen
```

Distance is measured as **rank**: `#12` means the jack is the 12th-nearest word
to where your ball stopped. `#1` is a *bacio*, a kiss on the jack.

## Three word sets ("maps of meaning")

Players choose where word positions come from (the **Words** button, or Help):

- **Common sense** (default, `web/data-sense/`): ConceptNet Numberbatch 19.08,
  which blends text statistics with a knowledge base of everyday facts ("bacon
  is a kind of meat", "a hen lays eggs"). Limited to 21,114 everyday words, 300
  dimensions, 6.3 MB. Courts are only dealt if one of the strongest throws can
  be explained word by word (`explainable()` in `engine.js`: every added word
  has cosine ≥ 0.3 with the jack, every subtracted word ≥ 0.3 with the start),
  and par is chosen among such throws.
- **Raw text** (`web/data/`): GloVe 6B 100d, learned only from which words
  appear together in Wikipedia and news. The classic word embedding: it can
  surprise ("ham" sits with football clubs, "eggs" nearer "meat" than "bacon").
- **AI tokens** (`web/data-tokens/`): GPT-2's own token table, the numbers a
  language model looks up for each token before anything else. 49,745 tokens
  (labelled as `web/tokens.js` shows them: "␣shoe" is " shoe"), 128 of 768
  dimensions, 6.4 MB. Neighbours include case and spacing variants ("␣Shoe",
  "Shoe") and word pieces. Every hand has two word pieces ("resso" for coffee),
  and tapping a tile shows its token ID and nearest tokens. 14 token puzzles
  (`web/data-tokens/puzzles.json`) turn on tokens with different embeddings,
  like ␣apple vs ␣Apple, or on word pieces; one dealt court in three on this
  map is one of them. Typed words map to
  their mid-sentence token. Built
  by `tools/build_token_data.py`; details in `docs/learning-ai.md`.

Why both exist: with raw-text words, the hand-made puzzles and dealt courts
often had best throws nobody could explain (bacon → eggs couldn't get better
than 20th; a typical dealt par read "doll − dragon + flour + puppy → peanut").
With common-sense words, 44 of the 60 puzzles pass the quality bar (32 with raw
text), and dealt pars read like "canal + disaster + avalanche + flow → flood".
Each set has its own Daily, tutorial example and puzzle retirement field
(`retired_sense` / `retired` / `retired_tokens` in `puzzles.json`; 35 puzzles
are active on the token map). An online room uses the host's set.

## Modes

- **Daily**: one court per day, the same for everyone. Four balls; the best
  counts. Copy an emoji summary to share.
- **Practice**: endless freshly dealt courts.
- **Puzzles**: hand-made courts (`web/data/puzzles.json`) with star ratings:
  44 active with common-sense words, 32 with raw text. The rest are marked
  `retired_sense` / `retired` with a reason (already solved at the start,
  couldn't be done well, or no sensible best throw); `tests/engine.test.js`
  fails if an active puzzle drifts into those states.
- **Versus**: against a bot or a friend on one device, with real bocce rules
  (the side farther from the jack throws next; the closer side scores a point
  per ball that beats the other's best; first to 5).
- **Online with friends** (under Versus): open a room, send the link, and
  everyone plays the same court at once on their own device, three balls each.
  You see each other's balls land; the words are revealed when the round ends.
  Up to six players. Browsers connect directly (WebRTC via the free PeerJS
  broker, `web/net.js`); the host's tab is the room, and nothing is stored.
- **Cat and mouse** (under Versus; direct link `#chase`): a chase through about
  1,300 everyday things on the common-sense map, moving by describing and
  doing words (car + fly → airplane). Each move adds or subtracts a card and
  hops to the nearest word you haven't visited (you don't see where until you
  play it). The mouse opens with a getaway of up to three cards; the cat
  catches it by landing within its 30 nearest words, can pounce with two
  cards every 5th move, and can pass for a fresh row of cards; trap words blow
  up the cat. The game says where everyone is and tells each move card by
  card (the bot's too), lists both journeys, and ends with a story of the
  chase. Against the bot or a friend on one device. Cards:
  `tools/chase_cards.txt` → `tools/build_terrains.py`. Design:
  `docs/game-modes.md`.
- **Tokens**: how a chatbot reads text, with GPT-2's real tokenizer
  (`web/tokens.js`). Split your own text into tokens and see their IDs, play
  "Guess the split" (eight rounds, each explaining a quirk that matters for
  chatbots), and follow the steps from tokens to a chosen next word, with a
  temperature slider that samples from the court's own vectors. In any game,
  "Show as AI tokens" under the hand relabels the tiles as GPT-2 tokens. Ideas
  and next steps for teaching how AI works: `docs/learning-ai.md`.

After every throw, **Why?** opens a reading of what each word did: each word's
exact share of the ball's similarity to the jack (`Space.explainThrow`), the
words crowding the jack before and after, and plain notes on the things that
tend to confuse people (subtracting a word that is itself close to the jack,
similarity rising while rank falls, landing somewhere none of the words point
to). It always says this is a reading of the numbers, not the model's reasons,
and links to a short explainer on why interpretability is hard. "See the
numbers" draws each word's vector as a strip, so you can watch the ball's
numbers come out of the words' numbers.

After every round you see par (the best throw the hand allowed), a few nearby
words placed on the court, and a "try your own words" box: type any throw to
see where it would have landed and say whether those words would have made the
court more fun. Suggestions are sent to the game server's `/api/suggestions`
(see `DEPLOY.md`) and wait in the browser until it answers. More mode ideas:
`docs/game-modes.md`.

## Teaching and sharing

- `web/teach.html` (live at https://wordbocce.davidreinstein.org/teach.html):
  a 30–45 minute lesson plan for intro NLP / ML / data-science classes, linked
  from the game's footer and the slides.
- `promo/README.md`: the promotion package. Draft posts (LessWrong, Show HN,
  Reddit, short messages), a pre-launch checklist and a tracker of where things
  were posted. `promo/og-card.html` is the source of `web/og-image.png`, the
  link-preview image.

## Run it

Play online: https://wordbocce.davidreinstein.org/ (GitHub Pages with a custom
domain, redeployed from `web/` on every push to main; the old
daaronr.github.io/word2vecgames address redirects there).

The game runs entirely in the browser. Nothing to install:

```bash
cd web && python3 -m http.server 8000     # then open http://localhost:8000
```

Or run the full server (game at `/`, legacy multiplayer lobby at `/classic`):

```bash
pip install -r requirements.txt
uvicorn word_bocce_mvp_fastapi:app --reload
```

The legacy API endpoints (`/match/...`, `/puzzle/{id}/solve`, `/classic`) also
need full embeddings: `python setup_embeddings.py --model glove-100` and
`export MODEL_PATH=./embeddings/glove-100.bin`.

## Layout

| Path | What |
| --- | --- |
| `web/` | The game: `index.html`, `app.js` (UI), `engine.js` (vector math, dealing, scoring), `style.css`, `presentation.html` (the maths, as slides), `teach.html` (lesson plan), `og-image.png` (link preview) |
| `promo/` | Promotion drafts, checklist and tracker (`promo/README.md`); `og-card.html` renders `web/og-image.png` |
| `web/data/` | `vectors.bin` (40k × 100 int8), `vocab.txt`, `pools.json` (card and jack word pools), `puzzles.json` |
| `web/tokens.js`, `web/tokens/` | GPT-2's tokenizer for the Tokens tab: `gpt2-merges.txt` (OpenAI's merge list) and `quiz.json` |
| `tools/build_web_data.py` | Rebuilds `web/data/` from a GloVe file and word norms |
| `tools/make_artifact.sh` | Stages `web/` for publishing as a claude.ai Artifact |
| `tests/engine.test.js` | `node tests/engine.test.js` checks the engine against the real vectors |
| `tests/tokens.test.js` | `node tests/tokens.test.js` checks the tokenizer against known GPT-2 IDs and the quiz's splits |
| `word_bocce_mvp_fastapi.py` | Server: static game plus the older multiplayer/puzzle API |
| `DEPLOY.md` | Static hosting, and the Linode server |
| `docs/original-design.md` | The original design document |
| `docs/learning-ai.md` | Making the game teach how AI works: what's built, the planned GPT-2 token map, more ideas |
| `archive/` | Superseded UI, docs and deploy configs |

## How dealing works

`engine.js` deals each court from a seed (the date, for Daily). It picks a jack
from ~1,250 vivid concrete nouns, then a start word that is related but not
close (cosine 0.12–0.35). The nine-tile hand has no random filler; every tile
is linked to the jack or the start word: two that pull toward the jack, two
worth subtracting (they carry the start word's flavour), two lures linked to
both (adding them drags the start along), and three near-misses that look like
they point at the jack but more weakly than the real pulls.

Balls, par and versus scoring are all judged by the jack's **rank** (similarity
only breaks ties), because rank is what players see. Par scores all 834 throws
the hand allows (up to three tiles, each ±) by similarity, then ranks the top 50
and keeps the best. Taking the most similar throw as par, as the game used to,
misses the rank-best throw in most deals.

New visitors land in a short guided tutorial before the Daily: hat + foot −
head → shoe with common-sense words, boat + sky − water → plane with raw text.

## Rebuilding the vector bundle

```bash
# GloVe 6B 100d (public domain), mirrored by gensim-data on GitHub
curl -L -o embeddings/glove-100.gz \
  https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz
# Brysbaert et al. (2014) concreteness norms, used to pick familiar card and jack words
curl -L -o embeddings/concreteness.txt \
  https://raw.githubusercontent.com/ArtsEngine/concreteness/master/Concreteness_ratings_Brysbaert_et_al_BRM.txt
python3 tools/build_web_data.py embeddings/glove-100.gz --lists embeddings --size 40000

# Common-sense set: ConceptNet Numberbatch 19.08 English (325 MB download), in the
# raw-text set's word order, everyday words only
curl -L -o embeddings/numberbatch-en.txt.gz \
  https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
python3 tools/build_web_data.py embeddings/numberbatch-en.txt.gz --vocab-from web/data/vocab.txt \
  --everyday 0.9 --lists embeddings --out web/data-sense
node tests/engine.test.js

# AI tokens set: GPT-2's token table from Hugging Face (fetches ~150 MB once into embeddings/)
python3 tools/build_token_data.py
node tests/engine.test.js
```

`tools/blocklist.txt` keeps slurs, profanity and a few grim words out of the
vocabulary entirely, so they never appear as tiles, jacks or landing labels.

## Credits

Raw-text vectors: GloVe 6B (Pennington, Socher & Manning, 2014), Public Domain
Dedication and License. Common-sense vectors: ConceptNet Numberbatch 19.08
(Speer, Chin & Havasi, 2017), CC BY-SA 4.0; the derived bundle in
`web/data-sense/` is shared under the same licence (see its `README.txt`).
Token map: GPT-2's token embeddings (OpenAI, 2019, modified MIT licence), see
`web/data-tokens/README.txt`.
Word familiarity: Brysbaert, Warriner & Kuperman (2014). Tokenizer: OpenAI's
GPT-2 merge list (2019, modified MIT licence), see `web/tokens/README.txt`.

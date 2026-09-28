# Word Bocce

A lawn game played in word-vector space. The yellow **jack** is a target word;
your ball starts as another word. Add and subtract word tiles to roll it closer:

```
king − man + woman  →  queen
```

Distance is measured as **rank**: `#12` means the jack is the 12th-nearest word
(out of 40,000) to where your ball stopped. `#1` is a *bacio*, a kiss on the jack.

## Modes

- **Daily**: one court per day, the same for everyone. Four balls; the best
  counts. Copy an emoji summary to share.
- **Practice**: endless freshly dealt courts.
- **Puzzles**: 60 hand-made courts (`web/data/puzzles.json`) with star ratings.
- **Versus**: against a bot or a friend on one device, with real bocce rules
  (the side farther from the jack throws next; the closer side scores a point
  per ball that beats the other's best; first to 5).

After every end you see the best throw the hand allowed, and the words nearest
the jack.

## Run it

Play online: https://daaronr.github.io/word2vecgames/ (GitHub Pages, redeployed
from `web/` on every push to main).

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
| `web/` | The game: `index.html`, `app.js` (UI), `engine.js` (vector math, dealing, scoring), `style.css`, `presentation.html` (the maths, as slides) |
| `web/data/` | `vectors.bin` (40k × 100 int8), `vocab.txt`, `pools.json` (card and jack word pools), `puzzles.json` |
| `tools/build_web_data.py` | Rebuilds `web/data/` from a GloVe file and word norms |
| `tools/make_artifact.sh` | Stages `web/` for publishing as a claude.ai Artifact |
| `tests/engine.test.js` | `node tests/engine.test.js` checks the engine against the real vectors |
| `word_bocce_mvp_fastapi.py` | Server: static game plus the older multiplayer/puzzle API |
| `DEPLOY.md` | Static hosting, and the Linode server |
| `docs/original-design.md` | The original design document |
| `archive/` | Superseded UI, docs and deploy configs |

## How dealing works

`engine.js` deals each court from a seed (the date, for Daily). It picks a jack
from ~1,250 vivid concrete nouns, then a start word that is related but not
close (cosine 0.12–0.35). The seven-tile hand holds two tiles that pull toward
the jack, one worth subtracting (it carries the start word's flavour), two
tempting near-misses, and two wildcards. Par is found by brute force over all
378 throws the hand allows.

## Rebuilding the vector bundle

```bash
# GloVe 6B 100d (public domain), mirrored by gensim-data on GitHub
curl -L -o embeddings/glove-100.gz \
  https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz
# Brysbaert et al. (2014) concreteness norms, used to pick familiar card and jack words
curl -L -o embeddings/concreteness.txt \
  https://raw.githubusercontent.com/ArtsEngine/concreteness/master/Concreteness_ratings_Brysbaert_et_al_BRM.txt
python3 tools/build_web_data.py embeddings/glove-100.gz --lists embeddings --size 40000
node tests/engine.test.js
```

`tools/blocklist.txt` keeps slurs, profanity and a few grim words out of the
vocabulary entirely, so they never appear as tiles, jacks or landing labels.

## Credits

Vectors: GloVe 6B (Pennington, Socher & Manning, 2014), Public Domain
Dedication and License. Word familiarity: Brysbaert, Warriner & Kuperman (2014).

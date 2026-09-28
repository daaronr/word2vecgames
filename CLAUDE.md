# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project Overview

"Word Bocce" is a word-vector game. Players start from a `start` word and add or
subtract word tiles to land as close as possible to a target word (the "jack"):
`ball = unit(v(start) + Σ ±v(tile))`, scored by cosine to the jack and by the
jack's **rank** among the ball's nearest words (rank 1 = "bacio").

## Where things live

- `web/` — **the current game**, fully client-side (no server needed).
  - `engine.js`: `Space` (int8 vectors → unit float32; `score`, `survey` for
    rank/nearest words, `allThrows` for par), `deal()` (seeded court + 7-tile
    hand), `courtBasis()` (2-D placement on the court), `tier()`. Works in the
    browser (`window.Bocce`) and in Node (`module.exports`).
  - `app.js`: UI and modes (daily / practice / puzzles / versus with a bot).
    Court is an SVG; distance from the jack is drawn on a log-rank scale.
  - `data/`: `vectors.bin` (int8, rows = `vocab.txt` order), `pools.json`
    (`cards`, `targets`), `puzzles.json` (the 60 curated puzzles; the server
    reads this file too).
- `tools/build_web_data.py` regenerates `web/data/` (see README).
  `tools/blocklist.txt` is the vocabulary safety filter.
- `tools/make_artifact.sh` stages `web/` for a claude.ai Artifact (strips the
  document skeleton, ships `vectors.bin` as `vectors.wasm` because Artifacts only
  serve known web types).
- `word_bocce_mvp_fastapi.py` — FastAPI server: mounts `web/` at `/`, and keeps
  the older server-side API (matches, `/puzzle/{id}/solve`, `/visualize`) used by
  the legacy UI at `/classic` (`archive/legacy-ui/index.html`). The legacy API
  needs `MODEL_PATH` (full GloVe/word2vec file); without it the server still
  serves the game.
- `archive/` — superseded UI, docs and deploy configs. Don't edit; don't
  resurrect without reason.

## Commands

```bash
node tests/engine.test.js                 # engine checks against the real vectors
cd web && python3 -m http.server 8000     # play locally
uvicorn word_bocce_mvp_fastapi:app --reload
```

## Conventions

- Keep the game static-hostable: no runtime dependency on the Python server.
- Card/jack words must stay familiar; change pools in `build_web_data.py`
  rather than hand-editing `pools.json`.
- Deals must stay deterministic for a given seed (Daily depends on it). If you
  change `deal()` or the data bundle, today's Daily changes for everyone.
- Visual identity: sage "clubhouse" ground, raked-gravel court, bottle-green
  boards, yellow jack, red/cobalt balls; Alfa Slab One (display), Atkinson
  Hyperlegible (UI), DM Mono (maths). Colours are CSS tokens in `style.css`
  with light and dark sets.
- Production is a Linode VPS (`DEPLOY.md`, `UPDATE_LINODE.sh`).

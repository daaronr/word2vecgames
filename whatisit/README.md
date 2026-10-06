# What Is It?

A guessing game about cryptic strings. Someone shows a web address, a brand name, a licence plate,
a patent title or a search phrase; everyone guesses what it really is; the reveal settles it. A judge
(the room, a free word-vector "robot", or an AI running on the player's own account) decides who was
closest. Started in this repository next to Word Bocce; self-contained so it can move out.

Design notes, costs, prior art and next steps: [DESIGN.md](DESIGN.md).

## Four trial versions (one page)

| Trial | Mode | Players | What it tests |
|---|---|---|---|
| A | Daily five | 1 | An NYT-style daily: free-text guesses scored by the robot judge, clues and "pick from four" at a cost, share grid |
| B | Party board | 2+ (one screen) | A points board of categories (100/200/300) or tic-tac-toe for two teams |
| C | Bring your own | 2+ | The original game: type anything, guess, look it up together, judge; Family-Feud bonus for results 2 and 3 |
| D | Bluff | 3+ (pass the phone) | Balderdash-style: write fakes, find the truth |

Every reveal asks "how fun was that one?"; every trial has a rate-this-version form; anyone can
suggest a mystery or a better clue.

## Where things live

- `src/` the game: `judge.js` (robot judge + AI-judge prompt/parser; browser and Node), `pick.js`
  (seeded randomness, the Daily's choice), `core.js` (content, storage, AI sources, shared UI),
  `daily.js`, `party.js`, `byo.js`, `bluff.js`, `app.js` (hub, suggest, settings, notes, boot), `style.css`.
- `content/*.json` the mysteries (`web`, `brand`, `plate`, `patent`, `search`, `double`,
  `double-plates`). Each has `prompt`, `ask`, `truth`, `more`, `key` (robot-judge key ideas, most
  important first), `clues` (vague to strong), `decoys` (tempting wrong answers), `url`, `source`,
  `checked`, `fun`, `difficulty`, `rating`. Plates add `dmv` (approved/denied); patents add `number`,
  `year`; search items add `top3`. Items marked `"status": "unverified"` are drafts shown with a flag.
- `site/` build output: `index.html` (the whole game in one file, for Netlify or any static host),
  `artifact.html` (the claude.ai Artifact version), `judge-vectors.bin` (robot-judge word vectors).
- `tools/build.mjs` builds `site/`; `tools/build_vectors.py` makes `judge-vectors.bin` from
  `../web/data-sense` (ConceptNet Numberbatch, CC BY-SA 4.0), reduced to 96 dimensions.
- `vendor/` the bundled Anthropic SDK used only for the bring-your-own-API-key judge.
- `tests/whatisit.test.js` content schema, robot-judge behaviour, AI-reply parsing, Daily determinism.

## Commands

```bash
node whatisit/tools/build.mjs                  # rebuild site/ after editing src/ or content/
python3 whatisit/tools/build_vectors.py        # only if the word vectors change
node whatisit/tests/whatisit.test.js
cd whatisit/site && python3 -m http.server 8000   # play locally at http://localhost:8000
```

## Hosting

- **Netlify (current trial):** `site/index.html` is deployed as a single file. The robot judge fetches
  `judge-vectors.bin` next to the page if present, otherwise from this repository on GitHub.
  Ratings go to Netlify Forms once form detection is switched on for the site (Site configuration >
  Forms); until then they stay in each player's browser (Settings > Copy them all).
- **Git-connected Netlify site:** set the base directory to `whatisit` (see `netlify.toml`); the
  vectors file then ships alongside the page.
- **claude.ai Artifact:** `site/artifact.html` plus `judge-vectors.bin` published as
  `judge-vectors.wasm`. There the AI judge runs on each viewer's own Claude plan (the `sample`
  capability) and ratings are shared through the artifact's database (`db`).

## Changing the Daily

The Daily is deterministic for a date and a content bundle (`src/pick.js`): editing `content/`
changes future (and today's) Dailies. That's fine during trials; freeze the bundle before a launch.

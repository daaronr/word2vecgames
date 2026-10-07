# What Is It?

A guessing game about cryptic things. Someone shows a web address, a brand name, a licence plate,
a patent title, a search phrase, a magnified detail of a picture, the start of a paper's title, an odd
song line, a news story late-night hosts joked about, or a headline that reads two ways; everyone
guesses what it really is; the reveal settles it. A judge decides who was closest: the room, the
mystery's answer key (likely guesses scored in advance), a free word-vector "robot", or an AI.
Started in this repository next to Word Bocce; self-contained so it can move out.

Design notes, costs, prior art and next steps: [DESIGN.md](DESIGN.md). Open items: [TODO.md](TODO.md).

## Four trial versions (one page)

| Trial | Mode | Players | What it tests |
|---|---|---|---|
| A | Daily five | 1 | An NYT-style daily: free-text guesses scored by the answer key, then the AI or the robot; clues and "pick from four" at a cost; share grid |
| B | Party board | 2+ (one screen) | A Jeopardy-style points board (unlock order, board control, Daily Double, penalties, Final with wagers), tic-tac-toe or Connect Four for two teams |
| C | Bring your own | 2+ | The original game: type anything, guess, look it up together, judge; Family-Feud bonus for results 2 and 3 |
| D | Bluff | 3+ (pass the phone) | Balderdash-style: write fakes, find the truth |

Every reveal asks "how fun was that one?"; every trial has a rate-this-version form; anyone can
suggest a mystery or a better clue.

## Where things live

- `src/` the game: `judge.js` (robot judge + AI-judge prompt/parser; browser and Node), `pick.js`
  (seeded randomness, the Daily's choice), `core.js` (content, storage, AI sources, shared UI),
  `daily.js`, `party.js`, `byo.js`, `bluff.js`, `app.js` (hub, suggest, settings, notes, boot), `style.css`.
- `content/*.json` the mysteries (`web`, `brand`, `plate`, `patent`, `search`, `double`,
  `double-plates`, `zoom`, `paper`, `lyric`, `latenight`, `headline`). Each has `prompt`, `ask`,
  `truth`, `more`, `key` (robot-judge key ideas, most important first), `clues` (vague to strong),
  `decoys` (tempting wrong answers), `graded` (the answer key: `{g, s, h}` = example guess, score
  0-100, spoiler-free hint), `url`, `source`, `checked`, `fun`, `difficulty`, `rating`. Plates add
  `dmv` (approved/denied); patents `number`, `year`; search items `top3`; pictures `image` (an SVG in
  `content/img/`, inlined by the build), `focus` and `zooms`; papers `full`, `authors`, `journal`,
  `year`, `field`; lyrics `song`, `year`, `pd`; late-night items `when`, `hosts` (only shows
  confirmed to have done a bit); headlines `where`. Items marked `"status": "unverified"` are drafts
  shown with a flag.
- `site/` build output: `index.html` (the whole game in one file, for Netlify or any static host),
  `artifact.html` (the claude.ai Artifact version), `judge-vectors.bin` (robot-judge word vectors).
- `tools/build.mjs` builds `site/`; `tools/vocab.mjs` checks which words the robot knows;
  `tools/build_vectors.py` makes `judge-vectors.bin` from
  `../web/data-sense` (ConceptNet Numberbatch, CC BY-SA 4.0), reduced to 96 dimensions.
- `netlify/functions/judge.mjs` the AI judge (`/api/judge`); `netlify/lib/judge-core.mjs` its testable
  logic; `netlify/lib/items.mjs` the answers it may judge (written by the build).
- `eval/` hand-scored guesses and the script that compares judges.
- `vendor/` the bundled Anthropic SDK used only for the bring-your-own-API-key judge.
- `tests/whatisit.test.js` content schema, robot-judge behaviour, AI-reply parsing, Daily determinism.

## Commands

```bash
node whatisit/tools/build.mjs                  # rebuild site/ after editing src/ or content/
python3 whatisit/tools/build_vectors.py        # only if the word vectors change
node whatisit/tests/whatisit.test.js
node whatisit/tests/judge-function.test.mjs  # the /api/judge logic with a fake model
node whatisit/eval/score_judges.mjs           # judges vs hand-scored guesses
cd whatisit/site && python3 -m http.server 8000   # play locally at http://localhost:8000
```

## Hosting on Netlify (with the AI judge)

One-time setup in the Netlify UI:
1. **Add new project > Import an existing project > GitHub**, pick `daaronr/word2vecgames`, or
   deploy from a terminal: `cd whatisit && npx netlify-cli deploy --prod`.
2. Branch: `main` (or this work's branch until it is merged). **Base directory: `whatisit`.**
   Build command and publish directory come from `netlify.toml`.
3. **Give the AI judge a key of your own.** The cheapest: a free Gemini API key from Google AI
   Studio (aistudio.google.com, "Get API key"; no card needed). In Netlify: **Project configuration >
   Environment variables > Add a variable**, `GEMINI_API_KEY` = your key, scope Functions, then
   redeploy. The free tier stops at its daily quota instead of billing you. An `ANTHROPIC_API_KEY` or
   `OPENAI_API_KEY` works too (set `JUDGE_PROVIDER` if you set more than one).
   Without your own key the AI judge rests and the game uses the answer keys and the robot.
4. **Keep Netlify's own AI from spending your credits.** Netlify AI Gateway also injects provider
   keys, billed to the team's Netlify credits; on the Free plan (300 credits a month for everything,
   15 per production deploy) running out pauses every project on the team. The function ignores
   those keys unless you set `JUDGE_USE_NETLIFY_CREDITS=1`. For belt and braces: **Team settings >
   AI enablement**, switch AI features off or set an AI inference credit limit.
5. **Project configuration > Forms > Enable form detection**, then redeploy, so ratings and
   suggestions land in the Forms tab.
6. **Testing with friends and family:** set `JUDGE_PASSCODE` to a code (or several, comma-separated)
   and send people `https://your-site/?pass=CODE`. The link saves the pass on their device (it can
   also be typed in Settings) and they get AI verdicts with nothing to set up; visitors without it
   get the answer key and the robot, so strangers can't use up the quota.
7. Optional limits (defaults in brackets): `JUDGE_DAILY_CAP` (200 model calls a day),
   `JUDGE_MONTHLY_CAP` (3000), `JUDGE_PER_VISITOR` (60 a day), `JUDGE_MODEL`, `JUDGE_OFF=1` to switch
   the AI off, `SHARE_URL` (link in the Daily's share text).

Each production deploy costs 15 credits, so test locally (`cd whatisit/site && python3 -m http.server`)
and deploy to production once per batch of changes.

Without the function (any other static host, or a plain file upload) the page still works: the
word-vector robot judges, and players can bring their own AI in Settings.

**claude.ai Artifact:** `site/artifact.html` plus `judge-vectors.bin` published as
`judge-vectors.wasm`. There the AI judge runs on each viewer's own Claude plan (the `sample`
capability) and ratings are shared through the artifact's database (`db`).

## Changing the Daily

The Daily is deterministic for a date and a content bundle (`src/pick.js`): editing `content/`
changes future (and today's) Dailies. That's fine during trials; freeze the bundle before a launch.

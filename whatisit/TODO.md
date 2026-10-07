# What Is It? to-do

Open items, newest first. Done items move to the changelog on the in-game Notes page
(`CHANGELOG` in `src/app.js`).

## Handoff between sessions

Several sessions work on this folder, on branch `claude/quirky-brahmagupta-l9gamm`. Before you stop,
add two or three lines here: what you changed, what's half-done, which files you're in. Pull first.

- 7 Oct, cloud session A (UI): folded options and explanations, feedback promise, changelog on the
  Notes page, `sfx()` sound cues, tester passes (`JUDGE_PASSCODE`). Prepared the repo split:
  `tools/split_repo.sh`, `whatisit/.gitignore`, `WII_REPO`/`WII_DIR` in `tools/build.mjs`. Not touching
  `netlify/functions/judge.mjs` or Netlify setup (session B has those).
- 7 Oct, local desktop session B (deploy + AI judge): site `whatisit-game` (team `daaronr`, Open
  Source plan, 10,000 credits a month shared by ~85 sites). Turned form detection on (it was off, so
  submissions were dropped). **Provisional:** the AI judge now runs Gemini 3.1 Flash-Lite through
  Netlify AI Gateway on the team's credits (`JUDGE_USE_NETLIFY_CREDITS=1`), capped at
  `JUDGE_MONTHLY_CREDITS=200` a month, counted from each call's token use (new
  `netlify/lib/providers.mjs`). Model test on the 217 hand-scored guesses: `eval/run_models.mjs`,
  results in `eval/results/api_*`. Fallback if credits become a worry: a free Gemini key from Google
  AI Studio as `GEMINI_API_KEY` and unset `JUDGE_USE_NETLIFY_CREDITS`. Site settings live in Netlify
  env vars, not the repo; deploy with `netlify deploy --prod` from `whatisit/` (no Git link yet).
  Deployed to production from 1bf6c2a; 4 forms registered; judge answering. To check spend: the
  `judge` Blobs store has `credits/YYYY-MM` (credits used this month) and `count/YYYY-MM-DD`.
  Open: decide whether to keep Netlify credits or switch to a free Gemini key; link the site to the
  new repo after the split (until then every deploy is manual).

## Needs David

- **Feedback from the 6 Oct play session.** Form detection was off until 7 Oct, so those notes never
  reached Netlify; they are still in the browser they were typed in (Settings > "Copy them all").
  New notes now go to Netlify Forms (Project > Forms). Paste or export them into a session to act on them. Same for the hypothes.is
  notes: export from the Hypothesis sidebar, or allow `hypothes.is` and `api.hypothes.is` in the
  environment's network settings.
- **Style sources.** `claude_code_misc_work` (style sheets and style recommendations) isn't reachable
  from cloud sessions: push it to GitHub or attach the files, then do the styling pass below.
- **Rewards model.** Look at how complainments.netlify.app rewards suggestions (blocked from cloud
  sessions) and decide what we offer: credit by name, points, extra plays, a monthly prize.
- **Own repo.** Recommended (see below). Needs a new, preferably private, GitHub repo.

## Build next

1. **Move to its own repo.** Once all sessions have pushed: create an empty private repo
   (e.g. `daaronr/whatisit`), then `bash whatisit/tools/split_repo.sh <its URL>` (keeps the history).
   After the split, in the new repo:
   - set the defaults in `tools/build.mjs` to `WII_REPO=daaronr/whatisit`, `WII_DIR=""`, branch `main`;
   - add a CLAUDE.md from the `whatisit/` section of word2vecgames' CLAUDE.md, plus this file's handoff rule;
   - `tools/build_vectors.py` reads `../web/data-sense`: keep it as a record (the vectors are already
     built into `site/judge-vectors.bin`), or copy the source vectors over;
   - Netlify: Project configuration > Build & deploy > Link the new repo; base directory empty;
   - in word2vecgames, replace `whatisit/` with a one-line pointer (or leave it frozen) and drop its
     CLAUDE.md/AGENTS.md section and `.gitignore` exceptions.

   Why: it shares nothing with Word Bocce at runtime (the vectors are already copied into `site/`),
   it has its own deploys, costs and Netlify site, and a private repo protects the content and answer
   keys (word2vecgames is public).
2. **Styling pass (visual and sound)** from David's style sheets: type scale, colour, motion on the
   reveal, a proper sound set (the current cues are synthesized placeholders in `sfx()`), and
   consistent habitats across the eleven categories.
3. **Reward suggestions.** Store suggestions with an id; when one ships, mark it used and credit the
   suggester in the changelog and on the mystery's reveal. Then points or prizes, per the rewards model.
4. **Grow answer keys from real guesses.** Export cached AI verdicts (Netlify Blobs `judge` store),
   review, and add common ones to `graded`, so more guesses are free over time.
5. **Verify drafts:** 40 brands, 5 double-take addresses, 5 headlines, 1 late-night item (marked
   `"status": "unverified"`).
6. **Online rooms** for remote play (reuse Word Bocce's PeerJS code in `web/net.js`).

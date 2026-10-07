# What Is It? to-do

Open items, newest first. Done items move to the changelog on the in-game Notes page
(`CHANGELOG` in `src/app.js`).

## Needs David

- **Feedback from the 6 Oct play session.** In-game notes on the Netlify site go to Netlify Forms
  (Project > Forms, once form detection is on) or stay in the browser they were typed in (Settings >
  "Copy them all"). Paste or export them into a session to act on them. Same for the hypothes.is
  notes: export from the Hypothesis sidebar, or allow `hypothes.is` and `api.hypothes.is` in the
  environment's network settings.
- **Style sources.** `claude_code_misc_work` (style sheets and style recommendations) isn't reachable
  from cloud sessions: push it to GitHub or attach the files, then do the styling pass below.
- **Rewards model.** Look at how complainments.netlify.app rewards suggestions (blocked from cloud
  sessions) and decide what we offer: credit by name, points, extra plays, a monthly prize.
- **Own repo.** Recommended (see below). Needs a new, preferably private, GitHub repo.

## Build next

1. **Move to its own repo.** `git subtree split --prefix=whatisit` keeps the history. Then: point
   `vectorUrls` in `tools/build.mjs` at the new repo, re-link the Netlify site (base directory becomes
   the repo root), move the CLAUDE.md/AGENTS.md section into the new repo's own CLAUDE.md, and leave
   a one-line pointer here. Reasons: it shares nothing with Word Bocce at runtime (the vectors are
   already copied into `site/`), it has its own deploys, costs and Netlify site, and keeping the
   content and answer keys in a private repo protects the work (this repo is public).
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

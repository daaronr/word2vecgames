# Word Bocce promotion package

Everything for telling people about [Word Bocce](https://wordbocce.davidreinstein.org/) lives here: drafts, a checklist, and a tracker of what went where. Nothing in this folder has been posted or sent. Update the tracker as things go out.

## What's here

| File | What it is |
| --- | --- |
| [lesswrong.md](lesswrong.md) | LessWrong post (~1,000 words): the game as a hands-on lesson in why interpretability is hard, three maps of meaning (including GPT-2's token table), where linear structure breaks |
| [show_hn.md](show_hn.md) | Show HN title and first comment |
| [reddit.md](reddit.md) | Posts for r/MachineLearning ([P]), r/LanguageTechnology, and a word-game subreddit |
| [short_posts.md](short_posts.md) | Interp Slack and Discord messages, ARENA and BlueDot, an email to NLP instructors, Semantle and Contexto framing, three Bluesky/Mastodon posts |
| [og-card.html](og-card.html) | Source for the link-preview image, [`web/og-image.png`](../web/og-image.png) (1200 × 630; how to regenerate is in the file) |
| [`web/teach.html`](../web/teach.html) | The teacher page, live at [wordbocce.davidreinstein.org/teach.html](https://wordbocce.davidreinstein.org/teach.html): a 30–45 minute lesson plan, linked from the game's footer and the slides |

Already live with this package: the preview image on the game, slides, and teacher pages, and a link on the last line of the Daily "Copy result" text.

The drafts were updated on 2026-10-01 for the AI tokens map (GPT-2's token table) and the Tokens tab, which landed after they were first written. If more features land before posting, re-check the "what it is" paragraphs.

The example used everywhere is the common-sense tutorial throw, hat + foot − head → shoe (shoe goes from 249th to 5th to 1st). It's checked in `tests/engine.test.js`. Other numbers in the drafts (ham's neighbors, eggs/meat/bacon ranks, cold being 16th-nearest to hot) were computed from the game's own vectors on 2026-10-01; they'll change only if the word bundles are rebuilt.

## Before posting

- [ ] **Real multiplayer test.** Play an online room with three to six people on different devices and networks, including a phone on mobile data. This is still open in the project's next steps, and HN or Reddit readers will try it.
- [ ] **Link previews.** Paste the game URL and the teacher-page URL into a Bluesky or Mastodon draft, a Slack DM to yourself, and a preview checker (for example opengraph.xyz) to confirm the new image shows. Some services cache old previews for a while.
- [ ] **Visit counting.** Decide on privacy-friendly counting before the first post, so you can see which channel worked. GitHub Pages has none built in. GoatCounter (free for non-commercial use) and Cloudflare Web Analytics are cookie-free; Plausible is paid. If you add one, use a tag per channel in the links you post, for example `?ref=lw`, `?ref=hn`, `?ref=reddit-ml`, `?ref=bsky`, `?ref=osmi`. The game reads only the part after `#`, so a `?ref=` tag doesn't affect play.
- [ ] **Venue rules.** Check each venue's self-promotion rules before posting: subreddit sidebars and pinned posts, HN's Show HN guidelines, and Slack or Discord pinned messages (or ask a moderator).
- [ ] **Known rough edges.** Fix these or decide to live with them, since technical readers will notice:
  - ~~Hard-coded "40,000 words" in the rank tooltip and vocabulary messages~~ Fixed 2026-10-01: they now say "every word on the map" and name the map in play.
  - ~~Rank tooltip didn't say that thrown words are left out~~ Fixed 2026-10-01: it now says so, as the drafts do.
  - The slides describe an earlier version of the game and still point to the old server address (it redirects).
- [ ] **A short clip.** Record 10–15 seconds of the tutorial (foot, throw, head, throw, bacio) for social posts.
- [ ] **Time to reply.** Post HN and Reddit only when you can answer comments for a few hours.

## Suggested order

1. **Bluesky and Mastodon**: low stakes, and a quick check that the preview image and the Daily share text look right in the wild.
2. **Open Source Mechanistic Interpretability Slack or EleutherAI**: a small, relevant audience that will give frank feedback on the "Why?" framing before the bigger posts.
3. **LessWrong**: the main written piece, and the best fit for the interpretability angle; the later posts can link to it.
4. **Show HN**: biggest reach, but one shot, so wait until the multiplayer test is done and the rough edges are fixed.
5. **Reddit** (r/MachineLearning, then r/LanguageTechnology, then a word-game subreddit), a few days apart, each tailored to its audience.
6. **NLP instructors, ARENA, and BlueDot**: after one real classroom or facilitator try of the teacher page, and timed for the start of a term or cohort.
7. **Semantle and Contexto players**: whenever, in a word-game community that allows links; this is the general-audience pitch.

## Tracker

| Channel | Draft | Status | Posted URL | Date |
| --- | --- | --- | --- | --- |
| Bluesky | [short_posts.md](short_posts.md#bluesky-and-mastodon) | draft | | |
| Mastodon | [short_posts.md](short_posts.md#bluesky-and-mastodon) | draft | | |
| Open Source Mech Interp Slack | [short_posts.md](short_posts.md#open-source-mechanistic-interpretability-slack-or-the-eleutherai-interpretability-channel) | draft | | |
| EleutherAI interpretability channel | [short_posts.md](short_posts.md#open-source-mechanistic-interpretability-slack-or-the-eleutherai-interpretability-channel) | draft | | |
| LessWrong | [lesswrong.md](lesswrong.md) | draft | | |
| Show HN | [show_hn.md](show_hn.md) | draft | | |
| r/MachineLearning | [reddit.md](reddit.md#rmachinelearning) | draft | | |
| r/LanguageTechnology | [reddit.md](reddit.md#rlanguagetechnology) | draft | | |
| r/wordgames (or similar) | [reddit.md](reddit.md#rwordgames-or-a-similar-word-game-subreddit) | draft | | |
| ARENA community | [short_posts.md](short_posts.md#arena-and-bluedot-ai-safety-fundamentals-course-communities) | draft | | |
| BlueDot community | [short_posts.md](short_posts.md#arena-and-bluedot-ai-safety-fundamentals-course-communities) | draft | | |
| Intro NLP instructors (email) | [short_posts.md](short_posts.md#email-to-intro-nlp-instructors) | draft | | |
| Semantle and Contexto players | [short_posts.md](short_posts.md#semantle-and-contexto-players) | draft | | |

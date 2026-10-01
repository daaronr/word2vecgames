# Short posts and messages (drafts)

Status: drafts, not posted or sent. Index and checklist: [README.md](README.md).

Check each channel's norms first. In Slack and Discord communities, look for a #projects, #show-and-tell, or #resources channel, or ask a moderator, rather than posting in a research channel by default.

---

## Open Source Mechanistic Interpretability Slack, or the EleutherAI interpretability channel

> I made a small browser game on word embeddings that might be a useful warm-up for people new to interp: https://wordbocce.davidreinstein.org/
>
> You add and subtract word vectors to roll a ball toward a target word. After each throw, a "Why?" panel splits the ball's cosine to the target into exact per-word shares, then says this is arithmetic, not the model's reasons. Example: in hat + foot − head → shoe, subtracting head *lowers* the similarity but moves shoe from 5th to 1st, because it pushes competitors back.
>
> To be clear, these are static embeddings: GloVe, ConceptNet Numberbatch, and GPT-2's input token table (one fixed vector per token, before any attention layer). Not transformer internals. I'd value a critical eye on how the panel frames attribution versus explanation, and on whether there's a better decomposition to show.

---

## ARENA and BlueDot (AI Safety Fundamentals) course communities

Best sent to course organizers, or posted where participants share resources, ideally near the start of a cohort.

> A possible pre-reading activity for people new to interpretability: Word Bocce, a free browser game on word embeddings. https://wordbocce.davidreinstein.org/
>
> It takes about 10 minutes to get the idea. You do vector arithmetic on words (hat + foot − head → shoe), and a "Why?" panel breaks each result into exact per-word contributions, then says why that still isn't an explanation. You can also switch between an embedding learned only from text and one that adds a knowledge graph, and watch "ham" go from football clubs to bacon. A third map is GPT-2's own token table, and a Tokens tab shows how GPT-2 splits text into tokens.
>
> It's static vectors (at most GPT-2's input embeddings), not a transformer's internals, so it's a warm-up rather than mech interp proper. There's a short lesson plan here if a facilitator wants to use it in a session: https://wordbocce.davidreinstein.org/teach.html

---

## Email to intro NLP instructors

Send individually, not as a mass mailing. Personalize the first line (their course, a syllabus topic) or don't send.

**Subject:** A free word-embedding game for your intro NLP class

> Hi [Name],
>
> I made a free browser game that might be useful for the word-embeddings week of [course]: https://wordbocce.davidreinstein.org/
>
> Students add and subtract word vectors to land near a target word. It shows cosine similarity and rank, breaks each throw into exact per-word contributions, and lets you switch between GloVe and ConceptNet Numberbatch, which makes polysemy and the effect of training data easy to see ("ham" is a football club in GloVe). No accounts or installs; it runs on phones.
>
> I wrote a 30–45 minute lesson plan with discussion questions on bias and on why attribution isn't explanation: https://wordbocce.davidreinstein.org/teach.html
>
> If you try it, I'd like to hear what didn't work. No need to reply otherwise.
>
> Best,
> David

---

## Semantle and Contexto players

For word-game communities (a subreddit, Discord, or forum where these players gather). Check whether outside links are allowed.

> If you like Semantle or Contexto, you might like Word Bocce: https://wordbocce.davidreinstein.org/
>
> It uses the same kind of word map, flipped around. You're told the target word, and you build a path to it by adding and subtracting other words (hat + foot − head → shoe). Like Contexto, you're scored by rank: 1st means your target is the closest word to where you landed. There's a Daily with a shareable result, plus puzzles and a versus mode.

---

## Bluesky and Mastodon

All three fit Bluesky's 300-character limit. Post one, see what happens, and space the others out over a week or two.

**1. The game** (attach `web/og-image.png`, or a 10–15 second screen recording of the tutorial: tap foot, throw, tap head twice, throw, bacio)

> I made a word game: bocce played on a map of meanings. Add and subtract words to roll your ball to the target.
>
> hat + foot − head → shoe
>
> Free, daily puzzle, works on a phone:
> https://wordbocce.davidreinstein.org/

**2. Two maps of meaning** (attach a screenshot of the Words dialog, or of the end-of-round "words closest in meaning to …" list in each word set)

> Same word, two training sets. In GloVe (Wikipedia and news), the nearest words to "ham" are Sunderland, Fulham and Middlesbrough. In ConceptNet Numberbatch: bacon, pork, sausage.
>
> You can switch between them in Word Bocce:
> https://wordbocce.davidreinstein.org/

**3. The "Why?" panel** (attach a screenshot of the "Why?" dialog after the tutorial's second throw, showing the bars and the caveat box)

> My word game has a "Why?" button. It splits each throw into exact per-word contributions, then says: this is our reading of the numbers, not the model's reasons.
>
> Exact attribution, still not an explanation. A tiny interpretability lesson:
> https://wordbocce.davidreinstein.org/

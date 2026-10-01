# Reddit posts (draft)

Status: drafts, not posted. Index and checklist: [README.md](README.md).

Check each subreddit's rules (sidebar and pinned posts) before posting. Some limit self-promotion, require flair, restrict which days project posts are allowed, or want a minimum of community participation first. Post one at a time, a few days apart, rather than cross-posting the same text, and stay around to answer comments.

---

## r/MachineLearning

Flair: [P] (Project). Check the current rules on self-promotion first.

**Title:** [P] Word Bocce: word-vector arithmetic as a browser game, with an exact per-word attribution panel that says it isn't an explanation

**Body:**

Word Bocce is a small browser game on static word embeddings: https://wordbocce.davidreinstein.org/

Your ball starts on a word, the jack (target) is another, and you add and subtract tiles to move toward it: ball = unit(v(start) + Σ ±v(tile)). The score is the jack's rank among the ball's nearest words, with similarity only breaking ties.

A few things that might be of interest:

- **Two embeddings, same game.** ConceptNet Numberbatch 19.08 (300d, 21,114 everyday words) is the default; GloVe 6B (100d, about 40k words) is a switch away. The differences are large and easy to see. In GloVe, "ham" sits with Sunderland, Fulham, and Middlesbrough, "orange" with other colors, and "apple" with Microsoft and IBM.
- **Attribution vs. explanation.** A "Why?" panel decomposes cos(ball, jack) = Σ sign·cos(word, jack) / |Σ sign·v(word)| into exact per-word shares, then says this is a reading of the numbers, not the model's reasons. In the tutorial (hat + foot − head → shoe), subtracting head *lowers* the similarity to shoe but moves it from 5th to 1st, by pushing competitors (toe, toes, beanie) back.
- **Selection effects.** With the default word set, courts are only dealt if one of the strongest throws passes a legibility test (every added word has cosine ≥ 0.3 with the jack, every subtracted word ≥ 0.3 with the start). That makes the default look more interpretable than the raw embedding is.
- **Rank skips the input words**, like classic analogy evaluation (Nissim et al., 2020, criticize this). Without it, king − man + woman lands nearest to king.

It runs entirely client-side (int8 vectors, brute-force scoring in JavaScript), and the source is here: https://github.com/daaronr/word2vecgames. Most of the code was written with Claude Code, with me directing and playtesting.

Feedback I'd find useful: better ways to present the decomposition, whether a contextual-embedding version would teach more or just be murkier, and bugs.

---

## r/LanguageTechnology

**Title:** A browser game for comparing GloVe with ConceptNet Numberbatch: polysemy, antonyms, and "king − man + woman"

**Body:**

I built a small game on word embeddings, Word Bocce: https://wordbocce.davidreinstein.org/

You start on one word and add or subtract word tiles to land near a target word, scored by the target's rank among the nearest words. You can switch between two embeddings, and comparing them turned out to be the most interesting part:

- **GloVe 6B** (100d, Wikipedia and news): "ham" is a football club (its nearest words are Sunderland, Fulham, Middlesbrough), "key" mostly means "important", and the nearest word to "cold" is "warm". From "eggs", "meat" is 7th-nearest and "bacon" 163rd.
- **ConceptNet Numberbatch 19.08** (300d, text plus a knowledge graph): ham sits with bacon, pork, and sausage, and eggs with hens and yolks. Antonyms still sit close, though: cold is the 16th-nearest word to hot.

Two honest notes. Rank excludes the words you threw, as standard analogy evaluation does. In the raw-text set, queen is already the 2nd-nearest word to king, and with king − man + woman the ball is nearest to king itself, so the famous example is flattered. And the "Why?" panel gives an exact per-word breakdown of each throw's similarity, which is arithmetic, not an explanation of what the model learned.

I also wrote a 30–45 minute lesson plan for intro NLP classes: https://wordbocce.davidreinstein.org/teach.html. Suggestions from people who teach this material would be very welcome. Source: https://github.com/daaronr/word2vecgames

---

## r/wordgames (or a similar word-game subreddit)

Check the rules first. Some word-game subreddits only allow links to your own game on certain days or in a weekly thread.

**Title:** Word Bocce: roll your word toward a target by adding and subtracting other words (free, daily puzzle)

**Body:**

I made a word game that works like bocce, played with meanings: https://wordbocce.davidreinstein.org/

Each round gives you a starting word, a target word (the "jack"), and nine word tiles. Add or subtract up to three tiles per throw to roll your ball closer. For example, hat + foot − head lands on shoe. Your score is how close the target ends up: 1st means it's the nearest word to your ball.

There's a Daily (the same court for everyone, with a shareable result), endless practice, hand-made puzzles with star ratings, and versus against a bot or friends. It's free, there's no sign-up, and it works on a phone. A 30-second tutorial runs the first time.

If you play Semantle or Contexto, it uses the same kind of word map, but you're given the target and have to build a route to it.

Some tiles are traps: they look related but pull you the wrong way. I'd love to hear which puzzles feel unfair or too easy.

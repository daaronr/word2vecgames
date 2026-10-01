# Show HN (draft)

Status: draft, not posted. Index and checklist: [README.md](README.md).

Post when you can stay around to answer comments for a few hours. Show HN rules: it has to be something people can try, with no sign-up wall (fine here). Don't ask anyone to upvote.

## Title (61 characters)

Show HN: Word Bocce – a lawn game played in word-vector space

URL: https://wordbocce.davidreinstein.org/

## First comment

I made a small browser game on word embeddings. Your ball starts on one word, the jack (target) is another, and you add and subtract word tiles to roll the ball closer, as in king − man + woman → queen. Distance is shown as rank: #1 means the jack is the nearest word to where your ball stopped. There's a Daily court (same for everyone), endless practice, hand-made puzzles, and versus against a bot, a friend on the same device, or up to six people online.

Some details that might interest people here:

- It's all static files on GitHub Pages. The vectors ship as int8 (6.5 MB for the default set), and every throw is scored in the browser by a brute-force scan of the vocabulary. The engine is about 300 lines of plain JavaScript.
- There are two word sets. The default is ConceptNet Numberbatch 19.08 (21,114 everyday words, 300 dimensions), which mixes text statistics with a knowledge graph of everyday facts. The other is GloVe 6B (about 40,000 words, 100 dimensions), learned only from Wikipedia and news. The difference is a good lesson in what training data does: in GloVe, "ham" sits with English football clubs (West Ham) and "orange" with other colors.
- Par is computed by trying all 834 throws a nine-tile hand allows (up to three tiles, each + or −), ranking the 50 most similar by the jack's rank, and keeping the best.
- After each throw, a "Why?" panel splits the ball's similarity to the jack into exact per-word shares (they add up, since the ball is a normalized sum of unit vectors). It then says plainly that this is arithmetic, not the model's reasons. Exact attribution still isn't explanation, even here.
- Rank skips the words you threw, as classic analogy tests do. Otherwise king − man + woman lands nearest to king.
- Online rooms are browser to browser over WebRTC (PeerJS for the handshake); the host's tab holds the game, and nothing is stored.

Most of the code was written with Claude Code, with me directing and playtesting. Source: https://github.com/daaronr/word2vecgames. There's also a lesson plan for teachers: https://wordbocce.davidreinstein.org/teach.html

I'd welcome bug reports, puzzles that feel wrong, and ideas for modes. One I'm unsure about: a version with contextual embeddings from a small transformer. Would that be more interesting, or just murkier?

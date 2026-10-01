# LessWrong post (draft)

Status: draft, not posted. Index and checklist: [README.md](README.md).

Title options (pick one):

- Word Bocce: a word-vector game whose "Why?" button admits it isn't an explanation
- A small game about why explaining even word embeddings is hard

Suggested tags: Interpretability (ML & AI), Machine Learning (ML), Practical. Before posting, swap in a `?ref=lw` link if visit counting is set up (see the checklist).

---

I made a small browser game, [Word Bocce](https://wordbocce.davidreinstein.org/). It's bocce played on a map of word meanings. Your ball starts on one word, the target (the jack) is another, and you add and subtract word tiles to roll the ball closer: hat + foot − head → shoe. Distance is shown as rank: #1 means the jack is the nearest word to where your ball stopped. It's free, there's no sign-up, and it works on a phone.

I'm posting it here because it turned into a hands-on lesson in why interpretability is hard, and I'd like feedback from people who think about that more than I do.

One caveat up front. These are static embeddings: one fixed vector per word or token. Two of the maps are classic word embeddings, GloVe (2014) and ConceptNet Numberbatch (2019). The third is GPT-2's own token table, the vectors the model looks up for each token before any attention layer runs. So at most this is the doorway into a transformer, not its internals. Treat it as a warm-up, or an intuition pump.

## The "Why?" panel

After every throw you can tap "Why?". Because the ball is the normalized sum of unit word vectors, its similarity to the jack splits exactly into one share per word:

cos(ball, jack) = Σ sign · cos(word, jack) / |Σ sign · word|

The panel shows those shares as bars, and they add up exactly. Here's the tutorial throw. Shoe starts as the 249th-nearest word to hat. Adding foot makes it 5th, with similarity 0.51, and the ball stops near "toe". Then subtracting head makes it 1st, a *bacio*, even though the similarity *drops* to 0.48. The shares are hat 0.16, foot 0.34, head −0.02. Subtracting head barely touched shoe; what changed was that "toe", "toes", and "beanie" stopped beating it.

So the bars are exact, and they still don't tell you why "toe" sat where it did, which is what you actually wanted to know. The panel ends every reading with "This is our reading of the numbers, not the model's actual reasons." It also flags patterns that confuse people: subtracting a word that is itself close to the jack, similarity rising while rank falls, and partial cancellation leaving a strange residue (on the raw-text map, bacon − meat lands among people's names, because Bacon is also a surname).

None of this is deep, but it helps to have it in a form you can poke at. If an exact attribution doesn't explain a three-word sum, where we know the entire computation, we should be at least as careful with attributions in much bigger models.

## A scoring rule I should be upfront about

The game's rank leaves out the words you threw. Without that, the ball from hat + foot − head is nearest to foot, then hat, then shoe, and king − man + woman is nearest to king. It's the convention [Nissim, van Noord & van der Goot (2020)](https://aclanthology.org/2020.cl-2.7/) criticize in the analogy literature. For a game I think it's the right rule (otherwise you'd keep "landing" on your own tile), but it flatters the famous analogy: in the raw-text set, queen is already the 2nd-nearest word to king before you throw anything, and in the common-sense set it's already 1st.

## Three maps of meaning

You can switch between three word sets:

- **Raw text**: GloVe 6B, 100 dimensions, about 40,000 words, learned only from which words appear together in Wikipedia and news.
- **Common sense** (the default): ConceptNet Numberbatch 19.08, 300 dimensions, 21,114 everyday words, which blends text statistics with a knowledge graph of everyday facts ("a hen lays eggs").
- **AI tokens**: GPT-2 small's token embedding table, 49,745 tokens, reduced from 768 to 128 dimensions with PCA. The neighbours are a quick lesson in what a language model's vocabulary is: case and spacing variants ("␣Shoe", "Shoe") and word pieces, not just words.

There's also a Tokens tab, using GPT-2's real tokenizer: split your own text, play an eight-round "guess the split" quiz, and step from tokens to a chosen next word with a temperature slider.

Same game, different training data, different "meaning". In the raw-text set, the nearest words to *ham* are Sunderland, Fulham, Middlesbrough, and Wigan (as in West Ham). *Orange* is a color, *apple* is Microsoft and IBM, and the nearest word to *cold* is *warm*. From *eggs*, "meat" is 7th-nearest and "bacon" 163rd. In the common-sense set, ham sits with bacon, pork, and sausage.

The design history is a small cautionary tale. I started with GloVe alone, and many hand-made puzzles had best throws nobody could explain: bacon → eggs couldn't do better than 20th, and a typical dealt par read "doll − dragon + flour + puppy → peanut". So the common-sense set became the default, and it only deals a court when one of its strongest throws passes a simple legibility test (every added word has cosine ≥ 0.3 with the jack, every subtracted word ≥ 0.3 with the start word). With common-sense words, 44 of the 60 hand-made puzzles pass the game's quality checks; with raw text, 32.

That's a selection effect worth naming. The default game looks more interpretable partly because I filtered for courts that look interpretable, which is roughly what happens whenever someone picks the examples for a paper or a demo.

## Linear structure, and where it breaks

The linear structure is real enough to build a game on: every dealt hand has tiles that pull toward the jack, tiles worth subtracting, and traps. The places it breaks are the instructive part:

1. **One vector per word.** Ham the meat and ham the football club share a point, and so do key the object and key meaning "important".
2. **Opposites sit close.** Even in the common-sense set, cold is the 16th-nearest word to hot, so in the hot → cold puzzle, *adding* "warm" moves you toward cold.
3. **Rank and similarity disagree.** As in the tutorial, a throw can lower the similarity and still win, because what matters is the crowd around the jack.

## What I'd like feedback on

1. Is the "Why?" framing right? Is there a better decomposition to show, or a clearer way to say what it doesn't tell you?
2. Is this useful as a warm-up before interpretability material, for example in ARENA or BlueDot-style courses? I wrote a [lesson plan for teachers](https://wordbocce.davidreinstein.org/teach.html); I'd welcome suggestions on it.
3. The token map uses GPT-2's input embeddings. Would a version built on contextual vectors (say, the residual stream partway through a small model) be worth making, or would it lose the clarity that makes this work?
4. Bugs, confusing wording, and puzzles that feel wrong.

Most of the code was written with Claude Code, an AI coding agent, with me directing and playtesting. The source is on [GitHub](https://github.com/daaronr/word2vecgames); the engine is about 300 lines of plain JavaScript, with tests that run it against the real vectors.

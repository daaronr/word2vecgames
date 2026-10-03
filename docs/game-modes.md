# Game-mode ideas

Ideas for multiplayer variants, noted 2026-09-29. None of these is built yet
except the first. All of them can run on the online-room code (`web/net.js` plus
the "party" section of `web/app.js`): the host's browser holds the state, and
only seeds, tile choices and words are sent between players.

## Built: simultaneous room ("Online with friends")

Everyone plays the same court at once, three balls each, and sees the others'
balls land with their words hidden until the round ends. The best ball wins the
round and scores a point for each of its balls that beats everyone else's best.
The aim is to get people trying the game together and to surface bugs.

## 1. Gift words (alternating, two players)

After each throw, the thrower must offer the opponent **two words**. Neither may
be a near-synonym of the jack; use the same test the deal uses (cosine to the
jack under 0.6, and not `related()` to the jack or the start word). The opponent
may add one of them to their hand. If they take it, the offerer can no longer
use it, though they can still use the word that wasn't taken.

- The tension: offer words that look useful but pull the wrong way, without
  handing over the tile you need.
- Needs: a turn-based room (alternating throws; the versus turn logic already
  exists), a word-entry step between throws, and per-player hands.

## 2. Guess the jack (one bowler, two or more guessers)

The bowler knows the start word and the jack, and has a wide hand. The guessers
see only the start word and where each ball lands (the "stopped near" words,
not the tiles). After the bowler's balls, each guesser names the jack. The
closest guess (by rank or similarity to the true jack) wins.

- To make the bowler try to be informative, the bowler could score with the
  best guesser, which makes it cooperative in the way Codenames is.
- Needs: hidden-information state per player (the host sends the jack only to
  the bowler), a guessing phase, and a rotating bowler role.

## 3. Calling your shot

In an alternating game where combinations can't be reused, a player can declare
"one-ball round". They throw once, and everyone else also gets only one ball.

- It needs a cost, or everyone would do it whenever they're first. Options:
  - If the caller doesn't win the round, each opponent scores 2 points.
  - The caller must name a target ("top 10") and loses points if they miss it.
  - Calling costs a point up front, and it's returned double on a win.
- Needs: the turn-based room from idea 1, and a declaration step before the
  first throw.

## 4. Feedback that improves the puzzles

Partly built. After any round, "Try your own words" lets a player type a throw
using any words, see where it would have landed, and say whether having those
words would have made the court more fun. Each suggestion is saved with its
seed, start, jack, hand, the throw, its rank, and the verdict.

Open question: suggestions are only stored in the player's browser for now.
They need somewhere to go, for example a small form endpoint or a shared
database. Once they're collected, they can:

- promote suggested words into `pools.json` (or weight them in `deal()`),
- flag deals where players repeatedly wanted a word that the deal filters out,
- tune the hand mix (pulls, sheds, lures, near-misses).

## 5. Cat and mouse (a chase) — first version built

Suggested 2026-10-03. One player is the mouse, the other the cat. Both start
on the same word. On each turn a player plays one card, added or subtracted,
and hops to a new word; the cat tries to land close enough to the mouse to
catch it, and the mouse tries to stay away. A caught mouse makes the cat the
new mouse.

### What the first version does (`renderChase` in `web/app.js`, `Terrain` in `web/engine.js`)

- **Terrain.** Every card and every landing spot comes from one small themed
  word set (`web/chase/terrains.json`). The first is "80s & 90s TV" (131
  words: seinfeld, kramer, homer, bart, mulder, scully, picard, baywatch,
  sitcom, cliffhanger, rerun, diner, precinct…), on the raw-text map, which
  knows the show and character names (the common-sense map doesn't).
- **Moves hop to words.** From word W, card C with sign s lands on the terrain
  word nearest to W + s·C (not W or C): seinfeld + bart − jerry → simpsons.
  An earlier version that kept adding cards onto a running total drifted into
  meaningless regions within a few moves.
- **Catching.** The cat catches the mouse when the mouse's word is among the 5
  terrain words nearest the cat's word (or they're on the same word). The
  mouse can also blunder into the cat.
- **Cards.** 16 shared face-up cards plus 3 private cards each. A played card
  is used up for both players, so the cat can't follow with the mouse's card.
  You can't play the word you're standing on.
- **Uno-style cards** in the shared row: Double (move twice) and Skip (the
  other player misses a turn).
- **Wild words.** Every third mouse move, the mouse may play any word on the
  map as a card.
- **Traps.** Two trap words per round. A cat that lands within the 2 words
  nearest a trap explodes, and the mouse gets all the remaining points.
- **Head start.** The mouse moves three times before the cat's first move.
  Without it the cat, starting on the same word, catches the mouse on move 1
  almost every time.
- **Pounce.** At its 5th and 10th moves the cat earns a pounce: two cards
  combined in one move (kept until used). This is the catch-up power that
  keeps the chase winnable for the cat.
- **Scoring.** The mouse earns 10 points for every cat move it survives
  (12 cat moves a round). A catch gives the cat 50 and makes it the mouse. If
  the mouse lasts all 12, it gets 50 more and stays the mouse. The cat may
  give up at any time; the mouse then gets half the remaining points.
- **Opponents.** The bot (greedy, choosing at random among its two best
  moves; it uses Skip when the cat is close and Double when the mouse is far)
  or a friend on the same device.
- **Map.** A 2-D sketch of the terrain (its two strongest directions, by PCA),
  with both players' trails, the cat's reach and the trap zones.

Balance, from bot-against-bot simulations (40 rounds each; the "human-like"
cat picks at random among its 3 best moves, against the game's bot mouse):

| Rules | Human-like cat catches | Typical catch |
| --- | --- | --- |
| 2-move head start, no pounce | 13 / 40 | move 1 (luck) |
| + pounce every 3rd move | 33 / 40 | move 3 |
| + pounce every 4th move | 29 / 40 | move 4 |
| + extra cat move every 4th turn | 19 / 40 | move 2 |
| 3-move head start, pounce every 5th (built) | 24 / 40 | move 5 |

A sharper cat catches more (37 / 40 with a pounce every 4th move). With cards
from the whole vocabulary the cat almost never catches anyone, since one card
throws the mouse anywhere, which is why the terrain matters. The test suite
checks the human-like bot cat catches the bot mouse in 3–11 of 12 games.

### Version 2 (simpler, friendlier)

After a first look: "simplify it, more handholding, more fun".

- **Simple rules by default**: one shared row of 14 cards (refilled from a
  deck, so it never runs out; version 1 could stall late in a round), add
  only, the cat's pounce and the traps. Private cards, subtracting, Double,
  Skip and wild words are behind an "Extras" box.
- **An illustrated intro** (first visit, and "How to play" any time) with a
  choice of playing the mouse or the cat.
- **A coach line every turn** saying what to do and what just happened.
- **Every card shows where it leads**; as the mouse, also how dangerous that
  spot is (Safe / Risky / Danger / Caught!). The cat gets destinations but no
  closeness labels: with full information a cat catches the mouse almost
  every time on a terrain this small (27–30 of 30 in simulations, whatever
  the catch radius), so the cat has to judge closeness of meaning itself. The
  bot cat picks among its 3 best moves, so a careful mouse can escape.
- **Characters**: a drawn cat and mouse glide across the map; the picked
  card's destination shows as a dotted arrow.
- **The big moments**: a pouncing-cat animation with a meow and a squeak for a
  catch, an explosion for a trap, a dancing mouse for an escape.

### Version 3 (everyday words, no previews, a story)

Feedback on version 2: the TV terrain didn't work (too narrow, and you have to
know the shows); cards showing where they lead "ruins the fun"; very different
cards led to the same word and an easy catch; the bot's moves were hard to
see; the mouse's opening head start didn't make sense; the cat had it too
easy. And a suggestion: end with a story of the journey.

- **Terrain: "Everyday things"**: the 1,285 concrete nouns the common-sense
  map uses as jacks (animals, food, places, jobs, objects), on that map
  (ConceptNet Numberbatch), where relationships mostly read sensibly:
  kitchen + car → garage, house − snow → mansion. Ten times as many words as
  the TV terrain, so different cards land in different places. The TV terrain
  stays in `terrains.json` with `hidden: true`.
- **No previews in play.** The intro's worked example is the only place
  where the game shows where a card leads; in the game, guessing that is the
  skill. Every card can be added or subtracted (tap once for +, twice for −).
- **No going back.** A player can't land on a word they've already visited
  (`visited` in `Terrain.hopTiles`); without this, players bounce between
  near-synonyms (highway, freeway, highway…). Near-copies of the start and of
  the cards (`related()`) are skipped too.
- **The getaway**: the mouse's first move plays up to three cards at once,
  each + or −, in place of three separate head-start moves. The bot mouse
  tries 150 random combinations (`Terrain.botGetaway`).
- **Catch radius 40** (of 1,285 words, about 3%), traps hit within 3 words.
  The map shades the cat's reach orange.
- **What just happened**: a banner after every move, the bot's included,
  shows the cards played and where they led ("The bot (cat) played hill +
  boyfriend → girlfriend. The mouse is hot (102nd nearest to the cat)."),
  and two "journey" columns list every move. Passed words stay labelled on
  the map.
- **The story of the chase**: at the end, a short story built from both
  journeys ("…the cat padded after it, past the girlfriend, the stove, the
  oven…"), "Tell it again" for another telling, and a button that copies a
  prompt with both journeys for any chatbot to write a better one.

Balance (bot against bot, 10 games each, 12 cat moves, pounce every 5th):

| Catch radius | Bot cat picks among its best | Bot mouse | Caught |
|---|---|---|---|
| 10 | 3 | 2 | 2–4 |
| 40 | 1 | 2 | 10 |
| 40 | 3 | 2 | 7 |
| 60 | 3 | 2 | 9 |
| 40 | 5 | 4 | 4 |
| 30 | 4 | 4 | 5 |
| **40** | **5** | **3** | **4** (shipped; `tests/engine.test.js` keeps it in 2–8) |

A person can't see where cards lead, so they play below the greedy bots; the
bots here are deliberately loose.

### Still to try

- **More terrains**: themed ones (food, sports) for variety, a GPT-2 token
  terrain (capitalised names live there), a picker once there are several.
- **Online rooms** (on `web/net.js`), and **several cats** racing to catch one
  mouse first.
- **Private cards the others can see but not use**, as suggested: a visible
  hand tells the cat where the mouse might run.
- **More Uno cards**: Reverse (swap roles for one turn), Draw two (take two
  cards from a face-down pile), Freeze (the other player must play a shared
  card).
- **Random wild turns** instead of every third move, and wild words for the
  cat too, rarer.
- **Calibration**: the pounce interval (every 5th move now) and head start,
  catch radius, round length (12 cat moves now; 20 was
  suggested), points for giving up (half now; a quarter was suggested), and
  whether the bot should bluff.
- **Traps the mouse can lure the cat onto**, scored for the mouse.

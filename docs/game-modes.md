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

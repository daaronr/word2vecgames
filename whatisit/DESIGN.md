# What Is It? Design notes

October 2026. Working notes for the trial versions in this folder: what the game is, how it is
built, what it would cost to run, who pays for the AI, how it might make money, what already exists,
and what to decide after the trials. Facts gathered on 6 October 2026; items marked *(unverified)*
came from search summaries and should be re-checked before anyone relies on them.

## 1. The game

Someone shows a cryptic string: a web address (`kafka.com`), a brand (`Liquid Death`), a
vanity plate (`KNKU OUT`), a patent title ("Method of exercising a cat") or a phrase to search
("Bring your own balls"). Everyone guesses what it really is, or what comes up first when you search
it. Then the reveal: the site, the product, the owner's own explanation, the patent drawing, the top
result. Whoever was closest wins the round; whoever was funniest can win a bonus.

The fun is the gap between what a name suggests and what it turns out to be. `kafka.com` is a PR
agency in Orlando run by two people called Kafka. `KNKU OUT` was an anaesthetist's plate the DMV
refused as a threat. "Method of exercising a cat" is a laser pointer. The best items pull your guess
firmly one way and land somewhere else; the next best are delightfully specific.

Working titles: *What Is It?* (used in the trials), *What Are They Selling?* (good for a brand-only
spin-off), *Really?*, *Name Game* (taken by others; avoid).

## 2. The four trial versions

All four are in one page, with a "how fun was that one?" rating after every reveal and a
rate-this-version form after each session.

| Trial | For | Core loop | What we learn |
|---|---|---|---|
| A. Daily five | 1 player, 5 minutes | Five mysteries a day from five of the ten everyday categories (a seeded draw per day). Type guesses; the judge shows how warm each is (0-100): answer key first, then the AI judge, then the robot. Up to 4 guesses, 3 clues (each cuts the points), "pick from four" as a fallback, a DMV side bet on plates, a share grid. After the reveal, "ask the AI judge" for a second opinion. | Is a solo daily fun? Is the robot judge fair enough? Do players prefer typing or picking? |
| B. Party board | 2+ in a room or on a call, one screen | A Jeopardy-style points board (3 rows of 100-300 or 5 rows of 200-1,000), tic-tac-toe or Connect Four for two teams. Points-board options: lower values in a column unlock higher ones (the default; on TV any square may be picked), whoever takes a square picks next, one hidden Daily Double (the picker answers alone for a wager), way-off guesses lose the value, and a Final mystery with wagers. Connect Four: the picking team chooses a column (each column is a category) and whichever team guesses closer drops its disc there. Guesses typed on a passed-round device, or on paper. Free judge, AI, or the room judges; a "funniest guess" bonus. | Which board? Do the Jeopardy rules add tension or just length? Does judging feel fair or fiddly? |
| C. Bring your own | 2+ | The original game: type any word, phrase or address; guess; search together via Google/DuckDuckGo/Bing/Wikipedia links; type what you found; judge. Optional Family-Feud bonus for results 2 and 3. | Is the free-form version more fun than curated items? |
| D. Bluff | 3+, pass the phone | Balderdash with real mysteries: everyone writes a believable fake, the truth and one of our decoys are mixed in, everyone votes. | Is fooling friends more fun than guessing? |

## 3. Design decisions

**Categories.** Six now: Dot-com mysteries, What are they selling?, Vanity plates, Patent office,
Top result, and Double take (PG-13 misreadings such as `expertsexchange.com`, off by default and never in
the Daily). Each mystery is drawn in its own habitat: an address bar, a shop sign, a California
plate, a patent front page with its INID codes, a search box. Candidates for later: app names, band
names, paint and nail-polish colour names (scored by colour distance), startup pitches, Wikipedia
oddities, product names in other languages.

**Vanity plates are the strongest category we found.** The California DMV applications dataset
(23,463 flagged applications, 2015-16, collected by Noah Veltman through a public-records request)
gives the owner's own explanation *and* the reviewer's worry *and* the verdict. That makes a
two-part reveal ("the owner says it means *I put people to sleep*; the DMV read it as a threat;
denied") and a free side bet (approved or denied?). The file is mostly vulgar, so every plate in the
game was picked by hand; 62 are in.

**Clues.** Every item has a ladder of three: vague, warmer, nearly gives it away. In the Daily
each clue multiplies the points by 0.85, 0.70, 0.55; on the party board each clue lowers the square's
value. Players can suggest a better clue from any reveal, which feeds the content loop below.

**Three judges.**

| Judge | How | Cost | Good at | Bad at |
|---|---|---|---|---|
| The room | People tap the winner | Free | Party play, humour, context | Solo play, arguments |
| Robot | Word vectors: compares the guess's words with each item's 2-5 key ideas (ConceptNet Numberbatch, the Word Bocce "common sense" map, 96 dims) | Free, instant, offline | Fair ranking of plain guesses; it explains each match; spots guesses that fell for a decoy | Generous with close cousins (shoes for a sock company); literal; knows no brand names |
| AI | An LLM reads every guess against the truth, scores 0-100 with a reason, and picks the funniest | About 0.015-1¢ a round | Nuance, partial credit, humour | Costs money; slower; needs an account |

**Tested (6 Oct 2026):** 31 mysteries, 217 realistic guesses, each scored by hand before any judge
saw them (`eval/judge-eval.json`; rerun with `node whatisit/eval/score_judges.mjs`).

| | Word vectors (robot) | Claude Haiku 4.5, same prompt the game uses |
|---|---|---|
| Picks the right winner | 27 of 31 rounds (87%) | 31 of 31 (100%) |
| Same order as the hand scores (Spearman, per round) | 0.81 | 0.95 |
| Correlation with the hand scores | 0.78 | 0.96 |
| Mean error | 18 points | 8 points |
| Wrong guesses scored warm | 13 of 74 | 4 of 74 |
| Right guesses scored cold | 8 of 72 | 0 of 72 |

The robot's misses are the ones players would notice: "a PR firm" for a PR agency scored 18 (it
doesn't know "PR"), "condiments" for mayonnaise 0, "an off-roader" 21, while "spray-on hair powder"
scored 89 for the comb-over because it shares the words hair and spray. A cheap model gets these
right and explains itself. Caveat: one person wrote both the guesses and the hand scores.

So the site now uses a small AI model as the judge (a Netlify Function, below), and the robot stays
only as the free fallback when the function is down or over its daily cap.

**Cheating.** In a solo game you can always search the answer, as you can look up a crossword. A
daily can ask for a first guess before any clue, show a timer, and rely on honour. Party play
polices itself.

**Search results are a moving target.** Engines differ and change, personalise by place and
person, and the APIs are shrinking (below). For curated "Top result" items we freeze one engine's
answer with a date; Bring-your-own does the live search on the players' own devices.

## 4. How it's built (trial version)

- One static page (`site/index.html`, about 480 KB with all 167 mysteries inlined), one Netlify
  Function for the AI judge (`netlify/functions/judge.mjs`), and the fallback robot's data file
  (`judge-vectors.bin`, 2.2 MB, fetched only when the AI judge is unavailable).
- Content is plain JSON with key ideas, clues, decoys, source and check date per item; a Node test
  checks the schema, that the robot knows every key idea, and that the Daily is deterministic.
- Ratings, version feedback, suggestions and robot-vs-AI comparisons go to the claude.ai artifact's
  database (when opened there), Netlify Forms (once switched on), or the player's browser.
- The same page runs as a claude.ai Artifact, where the AI judge runs on each viewer's own Claude
  plan and ratings land in a shared store we can read.

**Next steps if it goes further:**
- *Online play.* Word Bocce already has PeerJS rooms (`web/net.js`): the host's browser holds the
  state, players join by code from their phones. The same pattern gives a Jackbox-style party mode for
  free. (Inside claude.ai, the `room` capability does the same for an organisation's members.)
- *Content pipeline.* LLM-drafted items, human-checked, each with source and check date; a weekly
  link checker (domains lapse and get bought: zombo.com changed hands in February 2026); Wikidata
  for brands at scale ("company X makes product Y"); the DMV file for plates; Google Patents.
- *User-submitted URLs.* Never auto-open them. Show search results first; check addresses with
  Google's Web Risk API (Safe Browsing is non-commercial only); link out rather than embed, since
  most big sites refuse iframes; show a Wayback Machine snapshot when a site has changed.
- *Hosting at scale.* The page is cheap; the 2.2 MB vectors file is the bandwidth cost. On Netlify's
  credit plans bandwidth costs credits *(unverified: about 10 credits per GB; Free has 300 a month)*,
  so a popular version should shrink the file (64 dims and 12k words is under 1 MB) and serve it from a
  free CDN such as GitHub Pages or Cloudflare Pages.

**Live search for Bring-your-own, if the game ever does the searching itself:**

| Option | Price | Note |
|---|---|---|
| Google Custom Search JSON API | was $5/1k | Closed to new customers; shuts down 1 Jan 2027 |
| Google Programmable Search box | free with ads | From 2026 free engines cover 50 domains or fewer *(unverified)* |
| Bing Search API | n/a | Retired 11 Aug 2025 |
| Brave Search API | $5/1k, $5 monthly credit | Probably the simplest real web index |
| Exa / Tavily / SerpApi | $7/1k / 1k free then $8/1k / from $25/1k | |
| Claude's web search tool | $10/1k plus tokens | The AI can search and judge in one call; web fetch has no extra fee |

For the trials, players search on their own devices, which costs nothing and matches the in-person
game.

## 5. Token use and who pays

One judged round is about 700 tokens in (instructions, the truth, all guesses) and 150 out, one call
per round (not per guess). Prices from the Claude API docs (October 2026) and search summaries:

| Model | $/M tokens in / out | Per round | Per 1,000 rounds |
|---|---|---|---|
| Claude Opus 5.5, low effort | 4 / 20 | about 1¢ (thinking adds output) | about $10 |
| Claude Sonnet 5.5 | 2 / 10 | about 0.4¢ | about $4 |
| Claude Haiku 4.5 | 1 / 5 | 0.15¢ | $1.45 |
| Gemini 3.1 Flash-Lite *(unverified)* | 0.25 / 1.50 | 0.04¢ | $0.40 |
| OpenAI GPT-6 Luna *(unverified)* | 0.10 / 0.50 | 0.015¢ | $0.15 |

What that means:
- **Daily, robot-judged:** no AI cost at any scale.
- **Daily with an AI judge on every guess:** 10,000 players x 5 mysteries x about 2 calls is 100,000
  calls a day: $15-150 a day with a small model, about $1,000 a day with Opus. Too much for a free
  game, so the AI stays optional there.
- **Party play:** a 9-square game is 9 calls, under 2¢ with Haiku. A thousand parties a day is about
  $13. That can be paid for with credits or a party pack.
- Further savings: score all guesses in one call (the trials already do), cache the fixed
  instructions, use small models for scoring and keep big ones for writing clues offline.

**Letting players bring their own AI** (the trial already has the first three):

| Route | Player needs | Who pays | Friction |
|---|---|---|---|
| claude.ai Artifact (`sample` capability) | A Claude account | The viewer's own Claude plan | Low: one consent prompt. Working now |
| Copy the prompt into any chatbot, paste the reply back | Any chatbot, free tiers included | The player | Medium, but needs no setup. Working now |
| The player's own API key, called from the browser | An Anthropic key | The player | High; for testers. Working now (key stays in the browser) |
| OpenRouter sign-in (OAuth PKCE) | An OpenRouter account | The player's credits (some models are free) | Low after sign-in; works on a static site |
| Puter.js "user pays" | A Puter account | The player, after a free allowance | Low; one script tag |
| Sign in with ChatGPT | ChatGPT Plus/Pro | The player's ChatGPT plan | Low, but hosted apps are waitlisted *(unverified)* |
| Chrome's built-in Gemini Nano (Prompt API) | Desktop Chrome, about 22 GB free disk | Nobody | Free but desktop-only and a big download |
| A ChatGPT app (Apps SDK) or a Claude connector (MCP app) | An account on that platform | Probably the player's plan *(inferred)* | Distribution inside the chat apps; more work |

### What happened in the first week: the AI judge "ran out of credit"

The first deployment ran the AI judge through Netlify AI Gateway, which bills model calls to the
team's Netlify credits. On the Free plan those 300 credits a month pay for everything: each
production deploy costs 15 credits, AI calls cost 180 credits per dollar of model usage, and
bandwidth, requests and form submissions draw on the same pool. When the pool is empty, every
project on the team is paused ("Site not available") until the next month, and AI Gateway calls
stop. A dozen deploys alone use 180 credits, so a few days of testing can empty it. Calls were also
made per guess in the Daily (up to 20 per player per day), and judge-mode results were cached by
exact prompt, so the cache rarely hit.

### The fix: pay for as few calls as possible, and never from Netlify credits by default

1. **Answer keys (zero runtime cost).** Every mystery now carries `graded`: about ten likely guesses,
   scored in advance with a spoiler-free hint, written offline once. A guess that restates one of
   them (same content words, or a word-vector match of 0.85+) takes that score instantly, in the
   browser and again on the server, with no model call. Other guesses are nudged toward the
   nearest graded ones. Measured on the 217 hand-scored test guesses (which the key writers never saw):

   | Judge | Right winner | Rank agreement | Correlation | Mean error | Wrong scored warm | AI calls |
   |---|---|---|---|---|---|---|
   | Robot (word vectors) | 27/31 | 0.81 | 0.78 | 18 | 13 of 74 | 0 |
   | Robot + answer key | 31/31 | 0.90 | 0.91 | 12 | 1 of 74 | 0 |
   | Claude Haiku 4.5 | 31/31 | 0.95 | 0.96 | 8 | 4 of 74 | 217 |

   The answer key alone settles 78 of the 217 guesses (36%) with an average error of about 5 points
   against the human scores. Caveat: the gold scores and the keys were both written with Claude, so
   real players' ratings are the real test.
2. **Shared cache by meaning, not by prompt.** One guess at a curated mystery is cached under its
   normalized wording ("A laser pointer!" = "laser pointers"), shared by all players.
3. **The AI only for what's left,** with hard limits in `netlify/functions/judge.mjs`: 200 calls a day
   and 3,000 a month for the whole site, 60 a day per visitor, 20 requests a minute per visitor.
   When a limit is reached, or the provider says the money or quota ran out, the function answers
   "resting" and the game carries on with the answer key and the robot.
4. **Netlify credits only on purpose.** The function uses the first provider key you set yourself
   and ignores Netlify's gateway keys unless `JUDGE_USE_NETLIFY_CREDITS=1`. The recommended key is
   a free Gemini API key from Google AI Studio (no card): Gemini 3.1 Flash-Lite and 2.5 Flash-Lite
   are in the free tier, which allows on the order of a thousand requests a day per project (check
   the live quota in AI Studio) and simply stops when used up, so it cannot bill anyone. Google may
   use free-tier inputs to improve its models; the inputs here are game guesses. For more volume,
   switch the same key to paid tier 1 with a budget alert: Flash-Lite costs about $0.25/$1.50 per
   million tokens, roughly 0.02¢ a guess.
5. **Party play still uses the AI per round,** unless every guess is already in the answer key.
   A 9-square game is at most 9 calls.

Cost at scale with this design: if a third of guesses hit the key and the cache catches most
repeats of popular guesses, 10,000 Daily players at about 8 guesses each is roughly 30,000-50,000
calls a day: free up to the Gemini quota, then about $6-10 a day on Flash-Lite paid. That is the
point to add a supporter tier or ads (section 6), or to keep the Daily on the free judge and reserve
the AI for party play and supporters. Two more levers if needed: grow the answer keys from the
cache (re-scored offline) so hits rise over time, and run a small sentence-embedding model in the
browser for the robot.

Bring-your-own AI stays available in Settings for anyone who wants AI verdicts on every guess.

## 6. Making money

Roughly in order of how plausible they look:

1. **Free daily, paid extras.** The NYT model: the daily is free and shareable; a subscription
   adds the archive, practice packs, themed weeks and the AI judge. NYT reported 13.35 million
   subscribers in Q2 2026 and 11.2 billion game plays in 2025.
2. **Party packs.** A paid party mode (online rooms, AI host and judge, themed boards).
   Jackbox Party Pack 11 sells for $29.99 for five games; Death by AI, a free AI-judged party game,
   reached 20 million players in three months.
3. **Licensing to a publisher.** The Atlantic licensed Bracket City from its independent creator in
   2025; NYT bought Wordle for "low seven figures"; Arkadium distributes web games to 300+ partner
   sites and gives developers 75% of revenue.
4. **Sponsored rounds, clearly labelled.** "What are they selling?" is already an advertising
   format: a company pays to be the mystery (sponsored crosswords exist: Netflix in The Atlantic,
   Trader Joe's). Needs a visible "sponsored" tag and editorial control.
5. **Spin-offs with a buyer.** Link literacy for schools and phishing training ("where does this
   address really go?"); team icebreakers; a plate-reading game for road trips.
6. Small extras: affiliate links when an invented phrase is available as a domain; tips.

**Credits for feedback and suggestions.** The loop the user proposed: rating a mystery and suggesting
good ones earns credits that unlock more play (extra practice fives, party rounds with the AI judge).
To keep it honest: credit ratings only after the reveal; weight suggestions by how players later rate
them; cap credits per day; reward the best suggestion of the month with credit in the game and
something small. The trial's Suggest page asks which reward would make people suggest more (credit by
name, extra plays, a monthly prize, none needed), so the trials answer that question too.

## 7. Prior art: is this someone else's game?

Every ingredient exists; the combination looks new. No one owns the mechanic (next section).

| Thing | When | Closest part | Difference |
|---|---|---|---|
| *What's My Line?* (CBS) | 1950-67 | A panel guesses what a contestant does or sells | Yes/no questions about people's jobs |
| *Bumper Stumpers* (USA Network / Global) | 1987-90 | Teams decode vanity plates from a clue and letters revealed one by one | Almost certainly the vanity-plate show you remembered. Invented plates, buzzer play |
| *Site Unseen* | 2020 | A panel guesses what unusual URLs are, by yes/no questions | Only an IMDb listing; no scoring by closeness |
| Google Feud | 2015 | Family-Feud board of Google autocompletes | You guess how a query ends, not what it leads to |
| A Google a Day | 2011 | Daily trivia solved by searching | Search is the tool, not the thing guessed |
| "Pokémon or Big Data?", "Antidepressant or Tolkien?", "IKEA or Death" | 2013-18 | Guess what a name refers to | Two-way sorting, no open guesses |
| California Vanity app, VNTYPL8S | 2010s | Plates with meanings; match plates to owners | No reveal of real owners' reasons and DMV verdicts |
| Balderdash, Fictionary, Jackbox Fibbage | 1984-2014 | Fake answers vs the real one | Obscure words and facts, not names and addresses |
| Wavelength, Semantle, Contexto | 2019-22 | Scoring by closeness | Spectrum or word2vec closeness to one word |
| The New Yorker's Name Drop | 2021 | Progressive clues | Guess a person |
| Death by AI | 2024 | An AI judges players' answers | Survival stories, not real-world reveals |

## 8. Intellectual property

- **Mechanics can't be owned.** The US Copyright Office: copyright "does not protect the idea for a
  game, its name or title, or the method or methods for playing it". A patent on "guess, reveal,
  AI scores closeness" would almost certainly fail the *Alice* test (see *In re Smith*, 2016).
- **What can be owned:** the name (trademark), the code, the clue writing, the curated library and
  the ratings data. The edge is the library, the clues and the community, plus being first.
- **What to avoid:** copying another game's look (*Tetris v. Xio*, 2012) and names: Family Feud,
  Jeopardy!, Hollywood Squares, Balderdash, Fibbage, "Survey says". The trials use "points board",
  "tic-tac-toe" and "Bluff". NYT sent DMCA notices to Wordle clones in 2024 over the grid, colours and
  name, so a daily's look should be its own.
- **Using real names and sites:** naming brands to identify them is nominative fair use; no logos,
  no implied endorsement, a "not affiliated" line. Link out rather than host screenshots.
- **Content licences:** Numberbatch is CC BY-SA 4.0, so `judge-vectors.bin` must stay CC BY-SA (the
  game's code need not). Patents are public. The DMV dataset came from a public-records request and
  states no licence; individual plate texts are short and not personal, but ask before using it in a
  commercial product.
- **User submissions:** register a DMCA agent ($6 every three years) before taking them at scale;
  screen for slurs, personal details and harmful addresses.

## 9. A daily for NYT Games or The New Yorker

What such a daily needs, and how Trial A tries it:
- **Five minutes, no AI bill:** five mysteries, robot-judged, the same for everyone (seeded by date).
- **Fair-feeling scoring:** a 0-100 warmth meter with the matched words shown, "spot on" only when
  you name the main idea, a decoy check that tells you when you fell for the obvious reading.
- **A shareable result:** a five-square grid and a score out of 500.
- **A hook:** the vanity-plate side bet, and reveals worth reading (the DMV reviewer's notes, the
  patent's claims).

A New Yorker variant could lean on its Cartoon Caption Contest: the week's mystery, readers submit
the funniest wrong guess, readers vote. Precedents for getting in: Bracket City (licensed by The
Atlantic), Wordle (bought), Name Drop (built in-house).

## 10. Content so far

236 mysteries in eleven categories, each with an answer key:
- **Everyday categories:** 62 vanity plates (CA DMV file), 41 brands, 25 web addresses (including
  `kafka.com`), 15 patents (each checked against its text), 12 search phrases, 12 Double takes (PG-13).
- **New, October 2026** (from the first round of feedback):
  - *Up close* (15): a picture seen through a magnifying glass; each clue zooms out. Pictures are
    Microsoft's Fluent Emoji (MIT licence). A launch version wants our own macro photographs.
  - *Paper titles* (15): the part of a real paper's title before the colon ("Gorillas in our midst");
    guess the field and what it's about.
  - *Odd lyrics* (15): a strange line out of context; say what it means in the song. Public-domain
    songs only (traditional, or published before 1931), since publishers license and police lyrics.
    Modern pop lyrics would be funnier, but need a licence (LyricFind or Musixmatch) or a much
    stricter fair-use review.
  - *What's the deal with...?* (9): a news story that at least two late-night shows did bits on;
    guess the comic angle. No partisan politics, deaths or jokes about people's bodies.
  - *Headline, what?* (15): "crash blossoms", real headlines with an accidental second reading;
    say what the story was about. Five are marked as unconfirmed.

Checking was limited: the research helpers could search but most source pages were blocked, so
facts were confirmed from search-result summaries. Drafts are marked in the game. Still to verify:
40 brands, 5 double-take addresses, 5 headlines, 1 late-night story. Web swaps worth checking:
hasthelargehadroncolliderdestroyedtheworldyet.com, instantrimshot.com, findtheinvisiblecow.com,
endless.horse, milliondollarhomepage.com, hampsterdance.com, spacejam.com/1996, doesthedogdie.com,
windows93.net, nissan.com, lingscars.com.

More categories in the same spirit (explain what a cryptic phrase really means), not built yet:
- **Shop talk:** jargon out of context ("86 the salmon", "dead cat bounce", "yak shaving",
  "souls on board"): guess what it means and which job says it.
- **Lost in translation:** a film's title in another country, translated back ("The Incredible
  Journey in a Crazy Airplane" is *Airplane!* in Germany): name the film.
- **Nicknamed buildings:** the Gherkin, the Walkie-Talkie, the Cheesegrater: which city, and why.
- **Museum labels:** a strange object's catalogue description; what was it for?
- **Place-name stories:** Truth or Consequences, New Mexico; Boring, Oregon: how did it get the name?

## 11. Decide after the trials

1. Which version (or combination) to build on, from the ratings.
2. Which categories players rate highest (plates are the early favourite).
3. Whether the free judge (answer key plus robot) is good enough for the daily, so the AI is only
   needed for party play; whether to try in-browser sentence embeddings.
4. Typed guesses or multiple choice in the daily.
5. Online rooms for remote play (reuse Word Bocce's PeerJS code).
6. A name, and whether to keep it in this repo or move it out.

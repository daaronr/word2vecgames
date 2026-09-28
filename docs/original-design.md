1) Concept

Multiplayer spel (game) using word embeddings (e.g., Word2Vec) where each ronde (round) gives everyone the same startwoord (starting word) and doelwoord (target word). Players simultaneously play 2 kaarten (cards)—one openbaar (public) and one privé (private)—each chosen as add or subtract on the vector of the startwoord. The winner is the highest cosine similarity to the doelwoord after that single combined move. Also display the nearest actual word to each player’s resulting vector.

Dutch hints inline like this, with translations in parentheses.

2) Embeddings & Math

Model: word2vec-google-news-300 (300-dimensional). Alternatives: glove-wiki-gigaword-300, fastText (for better OOV handling).

Represent each token as vector v(word).

Player move result:

Let v₀ = v(startwoord), p = chosen public word, q = chosen private word.

Signs s₁, s₂ ∈ {+1, −1} for add/subtract.

v* = v₀ + s₁·v(p) + s₂·v(q)

Score = cos(v*, v(target)) = (v*·v_t) / (‖v*‖‖v_t‖)

Nearest word: argmax over vocabulary of cos(v*, v(w)), excluding out-of-bounds tokens as configured.

3) Deck, Cards, and Draws

Deck (“dek”): curated vocabulary of common, understandable tokens (e.g., 50k–200k tokens). Store only tokens present in the model.

Include joker cards (wildcards) at wildcard_ratio = 1/50 (configurable). A joker lets the player use any in-vocab token as their private card that round.

Per round:

Draw N_public = 10 unique public kaarten (words) shown to all.

Each player draws M_private = 2 private kaarten (words), hidden to others.

Cards are not consumed; draws are per-round fresh. New round → new draws.

4) Round Rules (Simultaneous Play)

Server selects startwoord and doelwoord (either random from the deck or from a themed list).

Server presents 10 public cards and deals 2 private cards to each player (with probability of joker).

Each player selects:

1 public card p and sign {+ / −},

1 private card q and sign {+ / −} (if joker, they may type any token; validate in-vocab).

Submit move → server computes v*, similarity S, and nearest word w_nearest.

After all moves (or timeout), reveal leaderboard for the round. Highest S wins.

Gelijkspel (tie): break on more precise similarity (4+ decimals). If still tied → shared win.

Optional: Closeness meter labels:

0.80–1.00: “heel dichtbij” (very close)

0.60–0.79: “warm”

0.40–0.59: “lauw”

0.20–0.39: “koud”

<0.20: “ijskoud” (ice-cold)

(English UI can show: “Very close”, “Getting warmer”, etc.)

5) OOV, Tokenization, Safety

Only use tokens in the embedding vocab. For GoogleNews, multiwords often use underscores (e.g., “New_York”). Either restrict deck to single tokens or store canonical forms with their tokenization.

If player enters a joker word that’s OOV → reject with suggestion.

Maintain a filter list to exclude slurs/profanity; sanitize user-entered joker tokens.

6) Performance Notes

MVP can brute-force nearest word search over a restricted vocab (e.g., the curated deck: 50k). For larger vocabs, use ANN index (FAISS or Annoy).

Pre-normalize all vectors and store as float32 to speed cosine.

Lazy load the model at startup; keep in memory.

7) Minimal Data Model (Python-ish)
class Card(TypedDict):
    token: str            # 'juice', 'ferment', or 'JOKER'
    type: Literal['PUBLIC','PRIVATE','JOKER']

class Player(TypedDict):
    id: str
    name: str
    private_cards: list[Card]
    submitted_move: bool
    result: dict | None   # {'similarity': float, 'nearest': str, 'vector': list[float]}

class RoundState(TypedDict):
    id: str
    start_word: str
    target_word: str
    public_cards: list[Card]   # len = 10
    moves_deadline_ts: int
    results: dict[str, dict]   # player_id -> result block

class Match(TypedDict):
    id: str
    players: dict[str, Player]
    deck_id: str
    settings: dict            # N_public, M_private, wildcard_ratio, timeout_secs
    rounds: list[RoundState]
    status: Literal['LOBBY','IN_PROGRESS','DONE']

8) Server API (FastAPI sketch)
POST /match
  body: {name, settings?}
  -> {match_id}

POST /match/{match_id}/join
  body: {player_name}
  -> {player_id, lobby_state}

POST /match/{match_id}/start
  -> {round_state}

GET  /match/{match_id}/round/current
  -> {round_state}

POST /match/{match_id}/round/{round_id}/submit
  body: {
    player_id,
    public_token, public_sign: '+'|'-',
    private_token, private_sign: '+'|'-',
    joker_override_token?: string  # only if private card is JOKER
  }
  -> {similarity, nearest_token, rank_preview?}

POST /match/{match_id}/round/{round_id}/resolve
  -> {leaderboard: [{player_id, similarity, nearest_token}]}

POST /match/{match_id}/round/next
  -> {round_state}

9) Core Functions (pseudocode)
# Preload
kv = load_keyedvectors(path)              # e.g., GoogleNews
norms = pre_normalize(kv)                 # store unit vectors for fast cosine

def cosine(u, v):                         # u, v are unit already
    return float(np.dot(u, v))

def vec(token):                           # returns unit vector
    return norms[token]                   # assume token exists

def move_vector(start, pub, s1, priv, s2):
    v = vec(start) + s1 * vec(pub) + s2 * vec(priv)
    v = v / np.linalg.norm(v)
    return v

def nearest_token(v, vocab_subset):
    # brute force MVP
    best = None; best_score = -2.0
    for t in vocab_subset:
        sc = float(np.dot(v, norms[t]))
        if sc > best_score:
            best_score, best = sc, t
    return best, best_score

def resolve_player_move(start, target, public_token, s1, private_token, s2, vocab_subset):
    v_star = move_vector(start, public_token, s1, private_token, s2)
    sim    = cosine(v_star, vec(target))
    near_t, near_sim = nearest_token(v_star, vocab_subset)
    return {"similarity": sim, "nearest": near_t, "nearest_sim": near_sim}

10) Minimal UI Flow (web)

Lobby: enter name, see players. Host clicks “Start”.

Round View:

Header: Startwoord (starting word), Doelwoord (target word).

Public cards (10 chips): each selectable with ± toggle.

Private cards (2 chips): each selectable with ± toggle. If one is JOKER, show an input box to type any token.

Submit button; show countdown timer.

Results View:

Leaderboard table: Player, Cosine (3 decimals), Nearest word, closeness meter label.

“Volgende ronde” (next round) button (host).

Optional: a 2D PCA projection dot for (start, target, each player result) for flavor—not required for MVP.

11) Game Settings (config)
{
  "vocab_source": "google-news-300",
  "deck_size": 100000,
  "N_public": 10,
  "M_private": 2,
  "wildcard_ratio": 0.02,
  "round_timeout_secs": 60,
  "nearest_vocab": "deck",   // 'deck' | 'full'
  "exclude_tokens_from_nearest": ["start", "target"] // true names replaced at runtime
}

12) Edge Cases

OOV: block selection / suggest alternatives.

Duplicate draws: re-roll to ensure uniqueness in public set and within a player’s private set.

No submissions: score equals cos(v(start), v(target)).

Token safety: filter disallowed tokens from deck and joker inputs.

Model unavailability: fallback to GloVe with same API surface.

13) Determinism & Testing

Seed the RNG for reproducible draws.

Unit tests:

move_vector normalizes properly.

Cosine monotonicity checks on synthetic vectors.

nearest_token returns one of top-k expected neighbors for known pairs (e.g., (king − man + woman) ~ queen).

Load test: 50 concurrent players submit within 1s.

14) Optional: Betting (“inzet”, stake)

Add an optional bet field (0–100) predicting your final cosine. Reward = 1 − |pred − actual|. Combine with rank points for final score. Can be added in a later sprint.

15) Build Notes

Backend: Python 3.11, FastAPI, Uvicorn, NumPy, (Gensim or KeyedVectors loader), Annoy/FAISS (optional).

Frontend: Any SPA or minimal HTML/JS; WebSocket for live updates (optional).

Persistence: In-memory for MVP; Redis if you want multi-process scale.

16) References (for implementers)

Mikolov, Tomas et al. (2013) “Efficient Estimation of Word Representations in Vector Space.”

Gensim KeyedVectors & Word2Vec usage (stable docs).

FAISS (ANN search) for nearest-neighbor scaling.i

17) Example Round (text I/O)
Round 3:
Startwoord: apple
Doelwoord: cider
Public: [fruit, juice, vinegar, sweet, ferment, press, orchard, yeast, sharp, barrel]
Your privé: [sugar, crush]   # (‘privé’ = private)
You play: +juice, +ferment
Server → similarity: 0.78, nearest: “cider”
Leaderboard:
1) You 0.78 (nearest: cider)
2) Dana 0.63 (nearest: apple_juice)
3) Rui 0.41 (nearest: vinegar)


This is everything needed to implement the MVP. If you want, I can generate the FastAPI scaffolding and a minimal deck loader next.

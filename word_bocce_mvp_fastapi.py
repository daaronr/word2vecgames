"""
Word Bocce — MVP backend
FastAPI server that runs the vector game using real word embeddings (Word2Vec/GloVe/FastText).

▶ How to run (with real embeddings):
1) Install deps:  pip install fastapi uvicorn[standard] numpy gensim annoy
2) Download a keyed vectors file, e.g. GoogleNews negative300 (binary):
   - https://code.google.com/archive/p/word2vec/
   - Or via Gensim once (gensim.downloader) and save to disk.
3) Set MODEL_PATH to the file path (e.g., GoogleNews-vectors-negative300.bin.gz):
   export MODEL_PATH=/path/to/GoogleNews-vectors-negative300.bin.gz
4) (Optional) limit deck size: export DECK_SIZE=100000 (default)
5) Run: uvicorn word_bocce_mvp_fastapi:app --reload

Notes:
- This server requires a *local* embeddings file; it does not auto-download.
- Uses unit-normalized vectors for cosine. Nearest search is over the curated deck
  with brute force (fast enough up to ~200k tokens). Annoy index included (optional).
"""

from __future__ import annotations
import os
import math
import time
import random
import threading
import json
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from gensim.models import KeyedVectors
except Exception as e:  # pragma: no cover
    KeyedVectors = None  # type: ignore

try:
    from annoy import AnnoyIndex
except Exception:
    AnnoyIndex = None  # type: ignore

# -------------------------------
# Config
# -------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH")
DECK_SIZE = int(os.environ.get("DECK_SIZE", "10000"))  # Smaller deck = more common words
N_PUBLIC = int(os.environ.get("N_PUBLIC", "10"))
M_PRIVATE = int(os.environ.get("M_PRIVATE", "2"))
WILDCARD_RATIO = float(os.environ.get("WILDCARD_RATIO", "0.2"))  # 20% chance for wildcards
ROUND_TIMEOUT_SECS = int(os.environ.get("ROUND_TIMEOUT_SECS", "120"))  # More time to think
USE_ANNOY = os.environ.get("USE_ANNOY", "0") == "1"
RNG_SEED = int(os.environ.get("RNG_SEED", "12345"))

# Embedding post-processing config
# EMBEDDING_PROFILE: Controls which embedding mode to use
#   - "classic_analogies": Standard word2vec with 3CosAdd scoring (for king-man+woman=queen)
#   - "intuitive_game": Enhanced with frequency weighting and other transformations
EMBEDDING_PROFILE = os.environ.get("EMBEDDING_PROFILE", "classic_analogies")

# FREQ_WEIGHT: 0-1, how much to de-emphasize frequent words (0.3 recommended for more intuitive results)
# Only used when EMBEDDING_PROFILE != "classic_analogies"
FREQ_WEIGHT = float(os.environ.get("FREQ_WEIGHT", "0.3"))
# NORMALIZE_METHOD: 'unit' (cosine-friendly), 'standardize', or 'none'
NORMALIZE_METHOD = os.environ.get("NORMALIZE_METHOD", "unit")

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# -------------------------------
# Data models
# -------------------------------
Sign = Literal["+", "-"]
CardType = Literal["PUBLIC", "PRIVATE", "JOKER"]
MatchStatus = Literal["LOBBY", "IN_PROGRESS", "DONE"]

class Card(BaseModel):
    token: str
    type: CardType

class PlayerResult(BaseModel):
    similarity: float
    nearest: str
    nearest_sim: float

class Player(BaseModel):
    id: str
    name: str
    private_cards: List[Card] = Field(default_factory=list)
    submitted_move: bool = False
    result: Optional[PlayerResult] = None

class RoundState(BaseModel):
    id: str
    start_word: str
    target_word: str
    public_cards: List[Card]
    moves_deadline_ts: int
    results: Dict[str, PlayerResult] = Field(default_factory=dict)

class Match(BaseModel):
    id: str
    players: Dict[str, Player] = Field(default_factory=dict)
    deck_id: str
    settings: Dict[str, object]
    rounds: List[RoundState] = Field(default_factory=list)
    status: MatchStatus = "LOBBY"

class CreateMatchReq(BaseModel):
    name: str
    settings: Optional[Dict[str, object]] = None

class JoinReq(BaseModel):
    player_name: str

class SubmitReq(BaseModel):
    player_id: str
    public_token: str
    public_sign: Sign
    private_token: str
    private_sign: Sign
    joker_override_token: Optional[str] = None

class ResolveResp(BaseModel):
    leaderboard: List[Dict[str, object]]

# -------------------------------
# Embedding space abstraction
# -------------------------------
class EmbeddingSpace:
    """
    Pluggable abstraction for word embeddings with post-processing capabilities.
    Subclass this to implement different preprocessing strategies.
    """
    def __init__(self, path: str, config: Optional[Dict] = None):
        """
        Load embeddings from path with optional configuration.

        Config options:
        - embedding_profile: str - 'classic_analogies' or 'intuitive_game'
        - freq_weight: float (0-1) - how much to weight by frequency (0 = no weighting, 1 = full weighting)
        - normalize_method: str - normalization method ('unit', 'standardize', 'none')
        - vocab_filter: List[str] - restrict to specific vocabulary
        """
        self.config = config or {}
        self.path = path
        self.embedding_profile = self.config.get('embedding_profile', 'classic_analogies')

        self._load_embeddings(path)
        self._apply_preprocessing()

        # For classic_analogies mode, precompute normalized matrix for efficient 3CosAdd
        if self.embedding_profile == 'classic_analogies':
            self._build_normalized_matrix()

    def _load_embeddings(self, path: str):
        """Load raw embeddings - override in subclasses"""
        if KeyedVectors is None:
            raise RuntimeError("gensim not installed. pip install gensim")

        self.raw_kv: KeyedVectors = KeyedVectors.load_word2vec_format(
            path, binary=path.endswith('.bin') or path.endswith('.bin.gz')
        )
        self.dim = int(self.raw_kv.vector_size)

    def _apply_preprocessing(self):
        """Apply post-processing transformations"""
        # Can be overridden for custom preprocessing
        pass

    def _build_normalized_matrix(self):
        """
        Build normalized matrix for efficient 3CosAdd computation.
        Only used in classic_analogies mode.
        """
        print(f"Building normalized matrix for {len(self.raw_kv.index_to_key)} words...")
        self.vocab = self.raw_kv.index_to_key
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}

        # Stack all vectors and normalize
        vectors = np.vstack([self.raw_kv.get_vector(word) for word in self.vocab])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self.matrix = (vectors / norms).astype(np.float32)
        print(f"Matrix shape: {self.matrix.shape}")

    def vector(self, word: str) -> np.ndarray:
        """Get (possibly preprocessed) vector for word"""
        if word not in self.raw_kv:
            raise KeyError(f"Word '{word}' not in vocabulary")

        # In classic_analogies mode, return pure normalized vector without any transformation
        if self.embedding_profile == 'classic_analogies':
            if hasattr(self, 'matrix') and word in self.word_to_idx:
                return self.matrix[self.word_to_idx[word]]
            else:
                # Fallback to raw_kv
                v = self.raw_kv.get_vector(word).astype(np.float32)
                n = np.linalg.norm(v)
                return v / n if n > 0 else v

        # For other profiles, apply transformations
        v = self.raw_kv.get_vector(word).astype(np.float32)

        # Apply frequency weighting if configured
        freq_weight = self.config.get('freq_weight', 0.0)
        if freq_weight > 0:
            # Words earlier in vocab are more frequent - downweight them slightly
            # to emphasize semantic meaning over frequency
            idx = self.raw_kv.key_to_index.get(word, len(self.raw_kv.index_to_key))
            freq_factor = 1.0 - (freq_weight * (1.0 - idx / len(self.raw_kv.index_to_key)))
            v = v * freq_factor

        # Normalize
        norm_method = self.config.get('normalize_method', 'unit')
        if norm_method == 'unit':
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
        elif norm_method == 'standardize':
            # Standardize each dimension
            v = (v - v.mean()) / (v.std() + 1e-8)

        return v

    def nearest(self, vec: np.ndarray, k: int = 10, vocab_filter: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Find k nearest words to vec, optionally filtered to vocab_filter.
        Returns list of (word, similarity) tuples.
        """
        candidates = vocab_filter if vocab_filter else self.raw_kv.index_to_key

        # Compute similarities
        results = []
        for word in candidates:
            if word not in self.raw_kv:
                continue
            word_vec = self.vector(word)
            # Cosine similarity (vectors are already normalized)
            sim = float(np.dot(vec, word_vec))
            results.append((word, sim))

        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def has(self, word: str) -> bool:
        """Check if word exists in vocabulary"""
        return word in self.raw_kv

    def most_similar_3cosadd(
        self,
        positive: List[str],
        negative: Optional[List[str]] = None,
        topn: int = 10,
        exclude: Optional[List[str]] = None,
        vocab_filter: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Classic 3CosAdd analogy scoring (Mikolov et al. 2013).

        Computes: score(w) = sum_{p in positive} cos(w, p) - sum_{n in negative} cos(w, n)

        This is the standard method that makes analogies like "king - man + woman ≈ queen" work.

        Args:
            positive: List of words to add (e.g., ['king', 'woman'])
            negative: List of words to subtract (e.g., ['man'])
            topn: Number of top results to return
            exclude: Words to exclude from results (typically the input words)
            vocab_filter: If provided, only consider words in this list

        Returns:
            List of (word, score) tuples, sorted by score descending
        """
        if not hasattr(self, 'matrix'):
            raise RuntimeError("3CosAdd requires classic_analogies mode with precomputed matrix")

        if negative is None:
            negative = []
        if exclude is None:
            exclude = []

        # Exclude input words from results
        exclude_set = set(exclude + positive + negative)

        # Determine candidate indices
        if vocab_filter is None:
            indices = np.arange(len(self.vocab))
        else:
            indices = np.array(
                [self.word_to_idx[w] for w in vocab_filter if w in self.word_to_idx],
                dtype=int,
            )

        # Get vectors for positive and negative words
        pos_vecs = np.stack([self.vector(w) for w in positive])
        neg_vecs = np.stack([self.vector(w) for w in negative]) if negative else None

        # Candidate vectors: (K, d)
        M = self.matrix[indices]

        # Compute cosine similarities
        # pos_sims: (K, |positive|) - cosine similarity to each positive word
        pos_sims = M @ pos_vecs.T
        scores = pos_sims.sum(axis=1)  # Sum over all positive words

        if neg_vecs is not None:
            # neg_sims: (K, |negative|) - cosine similarity to each negative word
            neg_sims = M @ neg_vecs.T
            scores -= neg_sims.sum(axis=1)  # Subtract all negative words

        # Exclude specified words by setting their scores to -inf
        for w in exclude_set:
            if w in self.word_to_idx:
                idx_in_vocab = self.word_to_idx[w]
                # Find position in our filtered indices array
                mask = indices == idx_in_vocab
                if mask.any():
                    scores[mask] = -1e9

        # Get top N indices
        top_idx = np.argsort(-scores)[:topn]
        result = [(self.vocab[indices[i]], float(scores[i])) for i in top_idx]

        return result


# -------------------------------
# Embedding store (refactored to use EmbeddingSpace)
# -------------------------------
class VectorStore:
    def __init__(self, path: Optional[str], embedding_config: Optional[Dict] = None):
        if not path:
            raise RuntimeError(
                "MODEL_PATH env var not set. Point it to a KeyedVectors file (e.g., GoogleNews .bin.gz)."
            )

        # Use EmbeddingSpace abstraction
        self.embedding_space = EmbeddingSpace(path, embedding_config)
        self.kv = self.embedding_space.raw_kv  # For backward compatibility
        self.dim = self.embedding_space.dim

        # Cache for deck tokens
        self._norms: Dict[str, np.ndarray] = {}
        self._deck_tokens: List[str] = []
        self._annoy: Optional[AnnoyIndex] = None

    def build_deck(self, size: int) -> List[str]:
        # Curate tokens: focus on common, readable words
        # GloVe/Word2Vec put most frequent words first, so we prioritize early tokens
        toks = []

        # Common word patterns to exclude
        exclude_patterns = ['_', '-', '.', '/', "'"]

        for t in self.kv.index_to_key:
            if len(toks) >= size:
                break

            # Must be alphabetic, lowercase, reasonable length
            if not (t.isalpha() and t.islower() and 3 <= len(t) <= 10):
                continue

            # Skip tokens with special characters
            if any(p in t for p in exclude_patterns):
                continue

            # Skip very uncommon letter patterns (heuristic: needs vowels)
            vowels = set('aeiou')
            if not any(c in vowels for c in t):
                continue

            toks.append(t)

        if len(toks) < max(1000, size // 10):
            # Fallback: just take first N alphabetic tokens
            toks = [t for t in self.kv.index_to_key[:size * 2]
                   if t.isalpha() and t.islower()][:size]

        self._deck_tokens = toks
        # Precompute unit vectors
        for t in toks:
            v = self.kv.get_vector(t).astype(np.float32)
            n = np.linalg.norm(v)
            self._norms[t] = (v / n) if n else v
        return toks

    def vec_unit(self, token: str) -> np.ndarray:
        """Get unit-normalized vector for token"""
        return self.embedding_space.vector(token)

    def cosine(self, u: np.ndarray, v: np.ndarray) -> float:
        return float(np.dot(u, v))

    def ensure_annoy(self) -> None:
        if AnnoyIndex is None:
            raise RuntimeError("annoy not installed. pip install annoy or set USE_ANNOY=0")
        if self._annoy is not None:
            return
        idx = AnnoyIndex(self.dim, metric='angular')
        for i, t in enumerate(self._deck_tokens):
            idx.add_item(i, self._norms[t])
        idx.build(20)
        self._annoy = idx

    def nearest(self, u: np.ndarray, k: int = 1, vocab_filter: Optional[List[str]] = None) -> Tuple[str, float]:
        """
        Find nearest word(s) to vector u.

        Args:
            u: Query vector
            k: Number of results (currently only k=1 is fully supported for backward compatibility)
            vocab_filter: Optional list of allowed words (domain restriction)

        Returns:
            Tuple of (word, similarity) for the nearest word
        """
        # If vocab filter is specified, use embedding_space.nearest()
        if vocab_filter:
            results = self.embedding_space.nearest(u, k=k, vocab_filter=vocab_filter)
            if results:
                return results[0]
            return "", -2.0

        # Search only over deck tokens (existing behavior)
        if USE_ANNOY:
            self.ensure_annoy()
            assert self._annoy is not None
            ids = self._annoy.get_nns_by_vector(u, k, include_distances=False)
            best_t = self._deck_tokens[ids[0]]
            sim = self.cosine(u, self._norms[best_t])
            return best_t, sim

        # Brute force over deck
        best_t = None
        best = -2.0
        for t, v in self._norms.items():
            sc = self.cosine(u, v)
            if sc > best:
                best, best_t = sc, t
        return best_t or "", best

# -------------------------------
# Game logic
# -------------------------------
class Game:
    def __init__(self, store: VectorStore):
        self.store = store
        self.matches: Dict[str, Match] = {}
        self.lock = threading.Lock()

    def new_match(self, name: str, settings: Optional[Dict[str, object]]) -> Match:
        mid = f"m_{int(time.time()*1000)}_{random.randint(1000,9999)}"
        s = {
            "N_public": N_PUBLIC,
            "M_private": M_PRIVATE,
            "wildcard_ratio": WILDCARD_RATIO,
            "timeout_secs": ROUND_TIMEOUT_SECS,
        }
        if settings:
            s.update(settings)
        m = Match(id=mid, deck_id="default", settings=s)
        with self.lock:
            self.matches[mid] = m
        return m

    def join(self, match_id: str, player_name: str) -> Tuple[Match, Player]:
        m = self._get_match(match_id)
        if m.status != "LOBBY":
            raise HTTPException(400, "Match already started")
        pid = f"p_{random.randint(100000,999999)}"
        p = Player(id=pid, name=player_name)
        m.players[pid] = p
        return m, p

    def start(self, match_id: str) -> RoundState:
        m = self._get_match(match_id)
        if m.status != "LOBBY":
            raise HTTPException(400, "Match already in progress or done")
        m.status = "IN_PROGRESS"
        rs = self._new_round(m)
        m.rounds.append(rs)
        return rs

    def current_round(self, match_id: str) -> RoundState:
        m = self._get_match(match_id)
        if not m.rounds:
            raise HTTPException(404, "No rounds yet")
        return m.rounds[-1]

    def submit(self, match_id: str, round_id: str, req: SubmitReq) -> PlayerResult:
        m = self._get_match(match_id)
        rs = self._get_round(m, round_id)
        p = m.players.get(req.player_id)
        if not p:
            raise HTTPException(404, "Unknown player")

        # Validate joker usage
        priv_token = req.private_token
        if priv_token.upper() == "JOKER":
            if not req.joker_override_token:
                raise HTTPException(400, "joker_override_token required for JOKER")
            priv_token = req.joker_override_token

        # Validate tokens exist
        for tok in [rs.start_word, rs.target_word, req.public_token, priv_token]:
            if tok not in store.kv:
                raise HTTPException(400, f"Token not in vocabulary: {tok}")

        # Compute move vector
        s1 = 1.0 if req.public_sign == "+" else -1.0
        s2 = 1.0 if req.private_sign == "+" else -1.0
        v = store.vec_unit(rs.start_word) + s1 * store.vec_unit(req.public_token) + s2 * store.vec_unit(priv_token)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        sim = store.cosine(v, store.vec_unit(rs.target_word))
        near_t, near_sim = store.nearest(v)

        res = PlayerResult(similarity=sim, nearest=near_t, nearest_sim=near_sim)
        p.submitted_move = True
        p.result = res
        rs.results[p.id] = res
        return res

    def resolve(self, match_id: str, round_id: str) -> ResolveResp:
        m = self._get_match(match_id)
        rs = self._get_round(m, round_id)
        board = [
            {
                "player_id": pid,
                "player_name": m.players[pid].name,
                "similarity": r.similarity,
                "nearest": r.nearest,
                "nearest_sim": r.nearest_sim,
            }
            for pid, r in rs.results.items()
        ]
        board.sort(key=lambda x: x["similarity"], reverse=True)
        return ResolveResp(leaderboard=board)

    def next_round(self, match_id: str) -> RoundState:
        m = self._get_match(match_id)
        rs = self._new_round(m)
        m.rounds.append(rs)
        return rs

    # --------------- helpers ---------------
    def _new_round(self, m: Match) -> RoundState:
        # Draw start/target from deck
        start_word, target_word = random.sample(deck_tokens, 2)
        # Public cards - ensure we get exactly N_public cards excluding start/target
        excluded = {start_word, target_word}
        available = [t for t in deck_tokens if t not in excluded]
        n_public = int(m.settings["N_public"])
        publics = [Card(token=t, type="PUBLIC") for t in random.sample(available, n_public)]
        # Deal private cards to players
        for p in m.players.values():
            p.submitted_move = False
            p.result = None
            privs: List[Card] = []
            used_tokens = {start_word, target_word}
            for _ in range(int(m.settings["M_private"])):
                if random.random() < float(m.settings["wildcard_ratio"]):
                    privs.append(Card(token="JOKER", type="JOKER"))
                else:
                    tok = random.choice([t for t in deck_tokens if t not in used_tokens])
                    used_tokens.add(tok)
                    privs.append(Card(token=tok, type="PRIVATE"))
            p.private_cards = privs
        rid = f"r_{int(time.time()*1000)}_{random.randint(100,999)}"
        deadline = int(time.time()) + int(m.settings["timeout_secs"])
        return RoundState(id=rid, start_word=start_word, target_word=target_word, public_cards=publics, moves_deadline_ts=deadline)

    def _get_match(self, match_id: str) -> Match:
        m = self.matches.get(match_id)
        if not m:
            raise HTTPException(404, "Unknown match")
        return m

    def _get_round(self, m: Match, round_id: str) -> RoundState:
        for r in m.rounds:
            if r.id == round_id:
                return r
        raise HTTPException(404, "Unknown round id")

# -------------------------------
# App init
# -------------------------------
app = FastAPI(title="Word Bocce MVP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embeddings and deck once
if MODEL_PATH is None:
    # Fail fast, but provide a helpful message at root endpoint
    store = None  # type: ignore
    deck_tokens: List[str] = []
else:
    # Configure embedding preprocessing
    embedding_config = {
        'embedding_profile': EMBEDDING_PROFILE,
        'freq_weight': FREQ_WEIGHT if EMBEDDING_PROFILE != 'classic_analogies' else 0.0,
        'normalize_method': NORMALIZE_METHOD,
    }
    print(f"Loading embeddings with profile: {EMBEDDING_PROFILE}")
    store = VectorStore(MODEL_PATH, embedding_config)
    deck_tokens = store.build_deck(DECK_SIZE)

game = Game(store) if store else None  # type: ignore

# -------------------------------
# Routes
# -------------------------------
@app.get("/api")
def api_root():
    if store is None:
        return {
            "status": "ok",
            "detail": "Set MODEL_PATH env var to a KeyedVectors file (e.g., GoogleNews-vectors-negative300.bin.gz) and restart.",
        }
    return {"status": "ok", "dim": store.dim, "deck_size": len(deck_tokens)}

@app.get("/", response_class=HTMLResponse)
def serve_game():
    try:
        with open("index.html", "r") as f:
            html_content = f.read()
            # Update API_URL to use current host
            html_content = html_content.replace(
                "const API_URL = 'http://localhost:8000';",
                "const API_URL = window.location.origin;"
            )
            return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)

@app.get("/presentation", response_class=HTMLResponse)
@app.get("/presentation.html", response_class=HTMLResponse)
def serve_presentation():
    try:
        with open("presentation.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>presentation.html not found</h1>", status_code=404)

@app.post("/match")
def create_match(req: CreateMatchReq):
    if store is None:
        raise HTTPException(503, "Embeddings not loaded. Set MODEL_PATH and restart.")
    m = game.new_match(req.name, req.settings)
    return m

@app.post("/match/{match_id}/join")
def join_match(match_id: str, req: JoinReq):
    m, p = game.join(match_id, req.player_name)
    return {"match": m, "player": p}

@app.post("/match/{match_id}/start")
def start_match(match_id: str):
    return game.start(match_id)

@app.get("/match/{match_id}/round/current")
def get_current_round(match_id: str):
    return game.current_round(match_id)

@app.get("/match/{match_id}/player/{player_id}")
def get_player_state(match_id: str, player_id: str):
    m = game._get_match(match_id)
    p = m.players.get(player_id)
    if not p:
        raise HTTPException(404, "Unknown player")
    return {"player": p, "match_status": m.status}

@app.post("/match/{match_id}/round/{round_id}/submit")
def submit_move(match_id: str, round_id: str, req: SubmitReq):
    return game.submit(match_id, round_id, req)

@app.post("/match/{match_id}/round/{round_id}/resolve")
def resolve_round(match_id: str, round_id: str):
    return game.resolve(match_id, round_id)

@app.post("/match/{match_id}/round/next")
def next_round(match_id: str):
    return game.next_round(match_id)

@app.get("/visualize/{word}")
def visualize_word(word: str, neighbors: int = 10):
    """Get word vector and its nearest neighbors for visualization"""
    if store is None:
        raise HTTPException(503, "Embeddings not loaded")

    if word not in store.kv:
        raise HTTPException(404, f"Word '{word}' not in vocabulary")

    # Get the word vector
    word_vec = store.vec_unit(word)

    # Find nearest neighbors
    from sklearn.decomposition import PCA
    import numpy as np

    # Collect vectors for PCA
    neighbor_words = []
    vectors = [word_vec]

    for t in deck_tokens[:5000]:  # Search in first 5000 words
        if t == word:
            continue
        sim = store.cosine(word_vec, store._norms[t])
        if len(neighbor_words) < neighbors:
            neighbor_words.append((t, sim))
            neighbor_words.sort(key=lambda x: x[1], reverse=True)
        elif sim > neighbor_words[-1][1]:
            neighbor_words[-1] = (t, sim)
            neighbor_words.sort(key=lambda x: x[1], reverse=True)

    # Add neighbor vectors
    for w, _ in neighbor_words:
        vectors.append(store._norms[w])

    # Reduce to 2D with PCA
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(np.array(vectors))

    # Format response
    points = [{"word": word, "x": float(coords_2d[0][0]), "y": float(coords_2d[0][1]), "similarity": 1.0}]
    for i, (w, sim) in enumerate(neighbor_words):
        points.append({
            "word": w,
            "x": float(coords_2d[i+1][0]),
            "y": float(coords_2d[i+1][1]),
            "similarity": float(sim)
        })

    return {"word": word, "points": points}

@app.get("/visualize/move/{start_word}/{target_word}")
def visualize_move(start_word: str, target_word: str,
                  public_word: str = None, private_word: str = None,
                  public_sign: str = "+", private_sign: str = "+"):
    """Visualize a word bocce move in 2D space"""
    if store is None:
        raise HTTPException(503, "Embeddings not loaded")

    words_to_check = [start_word, target_word]
    if public_word:
        words_to_check.append(public_word)
    if private_word:
        words_to_check.append(private_word)

    for w in words_to_check:
        if w not in store.kv:
            raise HTTPException(404, f"Word '{w}' not in vocabulary")

    from sklearn.decomposition import PCA
    import numpy as np

    # Calculate result vector if moves provided
    result_vec = None
    if public_word and private_word:
        s1 = 1.0 if public_sign == "+" else -1.0
        s2 = 1.0 if private_sign == "+" else -1.0
        result_vec = store.vec_unit(start_word) + s1 * store.vec_unit(public_word) + s2 * store.vec_unit(private_word)
        n = np.linalg.norm(result_vec)
        if n > 0:
            result_vec = result_vec / n

    # Collect vectors
    vectors = [
        store.vec_unit(start_word),
        store.vec_unit(target_word)
    ]
    labels = [start_word, target_word]

    if public_word:
        vectors.append(store.vec_unit(public_word))
        labels.append(public_word)
    if private_word:
        vectors.append(store.vec_unit(private_word))
        labels.append(private_word)
    if result_vec is not None:
        vectors.append(result_vec)
        labels.append("RESULT")

    # Reduce to 2D
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(np.array(vectors))

    # Format response
    points = []
    for i, label in enumerate(labels):
        points.append({
            "word": label,
            "x": float(coords_2d[i][0]),
            "y": float(coords_2d[i][1])
        })

    return {"points": points}

# -------------------------------
# Puzzle Mode Endpoints
# -------------------------------

# Load puzzles from JSON file
puzzles_data = []
try:
    with open("puzzles.json", "r") as f:
        puzzles_data = json.load(f)
except FileNotFoundError:
    print("Warning: puzzles.json not found. Puzzle mode will not be available.")
except json.JSONDecodeError:
    print("Warning: puzzles.json is invalid. Puzzle mode will not be available.")

@app.get("/puzzles")
def list_puzzles(difficulty: Optional[str] = None):
    """Get list of available puzzles, optionally filtered by difficulty"""
    if not puzzles_data:
        raise HTTPException(503, "Puzzles not available")

    result = puzzles_data
    if difficulty:
        result = [p for p in puzzles_data if p.get("difficulty") == difficulty.lower()]

    # Return summary without solution hints
    return [{
        "id": p["id"],
        "difficulty": p["difficulty"],
        "name": p["name"],
        "description": p["description"],
        "start_word": p["start_word"],
        "target_word": p["target_word"]
    } for p in result]

@app.get("/puzzle/{puzzle_id}")
def get_puzzle(puzzle_id: int):
    """Get a specific puzzle by ID"""
    if not puzzles_data:
        raise HTTPException(503, "Puzzles not available")

    puzzle = next((p for p in puzzles_data if p["id"] == puzzle_id), None)
    if not puzzle:
        raise HTTPException(404, f"Puzzle {puzzle_id} not found")

    # Filter out WILDCARD and cards too similar to target
    allowed = puzzle.get("allowed_cards", [])
    has_wildcard = "WILDCARD" in allowed
    filtered_allowed = []
    target_word = puzzle["target_word"]

    for card in allowed:
        if card == "WILDCARD":
            continue
        # Check if card is too similar to target
        if store and card in store.kv and target_word in store.kv:
            similarity = store.cosine(store.vec_unit(card), store.vec_unit(target_word))
            if similarity > 0.85:
                continue
        filtered_allowed.append(card)

    # Return modified puzzle
    result = puzzle.copy()
    result["allowed_cards"] = filtered_allowed
    result["has_wildcard"] = has_wildcard

    # Include domain_vocab if specified (for domain-restricted nearest neighbor search)
    if "domain_vocab" in puzzle:
        result["domain_vocab"] = puzzle["domain_vocab"]

    return result

class PuzzleSolution(BaseModel):
    public_token: str
    public_sign: int  # 1 or -1
    private_token: str
    private_sign: int  # 1 or -1

@app.post("/puzzle/{puzzle_id}/solve")
def solve_puzzle(puzzle_id: int, solution: PuzzleSolution):
    """Submit a solution to a puzzle and get rating"""
    if not puzzles_data:
        raise HTTPException(503, "Puzzles not available")

    if store is None:
        raise HTTPException(503, "Embeddings not loaded")

    puzzle = next((p for p in puzzles_data if p["id"] == puzzle_id), None)
    if not puzzle:
        raise HTTPException(404, f"Puzzle {puzzle_id} not found")

    # Check if used cards are in allowed list (or WILDCARD is allowed)
    allowed = puzzle.get("allowed_cards", [])
    has_wildcard = "WILDCARD" in allowed

    # Filter allowed cards to remove WILDCARD and cards too similar to target
    filtered_allowed = []
    target_word = puzzle["target_word"]
    for card in allowed:
        if card == "WILDCARD":
            continue
        # Check if card is too similar to target (would make puzzle trivial)
        if card in store.kv and target_word in store.kv:
            similarity = store.cosine(store.vec_unit(card), store.vec_unit(target_word))
            if similarity > 0.85:  # Too similar, skip this card
                continue
        filtered_allowed.append(card)

    # For WILDCARD puzzles, allow any reasonable word from deck
    if has_wildcard:
        # Check that used tokens are either in allowed list OR in vocabulary
        if solution.public_token not in filtered_allowed and solution.public_token not in deck_tokens:
            return {
                "valid": False,
                "error": f"'{solution.public_token}' is not valid for this puzzle"
            }
        if solution.private_token not in filtered_allowed and solution.private_token not in deck_tokens:
            return {
                "valid": False,
                "error": f"'{solution.private_token}' is not valid for this puzzle"
            }
    else:
        # Standard puzzle: must use cards from allowed list
        if solution.public_token not in filtered_allowed:
            return {
                "valid": False,
                "error": f"'{solution.public_token}' is not an allowed card for this puzzle"
            }
        if solution.private_token not in filtered_allowed:
            return {
                "valid": False,
                "error": f"'{solution.private_token}' is not an allowed card for this puzzle"
            }

    # Check if all words exist in vocabulary
    start_word = puzzle["start_word"]
    target_word = puzzle["target_word"]

    for word in [start_word, target_word, solution.public_token, solution.private_token]:
        if word not in store.kv:
            return {
                "valid": False,
                "error": f"Word '{word}' not in vocabulary"
            }

    # Calculate result vector and similarity to target
    v_start = store.vec_unit(start_word)
    v_target = store.vec_unit(target_word)

    # Use 3CosAdd if in classic_analogies mode
    if (store.embedding_space.embedding_profile == 'classic_analogies' and
        hasattr(store.embedding_space, 'most_similar_3cosadd')):
        # Build positive and negative lists for 3CosAdd
        positive = [start_word]
        negative = []

        s1 = float(solution.public_sign)
        s2 = float(solution.private_sign)

        if s1 > 0:
            positive.append(solution.public_token)
        else:
            negative.append(solution.public_token)

        if s2 > 0:
            positive.append(solution.private_token)
        else:
            negative.append(solution.private_token)

        # Compute 3CosAdd score for the target word
        pos_vecs = np.stack([store.embedding_space.vector(w) for w in positive])
        neg_vecs = np.stack([store.embedding_space.vector(w) for w in negative]) if negative else None

        v_target_norm = store.embedding_space.vector(target_word)

        similarity = float(np.sum(v_target_norm @ pos_vecs.T))
        if neg_vecs is not None:
            similarity -= float(np.sum(v_target_norm @ neg_vecs.T))

        # For nearest word finding, we still compute the result vector
        v_public = store.vec_unit(solution.public_token)
        v_private = store.vec_unit(solution.private_token)
        v_result = v_start + s1 * v_public + s2 * v_private
        v_result_unit = v_result / (np.linalg.norm(v_result) + 1e-12)
    else:
        # Use vector addition method (existing behavior)
        v_public = store.vec_unit(solution.public_token)
        v_private = store.vec_unit(solution.private_token)

        s1 = float(solution.public_sign)
        s2 = float(solution.private_sign)

        v_result = v_start + s1 * v_public + s2 * v_private
        v_result_unit = v_result / (np.linalg.norm(v_result) + 1e-12)

        # Calculate similarity to target
        similarity = float(store.cosine(v_result_unit, v_target))

    # Find nearest word - use domain vocab if specified
    domain_vocab = puzzle.get("domain_vocab", None)
    if domain_vocab:
        # Domain-restricted vocabulary for more intuitive results
        nearest_word, best_sim = store.nearest(v_result_unit, k=1, vocab_filter=domain_vocab)
    else:
        # Use default deck search
        nearest_word = start_word
        best_sim = -1.0
        for token in deck_tokens[:1000]:
            if token in [start_word, target_word]:
                continue
            v_tok = store._norms[token]
            sim = store.cosine(v_result_unit, v_tok)
            if sim > best_sim:
                best_sim = sim
                nearest_word = token

    # Calculate star rating and win condition
    thresholds = puzzle.get("stars_thresholds", {"3": 0.70, "2": 0.50, "1": 0.30})
    win_threshold = puzzle.get("win_threshold", thresholds["1"])  # Minimum to "pass"

    stars = 0
    if similarity >= thresholds["3"]:
        stars = 3
    elif similarity >= thresholds["2"]:
        stars = 2
    elif similarity >= thresholds["1"]:
        stars = 1

    # Determine if puzzle is passed
    passed = similarity >= win_threshold

    # Calculate progress indicator (0-100%)
    # Map similarity from [-1, 1] to [0, 100], with emphasis on positive similarities
    if similarity < 0:
        progress = max(0, (similarity + 1) * 25)  # -1 to 0 maps to 0-25%
    else:
        progress = 25 + (similarity * 75)  # 0 to 1 maps to 25-100%

    # Calculate distance metrics for better feedback
    start_to_target_sim = float(store.cosine(v_start, v_target))
    improvement = similarity - start_to_target_sim  # How much closer we got

    # Calculate all possible moves to find the best one
    all_moves = []
    best_move = None
    best_similarity = -1.0

    # Use 3CosAdd if in classic_analogies mode
    use_3cosadd = (store.embedding_space.embedding_profile == 'classic_analogies' and
                   hasattr(store.embedding_space, 'most_similar_3cosadd'))

    for pub_card in allowed:
        if pub_card not in store.kv:
            continue
        for priv_card in allowed:
            if priv_card not in store.kv:
                continue
            for pub_sign in [1, -1]:
                for priv_sign in [1, -1]:
                    if use_3cosadd:
                        # Use 3CosAdd scoring for classic_analogies mode
                        # Build positive and negative lists
                        positive = [start_word]
                        negative = []

                        if pub_sign > 0:
                            positive.append(pub_card)
                        else:
                            negative.append(pub_card)

                        if priv_sign > 0:
                            positive.append(priv_card)
                        else:
                            negative.append(priv_card)

                        # Get score for target word using 3CosAdd
                        # We'll compute it manually for the target
                        pos_vecs = np.stack([store.embedding_space.vector(w) for w in positive])
                        neg_vecs = np.stack([store.embedding_space.vector(w) for w in negative]) if negative else None

                        v_target_norm = store.embedding_space.vector(target_word)

                        # 3CosAdd score for target
                        move_sim = float(np.sum(v_target_norm @ pos_vecs.T))
                        if neg_vecs is not None:
                            move_sim -= float(np.sum(v_target_norm @ neg_vecs.T))
                    else:
                        # Use vector addition method (existing behavior)
                        v_pub = store.vec_unit(pub_card)
                        v_priv = store.vec_unit(priv_card)
                        v_res = v_start + float(pub_sign) * v_pub + float(priv_sign) * v_priv
                        v_res_unit = v_res / (np.linalg.norm(v_res) + 1e-12)

                        # Calculate similarity to target
                        move_sim = float(store.cosine(v_res_unit, v_target))

                    # Track this move
                    move_info = {
                        "public_card": pub_card,
                        "private_card": priv_card,
                        "public_sign": "+" if pub_sign > 0 else "-",
                        "private_sign": "+" if priv_sign > 0 else "-",
                        "similarity": round(move_sim, 4)
                    }
                    all_moves.append(move_info)

                    # Track best move
                    if move_sim > best_similarity:
                        best_similarity = move_sim
                        best_move = move_info.copy()

    # Sort all moves by similarity (best first)
    all_moves.sort(key=lambda m: m["similarity"], reverse=True)

    return {
        "valid": True,
        "similarity": similarity,
        "nearest_word": nearest_word,
        "stars": stars,
        "passed": passed,  # NEW: threshold-based win condition
        "win_threshold": win_threshold,  # NEW: what similarity is needed to pass
        "progress": round(progress, 1),  # NEW: progress indicator (0-100%)
        "improvement": round(improvement, 4),  # NEW: how much closer to target vs. start
        "start_similarity": round(start_to_target_sim, 4),  # NEW: baseline similarity
        "optimal_similarity": puzzle.get("optimal_similarity", 0.70),
        "public_used": solution.public_token,
        "private_used": solution.private_token,
        "public_sign": "+" if solution.public_sign > 0 else "-",
        "private_sign": "+" if solution.private_sign > 0 else "-",
        "best_move": best_move,
        "all_moves": all_moves[:20]  # Return top 20 moves to avoid huge payload
    }


# -------------------------------
# 2D Visualization Endpoint
# -------------------------------
class VisualizationRequest(BaseModel):
    words: List[str] = Field(..., description="List of words to visualize")
    start_word: Optional[str] = Field(None, description="Starting word")
    target_word: Optional[str] = Field(None, description="Target word")
    result_word: Optional[str] = Field(None, description="Result word after move")
    public_card: Optional[str] = Field(None, description="Public card used")
    public_sign: Optional[int] = Field(None, description="Public card sign (+1 or -1)")
    private_card: Optional[str] = Field(None, description="Private card used")
    private_sign: Optional[int] = Field(None, description="Private card sign (+1 or -1)")

@app.post("/visualize")
def visualize_words(req: VisualizationRequest):
    """
    Project word vectors into 2D space using PCA for visualization.
    Returns 2D coordinates for each word.
    """
    if not store or not store.kv:
        raise HTTPException(status_code=503, detail="Embeddings not loaded")

    # Filter words that exist in vocabulary
    valid_words = [w for w in req.words if w in store.kv]

    # Add special words if provided
    special_words = []
    if req.start_word and req.start_word in store.kv:
        special_words.append(("start", req.start_word))
    if req.target_word and req.target_word in store.kv:
        special_words.append(("target", req.target_word))
    if req.result_word and req.result_word in store.kv:
        special_words.append(("result", req.result_word))

    # Calculate intermediate steps if move details provided
    intermediate_steps = []
    if req.start_word and req.public_card and req.public_sign is not None:
        if req.start_word in store.kv and req.public_card in store.kv:
            # Step 1: start + public_card
            v_start = store.vec_unit(req.start_word)
            v_public = store.vec_unit(req.public_card)
            v_step1 = v_start + float(req.public_sign) * v_public
            v_step1_unit = v_step1 / (np.linalg.norm(v_step1) + 1e-12)
            nearest_step1, _ = store.nearest(v_step1_unit)
            special_words.append(("step1", nearest_step1))
            intermediate_steps.append({
                "step": 1,
                "word": nearest_step1,
                "operation": f"{req.start_word} {'+'  if req.public_sign > 0 else '-'} {req.public_card}"
            })

            # Step 2: step1 + private_card (if provided)
            if req.private_card and req.private_sign is not None and req.private_card in store.kv:
                v_private = store.vec_unit(req.private_card)
                v_step2 = v_step1 + float(req.private_sign) * v_private
                v_step2_unit = v_step2 / (np.linalg.norm(v_step2) + 1e-12)
                nearest_step2, _ = store.nearest(v_step2_unit)
                special_words.append(("step2", nearest_step2))
                intermediate_steps.append({
                    "step": 2,
                    "word": nearest_step2,
                    "operation": f"... {'+' if req.private_sign > 0 else '-'} {req.private_card}"
                })

    all_words = list(set(valid_words + [w for _, w in special_words]))

    if len(all_words) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 valid words")

    # Get vectors for all words
    vectors = np.array([store.kv[word] for word in all_words])

    # Simple PCA implementation (reduce to 2D)
    # Center the data
    mean_vec = np.mean(vectors, axis=0)
    centered = vectors - mean_vec

    # Compute covariance matrix
    cov_matrix = np.cov(centered.T)

    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Project onto first 2 principal components
    pca_components = eigenvectors[:, :2]
    coords_2d = centered @ pca_components

    # Normalize to [0, 1] range for easier plotting
    min_x, max_x = coords_2d[:, 0].min(), coords_2d[:, 0].max()
    min_y, max_y = coords_2d[:, 1].min(), coords_2d[:, 1].max()

    if max_x > min_x:
        coords_2d[:, 0] = (coords_2d[:, 0] - min_x) / (max_x - min_x)
    if max_y > min_y:
        coords_2d[:, 1] = (coords_2d[:, 1] - min_y) / (max_y - min_y)

    # Build response
    points = []
    for i, word in enumerate(all_words):
        point = {
            "word": word,
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
            "type": "card"
        }

        # Mark special words
        for special_type, special_word in special_words:
            if word == special_word:
                point["type"] = special_type
                break

        points.append(point)

    return {
        "points": points,
        "intermediate_steps": intermediate_steps,
        "variance_explained": float(eigenvalues[idx[0]] + eigenvalues[idx[1]]) / float(eigenvalues.sum()) if len(eigenvalues) > 0 else 0.0
    }


# -------------------------------
# Test/Sanity Check Endpoint for Classic Analogies
# -------------------------------
@app.get("/test/analogies")
def test_classic_analogies():
    """
    Test endpoint to verify classic word2vec analogies work correctly.
    Only functional when EMBEDDING_PROFILE=classic_analogies.
    """
    if not store or not store.embedding_space:
        raise HTTPException(503, "Embeddings not loaded")

    if store.embedding_space.embedding_profile != 'classic_analogies':
        return {
            "error": "Classic analogies require EMBEDDING_PROFILE=classic_analogies",
            "current_profile": store.embedding_space.embedding_profile,
            "message": "Set environment variable EMBEDDING_PROFILE=classic_analogies and restart"
        }

    # Classic analogy examples from Mikolov et al.
    test_cases = [
        {
            "name": "king - man + woman = ?",
            "positive": ["king", "woman"],
            "negative": ["man"],
            "expected": "queen"
        },
        {
            "name": "paris - france + germany = ?",
            "positive": ["paris", "germany"],
            "negative": ["france"],
            "expected": "berlin"
        },
        {
            "name": "tokyo - japan + france = ?",
            "positive": ["tokyo", "france"],
            "negative": ["japan"],
            "expected": "paris"
        },
        {
            "name": "good - bad + ugly = ?",
            "positive": ["good", "ugly"],
            "negative": ["bad"],
            "expected": "beautiful"
        }
    ]

    results = []
    for test in test_cases:
        try:
            # Check if all words exist
            missing = []
            for w in test["positive"] + test["negative"]:
                if not store.embedding_space.has(w):
                    missing.append(w)

            if missing:
                results.append({
                    "test": test["name"],
                    "status": "error",
                    "message": f"Words not in vocabulary: {missing}"
                })
                continue

            # Run 3CosAdd
            top_results = store.embedding_space.most_similar_3cosadd(
                positive=test["positive"],
                negative=test["negative"],
                topn=10
            )

            # Find rank of expected word
            words = [w for w, s in top_results]
            scores = {w: s for w, s in top_results}

            expected_rank = None
            expected_score = None
            if test["expected"] in words:
                expected_rank = words.index(test["expected"]) + 1
                expected_score = scores[test["expected"]]

            results.append({
                "test": test["name"],
                "expected": test["expected"],
                "top_10": words,
                "scores": scores,
                "expected_rank": expected_rank,
                "expected_score": expected_score,
                "status": "success" if expected_rank and expected_rank <= 3 else "warning"
            })

        except Exception as e:
            results.append({
                "test": test["name"],
                "status": "error",
                "message": str(e)
            })

    # Summary
    successful = sum(1 for r in results if r.get("expected_rank") and r["expected_rank"] == 1)
    top_3 = sum(1 for r in results if r.get("expected_rank") and r["expected_rank"] <= 3)

    return {
        "embedding_profile": EMBEDDING_PROFILE,
        "model_path": MODEL_PATH,
        "summary": {
            "total_tests": len(test_cases),
            "rank_1": successful,
            "top_3": top_3,
        },
        "results": results
    }


# -------------------------------
# Dev utility: quick local smoke (no HTTP)
# -------------------------------
if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("word_bocce_mvp_fastapi:app", host="0.0.0.0", port=8000, reload=True)

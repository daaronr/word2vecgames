# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is "Word Bocce" - a multiplayer word vector game that uses word embeddings (Word2Vec, GloVe, or FastText) to create a unique competitive experience. Players manipulate word vectors by adding/subtracting word embeddings to reach a target word, with the closest cosine similarity winning.

## Key Concepts

- **Vector Arithmetic**: Players start with a `start_word` and try to reach a `target_word` by adding/subtracting public and private card vectors
- **Card System**: Each round has 10 public cards (visible to all) and 2 private cards per player. Joker cards (wildcards) allow players to use any in-vocabulary token
- **Scoring**: Winner is determined by highest cosine similarity to target word after the combined move: `v* = v₀ + s₁·v(public) + s₂·v(private)`

## Running the Server

### Setup and Installation

```bash
# Install dependencies
pip install fastapi uvicorn[standard] numpy gensim annoy

# Download word embeddings (required)
# Option 1: GoogleNews-vectors-negative300.bin.gz from https://code.google.com/archive/p/word2vec/
# Option 2: Use gensim.downloader to get GloVe or other models
```

### Environment Configuration

```bash
# Required: Path to word embeddings file
export MODEL_PATH=/path/to/GoogleNews-vectors-negative300.bin.gz

# Optional configuration
export DECK_SIZE=100000              # Curated vocabulary size (default: 100000)
export N_PUBLIC=10                   # Public cards per round (default: 10)
export M_PRIVATE=2                   # Private cards per player (default: 2)
export WILDCARD_RATIO=0.02          # Joker probability (default: 0.02)
export ROUND_TIMEOUT_SECS=60        # Round time limit (default: 60)
export USE_ANNOY=1                  # Enable Annoy ANN index for faster nearest neighbor search (default: 0)
export RNG_SEED=12345               # Seed for reproducible draws (default: 12345)
```

### Starting the Server

```bash
uvicorn word_bocce_mvp_fastapi:app --reload
# Server runs on http://localhost:8000
```

## Architecture

### Core Components

1. **VectorStore** (`word_bocce_mvp_fastapi.py:119-192`)
   - Loads and manages word embeddings via gensim KeyedVectors
   - Pre-normalizes vectors to unit length for efficient cosine similarity
   - Builds curated deck from vocabulary (filters for lowercase, alphabetic, 3-12 chars)
   - Provides nearest neighbor search (brute-force or Annoy-based ANN)

2. **Game** (`word_bocce_mvp_fastapi.py:196-332`)
   - Manages match lifecycle: LOBBY → IN_PROGRESS → DONE
   - Handles round generation with random start/target word selection
   - Card dealing: public cards + private cards (with joker probability)
   - Move submission and vector computation
   - Leaderboard generation based on cosine similarity

3. **Data Models** (`word_bocce_mvp_fastapi.py:61-114`)
   - `Match`: Container for players, rounds, settings
   - `RoundState`: Current round's start/target words, public cards, deadline, results
   - `Player`: Identity, private cards, submission status, result
   - `Card`: Token and type (PUBLIC/PRIVATE/JOKER)

### API Endpoints

- `POST /match` - Create new match
- `POST /match/{match_id}/join` - Join a match in lobby
- `POST /match/{match_id}/start` - Start first round
- `GET /match/{match_id}/round/current` - Get current round state
- `GET /match/{match_id}/player/{player_id}` - Get player-specific state including private cards
- `POST /match/{match_id}/round/{round_id}/submit` - Submit player move (public card, private card, each with +/- sign)
- `POST /match/{match_id}/round/{round_id}/resolve` - Get leaderboard for round
- `POST /match/{match_id}/round/next` - Start next round

## Important Implementation Details

### Vector Math

The core computation in `submit()` (line 241-274):
- Validates tokens are in vocabulary
- Computes: `v* = v(start) + s₁·v(public) + s₂·v(private)` where signs are ±1
- Normalizes resulting vector
- Calculates cosine similarity to target: `cos(v*, v(target))`
- Finds nearest actual word in deck vocabulary

### Deck Curation

The `build_deck()` method filters embedding vocabulary to select "clean" tokens:
- Alphabetic, lowercase, 3-12 characters in length
- Fallback to first N tokens if filtering yields too few
- Pre-computes and caches unit-normalized vectors for all deck tokens

### Joker/Wildcard Handling

When a player has a JOKER card:
- They can specify any in-vocabulary token via `joker_override_token`
- Server validates the token exists in embeddings
- Vector computation proceeds normally with the custom token

### Performance Considerations

- All deck vectors are pre-normalized and cached as float32
- Nearest neighbor search defaults to brute-force over deck (fast up to ~200k tokens)
- Optional Annoy ANN index can be enabled with `USE_ANNOY=1` for larger decks
- Model is loaded once at startup and kept in memory

## Testing Strategy

Per design document, tests should cover:
- `move_vector` normalization correctness
- Cosine similarity monotonicity
- Known word analogy tests (e.g., king - man + woman ≈ queen)
- Concurrent player submissions (load testing)
- OOV token rejection
- Unique card drawing (no duplicates)

## Vocabulary and Token Safety

- Only in-vocabulary tokens are allowed
- Deck building should implement profanity/slur filtering (placeholder in current implementation)
- Joker inputs must be validated against both vocabulary and safety filter
- Multi-word tokens in GoogleNews use underscores (e.g., "New_York")

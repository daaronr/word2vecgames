# Word Bocce 🎯

A multiplayer word vector game where players manipulate word embeddings to reach target words!

## What is Word Bocce?

Word Bocce is a competitive game where players use vector arithmetic to navigate semantic space. Each round, all players start from the same starting word and try to reach a target word by adding or subtracting word vectors from public and private cards. The player whose final vector is closest to the target word wins!

### Game Mechanics

- **Start & Target**: Each round has a starting word and a target word
- **Cards**: Players get 10 shared public cards and 2 secret private cards
- **Moves**: Choose one public and one private card, each with + or - operation
- **Scoring**: Your result vector is compared to the target using cosine similarity
- **Winner**: Highest similarity wins!

Example: Start="apple", Target="cider" → Play +juice, +ferment → Win! 🎉

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Word Embeddings

Choose a model based on your needs:

```bash
# Small & fast (recommended for testing - ~130MB)
python setup_embeddings.py --model glove-100

# Medium quality (~380MB)
python setup_embeddings.py --model glove-300

# Best quality (~1.6GB, slower download)
python setup_embeddings.py --model google-news
```

### 3. Start the Backend Server

```bash
# Set the path to your downloaded embeddings
export MODEL_PATH=./embeddings/glove-100.bin

# Start the API server
uvicorn word_bocce_mvp_fastapi:app --reload
```

The backend will be running at `http://localhost:8000`

### 4. Start the Frontend Server

In a new terminal:

```bash
python run_frontend.py
```

The frontend will be running at `http://localhost:8080`

### 5. Play!

1. Open `http://localhost:8080` in your browser
2. Enter your name and create a match
3. Share the Match ID with friends to join
4. Start the game and compete!

## Game Files

- `word_bocce_mvp_fastapi.py` - FastAPI backend server
- `index.html` - Web-based game interface
- `setup_embeddings.py` - Helper script to download word embeddings
- `run_frontend.py` - Simple HTTP server for the frontend
- `requirements.txt` - Python dependencies
- `wordbocce_description.md` - Detailed game design document
- `CLAUDE.md` - Developer documentation

## Configuration

Customize game settings via environment variables:

```bash
export DECK_SIZE=100000              # Vocabulary size
export N_PUBLIC=10                   # Public cards per round
export M_PRIVATE=2                   # Private cards per player
export WILDCARD_RATIO=0.02          # Probability of JOKER cards
export ROUND_TIMEOUT_SECS=60        # Time limit per round
export USE_ANNOY=1                  # Use ANN for faster nearest neighbor search
export RNG_SEED=12345               # Seed for reproducible games
```

## API Endpoints

- `POST /match` - Create new match
- `POST /match/{id}/join` - Join a match
- `POST /match/{id}/start` - Start the game
- `GET /match/{id}/round/current` - Get current round state
- `GET /match/{id}/player/{player_id}` - Get player's private cards
- `POST /match/{id}/round/{round_id}/submit` - Submit your move
- `POST /match/{id}/round/{round_id}/resolve` - Get leaderboard
- `POST /match/{id}/round/next` - Start next round

## Architecture

The game uses:
- **Backend**: FastAPI with in-memory state management
- **Embeddings**: Word2Vec/GloVe via Gensim
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Vector Math**: NumPy for cosine similarity calculations

See `CLAUDE.md` for detailed architecture documentation.

## Development

### Running Tests

```bash
# Syntax check
python -m py_compile word_bocce_mvp_fastapi.py

# Start server in dev mode
uvicorn word_bocce_mvp_fastapi:app --reload --log-level debug
```

### How It Works

1. Embeddings are loaded and normalized at startup
2. Each word is represented as a 100-300 dimensional vector
3. Your move: `v_result = v_start + sign1×v_public + sign2×v_private`
4. Score: `cosine_similarity(v_result, v_target)`
5. The nearest actual word to your result is also displayed

## Troubleshooting

**"MODEL_PATH env var not set"**
- Make sure to run `export MODEL_PATH=./embeddings/your-model.bin` before starting the server

**"Token not in vocabulary"**
- Some words aren't in the embedding model. Try a different word or use the JOKER card

**Frontend can't connect to backend**
- Make sure the backend is running on port 8000
- Check for CORS errors in browser console
- Verify `API_URL` in index.html matches your backend

**Slow nearest neighbor search**
- Set `USE_ANNOY=1` to enable approximate nearest neighbor search
- Reduce `DECK_SIZE` to use a smaller vocabulary

## Credits

Inspired by word embedding games and the concept of "semantic bocce ball" where you try to get close to a target in vector space.

## License

tbd

# Classic Analogies Mode

This document describes the `classic_analogies` embedding profile that enables standard word2vec analogy behavior like `king - man + woman ≈ queen`.

## Overview

The Word Bocce game now supports two embedding profiles:

1. **`classic_analogies`** (default): Standard word2vec with 3CosAdd scoring for classic analogies
2. **`intuitive_game`**: Enhanced mode with frequency weighting and other transformations

## Configuration

Set the embedding profile via environment variable:

```bash
export EMBEDDING_PROFILE=classic_analogies  # Default
# or
export EMBEDDING_PROFILE=intuitive_game
```

## How Classic Analogies Mode Works

### 1. Pure Normalized Vectors

In `classic_analogies` mode, word vectors are:
- Loaded from the standard word2vec binary (e.g., GoogleNews-vectors-negative300.bin)
- L2-normalized only (no frequency weighting or other transformations)
- Stored in a precomputed matrix for efficient computation

### 2. 3CosAdd Scoring

The system implements the classic 3CosAdd scoring from Mikolov et al. (2013):

```
score(w) = Σ cos(w, positive_i) - Σ cos(w, negative_j)
```

For example, `king - man + woman`:
- positive = [king, woman]
- negative = [man]
- score(queen) = cos(queen, king) + cos(queen, woman) - cos(queen, man)

### 3. API Method

```python
results = store.embedding_space.most_similar_3cosadd(
    positive=['king', 'woman'],
    negative=['man'],
    topn=10
)
# Returns: [('queen', score), ('monarch', score), ...]
```

## Testing Classic Analogies

### Test Endpoint

Visit `/test/analogies` to verify that classic analogies work:

```bash
curl http://localhost:8000/test/analogies
```

This tests several classic analogies:
- king - man + woman = queen
- paris - france + germany = berlin
- tokyo - japan + france = paris
- good - bad + ugly = beautiful

### Example Response

```json
{
  "embedding_profile": "classic_analogies",
  "summary": {
    "total_tests": 4,
    "rank_1": 3,
    "top_3": 4
  },
  "results": [
    {
      "test": "king - man + woman = ?",
      "expected": "queen",
      "top_10": ["queen", "monarch", "princess", ...],
      "expected_rank": 1,
      "status": "success"
    }
  ]
}
```

## Using in Puzzles

Puzzles automatically use 3CosAdd scoring when `EMBEDDING_PROFILE=classic_analogies`.

### Example: King → Queen Puzzle

The puzzle file (puzzles.json) already includes a king→queen puzzle:

```json
{
  "id": 1,
  "name": "Royal Relations",
  "start_word": "king",
  "target_word": "queen",
  "allowed_cards": ["woman", "female", "male", "man", "royal", "crown"],
  "hint": "Gender transformation: king - man + woman"
}
```

When players submit moves:
- Move: king + woman - man
- System uses 3CosAdd to score: score(queen) = cos(queen,king) + cos(queen,woman) - cos(queen,man)
- This should yield a very high score if the embeddings are good

### Best Move Calculation

The system evaluates all possible card combinations using 3CosAdd to find the best move:

```python
# For each card combination:
positive = [start_word] + [cards with + sign]
negative = [cards with - sign]
score = 3CosAdd(target_word, positive, negative)
```

## Implementation Details

### EmbeddingSpace Class

Key changes:

```python
class EmbeddingSpace:
    def __init__(self, path, config):
        self.embedding_profile = config.get('embedding_profile', 'classic_analogies')

        if self.embedding_profile == 'classic_analogies':
            self._build_normalized_matrix()  # Precompute all vectors

    def vector(self, word):
        if self.embedding_profile == 'classic_analogies':
            return self.matrix[self.word_to_idx[word]]  # Pure normalized
        else:
            # Apply frequency weighting and other transformations
            ...

    def most_similar_3cosadd(self, positive, negative, topn=10):
        # Efficient matrix operations for 3CosAdd
        pos_vecs = stack([self.vector(w) for w in positive])
        neg_vecs = stack([self.vector(w) for w in negative]) if negative else None

        M = self.matrix  # All vocab vectors
        scores = (M @ pos_vecs.T).sum(axis=1)
        if neg_vecs:
            scores -= (M @ neg_vecs.T).sum(axis=1)

        return top_k(scores, topn)
```

### Puzzle Solving

When solving puzzles in `classic_analogies` mode:

```python
if embedding_profile == 'classic_analogies':
    # Build positive/negative lists
    positive = [start_word]
    if public_sign > 0:
        positive.append(public_card)
    else:
        negative.append(public_card)

    # Compute 3CosAdd score for target
    score = 3CosAdd(target_word, positive, negative)
else:
    # Use vector addition
    v_result = v_start + s1*v_public + s2*v_private
    score = cosine(v_result, v_target)
```

## Switching Between Modes

To switch between modes, restart the server with different environment variable:

```bash
# Classic mode (for standard analogies)
export EMBEDDING_PROFILE=classic_analogies
export MODEL_PATH=/path/to/GoogleNews-vectors-negative300.bin
uvicorn word_bocce_mvp_fastapi:app --reload

# Intuitive game mode (for enhanced gameplay)
export EMBEDDING_PROFILE=intuitive_game
export FREQ_WEIGHT=0.3
uvicorn word_bocce_mvp_fastapi:app --reload
```

## Key Differences

| Feature | classic_analogies | intuitive_game |
|---------|-------------------|----------------|
| Vector Preprocessing | None (pure L2-norm) | Frequency weighting |
| Scoring Method | 3CosAdd | Cosine similarity on combined vector |
| Best For | Standard word analogies | Intuitive gameplay |
| Example | king-man+woman=queen | More forgiving combinations |

## References

- Mikolov et al. (2013). "Linguistic Regularities in Continuous Space Word Representations"
- Original 3CosAdd implementation: gensim's `most_similar(positive=..., negative=...)`

## Troubleshooting

**Q: Analogies don't work well**
- Ensure you're using GoogleNews-vectors-negative300.bin or similar high-quality embeddings
- Verify EMBEDDING_PROFILE=classic_analogies
- Check /test/analogies endpoint for diagnostics

**Q: Matrix building takes too long**
- Normal for large vocabularies (3M words takes ~30 seconds)
- Matrix is built once at startup and cached

**Q: Out of memory**
- Large embeddings (3M words × 300 dims) need ~4GB RAM
- Consider using DECK_SIZE to limit vocabulary

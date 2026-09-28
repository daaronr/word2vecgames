# Classic Analogies Mode

This document describes the `classic_analogies` embedding profile that enables standard word2vec analogy behavior like `king - man + woman ≈ queen`.

## Overview

The Word Bocce game now supports two embedding profiles:

1. **`classic_analogies`** (default): Standard word2vec using vector arithmetic for scoring and 3CosAdd for nearest word lookup
2. **`intuitive_game`**: Enhanced mode with frequency weighting and other transformations

## ⚠️ Important: How Classic Mode Works

**Classic analogies mode uses TWO different methods:**

1. **Vector Arithmetic** (for scoring and ranking moves)
   - Score = cosine similarity between `result_vector` and `target`
   - `result_vector = start + sign₁·card₁ + sign₂·card₂`
   - Measures geometric distance in embedding space
   - Used for: player scores, best move rankings

2. **3CosAdd** (for finding nearest word)
   - Finds which vocabulary word best matches the analogy pattern
   - Used for: displaying the "nearest word" to your move
   - Example: `king - man + woman` → finds "queen" as nearest word

**Why both methods?**
- Vector arithmetic tells you **how close you got** to the target
- 3CosAdd tells you **which word** your move resembles most
- They serve different purposes and complement each other

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

### 2. Scoring: Vector Arithmetic

Player scores and move rankings use **vector arithmetic** (geometric distance):

```python
# Compute result vector
result_vector = start_vector + sign₁·card₁_vector + sign₂·card₂_vector

# Normalize
result_unit = result_vector / ||result_vector||

# Score is cosine similarity to target
score = cos(result_unit, target_vector)
```

**Example:** King → Queen with cards [woman, man, royal, crown]

| Move | Calculation | Score | Rank |
|------|-------------|-------|------|
| +woman +royal | king + woman + royal | 0.8323 | #1 |
| +royal +crown | king + royal + crown | 0.8178 | #3 |
| +woman -man | king + woman - man | 0.7699 | #15 |

**Why +woman +royal ranks higher than +woman -man?**
- Both "woman" and "royal" are semantically close to "queen"
- Adding them moves the vector geometrically closer to queen's position
- The classic analogy +woman -man still works (see next section)

### 3. Nearest Word: 3CosAdd

To find which vocabulary word best matches your move, we use **3CosAdd** from Mikolov et al. (2013):

```
score(w) = Σ cos(w, positive_i) - Σ cos(w, negative_j)
```

**Example:** `king + woman - man`
- positive = [king, woman]
- negative = [man]
- score(queen) = cos(queen, king) + cos(queen, woman) - cos(queen, man) = 0.786
- **Result: "queen" is the nearest word** ✓

This is why the classic analogy works - the **nearest word** to `king - man + woman` is indeed "queen", even though other moves might score higher geometrically.

### 4. Key Insight: Two Complementary Measures

**Vector Arithmetic (Geometric):**
- ✅ "How close did I get to the target?"
- ✅ Rankings show which moves are most effective
- ✅ Rewards adding target-related concepts

**3CosAdd (Analogical):**
- ✅ "Which word does this analogy pattern match?"
- ✅ Finds conceptual relationships (king:man::queen:woman)
- ✅ Shows you landed at a meaningful concept

**Example in practice:**
```
Move: king + woman - man
Vector score: 0.7699 (ranked #15)
Nearest word (3CosAdd): "queen" (rank #1 match!)

Interpretation: Your move gets 77% of the way to queen geometrically,
and "queen" is the best analogical match for the pattern you created.
```

### 5. API Method

```python
# For nearest word lookup
results = store.embedding_space.most_similar_3cosadd(
    positive=['king', 'woman'],
    negative=['man'],
    topn=10
)
# Returns: [('queen', 0.786), ('monarch', 0.699), ...]
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

The system evaluates all possible card combinations using **vector arithmetic** to find which move gets geometrically closest to the target:

```python
# For each card combination:
result_vector = start + sign₁·card₁ + sign₂·card₂
result_unit = normalize(result_vector)
score = cosine_similarity(result_unit, target)
```

**Example rankings for king → queen:**
```
#1  +woman +royal:  0.8323  (both concepts close to queen)
#2  +royal +crown:  0.8178  (very royal, but no gender shift)
#15 +woman -man:    0.7699  (classic analogy, conceptually elegant)
```

The classic analogy may not rank #1, but it produces the most **meaningful** nearest word ("queen" via 3CosAdd).

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
# Score calculation (vector arithmetic)
v_result = v_start + sign1·v_public + sign2·v_private
v_result_unit = normalize(v_result)
player_score = cosine(v_result_unit, v_target)

# Nearest word lookup (3CosAdd)
if classic_analogies:
    positive = [start_word, public_card if sign1>0 else []]
    negative = [public_card if sign1<0 else []]
    # (same for private card)

    nearest_word = most_similar_3cosadd(positive, negative, topn=1)
```

**Result display:**
```json
{
  "similarity": 0.7699,        // Vector arithmetic score
  "nearest_word": "queen",      // 3CosAdd nearest match
  "stars": 2,
  "passed": true
}
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
| Move Scoring | Vector arithmetic (geometric) | Vector arithmetic |
| Nearest Word | 3CosAdd (analogical) | Vector nearest neighbor |
| Best For | Classic analogies with meaningful nearest words | Intuitive gameplay |
| Example | king+woman-man finds "queen" as nearest | More forgiving combinations |

## Understanding the Dual Approach

**Why not use 3CosAdd for everything?**

3CosAdd was designed for analogy **retrieval** (finding which word matches a pattern), not for **ranking** which transformation is best. It has quirks:

```
3CosAdd scores (if used for ranking):
+royal +royal: 2.16  ❌ (can't use same card twice anyway)
+royal +crown: 2.13
+woman -man:   0.79  ✓ (classic analogy)

Vector arithmetic scores (actual implementation):
+woman +royal: 0.83  ✓ (adds related concepts)
+royal +crown: 0.82  ✓ (very royal)
+woman -man:   0.77  ✓ (gender transformation)
```

**Benefits of current approach:**
- ✅ Move rankings make intuitive sense (closer to target = higher rank)
- ✅ Classic analogies still work (nearest word via 3CosAdd)
- ✅ Best of both worlds: geometric scores + analogical interpretations
- ✅ Order matters appropriately (+woman -man ≠ +man -woman)

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

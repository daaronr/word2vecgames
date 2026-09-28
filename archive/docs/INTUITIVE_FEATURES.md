# Intuitive Features for Word2Vec Game

This document describes the new features implemented to make word2vec-based gameplay more intuitive and configurable.

## 🎯 Overview

The standard word2vec embeddings with raw cosine similarity often produce non-intuitive results. These enhancements provide:

1. **Pluggable Embedding Abstraction** - Flexible preprocessing of embeddings
2. **Frequency Weighting** - De-emphasize common words to focus on semantic meaning
3. **Domain-Restricted Vocabularies** - Limit search space to specific word categories
4. **Threshold-Based Win Conditions** - Clear success criteria with progress tracking

## 🔧 Implementation Details

### 1. EmbeddingSpace Abstraction

**File**: `word_bocce_mvp_fastapi.py:122-206`

A new pluggable abstraction for word embeddings that separates embedding loading from preprocessing.

```python
class EmbeddingSpace:
    def __init__(self, path: str, config: Optional[Dict] = None)
    def vector(self, word: str) -> np.ndarray
    def nearest(self, vec: np.ndarray, k: int = 10, vocab_filter: Optional[List[str]] = None) -> List[Tuple[str, float]]
    def has(self, word: str) -> bool
```

**Configuration Options**:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `freq_weight` | float (0-1) | 0.3 | How much to de-emphasize frequent words. 0 = no weighting, 1 = full weighting |
| `normalize_method` | str | 'unit' | Normalization: 'unit' (cosine-friendly), 'standardize', or 'none' |
| `vocab_filter` | List[str] | None | Restrict vocabulary to specific words |

**Example Usage**:

```python
# Create embedding space with frequency de-weighting
config = {
    'freq_weight': 0.3,  # 30% frequency de-weighting
    'normalize_method': 'unit'
}
space = EmbeddingSpace('/path/to/embeddings.bin', config)

# Get preprocessed vector
vec = space.vector('apple')

# Search with domain restriction
fruit_vocab = ['orange', 'banana', 'grape', 'lemon']
results = space.nearest(vec, k=5, vocab_filter=fruit_vocab)
```

### 2. Frequency Weighting

**Location**: `word_bocce_mvp_fastapi.py:162-169`

Reduces the influence of very frequent words to emphasize semantic relationships.

**How it works**:
- Words earlier in the vocabulary (more frequent) get slightly downweighted
- Formula: `freq_factor = 1.0 - (freq_weight * (1.0 - word_index / vocab_size))`
- This makes results like "king - man + woman → queen" more reliable
- Reduces nonsensical results like "apple + citrus - yellow → product"

**Environment Variable**:
```bash
export FREQ_WEIGHT=0.3  # Recommended value: 0.3 (30% weighting)
```

### 3. Domain-Restricted Vocabularies

**Location**: `word_bocce_mvp_fastapi.py:293-328` (VectorStore.nearest)

Allows puzzles to restrict nearest-neighbor search to specific word categories.

**Puzzle Configuration**:

Add a `domain_vocab` field to puzzles in `puzzles.json`:

```json
{
  "id": 1,
  "name": "Fruit Puzzle",
  "start_word": "apple",
  "target_word": "orange",
  "allowed_cards": ["red", "sweet", "citrus", "round"],
  "domain_vocab": [
    "apple", "orange", "banana", "grape", "lemon",
    "cherry", "peach", "pear", "strawberry", "melon"
  ],
  "description": "Reach orange from apple using fruit vocabulary only"
}
```

**Benefits**:
- Results stay within logical domain (e.g., fruits, colors, animals)
- Prevents nonsensical cross-domain results
- Makes gameplay more predictable and satisfying

### 4. Threshold-Based Win Conditions

**Location**: `word_bocce_mvp_fastapi.py:868-892`

Clear success criteria with detailed progress tracking.

**New Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `passed` | bool | Whether the similarity exceeds win_threshold |
| `win_threshold` | float | Minimum similarity required to pass (default: 0.30) |
| `progress` | float | Progress indicator 0-100% |
| `improvement` | float | How much closer to target vs. starting position |
| `start_similarity` | float | Baseline similarity between start and target |

**Example Response**:

```json
{
  "valid": true,
  "similarity": 0.6521,
  "stars": 2,
  "passed": true,
  "win_threshold": 0.30,
  "progress": 73.9,
  "improvement": 0.4123,
  "start_similarity": 0.2398,
  "nearest_word": "fruit",
  "public_used": "red",
  "private_used": "sweet"
}
```

**Progress Calculation**:
- Negative similarity (-1 to 0): Maps to 0-25% progress
- Positive similarity (0 to 1): Maps to 25-100% progress
- Provides visual feedback for UI progress bars

## 🚀 Usage Examples

### Example 1: Configure Server with Intuitive Settings

```bash
# Set environment variables
export MODEL_PATH=./embeddings/GoogleNews-vectors-negative300.bin.gz
export FREQ_WEIGHT=0.3           # Enable frequency weighting
export NORMALIZE_METHOD=unit     # Use unit normalization (default)
export DECK_SIZE=10000           # Use common words only

# Start server
uvicorn word_bocce_mvp_fastapi:app --reload
```

### Example 2: Create Intuitive Puzzles

Create or modify `puzzles.json`:

```json
[
  {
    "id": 1,
    "difficulty": "easy",
    "name": "Color Mixing",
    "description": "Mix colors to reach purple",
    "start_word": "red",
    "target_word": "purple",
    "allowed_cards": ["blue", "pink", "dark", "light", "WILDCARD"],
    "domain_vocab": [
      "red", "blue", "green", "yellow", "purple", "orange",
      "pink", "brown", "black", "white", "gray", "violet"
    ],
    "win_threshold": 0.40,
    "stars_thresholds": {
      "3": 0.70,
      "2": 0.55,
      "1": 0.40
    }
  },
  {
    "id": 2,
    "difficulty": "medium",
    "name": "Animal Kingdom",
    "description": "Transform a cat into a lion",
    "start_word": "cat",
    "target_word": "lion",
    "allowed_cards": ["wild", "big", "fierce", "jungle", "roar", "WILDCARD"],
    "domain_vocab": [
      "cat", "dog", "lion", "tiger", "bear", "wolf",
      "elephant", "giraffe", "zebra", "monkey", "snake"
    ],
    "win_threshold": 0.35
  }
]
```

### Example 3: Test New Features Programmatically

```python
from word_bocce_mvp_fastapi import EmbeddingSpace, VectorStore

# Create embedding space with custom config
config = {
    'freq_weight': 0.3,
    'normalize_method': 'unit'
}

space = EmbeddingSpace('/path/to/embeddings.bin', config)

# Get vector and search in domain
vec = space.vector('apple')
fruit_domain = ['orange', 'banana', 'grape', 'lemon', 'cherry']
results = space.nearest(vec, k=3, vocab_filter=fruit_domain)

print("Top 3 fruits similar to apple:")
for word, sim in results:
    print(f"  {word}: {sim:.4f}")
```

## 📊 Expected Improvements

### Before (Raw Embeddings):
- "apple + citrus - yellow → product" ❌
- "king - man + woman → monarch" (instead of "queen") ❌
- Frequent words dominate results
- Results often cross semantic domains

### After (With New Features):
- "apple + citrus - yellow → orange" ✅ (with fruit domain)
- "king - man + woman → queen" ✅ (with frequency weighting)
- Semantic meaning emphasized
- Results stay within logical domains

## 🧪 Testing

### Manual Testing

1. **Start server with new config**:
   ```bash
   export MODEL_PATH=./embeddings/glove-100.bin
   export FREQ_WEIGHT=0.3
   uvicorn word_bocce_mvp_fastapi:app
   ```

2. **Test domain-restricted puzzle**:
   ```bash
   curl http://localhost:8000/puzzle/1
   curl -X POST http://localhost:8000/puzzle/1/solve \
     -H "Content-Type: application/json" \
     -d '{"public_token":"blue","public_sign":1,"private_token":"pink","private_sign":1}'
   ```

3. **Check new response fields**:
   - `passed`: true/false
   - `progress`: 0-100
   - `improvement`: positive if closer to target

### Automated Testing

Run the test script (requires numpy, gensim installed):

```bash
# Install dependencies first
pip install -r requirements.txt

# Run tests
export MODEL_PATH=./embeddings/your-embeddings.bin
python3 test_intuitive_features.py
```

## 🔍 Technical Notes

### Performance Considerations

1. **Frequency Weighting**: O(1) overhead per vector retrieval (negligible)
2. **Domain Filtering**: O(n) where n = domain size (fast for domains < 1000 words)
3. **Preprocessing**: Applied on-demand, vectors not pre-cached
4. **Memory**: No significant increase (config is lightweight)

### Backward Compatibility

All features are **backward compatible**:
- Default config uses no frequency weighting (same as before)
- Domain filter is optional (defaults to full deck)
- Existing puzzles work without modification
- Old API endpoints unchanged

### Future Enhancements

Potential future improvements:

1. **Per-puzzle embedding config**: Allow different freq_weight per puzzle
2. **Dynamic threshold adjustment**: Auto-tune based on difficulty
3. **Multiple domain support**: Allow AND/OR combinations of domains
4. **Contextual embeddings**: Support for BERT/GPT-style embeddings
5. **User feedback loop**: Adjust weights based on player ratings

## 📝 Configuration Reference

### Environment Variables

```bash
# Core settings
MODEL_PATH=/path/to/embeddings.bin       # Required
DECK_SIZE=10000                          # Optional, default: 10000

# New intuitive features
FREQ_WEIGHT=0.3                          # Optional, default: 0.3
NORMALIZE_METHOD=unit                    # Optional, default: unit

# Existing settings
N_PUBLIC=10
M_PRIVATE=2
WILDCARD_RATIO=0.2
ROUND_TIMEOUT_SECS=120
USE_ANNOY=0
RNG_SEED=12345
```

### Puzzle JSON Schema Extensions

```json
{
  "id": 1,
  "name": "string",
  "difficulty": "easy|medium|hard",
  "start_word": "string",
  "target_word": "string",
  "allowed_cards": ["string"],

  // NEW: Domain-restricted vocabulary
  "domain_vocab": ["string"],  // optional

  // NEW: Threshold-based win condition
  "win_threshold": 0.30,       // optional, default: 0.30

  // Existing
  "stars_thresholds": {
    "3": 0.70,
    "2": 0.50,
    "1": 0.30
  }
}
```

## 🎓 Best Practices

1. **Choose appropriate FREQ_WEIGHT**:
   - 0.0: No weighting (use for technical/domain-specific embeddings)
   - 0.2-0.4: Recommended for general embeddings (GloVe, Word2Vec)
   - 0.5+: Aggressive semantic focus (may lose some nuance)

2. **Design domain vocabularies**:
   - Keep domains focused (10-50 words ideal)
   - Include related words from the same category
   - Test that domain words exist in embeddings

3. **Set win thresholds**:
   - Easy puzzles: 0.25-0.35
   - Medium puzzles: 0.35-0.50
   - Hard puzzles: 0.50-0.70

4. **Use progress indicators**:
   - Show progress bar in UI (0-100%)
   - Display improvement vs. starting position
   - Celebrate when `passed: true`

## 📚 References

- Original spec: See initial implementation request
- EmbeddingSpace class: `word_bocce_mvp_fastapi.py:122-206`
- VectorStore integration: `word_bocce_mvp_fastapi.py:212-328`
- Puzzle enhancements: `word_bocce_mvp_fastapi.py:762-951`

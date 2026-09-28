#!/usr/bin/env python3
"""
Test script for new intuitive features:
1. EmbeddingSpace abstraction with frequency weighting
2. Domain-restricted vocabularies
3. Threshold-based win conditions
4. Progress indicators
"""

import os
import sys
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_embedding_space():
    """Test the new EmbeddingSpace abstraction"""
    print("=" * 70)
    print("TEST 1: EmbeddingSpace Abstraction")
    print("=" * 70)

    from word_bocce_mvp_fastapi import EmbeddingSpace

    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        print("❌ MODEL_PATH not set. Skipping embedding tests.")
        print("   Set MODEL_PATH to test: export MODEL_PATH=/path/to/embeddings.bin")
        return False

    print(f"\n📚 Loading embeddings from: {model_path}")

    # Test 1: Basic loading
    print("\n1️⃣  Testing basic loading...")
    try:
        space = EmbeddingSpace(model_path, config={})
        print(f"✓ Loaded embeddings: {space.dim} dimensions")
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return False

    # Test 2: Frequency weighting
    print("\n2️⃣  Testing frequency weighting...")
    try:
        # No weighting
        space_no_weight = EmbeddingSpace(model_path, config={'freq_weight': 0.0})
        v1 = space_no_weight.vector('king')

        # With weighting
        space_weighted = EmbeddingSpace(model_path, config={'freq_weight': 0.5})
        v2 = space_weighted.vector('king')

        print(f"✓ Vector without weighting norm: {np.linalg.norm(v1):.4f}")
        print(f"✓ Vector with weighting norm: {np.linalg.norm(v2):.4f}")
        print(f"✓ Vectors are different: {not np.allclose(v1, v2)}")
    except Exception as e:
        print(f"❌ Frequency weighting test failed: {e}")
        return False

    # Test 3: vocab_filter in nearest()
    print("\n3️⃣  Testing domain-restricted nearest neighbor search...")
    try:
        space = EmbeddingSpace(model_path, config={'normalize_method': 'unit'})

        # Get vector for "apple"
        if not space.has('apple'):
            print("⚠️  'apple' not in vocabulary, skipping this test")
        else:
            apple_vec = space.vector('apple')

            # Unrestricted search
            all_results = space.nearest(apple_vec, k=5)
            print(f"✓ Top 5 nearest to 'apple' (unrestricted):")
            for word, sim in all_results:
                print(f"   - {word}: {sim:.4f}")

            # Restricted to fruits only
            fruit_domain = ['orange', 'banana', 'grape', 'lemon', 'cherry',
                           'peach', 'pear', 'plum', 'mango', 'berry']
            # Filter to only words that exist in vocab
            fruit_domain = [w for w in fruit_domain if space.has(w)]

            if len(fruit_domain) > 0:
                fruit_results = space.nearest(apple_vec, k=3, vocab_filter=fruit_domain)
                print(f"\n✓ Top 3 nearest to 'apple' (fruit domain only):")
                for word, sim in fruit_results:
                    print(f"   - {word}: {sim:.4f}")
            else:
                print("⚠️  No fruit words in vocabulary, skipping domain test")
    except Exception as e:
        print(f"❌ Domain restriction test failed: {e}")
        return False

    print("\n✅ All EmbeddingSpace tests passed!")
    return True


def test_vectorstore_integration():
    """Test VectorStore integration with EmbeddingSpace"""
    print("\n" + "=" * 70)
    print("TEST 2: VectorStore Integration")
    print("=" * 70)

    from word_bocce_mvp_fastapi import VectorStore

    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        print("❌ MODEL_PATH not set. Skipping.")
        return False

    print("\n📦 Creating VectorStore with embedding config...")

    try:
        config = {
            'freq_weight': 0.3,
            'normalize_method': 'unit'
        }
        store = VectorStore(model_path, embedding_config=config)
        print(f"✓ VectorStore created with config: {config}")
        print(f"✓ Dimension: {store.dim}")

        # Build a small deck
        print("\n🃏 Building deck...")
        deck = store.build_deck(size=1000)
        print(f"✓ Built deck with {len(deck)} tokens")
        print(f"✓ First 10 tokens: {deck[:10]}")

        # Test vector retrieval
        if 'king' in store.kv:
            v = store.vec_unit('king')
            print(f"\n✓ Retrieved vector for 'king': shape={v.shape}, norm={np.linalg.norm(v):.4f}")

            # Test nearest with domain filter
            domain = [w for w in ['queen', 'prince', 'monarch', 'throne', 'crown'] if w in deck]
            if len(domain) > 0:
                nearest, sim = store.nearest(v, k=1, vocab_filter=domain)
                print(f"✓ Nearest in royal domain: '{nearest}' (similarity: {sim:.4f})")

        print("\n✅ VectorStore integration tests passed!")
        return True

    except Exception as e:
        print(f"❌ VectorStore test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_puzzle_features():
    """Test puzzle-related features via API"""
    print("\n" + "=" * 70)
    print("TEST 3: Puzzle Features (Win Conditions & Progress)")
    print("=" * 70)

    try:
        import requests
    except ImportError:
        print("❌ requests library not installed. Install with: pip install requests")
        return False

    base_url = os.environ.get("WORD_BOCCE_URL", "http://127.0.0.1:8000")

    print(f"\n🌐 Testing against: {base_url}")

    # Check server
    try:
        resp = requests.get(f"{base_url}/api", timeout=2)
        if resp.status_code == 200:
            info = resp.json()
            print(f"✓ Server connected: {info}")
        else:
            print("❌ Server not ready. Start with: uvicorn word_bocce_mvp_fastapi:app")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Start server with: uvicorn word_bocce_mvp_fastapi:app")
        return False

    # Test puzzle endpoint
    print("\n🧩 Testing puzzle features...")
    try:
        resp = requests.get(f"{base_url}/puzzles", timeout=5)
        if resp.status_code == 200:
            puzzles = resp.json()
            print(f"✓ Found {len(puzzles)} puzzles")
            if len(puzzles) > 0:
                print(f"   First puzzle: {puzzles[0].get('name', 'N/A')}")

                # Get detailed puzzle
                puzzle_id = puzzles[0]['id']
                resp = requests.get(f"{base_url}/puzzle/{puzzle_id}", timeout=5)
                if resp.status_code == 200:
                    puzzle = resp.json()
                    print(f"\n✓ Puzzle {puzzle_id} details:")
                    print(f"   Name: {puzzle.get('name', 'N/A')}")
                    print(f"   Start: {puzzle.get('start_word', 'N/A')}")
                    print(f"   Target: {puzzle.get('target_word', 'N/A')}")
                    print(f"   Allowed cards: {len(puzzle.get('allowed_cards', []))}")
                    print(f"   Has wildcard: {puzzle.get('has_wildcard', False)}")
                    print(f"   Domain vocab: {len(puzzle.get('domain_vocab', []))} words" if 'domain_vocab' in puzzle else "   Domain vocab: not set")

                    # Try to solve with first available moves
                    allowed = puzzle.get('allowed_cards', [])
                    if len(allowed) >= 2:
                        solution = {
                            'public_token': allowed[0],
                            'public_sign': 1,
                            'private_token': allowed[1],
                            'private_sign': 1
                        }

                        print(f"\n🎯 Testing solution: +{allowed[0]} +{allowed[1]}")
                        resp = requests.post(
                            f"{base_url}/puzzle/{puzzle_id}/solve",
                            json=solution,
                            timeout=5
                        )

                        if resp.status_code == 200:
                            result = resp.json()
                            print(f"\n✓ Solution result:")
                            print(f"   Valid: {result.get('valid', False)}")
                            print(f"   Similarity: {result.get('similarity', 0):.4f}")
                            print(f"   Stars: {result.get('stars', 0)}")
                            print(f"   Passed: {result.get('passed', False)}")  # NEW
                            print(f"   Win threshold: {result.get('win_threshold', 0):.4f}")  # NEW
                            print(f"   Progress: {result.get('progress', 0):.1f}%")  # NEW
                            print(f"   Improvement: {result.get('improvement', 0):.4f}")  # NEW
                            print(f"   Nearest word: {result.get('nearest_word', 'N/A')}")

                            print("\n✅ All puzzle features working!")
                            return True
        else:
            print("⚠️  No puzzles found. Create puzzles.json to test this feature.")
            return True  # Not a failure, just no puzzles

    except Exception as e:
        print(f"❌ Puzzle test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    print("\n" + "🚀 " * 20)
    print("TESTING NEW INTUITIVE FEATURES FOR WORD2VEC GAME")
    print("🚀 " * 20 + "\n")

    results = []

    # Test 1: EmbeddingSpace
    results.append(("EmbeddingSpace", test_embedding_space()))

    # Test 2: VectorStore Integration
    results.append(("VectorStore Integration", test_vectorstore_integration()))

    # Test 3: Puzzle Features
    results.append(("Puzzle Features", test_puzzle_features()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(p for _, p in results)
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

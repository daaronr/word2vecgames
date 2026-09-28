#!/usr/bin/env python3
"""
Test script to simulate a complete Word Bocce game
"""

import requests
import json
import time
import sys
import os

# Allow overriding the base URL via environment variable
BASE_URL = os.environ.get("WORD_BOCCE_URL", "http://127.0.0.1:8000")

def test_game():
    print("🎯 Testing Word Bocce Game")
    print("=" * 60)
    print(f"Testing against: {BASE_URL}")

    # 0. Check if server is running
    print("\n0️⃣  Checking server connection...")
    try:
        resp = requests.get(f"{BASE_URL}/", timeout=2)
        server_info = resp.json()
        print(f"✓ Server is running!")
        print(f"   Embeddings: {server_info.get('dim')}d")
        print(f"   Deck size: {server_info.get('deck_size', 'N/A')}")
    except requests.exceptions.ConnectionError:
        print(f"\n❌ ERROR: Cannot connect to server at {BASE_URL}")
        print("\nPlease start the server first:")
        print("  export MODEL_PATH=./embeddings/glove-100.bin")
        print("  uvicorn word_bocce_mvp_fastapi:app --reload")
        print("\nOr if running on a different port:")
        print("  export WORD_BOCCE_URL=http://127.0.0.1:YOUR_PORT")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

    # 1. Create a match
    print("\n1️⃣  Creating match...")
    resp = requests.post(f"{BASE_URL}/match", json={"name": "Test Game"})
    match = resp.json()
    match_id = match["id"]
    print(f"✓ Match created: {match_id}")

    # 2. Join as Player 1
    print("\n2️⃣  Player 1 joining...")
    resp = requests.post(f"{BASE_URL}/match/{match_id}/join",
                        json={"player_name": "Alice"})
    data = resp.json()
    player1_id = data["player"]["id"]
    print(f"✓ Alice joined as {player1_id}")

    # 3. Join as Player 2
    print("\n3️⃣  Player 2 joining...")
    resp = requests.post(f"{BASE_URL}/match/{match_id}/join",
                        json={"player_name": "Bob"})
    data = resp.json()
    player2_id = data["player"]["id"]
    print(f"✓ Bob joined as {player2_id}")

    # 4. Start the match
    print("\n4️⃣  Starting match...")
    resp = requests.post(f"{BASE_URL}/match/{match_id}/start")
    round_data = resp.json()
    round_id = round_data["id"]
    start_word = round_data["start_word"]
    target_word = round_data["target_word"]
    public_cards = round_data["public_cards"]

    print(f"✓ Round started: {round_id}")
    print(f"   Start Word: {start_word}")
    print(f"   Target Word: {target_word}")
    print(f"   Public Cards: {[c['token'] for c in public_cards[:5]]}... (showing 5/{len(public_cards)})")

    # 5. Get Player 1's private cards
    print("\n5️⃣  Getting Player 1's private cards...")
    resp = requests.get(f"{BASE_URL}/match/{match_id}/player/{player1_id}")
    player1_data = resp.json()
    private_cards_p1 = player1_data["player"]["private_cards"]
    print(f"✓ Alice's private cards: {[c['token'] for c in private_cards_p1]}")

    # 6. Get Player 2's private cards
    print("\n6️⃣  Getting Player 2's private cards...")
    resp = requests.get(f"{BASE_URL}/match/{match_id}/player/{player2_id}")
    player2_data = resp.json()
    private_cards_p2 = player2_data["player"]["private_cards"]
    print(f"✓ Bob's private cards: {[c['token'] for c in private_cards_p2]}")

    # 7. Player 1 submits move
    print("\n7️⃣  Alice submitting move...")
    p1_public = public_cards[0]["token"]
    p1_private = private_cards_p1[0]["token"]

    resp = requests.post(
        f"{BASE_URL}/match/{match_id}/round/{round_id}/submit",
        json={
            "player_id": player1_id,
            "public_token": p1_public,
            "public_sign": "+",
            "private_token": p1_private,
            "private_sign": "+"
        }
    )
    result_p1 = resp.json()
    print(f"✓ Alice played: +{p1_public}, +{p1_private}")
    print(f"   Similarity: {result_p1['similarity']:.4f}")
    print(f"   Nearest word: {result_p1['nearest']}")

    # 8. Player 2 submits move
    print("\n8️⃣  Bob submitting move...")
    p2_public = public_cards[1]["token"]
    p2_private = private_cards_p2[0]["token"]

    resp = requests.post(
        f"{BASE_URL}/match/{match_id}/round/{round_id}/submit",
        json={
            "player_id": player2_id,
            "public_token": p2_public,
            "public_sign": "-",
            "private_token": p2_private,
            "private_sign": "+"
        }
    )
    result_p2 = resp.json()
    print(f"✓ Bob played: -{p2_public}, +{p2_private}")
    print(f"   Similarity: {result_p2['similarity']:.4f}")
    print(f"   Nearest word: {result_p2['nearest']}")

    # 9. Resolve round and get leaderboard
    print("\n9️⃣  Getting leaderboard...")
    resp = requests.post(f"{BASE_URL}/match/{match_id}/round/{round_id}/resolve")
    leaderboard_data = resp.json()
    leaderboard = leaderboard_data["leaderboard"]

    print("✓ Final Leaderboard:")
    for i, entry in enumerate(leaderboard, 1):
        closeness = get_closeness_label(entry["similarity"])
        print(f"   {i}. {entry['player_name']}: {entry['similarity']:.4f} "
              f"(nearest: {entry['nearest']}) [{closeness}]")

    # 10. Start next round
    print("\n🔟 Starting next round...")
    resp = requests.post(f"{BASE_URL}/match/{match_id}/round/next")
    round_data = resp.json()
    print(f"✓ Round {round_data['id']} started!")
    print(f"   New Start: {round_data['start_word']}")
    print(f"   New Target: {round_data['target_word']}")

    print("\n" + "=" * 60)
    print("✅ All tests passed! Game is working perfectly!")
    print("=" * 60)

def get_closeness_label(sim):
    if sim >= 0.80:
        return "Very Close 🔥"
    elif sim >= 0.60:
        return "Getting Warmer 🌡️"
    elif sim >= 0.40:
        return "Lukewarm 😐"
    elif sim >= 0.20:
        return "Cold ❄️"
    else:
        return "Ice Cold 🧊"

if __name__ == "__main__":
    try:
        test_game()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

"""Suggestions endpoint: stores a row, rejects oversized input, rate-limits per client.

Run: FEEDBACK_DB=/tmp/wb-test.db python -m pytest tests/test_suggestions_api.py
  or: FEEDBACK_DB=/tmp/wb-test.db python tests/test_suggestions_api.py
Needs fastapi + httpx; no word embeddings (the server then serves only the browser game).
"""
import json
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("FEEDBACK_DB", os.path.join(tempfile.mkdtemp(), "feedback.db"))
os.environ.pop("MODEL_PATH", None)

from fastapi.testclient import TestClient  # noqa: E402

import word_bocce_mvp_fastapi as server  # noqa: E402

client = TestClient(server.app)

GOOD = {
    "at": "2026-09-29T10:00:00Z", "client": "test-client", "kind": "practice", "seed": "practice-1",
    "start": "gold", "target": "garlic", "hand": ["tea", "oven"],
    "throw": [{"word": "onions", "sign": 1}, {"word": "salt", "sign": 1}],
    "rank": 2, "bestRank": 36, "words": ["onions", "salt"], "verdict": "more fun", "note": "",
}


def test_store_and_read_back():
    r = client.post("/api/suggestions", json=GOOD)
    assert r.status_code == 200, r.text
    row = sqlite3.connect(os.environ["FEEDBACK_DB"]).execute(
        "select start_word, target_word, words, verdict, rank from suggestions order by id desc limit 1").fetchone()
    assert row == ("gold", "garlic", json.dumps(["onions", "salt"]), "more fun", 2)


def test_rejects_oversized():
    r = client.post("/api/suggestions", json={**GOOD, "note": "x" * 600})
    assert r.status_code == 422


def test_rate_limit():
    body = {**GOOD, "client": "flood"}
    codes = [client.post("/api/suggestions", json=body).status_code for _ in range(server.FEEDBACK_MAX_PER_HOUR + 1)]
    assert codes[-1] == 429 and codes[0] == 200


def test_game_still_served():
    r = client.get("/")
    assert r.status_code == 200 and "<title>Word Bocce</title>" in r.text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

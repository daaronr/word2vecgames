"""With GAME_URL set, game pages redirect to the published copy; the API and /classic stay here.

Run: python tests/test_game_redirect.py   (needs fastapi + httpx; no word embeddings)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["GAME_URL"] = "https://example.org/game/"
os.environ.setdefault("FEEDBACK_DB", os.path.join(tempfile.mkdtemp(), "feedback.db"))
os.environ.pop("MODEL_PATH", None)

from fastapi.testclient import TestClient  # noqa: E402

import word_bocce_mvp_fastapi as server  # noqa: E402

client = TestClient(server.app)


def test_game_pages_redirect():
    for path, want in [("/", "https://example.org/game/"), ("/presentation.html", "https://example.org/game/presentation.html"),
                       ("/data/vectors.bin?x=1", "https://example.org/game/data/vectors.bin?x=1")]:
        r = client.get(path, follow_redirects=False)
        assert r.status_code == 302 and r.headers["location"] == want, (path, r.status_code, r.headers.get("location"))


def test_api_classic_and_suggestions_stay():
    assert client.get("/api").status_code == 200
    assert client.get("/classic").status_code == 200
    r = client.post("/api/suggestions", json={"start": "hat", "target": "shoe", "client": "redirect-test"})
    assert r.status_code == 200, r.text


if __name__ == "__main__":
    test_game_pages_redirect()
    test_api_classic_and_suggestions_stay()
    print("redirect ok")

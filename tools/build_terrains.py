#!/usr/bin/env python3
"""
Fill in the cards of the Cat and mouse terrains (web/chase/terrains.json).

A terrain's words are where players stand (things: dog, kitchen, car). Its cards, if it has a
"cards" list, are what they play onto them: describing and doing words from tools/chase_cards.txt
(wild, cold, fly), which tend to move you somewhere that makes sense (car + fly → airplane).
Keeps the card words that are on the terrain's map and not blocklisted.

Usage: python3 tools/build_terrains.py && node tests/engine.test.js
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAPS = {"sense": "data-sense", "text": "data"}


def words(path):
    with open(path, encoding="utf8") as f:
        return [w for line in f if not line.startswith("#") for w in line.split()]


def main():
    path = os.path.join(ROOT, "web", "chase", "terrains.json")
    terrains = json.load(open(path, encoding="utf8"))
    block = {w.lower() for w in words(os.path.join(HERE, "blocklist.txt"))}
    cards = list(dict.fromkeys(words(os.path.join(HERE, "chase_cards.txt"))))
    for t in terrains:
        if "cards" not in t:
            continue
        vocab = set(open(os.path.join(ROOT, "web", MAPS[t["map"]], "vocab.txt"), encoding="utf8").read().split("\n"))
        t["cards"] = [w for w in cards if w in vocab and w not in block]
        missing = [w for w in cards if w not in vocab]
        print(f"{t['id']}: {len(t['cards'])} cards" + (f"; not on the map: {' '.join(missing)}" if missing else ""))
    with open(path, "w", encoding="utf8") as f:
        f.write("[\n" + ",\n".join(json.dumps(t, ensure_ascii=False) for t in terrains) + "\n]\n")


if __name__ == "__main__":
    main()

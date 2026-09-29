#!/usr/bin/env python3
"""
Build the compact word-vector bundle used by the in-browser game (web/).

Reads a GloVe/word2vec text file (optionally .gz, with or without a
"count dim" header line) and writes:

  web/data/vocab.txt    one word per line (row order of vectors.bin)
  web/data/vectors.bin  int8 matrix, rows unit-normalised then scaled per row
  web/data/pools.json   {"dim", "count", "cards": [...], "targets": [...]}

Card and target pools come from curated common-word lists, intersected with
the embedding vocabulary, so the game deals familiar words.

Usage:
  python tools/build_web_data.py embeddings/glove-100.gz \
      --lists embeddings --size 40000

Word norms (put in --lists dir, CRLF is fine):
  concreteness.txt  Brysbaert, Warriner & Kuperman (2014) concreteness ratings,
                    github.com/ArtsEngine/concreteness
The GloVe 6B vectors are public-domain (PDDL); gensim-data mirrors them on GitHub:
  https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz
"""
import argparse
import gzip
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORD_RE = re.compile(r"^[a-z]{3,14}$")


def read_list(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf8") as f:
        return [w.strip().lower() for w in f if w.strip() and not w.startswith("#")]


def load_vectors(path, limit, only=None):
    """Read up to `limit` rows; with `only` (a set), read the whole file but keep just those words."""
    opener = gzip.open if path.endswith(".gz") else open
    words, vecs = [], []
    with opener(path, "rt", encoding="utf8") as f:
        first = f.readline().rstrip().split(" ")
        if len(first) != 2 and (only is None or first[0] in only):  # no header: first line is a vector
            words.append(first[0])
            vecs.append(np.asarray(first[1:], dtype=np.float32))
        for line in f:
            if only is None and len(words) >= limit:
                break
            w, _, rest = line.rstrip().partition(" ")
            if only is not None and w not in only:
                continue
            words.append(w)
            vecs.append(np.asarray(rest.split(" "), dtype=np.float32))
    return words, np.vstack(vecs)


def reduce_dims(X, dims):
    """Keep the `dims` strongest directions (PCA on unit rows), so the bundle stays small."""
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[:dims].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vectors")
    ap.add_argument("--lists", default="embeddings")
    ap.add_argument("--size", type=int, default=40000, help="vocabulary rows to export")
    ap.add_argument("--scan", type=int, default=120000, help="rows of the source file to scan")
    ap.add_argument("--out", default=os.path.join(ROOT, "web", "data"))
    ap.add_argument("--vocab-from", help="use this vocab.txt's words, in its order (e.g. web/data/vocab.txt), "
                    "for sources not sorted by frequency such as ConceptNet Numberbatch")
    ap.add_argument("--dims", type=int, default=0, help="reduce to this many dimensions (PCA); 0 keeps all")
    ap.add_argument("--everyday", type=float, default=0,
                    help="keep only words at least this share of people know (e.g. 0.9, from the concreteness "
                         "norms, plus their plurals) and the game's own words; 0 keeps all")
    args = ap.parse_args()

    stop = set(open(os.path.join(HERE, "stopwords.txt")).read().split())
    block = set(read_list(os.path.join(HERE, "blocklist.txt")))
    puzzles = json.load(open(os.path.join(ROOT, "web", "data", "puzzles.json")))
    puzzle_words = set()
    for p in puzzles:
        puzzle_words |= {p["start_word"], p["target_word"]}
        puzzle_words |= {c for c in p.get("allowed_cards", []) if c != "WILDCARD"}

    order = read_list(args.vocab_from) if args.vocab_from else None
    words, M = load_vectors(args.vectors, args.scan, set(order) | puzzle_words if order else None)
    if order:  # put the rows in the given (frequency) order; that order is what `rank` means below
        pos = {w: i for i, w in enumerate(words)}
        in_order = set(order)
        present = [w for w in order if w in pos] + sorted(w for w in puzzle_words if w in pos and w not in in_order)
        M = M[[pos[w] for w in present]]
        words = present
    rank = {w: i for i, w in enumerate(words)}
    print(f"scanned {len(words)} rows, dim {M.shape[1]}")
    if args.dims:
        M = reduce_dims(M, args.dims)
        print(f"reduced to {args.dims} dimensions")

    def ok(w):
        return WORD_RE.match(w) and w not in stop and w not in block

    everyday = None
    if args.everyday:
        known = set()
        with open(os.path.join(args.lists, "concreteness.txt"), encoding="utf8") as f:
            next(f)
            for line in f:
                c = line.rstrip("\r\n").split("\t")
                if c[1] == "0" and float(c[6]) >= args.everyday:
                    known.add(c[0].lower())

        def everyday(w):
            return (w in known or (w.endswith("s") and w[:-1] in known)
                    or (w.endswith("es") and w[:-2] in known) or (w.endswith("ies") and w[:-3] + "y" in known))

    keep = [i for i, w in enumerate(words) if ok(w) and (everyday is None or everyday(w) or w in puzzle_words)][: args.size]
    keep_set = set(keep)
    for w in puzzle_words:  # puzzles must always be playable
        if w in rank and rank[w] not in keep_set:
            keep.append(rank[w])
            keep_set.add(rank[w])
    missing = sorted(w for w in puzzle_words if w not in rank)
    if missing:
        print("puzzle words missing from embeddings:", missing)
    vocab = [words[i] for i in keep]
    X = M[keep]
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    scale = np.abs(X).max(axis=1, keepdims=True)
    Q = np.round(X / scale * 127).astype(np.int8)

    # Pools: familiar words, using Brysbaert et al. (2014) concreteness norms,
    # which also carry SUBTLEX frequency, % of raters who knew the word, and POS.
    norms = {}
    with open(os.path.join(args.lists, "concreteness.txt"), encoding="utf8") as f:
        next(f)
        for line in f:
            c = line.rstrip("\r\n").split("\t")
            if c[1] == "0":  # skip bigrams
                norms[c[0]] = dict(conc=float(c[2]), known=float(c[6]), freq=float(c[7]), pos=c[8])
    extra = " ".join(read_list(os.path.join(HERE, "targets_extra.txt"))).split()
    in_vocab = set(vocab)

    def pool(pred, cap, add=()):
        ws = {w for w, n in norms.items() if pred(n)} | set(add)
        ws = [w for w in ws if w in in_vocab and ok(w) and rank[w] < cap]
        return sorted(ws, key=lambda w: rank[w])

    # Cards: everyday nouns and adjectives, concrete or abstract.
    cards = pool(lambda n: n["pos"] in ("Noun", "Adjective") and n["freq"] >= 150 and n["known"] >= 0.97,
                 cap=30000)
    # Targets ("jacks"): vivid, concrete, well-known nouns plus a hand-picked list.
    targets = pool(lambda n: n["pos"] == "Noun" and n["conc"] >= 4.2 and n["freq"] >= 200 and n["known"] >= 0.98,
                   cap=25000, add=extra)
    targets = [w for w in targets if not (w.endswith("s") and w[:-1] in norms)]  # singular jacks read better

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "vocab.txt"), "w") as f:
        f.write("\n".join(vocab))
    Q.tofile(os.path.join(args.out, "vectors.bin"))
    with open(os.path.join(args.out, "pools.json"), "w") as f:
        json.dump({"dim": int(Q.shape[1]), "count": len(vocab), "source": os.path.basename(args.vectors),
                   "cards": cards, "targets": targets}, f, separators=(",", ":"))
    print(f"vocab {len(vocab)}, cards {len(cards)}, targets {len(targets)}, "
          f"vectors.bin {Q.nbytes/1e6:.1f} MB")


if __name__ == "__main__":
    main()

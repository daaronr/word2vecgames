#!/usr/bin/env python3
"""
Build web/data-tokens/: the "AI tokens" map, from GPT-2's own token table.

GPT-2 (OpenAI, 2019, modified MIT licence) turns each of its 50,257 tokens into a row of 768
numbers before anything else happens; that table (`wte`) is the map this builds. Only that one
tensor is fetched, with an HTTP range request (about 150 MB), not the whole 550 MB model.

Steps: drop tokens that make bad court labels (pieces of characters, control characters, very
long runs, blocklisted words), mean-centre the rest (GPT-2's rows share one large common
direction that otherwise swamps cosine similarity), keep the strongest `--dims` directions (PCA),
and quantise to int8 as build_web_data.py does.

Writes web/data-tokens/:
  vocab.txt    one token label per line, as web/tokens.js labels it (␣ = a leading space)
  vectors.bin  int8 rows in vocab.txt order
  pools.json   {"dim", "count", "cards", "targets"}: the familiar words from web/data-sense
               and web/data whose mid-sentence form (" word") is a single GPT-2 token

Usage:
  python3 tools/build_token_data.py              # downloads into embeddings/ once
  node tests/engine.test.js
"""
import argparse
import json
import os
import struct
import subprocess
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
URL = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"


def fetch_range(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def load_wte(cache):
    if os.path.exists(cache):
        return np.load(cache)
    n = struct.unpack("<Q", fetch_range(URL, 0, 7))[0]
    header = json.loads(fetch_range(URL, 8, 8 + n - 1))
    info = header["wte.weight"]
    a, b = info["data_offsets"]
    print(f"fetching wte {info['shape']} ({(b - a) / 1e6:.0f} MB)")
    raw = fetch_range(URL, 8 + n + a, 8 + n + b - 1)
    W = np.frombuffer(raw, dtype="<f4").reshape(info["shape"]).copy()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, W)
    return W


def token_labels():
    """Labels for every GPT-2 token, from web/tokens.js (so the court and the Tokens tab agree)."""
    js = (
        "const fs=require('fs');const {Tokenizer}=require(process.argv[1]+'/web/tokens.js');"
        "const T=new Tokenizer(fs.readFileSync(process.argv[1]+'/web/tokens/gpt2-merges.txt','utf8'));"
        "const out=[];for(let i=0;i<T.size;i++){const l=T.label(i);out.push([l.text,l.partial]);}"
        "process.stdout.write(JSON.stringify(out));"
    )
    return json.loads(subprocess.check_output(["node", "-e", js, ROOT]))


def read_list(path):
    with open(path, encoding="utf8") as f:
        return [w.strip().lower() for w in f if w.strip() and not w.startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "embeddings", "gpt2-wte.npy"))
    ap.add_argument("--dims", type=int, default=128)
    ap.add_argument("--out", default=os.path.join(ROOT, "web", "data-tokens"))
    args = ap.parse_args()

    W = load_wte(args.cache).astype(np.float64)
    labels = token_labels()
    assert len(labels) == W.shape[0]
    block = set(read_list(os.path.join(HERE, "blocklist.txt")))

    def ok(i):
        text, partial = labels[i]
        core = text.lstrip("␣").lower()
        if partial or i == len(labels) - 1:            # pieces of characters, <|endoftext|>
            return False
        if not core or len(text) > 20 or any(c in text for c in "↵⇥␍"):
            return False
        if any(ord(c) < 32 or 0x7f <= ord(c) < 0xa0 for c in text):
            return False
        return core not in block and not any(b in core for b in block if len(b) >= 5)

    keep = [i for i in range(W.shape[0]) if ok(i)]
    X = W[keep]
    X = X - W.mean(axis=0, keepdims=True)              # centre on the whole table's mean
    _, _, vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    X = X @ vt[: args.dims].T
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q = np.round(X / np.abs(X).max(axis=1, keepdims=True) * 127).astype(np.int8)
    vocab = [labels[i][0] for i in keep]
    assert len(set(vocab)) == len(vocab), "token labels must be unique"

    # Pools: the word sets' familiar words that GPT-2 keeps whole after a space.
    have = set(vocab)
    pools = [json.load(open(os.path.join(ROOT, "web", d, "pools.json"))) for d in ("data-sense", "data")]
    def merged(key):
        seen, out = set(), []
        for p in pools:
            for w in p[key]:
                t = "␣" + w
                if t in have and t not in seen:
                    seen.add(t)
                    out.append(t)
        return out
    cards, targets = merged("cards"), merged("targets")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "vocab.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(vocab))
    Q.tofile(os.path.join(args.out, "vectors.bin"))
    with open(os.path.join(args.out, "pools.json"), "w", encoding="utf8") as f:
        json.dump({"dim": int(Q.shape[1]), "count": len(vocab), "source": "gpt2 wte",
                   "cards": cards, "targets": targets}, f, ensure_ascii=False, separators=(",", ":"))
    print(f"tokens {len(vocab)} of {W.shape[0]}, dims {args.dims}, cards {len(cards)}, targets {len(targets)}, "
          f"vectors.bin {Q.nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

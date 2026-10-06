"""Build the compact word-vector file the robot judge uses.

Source: the Word Bocce "common sense" set in web/data-sense/ (ConceptNet Numberbatch 19.08,
21,114 everyday English words, 300 dims, int8; CC BY-SA 4.0). We centre the vectors, keep the
top principal components (96 by default; cosine similarities stay ~0.86-correlated with the full
300 dims), and re-quantise each row to int8 so it fits in about 2 MB.

Output: one little-endian binary file
    b"WIIV" | uint32 n_words | uint32 dim | uint32 vocab_bytes | vocab (UTF-8, "\n"-joined) | int8[n*dim]

Usage:
    python3 whatisit/tools/build_vectors.py            # writes whatisit/site/judge-vectors.bin
    python3 whatisit/tools/build_vectors.py --dim 128
"""
import argparse
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "web" / "data-sense"))
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--out", default=str(ROOT / "whatisit" / "site" / "judge-vectors.bin"))
    args = ap.parse_args()

    src = Path(args.src)
    vocab = [w for w in (src / "vocab.txt").read_text(encoding="utf-8").split("\n") if w]
    raw = np.fromfile(src / "vectors.bin", dtype=np.int8)
    full_dim = raw.size // len(vocab)
    assert raw.size == len(vocab) * full_dim, "vectors.bin does not match vocab.txt"

    X = raw.reshape(len(vocab), full_dim).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[: args.dim].T
    Q = np.round(P / (np.abs(P).max(axis=1, keepdims=True) + 1e-9) * 127).astype(np.int8)

    vb = "\n".join(vocab).encode("utf-8")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"WIIV")
        f.write(struct.pack("<III", len(vocab), args.dim, len(vb)))
        f.write(vb)
        f.write(Q.tobytes())
    print(f"{out}: {len(vocab)} words x {args.dim} dims, {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

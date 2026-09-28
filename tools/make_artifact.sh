#!/bin/sh
# Stage web/ as a claude.ai Artifact: the host wraps the page in its own
# <!doctype>/<head>/<body>, so strip ours and copy the assets alongside.
set -e
OUT=${1:-dist/artifact}
mkdir -p "$OUT/data"
sed -e '/^<!doctype html>$/d' -e '/^<html/d' -e '/^<\/html>$/d' -e '/^<head>$/d' -e '/^<\/head>$/d' \
    -e '/^<body>$/d' -e '/^<\/body>$/d' -e '/<meta charset/d' -e '/<meta name="viewport"/d' web/index.html > "$OUT/index.html"
cp web/engine.js web/app.js web/style.css web/presentation.html "$OUT/"
cp web/data/vocab.txt web/data/pools.json web/data/puzzles.json "$OUT/data/"
# Artifacts only serve known web types; ship the raw int8 bytes under .wasm
# (fetched as an ArrayBuffer, never instantiated).
cp web/data/vectors.bin "$OUT/data/vectors.wasm"
sed -i 's|<script src="engine.js"></script>|<script>window.WORD_BOCCE_VECTORS = "vectors.wasm";</script>\n<script src="engine.js"></script>|' "$OUT/index.html"
echo "staged $OUT"

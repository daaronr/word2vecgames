#!/usr/bin/env bash
# Copy whatisit/ into its own repository, with its git history, without touching your working tree.
#
#   bash whatisit/tools/split_repo.sh git@github.com:daaronr/whatisit.git [source-branch]
#
# Run from inside a clone of word2vecgames, after every session working on whatisit has pushed.
# The new repository should be empty (no README). Afterwards see "After the split" in TODO.md.
set -euo pipefail
NEW_REMOTE=${1:?usage: split_repo.sh <new repo URL> [source branch]}
SRC=${2:-claude/quirky-brahmagupta-l9gamm}
cd "$(git rev-parse --show-toplevel)"
git fetch origin "$SRC"
git subtree split --prefix=whatisit "origin/$SRC" -b whatisit-only
git push "$NEW_REMOTE" whatisit-only:main
git branch -D whatisit-only
echo "Pushed whatisit/ (with history) to $NEW_REMOTE as main."

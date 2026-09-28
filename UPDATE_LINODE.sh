#!/bin/bash
# Update Word Bocce on the Linode server (run there, as root).
# The app is a git checkout at /opt/word2vecgames served by the "word-bocce" systemd unit
# (uvicorn on port 8000, system Python, no nginx in front).
set -e
cd /opt/word2vecgames
echo "Pulling latest main..."
git fetch origin main
git checkout main
git pull --ff-only origin main
# System Python is "externally managed" (PEP 668), so pip refuses to install into it.
# Dependencies are already installed; this only warns if requirements.txt now needs more.
pip3 install -q -r requirements.txt 2>/dev/null || echo "Note: pip skipped (externally managed Python). If requirements.txt changed, install the new packages by hand."
echo "Restarting service..."
systemctl restart word-bocce
sleep 2
systemctl status word-bocce --no-pager | head -5
echo "Done. The game is at http://45.79.160.157:8000/, the old multiplayer UI at /classic."

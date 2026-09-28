#!/bin/bash
# Update Word Bocce on the Linode server (run there, as root).
# The app is a git checkout at /opt/word2vecgames served by the "wordbocce" systemd unit.
set -e
cd /opt/word2vecgames
echo "Pulling latest main..."
git fetch origin main
git checkout main
git pull --ff-only origin main
pip3 install -q -r requirements.txt
echo "Restarting service..."
systemctl restart wordbocce
sleep 2
systemctl status wordbocce --no-pager | head -5
echo "Done. The game is at /, the old multiplayer UI at /classic."

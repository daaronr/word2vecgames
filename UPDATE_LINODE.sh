#!/bin/bash
# Update Word Bocce on Linode without Git

set -e

echo "🔄 Updating Word Bocce on Linode..."

# Go to app directory
cd /opt/word2vecgames || exit 1

# Backup current files
echo "📦 Creating backup..."
cp index.html index.html.backup
cp word_bocce_mvp_fastapi.py word_bocce_mvp_fastapi.py.backup
cp puzzles.json puzzles.json.backup

# Download latest files from GitHub
echo "⬇️  Downloading latest files..."
REPO_BASE="https://raw.githubusercontent.com/daaronr/word2vecgames/main"

curl -fsSL "${REPO_BASE}/index.html" -o index.html
curl -fsSL "${REPO_BASE}/word_bocce_mvp_fastapi.py" -o word_bocce_mvp_fastapi.py
curl -fsSL "${REPO_BASE}/puzzles.json" -o puzzles.json
curl -fsSL "${REPO_BASE}/presentation.html" -o presentation.html

echo "✅ Files updated!"

# Restart service
echo "🔄 Restarting service..."
systemctl restart wordbocce

# Check status
echo "✅ Checking service status..."
sleep 2
systemctl status wordbocce --no-pager

echo ""
echo "🎉 Update complete! Your game is now live with:"
echo "   - 30 culturally-referenced puzzles"
echo "   - Best move display"
echo "   - All moves comparison"
echo "   - 2D word space visualization"
echo ""
echo "Visit: http://45.79.160.157"

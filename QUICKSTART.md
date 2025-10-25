# 🎯 Word Bocce - Quickstart Guide

Get playing in 3 minutes!

## Step 1: Install (1 minute)

```bash
pip install -r requirements.txt
```

## Step 2: Download Embeddings (1-5 minutes depending on size)

```bash
# Fast download, good for testing (~130MB)
python setup_embeddings.py --model glove-100
```

## Step 3: Start the Game (10 seconds)

```bash
./start.sh
```

That's it! Open http://localhost:8080 and start playing! 🎉

---

## Manual Start (if start.sh doesn't work)

**Terminal 1 - Backend:**
```bash
export MODEL_PATH=./embeddings/glove-100.bin
uvicorn word_bocce_mvp_fastapi:app --reload
```

**Terminal 2 - Frontend:**
```bash
python run_frontend.py
```

**Browser:**
Open http://localhost:8080

---

## First Game Tutorial

1. **Create a Match**
   - Enter your name
   - Click "Create New Match"
   - Share the Match ID with friends (or play solo for testing)

2. **Start Round**
   - Click "Start Game"
   - You'll see:
     - Starting word (e.g., "apple")
     - Target word (e.g., "cider")
     - 10 public cards everyone can see
     - 2 private cards only you see

3. **Make Your Move**
   - Pick one public card (click + or -)
   - Pick one private card (click + or -)
   - Click "Submit Move"

4. **See Results**
   - Click "View Results" to see the leaderboard
   - Highest cosine similarity wins!
   - See what word your vector landed on

5. **Next Round**
   - Click "Next Round" to play again

---

## Pro Tips

- **Think semantically**: If target is "cider", words like "ferment", "juice", "apple" point in the right direction
- **Use subtraction**: Sometimes you need to move *away* from concepts
- **JOKER cards**: If you get a 🃏, you can type ANY word!
- **Nearest word**: Shows what actual word your vector is closest to - helps you understand the space

---

## Troubleshooting

**Can't connect?**
- Make sure both backend (port 8000) and frontend (port 8080) are running

**"Token not in vocabulary"?**
- Try a simpler word, or use the google-news model which has more words

**Too slow?**
- Set `export USE_ANNOY=1` before starting backend
- Or reduce deck size: `export DECK_SIZE=50000`

---

## Next Steps

- Read [README.md](README.md) for full documentation
- Check [wordbocce_description.md](wordbocce_description.md) for game design details
- See [CLAUDE.md](CLAUDE.md) for developer documentation

**Have fun playing Word Bocce!** 🎯

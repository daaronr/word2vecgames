# Word Bocce - Deployment Guide

Multiple ways to deploy and share Word Bocce with friends!

## 🚀 Quick Deploy Options

### Option 1: Docker (Easiest!)

```bash
# Build and run with one command
docker-compose up

# Or manually:
docker build -t wordbocce .
docker run -p 8000:8000 wordbocce
```

Then visit `http://localhost:8000` (backend serves the frontend too!)

**Pros**: No Python installation needed, works everywhere
**Cons**: Larger download (~500MB with embeddings)

---

### Option 2: Local Python (Fastest)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download embeddings (one-time, ~2 minutes)
python setup_embeddings.py --model glove-100

# 3. Start everything with one command!
./start.sh
```

Visit `http://localhost:8080`

**Pros**: Fast startup, easy to modify
**Cons**: Requires Python 3.9+

---

### Option 3: Share on LAN (Play with Friends)

Share the game with friends on the same WiFi:

```bash
# 1. Start the servers
./start.sh

# 2. Find your IP address
# Mac: ifconfig | grep "inet " | grep -v 127.0.0.1
# Linux: hostname -I
# Windows: ipconfig

# 3. Share this URL with friends:
# http://YOUR_IP:8080
```

Example: If your IP is `192.168.1.100`, friends visit `http://192.168.1.100:8080`

---

### Option 4: Cloud Deploy (Share with Anyone!)

#### Railway.app (Free tier available)

1. Sign up at [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub"
3. Connect your Word Bocce repo
4. Railway auto-detects the Dockerfile
5. Get a public URL like `wordbocce.up.railway.app`

#### Render.com (Free tier available)

1. Sign up at [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your repo
4. Render builds from Dockerfile
5. Get a public URL

#### fly.io (Free tier)

```bash
# Install fly CLI
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Deploy!
fly launch
fly deploy
```

---

## 📊 Deployment Comparison

| Method | Setup Time | Cost | Best For |
|--------|-----------|------|----------|
| Docker Compose | 5 min | Free | Local development |
| Local Python | 3 min | Free | Quick testing |
| LAN Sharing | 5 min | Free | House parties |
| Railway/Render | 10 min | Free tier | Public sharing |
| fly.io | 10 min | Free tier | Advanced users |

---

## 🔧 Configuration

All deployments use these environment variables:

```bash
MODEL_PATH=./embeddings/glove-100.bin  # Path to embeddings
DECK_SIZE=10000                         # Vocabulary size (10k = common words)
WILDCARD_RATIO=0.2                      # 20% chance for JOKER cards
ROUND_TIMEOUT_SECS=120                  # 2 minutes per round
```

To customize, edit:
- `docker-compose.yml` for Docker
- `start.sh` for local deployment
- Environment variables in cloud platform settings

---

## 🎮 Testing Your Deployment

```bash
# Test the backend API
curl http://localhost:8000/

# Should return:
# {"status":"ok","dim":100,"deck_size":10000}

# Test frontend
open http://localhost:8080  # Mac
xdg-open http://localhost:8080  # Linux
start http://localhost:8080  # Windows
```

---

## 🐛 Troubleshooting

**Port already in use:**
```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different ports
export BACKEND_PORT=8001
export FRONTEND_PORT=8081
```

**Embeddings not found:**
```bash
# Download again
python setup_embeddings.py --model glove-100

# Verify file exists
ls -lh embeddings/glove-100.bin
```

**Docker build slow:**
```bash
# The first build downloads embeddings (~130MB)
# Subsequent builds use cache and are fast
docker-compose build --no-cache  # Force rebuild
```

**Can't connect on LAN:**
- Check firewall settings
- Make sure you're on the same network
- Try `0.0.0.0` instead of `127.0.0.1` in backend

---

## 📱 Mobile Considerations

The game works on mobile browsers! Make sure to:
1. Use responsive design (already implemented)
2. Deploy with HTTPS for PWA support
3. Test touch interactions

---

## 🔒 Security Notes

**For local/LAN deployment:**
- No security needed for trusted networks

**For public deployment:**
- Enable HTTPS (Railway/Render do this automatically)
- Consider adding authentication for private games
- Rate limit API endpoints to prevent abuse

---

## 🚢 Production Checklist

- [ ] Embeddings downloaded
- [ ] Dependencies installed
- [ ] Backend starts without errors
- [ ] Frontend loads correctly
- [ ] Can create and join matches
- [ ] Cards display properly
- [ ] Moves can be submitted
- [ ] Leaderboard works
- [ ] Next round works

Test with: `python test_game.py`

---

## 📈 Scaling

**For more players:**
- Increase server resources (2GB RAM minimum)
- Use Redis for match state (not implemented in MVP)
- Deploy multiple instances behind load balancer

**For better performance:**
- Set `USE_ANNOY=1` for faster nearest neighbor search
- Reduce `DECK_SIZE` for faster startup
- Use CDN for static files

---

## 🎓 Next Steps

After deploying:
1. Share the URL with friends
2. Create a match and test gameplay
3. Check the presentation: `http://YOUR_URL/presentation.html`
4. Monitor server logs for issues
5. Have fun! 🎯

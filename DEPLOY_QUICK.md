# 🚀 Quick Deployment Guide

Get Word Bocce online in under 10 minutes!

---

## Option 1: Railway (Recommended ⭐)

**Why Railway**: Free tier, auto-deploy from GitHub, takes 5 minutes.

### Steps:

1. **Sign up**: https://railway.app
2. **New Project** → "Deploy from GitHub repo"
3. **Select** your `word2vecgames` repository
4. **Add Environment Variable**:
   - `MODEL_PATH` = `./embeddings/glove-100.bin`
5. **Generate Domain**: Settings → Generate Domain
6. **Done!** You'll get: `https://word-bocce.up.railway.app`

**Free Tier**: $5 credit/month, auto-sleeps after inactivity

---

## Option 2: Render.com

**Why Render**: Similar to Railway, good free tier.

### Steps:

1. **Sign up**: https://render.com
2. **New** → **Web Service** → Connect GitHub
3. **Settings**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn word_bocce_mvp_fastapi:app --host 0.0.0.0 --port $PORT`
4. **Environment**:
   - `MODEL_PATH` = `./embeddings/glove-100.bin`
5. **Create** → Auto-deploys!

**Free Tier**: 750 hours/month (enough for hobby use)

---

## Option 3: Fly.io (Advanced)

**Why Fly.io**: Best global performance, generous free tier.

### Steps:

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Deploy (from project directory)
flyctl launch --now

# Open in browser
flyctl open
```

**Free Tier**: 3 shared VMs, 160GB bandwidth/month

---

## Option 4: Your Linode VPS

You mentioned you have Linode. See `LINODE_DEPLOY.md` for full guide.

**Quick version**:
```bash
# SSH into Linode
ssh root@your-linode-ip

# Clone repo
git clone https://github.com/yourusername/word2vecgames.git
cd word2vecgames

# Install dependencies
pip3 install -r requirements.txt

# Download embeddings
python3 setup_embeddings.py --model glove-100

# Run with systemd or screen
screen -S wordbocce
export MODEL_PATH=./embeddings/glove-100.bin
uvicorn word_bocce_mvp_fastapi:app --host 0.0.0.0 --port 8000
```

Then access at: `http://your-linode-ip:8000`

---

## 🔒 Adding Password Protection

### Method 1: HTTP Basic Auth (Simplest)

Add to `word_bocce_mvp_fastapi.py`:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

security = HTTPBasic()

def verify_password(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "demo")
    correct_password = secrets.compare_digest(credentials.password, "bocce2025")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Add to protected routes
@app.get("/", dependencies=[Depends(verify_password)])
async def read_index():
    return FileResponse("index.html")
```

**Users will see browser login popup**: Username: `demo`, Password: `bocce2025`

### Method 2: Environment Variable Password

```python
import os
from fastapi import Header, HTTPException

SECRET_TOKEN = os.getenv("ACCESS_TOKEN", "your-secret-token")

def verify_token(x_access_token: str = Header(...)):
    if x_access_token != SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid access token")
```

Set `ACCESS_TOKEN` in Railway/Render environment variables.

### Method 3: Simple Query Parameter (Quick & Dirty)

```python
from fastapi import Query, HTTPException

@app.get("/")
async def read_index(access: str = Query(None)):
    if access != "demo2025":
        raise HTTPException(status_code=403, detail="Access denied")
    return FileResponse("index.html")
```

Share link: `https://your-app.com/?access=demo2025`

---

## 📊 Monitoring & Logs

### Railway
- Dashboard → Deployments → View Logs
- Metrics tab shows CPU/RAM usage

### Render
- Dashboard → Logs tab (real-time)
- Metrics tab for performance

### Fly.io
```bash
flyctl logs        # View logs
flyctl status      # Check status
flyctl scale count 1  # Scale instances
```

---

## 🆓 Cost Comparison

| Platform | Free Tier | Paid Tier | Auto-Sleep |
|----------|-----------|-----------|------------|
| **Railway** | $5/month credit | $5+/month usage | ✅ Yes |
| **Render** | 750 hours/month | $7+/month | ✅ Yes (15 min) |
| **Fly.io** | 3 VMs, 160GB | $0.02/hour VM | ❌ No |
| **Linode** | None | $5/month VPS | ❌ No |

**Recommendation**: Start with **Railway** for easiest setup, move to Linode if you need 24/7 uptime.

---

## 🔧 Troubleshooting

### "Application failed to start"
- Check environment variables are set
- Verify `MODEL_PATH` points to correct embeddings file
- Check logs for Python errors

### "502 Bad Gateway"
- App might be starting (wait 30 seconds)
- Check port configuration (should be `$PORT` not hardcoded)

### "Out of memory"
- Embeddings file is large (~330MB)
- Upgrade to paid tier for more RAM
- Or use smaller model: `glove-50` instead of `glove-100`

### "CORS errors in browser"
- Update API_URL in `index.html` to match your deployed URL
- Or add CORS middleware (see `word_bocce_mvp_fastapi.py`)

---

## 🎯 Next Steps After Deployment

1. **Update API_URL**: In `index.html`, change:
   ```javascript
   const API_URL = 'https://your-deployed-url.com';
   ```

2. **Test**: Open `/presentation.html` to see demo slides

3. **Share**: Post on:
   - Product Hunt
   - Hacker News (Show HN)
   - Reddit r/MachineLearning, r/WebGames
   - Twitter with #NLP #WordEmbeddings hashtags

4. **Monitor**: Check logs for errors or usage patterns

5. **Iterate**: Get feedback and improve!

---

## 📝 Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./embeddings/glove-100.bin` | Path to word embeddings |
| `DECK_SIZE` | `10000` | Vocabulary size |
| `WILDCARD_RATIO` | `0.2` | Joker card frequency |
| `ROUND_TIMEOUT_SECS` | `120` | Time limit per round |
| `PORT` | `8000` | Server port (auto-set by platform) |

---

## 🆘 Need Help?

- **Railway Docs**: https://docs.railway.app
- **Render Docs**: https://render.com/docs
- **Fly.io Docs**: https://fly.io/docs
- **Issues**: Open GitHub issue in your repo

Good luck with your deployment! 🎉

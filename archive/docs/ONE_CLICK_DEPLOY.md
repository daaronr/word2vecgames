# 🚀 One-Click Deploy Word Bocce

Deploy Word Bocce to the cloud in minutes - **completely free!**

## 🎯 Quick Summary

After deploying, you'll get a public URL like:
- `https://wordbocce-xyz.up.railway.app`
- `https://wordbocce.onrender.com`
- `https://wordbocce.fly.dev`

**People can just click the link and play - no setup required!**

---

## Option 1: Railway (Easiest!) ⭐

**Deploy Time: 5 minutes**

### Steps:

1. **Sign up**: https://railway.app (free with GitHub)

2. **Click "New Project"** → **"Deploy from GitHub repo"**

3. **Select** your `word2vecgames` repository

4. **Railway auto-detects** the Dockerfile and deploys!

5. **Get your URL**: Railway generates one like `wordbocce-production.up.railway.app`

6. **Done!** Share your URL with friends

### What You Get:
- ✅ Free 500 hours/month (plenty for a game!)
- ✅ Auto-deploy on git push
- ✅ HTTPS included
- ✅ Custom domain support

### Pro Tips:
- Set a custom domain: Settings → Domains → Add Custom Domain
- Monitor usage: Dashboard shows requests, CPU, memory
- View logs: Click your service → Logs tab

---

## Option 2: Render (Most Reliable)

**Deploy Time: 7 minutes**

### Steps:

1. **Sign up**: https://render.com (free with GitHub)

2. **Click "New +"** → **"Web Service"**

3. **Connect** your GitHub repository

4. **Render auto-detects** the `render.yaml` file

5. **Click "Create Web Service"**

6. **Wait** for build (~5-10 minutes, downloads embeddings)

7. **Get your URL**: Like `wordbocce.onrender.com`

### What You Get:
- ✅ Free 750 hours/month
- ✅ Auto-deploy on git push
- ✅ Zero-downtime deploys
- ✅ Free SSL certificates

### Note:
- Free tier "spins down" after 15 min of inactivity
- First request after sleep takes ~30 seconds to wake up
- Stays awake while people are playing

---

## Option 3: Fly.io (Most Advanced)

**Deploy Time: 8 minutes**

### Steps:

```bash
# 1. Install fly CLI
curl -L https://fly.io/install.sh | sh

# 2. Sign up / login
fly auth login

# 3. Launch from your repo directory
cd /path/to/word2vecgames
fly launch

# 4. Answer prompts:
#    - App name: wordbocce (or your choice)
#    - Region: Choose closest to you
#    - Database: No
#    - Deploy now: Yes

# 5. Your app is live!
# URL: https://wordbocce.fly.dev
```

### What You Get:
- ✅ Free tier: 3 shared-cpu VMs
- ✅ Always-on (doesn't sleep!)
- ✅ Global CDN
- ✅ CLI for easy management

### Useful Commands:
```bash
fly status                 # Check if running
fly logs                   # View logs
fly deploy                 # Redeploy
fly open                   # Open in browser
```

---

## Option 4: Heroku (Classic Choice)

**Deploy Time: 10 minutes**

### Steps:

1. **Sign up**: https://heroku.com

2. **Install Heroku CLI**:
   ```bash
   brew install heroku/brew/heroku  # Mac
   # or download from heroku.com/cli
   ```

3. **Deploy**:
   ```bash
   cd /path/to/word2vecgames
   heroku login
   heroku create wordbocce
   git push heroku main
   heroku open
   ```

### What You Get:
- ✅ Free 550-1000 hours/month
- ✅ Mature platform
- ✅ Lots of add-ons
- ✅ Auto HTTPS

### Note:
- Requires credit card for verification (not charged)
- Free tier sleeps after 30min inactivity

---

## Comparison Table

| Platform | Deploy Speed | Free Tier | Sleep? | Best For |
|----------|-------------|-----------|---------|----------|
| **Railway** | ⚡ Fastest | 500 hrs | No | Quick start |
| **Render** | 🔨 Slow build | 750 hrs | Yes (15min) | Reliability |
| **Fly.io** | 🚀 Fast | 3 VMs | No | Always-on |
| **Heroku** | ⏱️ Medium | 550 hrs | Yes (30min) | Classic |

---

## After Deployment

### 1. Test Your Deployment

Visit your URL (e.g., `https://wordbocce.up.railway.app`)

You should see:
- ✅ Word Bocce game interface
- ✅ "Quick Play" button works
- ✅ Can create and join matches
- ✅ Shareable links work

### 2. Share Your Game

**Quick Play Flow:**
1. Player 1 clicks "Quick Play"
2. Gets a link: `https://yourapp.com/?match=m_1234567_5678`
3. Shares link with Friend
4. Friend clicks link, enters name, auto-joins
5. Player 1 clicks "Start Game"
6. Play!

**Example Share Message:**
```
🎯 Let's play Word Bocce!

Click this link to join my game:
https://wordbocce.up.railway.app/?match=m_1234567_5678

It's a word vector game where you navigate semantic space!
```

### 3. Share on Social Media

**Twitter/X:**
```
Just deployed Word Bocce - a multiplayer game using word embeddings!

Try it: https://wordbocce.up.railway.app

#AI #MachineLearning #WordGames #NLP
```

**LinkedIn:**
```
Excited to share my latest project: Word Bocce!

A multiplayer game that teaches players about word embeddings
and semantic space through competitive gameplay.

Try it: https://wordbocce.up.railway.app
```

---

## Customization

### Custom Domain

**Railway:**
1. Settings → Domains → Add Custom Domain
2. Add CNAME: `wordbocce.com` → `yourapp.up.railway.app`

**Render:**
1. Settings → Custom Domains → Add Domain
2. Add CNAME as instructed

**Result:** `https://wordbocce.com` instead of platform subdomain

### Environment Variables

To change game settings, set these in your platform's dashboard:

```
DECK_SIZE=15000              # More word variety
WILDCARD_RATIO=0.3           # 30% JOKER cards
ROUND_TIMEOUT_SECS=180       # 3-minute rounds
```

**Where to set:**
- Railway: Variables tab
- Render: Environment → Add Environment Variable
- Fly.io: `fly secrets set KEY=value`
- Heroku: `heroku config:set KEY=value`

---

## Monitoring & Analytics

### Railway
- Dashboard → Your Service → Metrics
- See: Requests/sec, CPU, Memory, Build times

### Render
- Your Service → Metrics tab
- See: Response times, Memory, CPU

### Google Analytics (Optional)

Add to `index.html` before `</head>`:
```html
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

---

## Troubleshooting

### Build Failed

**"Download embeddings failed":**
- Build timeout on free tier
- Solution: Use smaller `glove-100` model (default)
- Already configured in `Dockerfile`

**"Port already in use":**
- Your app isn't using `$PORT` environment variable
- Fixed in our Dockerfile: `--port ${PORT}`

### App Crashes

**Check logs:**
```bash
# Railway: Dashboard → Logs
# Render: Service → Logs tab
# Fly.io: fly logs
# Heroku: heroku logs --tail
```

**Common issues:**
1. Out of memory: Reduce `DECK_SIZE`
2. Build timeout: Already using fast `glove-100`
3. Port binding: Already fixed in Dockerfile

### Slow Response

**First request slow (30s+):**
- App is "waking up" from sleep (Render, Heroku)
- Solution: Use Railway or Fly.io (no sleep)
- Or: Keep app warm with uptime monitoring

**Always slow:**
- Embeddings not loaded
- Check logs for errors
- Verify `MODEL_PATH` is correct

---

## Scaling Up

When you outgrow free tier:

### Railway: $5/month
- 500 hrs → Unlimited
- Shared CPU → Dedicated CPU

### Render: $7/month
- No sleep
- More resources

### Fly.io: $5-10/month
- Scale VMs
- More regions

---

## Security Considerations

**Current MVP:**
- No authentication (anyone can create matches)
- No data persistence (matches lost on restart)
- No rate limiting

**For production:**
- Add user accounts (Auth0, Firebase)
- Add Redis for match persistence
- Add rate limiting (FastAPI middleware)
- Add profanity filter for joker cards

See `DEPLOY.md` for production hardening steps.

---

## Success Checklist

Before sharing publicly:

- [ ] App deployed and accessible
- [ ] Quick Play works
- [ ] Shareable links work
- [ ] Can create and join matches
- [ ] Cards display correctly
- [ ] Moves can be submitted
- [ ] Leaderboard shows results
- [ ] Next round works
- [ ] Presentation page loads (`/presentation`)

---

## Share Your Deployment!

Once deployed, share it:

1. **Add to your portfolio/resume**
2. **Post on social media** (Twitter, LinkedIn, Reddit)
3. **Submit to**:
   - Product Hunt: https://producthunt.com
   - Hacker News: https://news.ycombinator.com/submit
   - Reddit r/machinelearning, r/gamedev

4. **Add to your GitHub README**:
   ```markdown
   ## 🎮 Play Online

   Try Word Bocce: https://wordbocce.up.railway.app
   ```

---

## 🎉 You're Done!

Your game is now publicly accessible and shareable. Anyone can click your link and start playing - no installation, no setup!

**Need help?** Open an issue on GitHub: https://github.com/daaronr/word2vecgames/issues

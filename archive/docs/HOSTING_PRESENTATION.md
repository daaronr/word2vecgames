# Hosting the Word Bocce Presentation

Quick guide to publicly host `presentation.html` so you can share it with others.

## Option 1: GitHub Pages (Easiest!) ⭐

Since your code is already on GitHub, this is the fastest option:

### Setup (One-time, 2 minutes)

1. Go to your GitHub repo: https://github.com/daaronr/word2vecgames
2. Click **Settings** → **Pages** (left sidebar)
3. Under "Source", select **main** branch
4. Click **Save**
5. Wait 1-2 minutes for deployment

### Your presentation will be live at:
```
https://daaronr.github.io/word2vecgames/presentation.html
```

**That's it!** GitHub Pages automatically serves all HTML files.

### Benefits:
- ✅ Free
- ✅ Automatic HTTPS
- ✅ Auto-updates when you push changes
- ✅ Custom domain support
- ✅ No setup needed

---

## Option 2: Netlify Drop (Super Fast)

1. Go to https://app.netlify.com/drop
2. Drag and drop just the `presentation.html` file
3. Get instant URL like: `https://random-name-123.netlify.app/presentation.html`

**Benefits:**
- ✅ Instant deployment (30 seconds)
- ✅ Free tier
- ✅ Custom domain support

---

## Option 3: Vercel (Professional)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy from your repo directory
cd /Users/yosemite/githubs/word2vecgames
vercel --prod

# Answer prompts, get URL like:
# https://word2vecgames.vercel.app/presentation.html
```

**Benefits:**
- ✅ Fast CDN
- ✅ Auto-deploys from GitHub
- ✅ Analytics included

---

## Option 4: Surge.sh (Developer-Friendly)

```bash
# Install
npm install -g surge

# Deploy just the presentation
cd /Users/yosemite/githubs/word2vecgames
surge presentation.html

# Get URL like:
# https://word-bocce-presentation.surge.sh
```

---

## Option 5: Cloudflare Pages

1. Go to https://pages.cloudflare.com
2. Connect your GitHub repo
3. Set build command: (none)
4. Set output directory: `/`
5. Deploy!

URL: `https://word2vecgames.pages.dev/presentation.html`

**Benefits:**
- ✅ Cloudflare's global CDN (fastest)
- ✅ Unlimited bandwidth
- ✅ Free SSL

---

## Recommended: GitHub Pages

**Why?** Because:
1. Your code is already there
2. Zero configuration needed
3. Professional URL
4. Updates automatically when you push

### Enable it now:

```bash
# Already done - code is pushed!
# Just go to repo settings → Pages → Enable
```

---

## Custom Domain (Optional)

Once hosted, you can add a custom domain:

### For GitHub Pages:
1. Buy domain (e.g., wordbocce.com)
2. Add CNAME file to repo:
   ```
   echo "wordbocce.com" > CNAME
   git add CNAME && git commit -m "Add custom domain" && git push
   ```
3. Configure DNS:
   - Add CNAME record: `www` → `daaronr.github.io`
   - Add A records for apex domain

### Result:
- https://wordbocce.com/presentation.html

---

## Sharing the Presentation

Once hosted, share with:

### Direct Link
```
https://daaronr.github.io/word2vecgames/presentation.html
```

### Social Media
```
🎯 Check out Word Bocce - a game where you navigate semantic space
using word vector arithmetic!

Live demo: https://daaronr.github.io/word2vecgames/presentation.html

#AI #MachineLearning #WordEmbeddings #GameDev
```

### Email
```
Subject: Word Bocce - Vector Space Word Game

I built a multiplayer game using word embeddings!

See how it works:
https://daaronr.github.io/word2vecgames/presentation.html

Try it yourself:
https://github.com/daaronr/word2vecgames
```

---

## Analytics (Optional)

Add Google Analytics or Plausible to track visitors:

```html
<!-- Add to presentation.html <head> section -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
```

---

## Next Steps

1. **Enable GitHub Pages** (2 minutes)
2. **Share the URL** on social media
3. **Add to your portfolio/resume**
4. **Submit to:**
   - Product Hunt
   - Hacker News (Show HN)
   - Reddit r/gamedev, r/MachineLearning
   - LinkedIn

---

## Troubleshooting

**GitHub Pages not working?**
- Check Settings → Pages is enabled
- Verify branch is set to "main"
- Wait 2-3 minutes for first deployment
- Check https://github.com/daaronr/word2vecgames/actions for build status

**404 Error?**
- Make sure file is named exactly `presentation.html` (case-sensitive)
- Check the file is in root directory (not in a subfolder)

**Need help?**
- Check GitHub Pages docs: https://pages.github.com
- Open an issue in your repo

---

## Pro Tip: Create a Landing Page

Create `index.html` in your repo root:

```html
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url=/presentation.html">
</head>
<body>
    Redirecting to presentation...
</body>
</html>
```

Now visitors to `https://daaronr.github.io/word2vecgames/` automatically see the presentation!

---

**🎯 Start sharing your Word Bocce presentation today!**

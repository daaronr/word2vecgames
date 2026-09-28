# 🚀 Deploy Word Bocce to Your Linode NOW

**Your Linode IP:** `45.79.160.157`

---

## Quick Deploy (10 minutes)

### Step 1: SSH into your Linode

```bash
ssh root@45.79.160.157
```

### Step 2: Run the automated deployment script

```bash
curl -fsSL https://raw.githubusercontent.com/daaronr/word2vecgames/main/deploy_to_linode.sh | bash
```

**OR** manually:

```bash
# Clone repo
cd /opt
git clone https://github.com/daaronr/word2vecgames.git
cd word2vecgames

# Run deployment script
chmod +x deploy_to_linode.sh
./deploy_to_linode.sh
```

### Step 3: Access your game!

Once deployment completes, visit:

**🎮 http://45.79.160.157**

---

## What the script does:

1. ✅ Updates system packages
2. ✅ Installs Python, nginx, dependencies
3. ✅ Clones your repository
4. ✅ Downloads word embeddings (~330MB)
5. ✅ Creates systemd service (auto-restart)
6. ✅ Configures nginx reverse proxy
7. ✅ Sets up firewall (UFW)
8. ✅ Starts the application

---

## Useful Commands

**Check if running:**
```bash
systemctl status wordbocce
```

**View logs:**
```bash
journalctl -u wordbocce -f
```

**Restart service:**
```bash
systemctl restart wordbocce
```

**Update to latest code:**
```bash
cd /opt/word2vecgames
git pull
systemctl restart wordbocce
```

**Stop service:**
```bash
systemctl stop wordbocce
```

---

## Adding a Domain (Optional)

If you have a domain like `wordbocce.com`:

1. **Point DNS to your Linode:**
   - A record: `@` → `45.79.160.157`
   - A record: `www` → `45.79.160.157`

2. **Update nginx config:**
   ```bash
   nano /etc/nginx/sites-available/wordbocce
   ```

   Change:
   ```nginx
   server_name 45.79.160.157;
   ```

   To:
   ```nginx
   server_name wordbocce.com www.wordbocce.com;
   ```

3. **Add SSL (HTTPS):**
   ```bash
   certbot --nginx -d wordbocce.com -d www.wordbocce.com
   ```

4. **Restart nginx:**
   ```bash
   systemctl restart nginx
   ```

---

## Troubleshooting

**Can't connect to server:**
```bash
# Check if service is running
systemctl status wordbocce

# Check logs for errors
journalctl -u wordbocce -n 50
```

**Port 80 blocked:**
```bash
# Check firewall
ufw status

# Allow HTTP
ufw allow 80/tcp
```

**Out of memory:**
```bash
# Check memory usage
free -h

# May need to upgrade Linode to 2GB plan
```

**Need to restart:**
```bash
systemctl restart wordbocce
systemctl restart nginx
```

---

## Next Steps After Deployment

1. ✅ Test the game at http://45.79.160.157
2. ✅ Share the presentation: http://45.79.160.157/presentation.html
3. ✅ Add a custom domain (optional)
4. ✅ Add SSL/HTTPS (optional but recommended)
5. ✅ Share with friends and get feedback!

---

## Cost

If using a dedicated Linode:
- **Nanode 1GB:** $5/month (adequate for testing)
- **Linode 2GB:** $10/month (better for production)

If adding to existing Linode: **$0** (already paying for it!)

---

## Security Notes

- The deployment script sets up a basic firewall (UFW)
- Only ports 22 (SSH), 80 (HTTP), and 443 (HTTPS) are open
- Consider adding fail2ban for SSH protection
- Add HTTPS with Let's Encrypt (free SSL)

**Add fail2ban (optional):**
```bash
apt install fail2ban -y
systemctl enable fail2ban
systemctl start fail2ban
```

---

## Support

- **Full guide:** See `LINODE_DEPLOY.md` for detailed instructions
- **Logs location:** `/var/log/nginx/` and `journalctl -u wordbocce`
- **App location:** `/opt/word2vecgames/`

Good luck with your deployment! 🎉

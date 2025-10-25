# 🚀 Deploy Word Bocce to Linode VPS

Deploy Word Bocce to your own Linode VPS server for full control and customization.

**Deploy Time: 15-20 minutes**

---

## Prerequisites

- Linode account (https://linode.com)
- Domain name (optional, but recommended)
- SSH client

---

## Option 1: Docker Deployment (Recommended) 🐳

### Step 1: Create Linode Instance

1. **Log into Linode Cloud Manager**: https://cloud.linode.com

2. **Create a Linode**:
   - Click "Create" → "Linode"
   - **Distribution**: Ubuntu 22.04 LTS
   - **Region**: Choose closest to your target audience
   - **Linode Plan**:
     - Nanode 1GB ($5/month) - Good for testing
     - Linode 2GB ($10/month) - Recommended for production
   - **Linode Label**: `wordbocce`
   - **Root Password**: Create a strong password
   - Click "Create Linode"

3. **Wait for provisioning** (~30 seconds)

4. **Note your IP address** (shown in dashboard)

### Step 2: Initial Server Setup

SSH into your server:

```bash
ssh root@YOUR_LINODE_IP
```

Update system packages:

```bash
apt update && apt upgrade -y
```

Install Docker and Docker Compose:

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install docker-compose -y

# Start Docker service
systemctl start docker
systemctl enable docker

# Verify installation
docker --version
docker-compose --version
```

### Step 3: Clone and Deploy Application

```bash
# Install git
apt install git -y

# Clone repository
cd /opt
git clone https://github.com/daaronr/word2vecgames.git
cd word2vecgames

# Create environment file
cat > .env << EOF
MODEL_PATH=./embeddings/glove-100.bin
DECK_SIZE=10000
WILDCARD_RATIO=0.2
ROUND_TIMEOUT_SECS=120
PORT=8000
EOF

# Build and run with Docker Compose
docker-compose up -d --build
```

### Step 4: Configure Firewall

```bash
# Install ufw firewall
apt install ufw -y

# Allow SSH (important - don't lock yourself out!)
ufw allow 22/tcp

# Allow HTTP and HTTPS
ufw allow 80/tcp
ufw allow 443/tcp

# Enable firewall
ufw --force enable

# Check status
ufw status
```

### Step 5: Setup Nginx Reverse Proxy

```bash
# Install Nginx
apt install nginx -y

# Create Nginx configuration
cat > /etc/nginx/sites-available/wordbocce << 'EOF'
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# Enable site
ln -s /etc/nginx/sites-available/wordbocce /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default

# Test configuration
nginx -t

# Restart Nginx
systemctl restart nginx
systemctl enable nginx
```

**Replace `YOUR_DOMAIN_OR_IP`** in the config file:

```bash
# If you have a domain:
sed -i 's/YOUR_DOMAIN_OR_IP/wordbocce.yourdomain.com/g' /etc/nginx/sites-available/wordbocce

# If using IP only:
sed -i 's/YOUR_DOMAIN_OR_IP/YOUR_LINODE_IP/g' /etc/nginx/sites-available/wordbocce

# Reload Nginx
systemctl reload nginx
```

### Step 6: SSL Certificate (Optional but Recommended)

If you have a domain name:

```bash
# Install Certbot
apt install certbot python3-certbot-nginx -y

# Get SSL certificate
certbot --nginx -d wordbocce.yourdomain.com

# Follow prompts:
# - Enter email address
# - Agree to terms
# - Choose to redirect HTTP to HTTPS (recommended)

# Certbot will auto-renew. Test renewal:
certbot renew --dry-run
```

### Step 7: Test Your Deployment

Visit your server:
- With domain: `https://wordbocce.yourdomain.com`
- With IP: `http://YOUR_LINODE_IP`

You should see the Word Bocce game interface!

---

## Option 2: Direct Python Deployment (No Docker)

### Steps 1-4: Same as Docker option (create server, SSH, update, firewall)

### Step 5: Install Python and Dependencies

```bash
# Install Python and pip
apt install python3 python3-pip python3-venv -y

# Clone repository
cd /opt
git clone https://github.com/daaronr/word2vecgames.git
cd word2vecgames

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download embeddings
python setup_embeddings.py --model glove-100 --output ./embeddings

# Set environment variables
export MODEL_PATH=./embeddings/glove-100.bin
export DECK_SIZE=10000
export WILDCARD_RATIO=0.2
export ROUND_TIMEOUT_SECS=120
```

### Step 6: Create Systemd Service

```bash
cat > /etc/systemd/system/wordbocce.service << 'EOF'
[Unit]
Description=Word Bocce Game Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/word2vecgames
Environment="MODEL_PATH=./embeddings/glove-100.bin"
Environment="DECK_SIZE=10000"
Environment="WILDCARD_RATIO=0.2"
Environment="ROUND_TIMEOUT_SECS=120"
ExecStart=/opt/word2vecgames/venv/bin/uvicorn word_bocce_mvp_fastapi:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable wordbocce
systemctl start wordbocce

# Check status
systemctl status wordbocce
```

### Step 7: Setup Nginx and SSL

Same as Docker Option Step 5 and 6.

---

## Domain Configuration

If you have a domain name, point it to your Linode:

### At Your Domain Registrar:

Add an **A Record**:
```
Type: A
Name: wordbocce (or @ for root domain)
Value: YOUR_LINODE_IP
TTL: 300 (or default)
```

Wait 5-15 minutes for DNS propagation, then visit:
- `http://wordbocce.yourdomain.com`

Then run Certbot (Step 6) to add HTTPS.

---

## Management Commands

### Docker Deployment

```bash
# View logs
docker-compose logs -f

# Restart application
docker-compose restart

# Stop application
docker-compose down

# Update application
cd /opt/word2vecgames
git pull origin main
docker-compose up -d --build

# View running containers
docker ps
```

### Direct Python Deployment

```bash
# View logs
journalctl -u wordbocce -f

# Restart application
systemctl restart wordbocce

# Stop application
systemctl stop wordbocce

# Update application
cd /opt/word2vecgames
git pull origin main
systemctl restart wordbocce
```

---

## Monitoring

### Check Application Health

```bash
# Check if app is running
curl http://localhost:8000/api

# Check Nginx
systemctl status nginx

# Check Docker (if using)
docker ps

# Check Python service (if not using Docker)
systemctl status wordbocce
```

### View Logs

```bash
# Docker logs
docker-compose logs -f wordbocce

# Python service logs
journalctl -u wordbocce -f --lines=100

# Nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

### Resource Usage

```bash
# Check memory and CPU
htop

# Check disk space
df -h

# Check Docker stats (if using Docker)
docker stats
```

---

## Backups

### Automated Backups with Linode

1. **In Linode Cloud Manager**:
   - Your Linode → Backups tab
   - Enable Backups ($2/month for 1GB Linode)
   - Automatic daily, weekly snapshots

### Manual Backups

```bash
# Backup match data (if you add Redis later)
# For now, matches are in-memory only

# Backup configuration
tar -czf wordbocce-backup-$(date +%Y%m%d).tar.gz \
    /opt/word2vecgames \
    /etc/nginx/sites-available/wordbocce \
    /etc/systemd/system/wordbocce.service

# Download to local machine
scp root@YOUR_LINODE_IP:/root/wordbocce-backup-*.tar.gz .
```

---

## Scaling and Performance

### Increase Resources

If your server is slow:

1. **Resize Linode**:
   - Linode Cloud Manager → Settings → Resize
   - Choose larger plan (2GB → 4GB)
   - Server will reboot

2. **Optimize Docker**:
   ```bash
   # Limit memory usage
   docker-compose down
   # Edit docker-compose.yml, add under 'wordbocce' service:
   #   mem_limit: 1g
   #   cpus: 1
   docker-compose up -d
   ```

### Enable Caching

Add caching to Nginx for static assets:

```bash
cat >> /etc/nginx/sites-available/wordbocce << 'EOF'

    # Cache static files
    location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
EOF

systemctl reload nginx
```

---

## Security Hardening

### Create Non-Root User

```bash
# Create user
adduser wordbocce
usermod -aG sudo,docker wordbocce

# Switch to new user
su - wordbocce

# Update service to run as this user
sudo sed -i 's/User=root/User=wordbocce/g' /etc/systemd/system/wordbocce.service
sudo systemctl daemon-reload
sudo systemctl restart wordbocce
```

### Disable Root SSH

```bash
# Edit SSH config
sudo nano /etc/ssh/sshd_config

# Change this line:
# PermitRootLogin yes
# To:
# PermitRootLogin no

# Restart SSH
sudo systemctl restart sshd
```

### Fail2Ban (Prevent Brute Force)

```bash
sudo apt install fail2ban -y
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### Automatic Updates

```bash
sudo apt install unattended-upgrades -y
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

---

## Troubleshooting

### Cannot Access Game

**Check if application is running**:
```bash
# Docker
docker ps

# Direct Python
systemctl status wordbocce
```

**Check if port 8000 is listening**:
```bash
netstat -tlnp | grep 8000
```

**Check Nginx**:
```bash
nginx -t
systemctl status nginx
```

### 502 Bad Gateway

**Application is not running**:
```bash
# Docker
docker-compose up -d

# Python
systemctl start wordbocce
```

**Check application logs**:
```bash
# Docker
docker-compose logs

# Python
journalctl -u wordbocce -n 50
```

### Out of Memory

**Check memory usage**:
```bash
free -h
```

**If low, resize Linode or reduce DECK_SIZE**:
```bash
# Edit .env file
nano /opt/word2vecgames/.env
# Change: DECK_SIZE=5000

# Restart
docker-compose restart
# or
systemctl restart wordbocce
```

### SSL Certificate Errors

**Renew certificate**:
```bash
sudo certbot renew --force-renewal
sudo systemctl reload nginx
```

---

## Cost Estimates

### Linode Pricing

| Plan | RAM | Storage | Transfer | Price/mo |
|------|-----|---------|----------|----------|
| Nanode | 1 GB | 25 GB | 1 TB | $5 |
| Linode 2GB | 2 GB | 50 GB | 2 TB | $10 |
| Linode 4GB | 4 GB | 80 GB | 4 TB | $20 |

**Recommended**: Start with Nanode ($5/mo), upgrade if needed.

### Additional Costs

- Domain name: $10-15/year (optional)
- Backups: $2/month (optional)
- **SSL Certificate**: FREE with Let's Encrypt

**Total**: $5-10/month

---

## Comparison: Linode vs Cloud Platforms

| Feature | Linode VPS | Railway/Render |
|---------|-----------|----------------|
| **Cost** | $5-10/mo | Free tier available |
| **Control** | Full server access | Limited |
| **Setup Time** | 15-20 min | 5-7 min |
| **Scalability** | Manual | Automatic |
| **Maintenance** | You manage | Managed |
| **Custom Domain** | Full DNS control | CNAME only |
| **Best For** | Learning, full control | Quick deployment |

---

## Next Steps

After deploying to Linode:

1. **Test the deployment**: Create matches, play games
2. **Monitor resource usage**: Check memory, CPU with `htop`
3. **Set up monitoring**: Consider UptimeRobot or Pingdom
4. **Configure backups**: Enable Linode Backups
5. **Add custom domain**: Point your domain to the server
6. **Enable SSL**: Run Certbot for HTTPS
7. **Share your game**: Send friends your URL!

---

## Advanced: Multiple Linode Instances

For high availability, deploy to multiple Linode regions:

1. Deploy to 3 different regions (e.g., US East, US West, EU)
2. Set up DNS load balancing with your domain provider
3. Configure health checks
4. Traffic routes to closest/healthiest server

**Cost**: $15/month (3 x Nanode)

---

## Support

**Linode Documentation**: https://www.linode.com/docs/

**Community Support**: https://www.linode.com/community/questions/

**GitHub Issues**: https://github.com/daaronr/word2vecgames/issues

---

## Summary

You now have Word Bocce running on your own Linode VPS with:

- ✅ Full control over the server
- ✅ Custom domain with SSL (optional)
- ✅ Automatic restarts and monitoring
- ✅ Nginx reverse proxy for better performance
- ✅ Firewall configured for security
- ✅ Easy updates with git pull

**Your game is live at**: `http://YOUR_LINODE_IP` or `https://wordbocce.yourdomain.com`

Share the link and start playing!

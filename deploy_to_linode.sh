#!/bin/bash
# Quick deployment script for Word Bocce on Linode
# Run this ON YOUR LINODE SERVER (after SSHing in)

set -e  # Exit on any error

echo "🚀 Deploying Word Bocce to Linode..."

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install dependencies
echo "🐍 Installing Python and dependencies..."
apt install -y python3 python3-pip git nginx certbot python3-certbot-nginx

# Clone repository (if not already cloned)
if [ ! -d "/opt/word2vecgames" ]; then
    echo "📥 Cloning repository..."
    cd /opt
    git clone https://github.com/daaronr/word2vecgames.git
    cd word2vecgames
else
    echo "📥 Updating repository..."
    cd /opt/word2vecgames
    git pull
fi

# Install Python requirements
echo "📚 Installing Python packages..."
pip3 install -r requirements.txt

# Download embeddings
echo "🧠 Downloading word embeddings (this may take a few minutes)..."
python3 setup_embeddings.py --model glove-100 --output ./embeddings

# Create systemd service
echo "⚙️  Creating systemd service..."
cat > /etc/systemd/system/wordbocce.service << 'EOF'
[Unit]
Description=Word Bocce Game Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/word2vecgames
Environment="MODEL_PATH=./embeddings/glove-100.bin"
Environment="PORT=8000"
ExecStart=/usr/bin/python3 -m uvicorn word_bocce_mvp_fastapi:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Configure nginx
echo "🌐 Configuring nginx..."
cat > /etc/nginx/sites-available/wordbocce << 'EOF'
server {
    listen 80;
    server_name 45.79.160.157;

    location / {
        proxy_pass http://127.0.0.1:8000;
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

# Enable nginx site
ln -sf /etc/nginx/sites-available/wordbocce /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test nginx config
nginx -t

# Configure firewall
echo "🔥 Configuring firewall..."
ufw allow 22/tcp   # SSH
ufw allow 80/tcp   # HTTP
ufw allow 443/tcp  # HTTPS
ufw --force enable

# Start services
echo "🚀 Starting services..."
systemctl daemon-reload
systemctl enable wordbocce
systemctl start wordbocce
systemctl restart nginx

# Check status
echo ""
echo "✅ Deployment complete!"
echo ""
echo "🎮 Word Bocce is now running at:"
echo "   http://45.79.160.157"
echo ""
echo "📊 Check service status:"
echo "   systemctl status wordbocce"
echo ""
echo "📝 View logs:"
echo "   journalctl -u wordbocce -f"
echo ""
echo "🔄 Restart service:"
echo "   systemctl restart wordbocce"
echo ""

# Show status
systemctl status wordbocce --no-pager

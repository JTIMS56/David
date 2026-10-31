# Popper FX Platform — Deployment Guide

## Recommended hosting: DigitalOcean (~$39/month)

| Service | Cost | Notes |
|---------|------|-------|
| Droplet (2 GB RAM, 1 vCPU) | $12/month | Runs Docker + nginx + app |
| Managed PostgreSQL (1 GB) | $15/month | Automatic backups, failover |
| **Total** | **~$27/month** | Can start smaller, scale up |

DigitalOcean is recommended because:
- One-click Docker Droplet setup
- Managed PostgreSQL with automatic daily backups
- Built-in firewall (no iptables config)
- Simple DNS management
- $200 free credit for new accounts

---

## Step 1 — Create your server (15 minutes)

1. Sign up at [digitalocean.com](https://digitalocean.com)
2. **Create → Droplet**
   - Choose region: closest to you
   - Image: **Docker** (in the Marketplace tab)
   - Size: **Basic / Regular — 2 GB / 1 vCPU ($12/month)**
   - Authentication: SSH key (add your public key `~/.ssh/id_rsa.pub`)
   - Hostname: `david-fx`
3. Note the server IP address (you'll need it throughout)
4. SSH in: `ssh root@<YOUR_IP>`

---

## Step 2 — Point a domain at your server (optional but required for HTTPS)

1. Buy a domain (Namecheap, Cloudflare, etc.) — ~$12/year
2. Add an **A record**: `@` → `<YOUR_IP>`
3. Wait 5 minutes for DNS propagation
4. Verify: `ping yourdomain.com` should resolve to your IP

> **Skip this for testing**: you can use a self-signed certificate and access via IP.

---

## Step 3 — Deploy the application (10 minutes)

```bash
# On your server (SSH'd in as root):

# Clone the repo
git clone https://github.com/jtims56/david.git /opt/david
cd /opt/david

# Copy and configure environment
cp .env.production .env
nano .env   # Fill in all values (see below)

# Generate secure keys
python3 -c "import secrets; print('API_KEY=' + secrets.token_urlsafe(32))"
python3 -c "import secrets; print('WS_TOKEN=' + secrets.token_urlsafe(32))"
python3 -c "import secrets; print('POSTGRES_PASSWORD=' + secrets.token_urlsafe(24))"
# Paste each output into .env

# Set your domain
sed -i 's/yourdomain.com/YOUR_ACTUAL_DOMAIN/g' .env nginx/nginx.conf
```

---

## Step 4 — Get SSL certificate (Let's Encrypt)

```bash
# Create SSL directory
mkdir -p nginx/ssl

# Run certbot (replace yourdomain.com with your actual domain)
docker run --rm \
  -v $(pwd)/nginx/ssl:/etc/letsencrypt \
  -v $(pwd)/nginx/certbot_www:/var/www/certbot \
  -p 80:80 \
  certbot/certbot certonly \
  --standalone \
  --email your@email.com \
  --agree-tos \
  --no-eff-email \
  -d yourdomain.com

# Certificates are now in nginx/ssl/live/yourdomain.com/
# Update nginx.conf to use the correct path:
sed -i 's|/etc/nginx/ssl/fullchain.pem|/etc/nginx/ssl/live/yourdomain.com/fullchain.pem|g' nginx/nginx.conf
sed -i 's|/etc/nginx/ssl/privkey.pem|/etc/nginx/ssl/live/yourdomain.com/privkey.pem|g' nginx/nginx.conf
```

### Testing with self-signed cert (no domain needed):

```bash
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/privkey.pem \
  -out nginx/ssl/fullchain.pem \
  -subj "/CN=localhost"
```

---

## Step 5 — Launch

```bash
./deploy.sh --build
```

This will:
1. Build the Docker image (compiles C++ Dirac engine — takes ~2 min first time)
2. Start PostgreSQL
3. Start the FastAPI application
4. Start nginx reverse proxy
5. Run health checks

**First deploy takes ~3-5 minutes. Subsequent deploys take ~30 seconds.**

---

## Step 6 — Verify it works

```bash
# From your local machine:
curl https://yourdomain.com/health
# Expected: {"status":"ok"}

curl https://yourdomain.com/api/status \
  -H "X-API-Key: YOUR_API_KEY"
# Expected: JSON with portfolio + agent status

# Open dashboard in browser:
# https://yourdomain.com
```

---

## Day-to-day operations

### View logs
```bash
docker compose logs -f app        # application logs
docker compose logs -f nginx      # nginx access/error logs
docker compose logs -f db         # database logs
```

### Update to latest version
```bash
./deploy.sh    # pulls git, restarts services
```

### Emergency kill switch
```bash
# Via API:
curl -X POST https://yourdomain.com/api/risk/kill-switch \
  -H "X-API-Key: YOUR_API_KEY"

# Via Docker exec (if API is down):
docker compose exec app \
  curl -X POST http://localhost:8000/api/risk/kill-switch \
  -H "X-API-Key: YOUR_API_KEY"
```

### Backup database
```bash
docker compose exec db \
  pg_dump -U david david_fx | gzip > backup_$(date +%Y%m%d).sql.gz
```

### Run tests
```bash
docker compose exec app pytest tests/ -v
```

---

## Renew SSL certificate (every 90 days)

```bash
docker compose run certbot renew
docker compose restart nginx
```

**Automate this** — add to cron on the server:
```bash
# Edit crontab: crontab -e
0 0 1 * * cd /opt/david && docker compose run --rm certbot renew && docker compose restart nginx
```

---

## Security checklist before go-live

- [ ] `API_KEY` set to a random 32+ character string
- [ ] `WS_TOKEN` set to a random 32+ character string  
- [ ] `POSTGRES_PASSWORD` set to a random string
- [ ] `ALLOWED_ORIGINS` set to your exact domain (not `*`)
- [ ] `TRADING_MODE=paper` (keep until fully validated)
- [ ] HTTPS working (check browser padlock)
- [ ] DigitalOcean firewall: only ports 22, 80, 443 open
- [ ] SSH key-only access (no password SSH)
- [ ] `git secret` or similar for `.env` if stored in git

---

## Cost breakdown

| Item | Monthly |
|------|---------|
| DigitalOcean Droplet (2GB) | $12 |
| Managed PostgreSQL (Dev) | $15 |
| Domain (annual, amortized) | ~$1 |
| **Total** | **~$28/month** |

For production with real money, upgrade to:
- Droplet: 4GB ($24) for reliability
- Managed PostgreSQL: Standard (replicated) — $50/month
- **Total: ~$75/month**

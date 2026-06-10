#!/usr/bin/env bash
# deploy.sh — Zero-downtime deploy for David FX Trading Platform
#
# Usage:
#   ./deploy.sh           # deploy latest from current branch
#   ./deploy.sh --build   # force rebuild Docker image
#
# Prerequisites: Docker, Docker Compose v2, .env file in project root
set -euo pipefail

COMPOSE="docker compose"
APP_SERVICE="app"
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${BOLD}[deploy]${NC} $*"; }
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; }
die()  { echo -e "${RED}✗${NC} $*" >&2; exit 1; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────
log "Running pre-flight checks..."

[ -f .env ] || die ".env file not found. Copy .env.production to .env and fill in values."

# Check required env vars
required=(API_KEY WS_TOKEN POSTGRES_PASSWORD ANTHROPIC_API_KEY)
missing=()
for var in "${required[@]}"; do
    val=$(grep -E "^${var}=" .env 2>/dev/null | cut -d= -f2-)
    if [ -z "$val" ] || [[ "$val" == *"CHANGE_ME"* ]]; then
        missing+=("$var")
    fi
done
if [ ${#missing[@]} -gt 0 ]; then
    die "Missing or placeholder values in .env: ${missing[*]}"
fi
ok "Environment variables look good"

# Check SSL certs exist (skip in dev)
ENVIRONMENT=$(grep -E "^ENVIRONMENT=" .env 2>/dev/null | cut -d= -f2- || echo "development")
if [ "$ENVIRONMENT" = "production" ]; then
    [ -f nginx/ssl/fullchain.pem ] || die "SSL cert not found at nginx/ssl/fullchain.pem. Run certbot first."
    [ -f nginx/ssl/privkey.pem   ] || die "SSL key not found at nginx/ssl/privkey.pem. Run certbot first."
    ok "SSL certificates present"
fi

# ── Pull latest code ──────────────────────────────────────────────────────────
log "Pulling latest code..."
git pull --ff-only origin "$(git branch --show-current)"
ok "Code up to date ($(git rev-parse --short HEAD))"

# ── Build image ───────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--build" ]] || ! $COMPOSE images "$APP_SERVICE" | grep -q "$APP_SERVICE"; then
    log "Building Docker image (this compiles C++ engine, ~2 min)..."
    $COMPOSE build --no-cache "$APP_SERVICE"
    ok "Image built"
else
    log "Using existing image (pass --build to force rebuild)"
fi

# ── Start / restart services ──────────────────────────────────────────────────
log "Starting database..."
$COMPOSE up -d db
log "Waiting for database to be healthy..."
until $COMPOSE exec -T db pg_isready -q 2>/dev/null; do
    printf '.'
    sleep 2
done
echo
ok "Database ready"

log "Deploying application..."
$COMPOSE up -d --remove-orphans app nginx
ok "Services started"

# ── Health check ──────────────────────────────────────────────────────────────
log "Waiting for app to pass health check..."
for i in $(seq 1 30); do
    if $COMPOSE exec -T app curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        ok "App is healthy"
        break
    fi
    if [ "$i" -eq 30 ]; then
        warn "App health check timed out. Checking logs:"
        $COMPOSE logs --tail=50 "$APP_SERVICE"
        die "Deployment failed — app not healthy after 60 seconds"
    fi
    sleep 2
done

# ── Final status ──────────────────────────────────────────────────────────────
echo
echo -e "${BOLD}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  David FX Platform deployed successfully${NC}"
echo -e "${BOLD}═══════════════════════════════════════════${NC}"
$COMPOSE ps
echo
log "View logs:     docker compose logs -f app"
log "Stop:          docker compose down"
log "Emergency:     docker compose exec app curl -X POST http://localhost:8000/api/risk/kill-switch -H 'X-API-Key: \$API_KEY'"

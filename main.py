"""
David — Agentic AI FX Trading Platform
────────────────────────────────────────────────────────────────────────────
Entry point. Starts FastAPI with WebSocket support, initialises DB,
launches market data feed, order monitor, and the AI trading agent.

Run with:  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from database import init_db
from services.market_data import market_data
from services.order_service import order_service
from services.portfolio_service import portfolio_service
from services.risk_manager import RiskManager, risk_manager as _rm_sentinel
import services.risk_manager as _risk_mod
from agents.trading_agent import trading_agent
from api.routes import router
from services.risk_gate import risk_gate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("david")

# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)
        logger.info(f"WS client connected (total={len(self._connections)})")

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def broadcast(self, data: dict) -> None:
        dead: Set[WebSocket] = set()
        message = json.dumps(data, default=str)
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        self._connections -= dead


manager = ConnectionManager()


# ── Price broadcast task ──────────────────────────────────────────────────────

async def broadcast_prices(interval: float = 2.0) -> None:
    while True:
        try:
            all_prices = market_data.get_all_prices()
            rates = {}
            for pair, bar in all_prices.items():
                from services.market_data import PAIR_CONFIG
                pip = PAIR_CONFIG.get(pair, {}).get("pip", 0.0001)
                rates[pair] = {
                    "bid": bar.bid,
                    "ask": bar.ask,
                    "mid": bar.mid,
                    "spread_pips": round((bar.ask - bar.bid) / pip, 1),
                }
            state = await portfolio_service.get_state()
            await manager.broadcast({
                "event": "prices",
                "rates": rates,
                "portfolio": {
                    "balance": state["balance"],
                    "equity": state["equity"],
                    "unrealised_pnl": state["unrealised_pnl"],
                    "daily_pnl": state["daily_pnl"],
                    "open_positions": state["open_positions"],
                },
                "agent": trading_agent.get_status(),
                "timestamp": datetime.utcnow().isoformat(),
            })
        except Exception:
            logger.exception("Error in price broadcast loop")
        await asyncio.sleep(interval)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising David FX Trading Platform...")

    # Database
    await init_db()
    logger.info("Database initialised")

    # Market data
    await market_data.start()
    logger.info(f"Market data started (mode={settings.market_data_mode})")

    # Wire up portfolio → risk manager
    _risk_mod.risk_manager = RiskManager(portfolio_service)

    # Apply config-driven risk limits to the hard gate
    risk_gate._max_drawdown_pct = settings.max_drawdown_pct

    # Wire up order service broadcast
    order_service.set_broadcast(manager.broadcast)

    # Start order monitor (SL/TP checks)
    await order_service.start_monitor()

    # Seed portfolio snapshot
    await portfolio_service.snapshot()

    # Start price broadcast
    broadcast_task = asyncio.create_task(broadcast_prices())

    # Auto-start agent only when explicitly configured
    if settings.anthropic_api_key:
        await trading_agent.start()
        logger.info("Trading agent started (mode=ai)")
    elif getattr(settings, "demo_mode", False):
        await trading_agent.start()
        logger.info("Trading agent started (mode=demo)")

    logger.info(f"Platform ready — balance: ${settings.initial_balance:,.2f}")
    yield

    # Shutdown
    broadcast_task.cancel()
    await trading_agent.stop()
    await order_service.stop_monitor()
    await market_data.stop()
    logger.info("Platform shut down cleanly")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="David — Agentic AI FX Trading Platform",
    version="1.0.0",
    description="Autonomous Claude-powered FX trading agent with real-time dashboard",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)

app.include_router(router)

# Static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    return FileResponse(str(static_dir / "index.html"))


# ── Health checks (public — no auth required) ─────────────────────────────────

@app.get("/health", tags=["ops"], include_in_schema=False)
async def health():
    """Liveness probe — returns 200 as long as the process is alive."""
    return {"status": "ok"}


@app.get("/readiness", tags=["ops"], include_in_schema=False)
async def readiness():
    """
    Readiness probe — checks DB connectivity and market data feed.
    Returns 503 if not ready so the load balancer withholds traffic.
    """
    issues = []
    # DB check
    try:
        from database import AsyncSessionLocal
        from sqlalchemy import text
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        issues.append(f"db: {exc}")

    # Market data check
    try:
        prices = market_data.get_all_prices()
        if not prices:
            issues.append("market_data: no prices available")
    except Exception as exc:
        issues.append(f"market_data: {exc}")

    if issues:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "issues": issues},
        )
    return {"status": "ready", "pairs": len(market_data.get_all_prices())}


# ── WebSocket ─────────────────────────────────────────────────────────────────

_WS_CONTROL_CMDS = {"run_agent", "start_agent", "stop_agent"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    _authenticated = not bool(settings.ws_token)   # no token configured → open
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except Exception:
                continue

            cmd = msg.get("cmd")

            # Auth handshake: client sends {"cmd": "auth", "token": "<secret>"}
            if cmd == "auth":
                if not settings.ws_token or msg.get("token") == settings.ws_token:
                    _authenticated = True
                    await websocket.send_text(json.dumps({"event": "auth", "ok": True}))
                else:
                    await websocket.send_text(json.dumps({"event": "auth", "ok": False, "reason": "bad token"}))
                continue

            # Control commands require authentication when a token is configured
            if cmd in _WS_CONTROL_CMDS and not _authenticated:
                await websocket.send_text(json.dumps({"event": "error", "reason": "not authenticated"}))
                continue

            if cmd == "run_agent":
                asyncio.create_task(trading_agent.run_once())
            elif cmd == "start_agent":
                asyncio.create_task(trading_agent.start())
            elif cmd == "stop_agent":
                asyncio.create_task(trading_agent.stop())

    except WebSocketDisconnect:
        manager.disconnect(websocket)

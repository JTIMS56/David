"""
Popper — Agentic AI FX Trading Platform
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

from config import settings, APP_VERSION
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
logger = logging.getLogger("popper")

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


# ── Forecast evaluation task ──────────────────────────────────────────────────

async def evaluate_forecasts(interval: float = 60.0) -> None:
    """Check past-horizon DHJ forecasts every minute and record outcomes."""
    from sqlalchemy import and_, select as _select
    from database import AsyncSessionLocal as _ASL
    from models.orm import ForecastLog, EnsembleForecastLog
    from services.market_data import PAIR_CONFIG

    while True:
        await asyncio.sleep(interval)
        try:
            now = datetime.utcnow()
            # Never grade a forecast against a frozen feed. Doing so records a
            # 0-pip "actual move" and a false miss, silently poisoning the
            # accuracy record — exactly the failure that went undetected for
            # eleven days when an unsupported instrument killed the feed.
            _fh = market_data.feed_health()
            if not _fh["healthy"]:
                logger.critical(
                    "Forecast evaluation SKIPPED — price feed stale (%.0fs). "
                    "Outcomes would be recorded against frozen prices.",
                    _fh["newest_age_seconds"] or -1,
                )
                continue
            async with _ASL() as db:
                q = (
                    _select(ForecastLog)
                    .where(
                        and_(
                            ForecastLog.horizon_at <= now,
                            ForecastLog.evaluated_at.is_(None),
                        )
                    )
                    .limit(50)
                )
                result = await db.execute(q)
                pending = result.scalars().all()
                evaluated = 0
                for log in pending:
                    bar = market_data.get_price(log.pair)
                    if bar is None:
                        continue
                    pip = PAIR_CONFIG.get(log.pair, {}).get("pip", 0.0001)
                    actual_pips = round((bar.mid - log.spot_price) / pip, 1)
                    log.outcome_price     = round(bar.mid, 6)
                    log.actual_move_pips  = actual_pips
                    log.direction_correct = (
                        (actual_pips > 0 and log.expected_direction == "UP") or
                        (actual_pips < 0 and log.expected_direction == "DOWN")
                    )
                    if log.bs_expected_direction:
                        log.bs_direction_correct = (
                            (actual_pips > 0 and log.bs_expected_direction == "UP") or
                            (actual_pips < 0 and log.bs_expected_direction == "DOWN")
                        )
                    log.evaluated_at = now
                    evaluated += 1
                # ── Ensemble shadow forecasts (same horizon logic) ───────────
                eq = (
                    _select(EnsembleForecastLog)
                    .where(
                        and_(
                            EnsembleForecastLog.horizon_at <= now,
                            EnsembleForecastLog.evaluated_at.is_(None),
                        )
                    )
                    .limit(50)
                )
                eresult = await db.execute(eq)
                epending = eresult.scalars().all()
                e_evaluated = 0
                for elog in epending:
                    bar = market_data.get_price(elog.pair)
                    if bar is None:
                        continue
                    pip = PAIR_CONFIG.get(elog.pair, {}).get("pip", 0.0001)
                    e_actual = round((bar.mid - elog.spot_price) / pip, 1)
                    elog.outcome_price     = round(bar.mid, 6)
                    elog.actual_move_pips  = e_actual
                    elog.direction_correct = (
                        (e_actual > 0 and elog.direction == "UP") or
                        (e_actual < 0 and elog.direction == "DOWN")
                    )
                    elog.evaluated_at = now
                    e_evaluated += 1

                if evaluated or e_evaluated:
                    await db.commit()
                    logger.info(
                        "Forecast evaluator: marked %d DHJ, %d ensemble forecast(s)",
                        evaluated, e_evaluated,
                    )
        except Exception:
            logger.exception("Error in forecast evaluation loop")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising Popper FX Trading Platform — version %s", APP_VERSION)

    # Database
    await init_db()
    logger.info("Database initialised")

    # Market data
    # Correct the price-feed mode BEFORE starting the feed. Sending live orders
    # to OANDA while pricing decisions from a different source is the exact
    # decision/execution desync that produced phantom fills and fake slippage
    # earlier in this project, and a non-OANDA feed cannot supply index or metal
    # instruments at all. This is an invariant, not a preference, so enforce it
    # rather than warn and continue.
    if (settings.trading_mode == "oanda" and settings.oanda_api_key
            and settings.market_data_mode != "oanda"):
        logger.critical(
            "MISCONFIGURATION CORRECTED: trading_mode=oanda but "
            "market_data_mode=%s. Forcing market_data_mode=oanda — orders must "
            "never be priced from a different feed than they execute against. "
            "Set MARKET_DATA_MODE=oanda in the environment to make this explicit.",
            settings.market_data_mode,
        )
        settings.market_data_mode = "oanda"

    await market_data.start()
    logger.info(f"Market data started (mode={settings.market_data_mode})")

    # Load REAL recent candles so indicators are valid from cycle 1 rather than
    # waiting for live ticks to accumulate after every restart. Runs whenever
    # OANDA credentials exist, independent of feed mode.
    if settings.oanda_api_key and settings.oanda_account_id:
        _bs = await market_data.bootstrap_history()
        logger.info("History bootstrap: %d instruments ready", len(_bs["loaded"]))
        if _bs["failed"]:
            logger.critical(
                "History bootstrap FAILED for %s — these instruments cannot be "
                "traded until live ticks accumulate. Check instrument names and "
                "OANDA entitlements.", _bs["failed"],
            )
    if settings.trading_mode == "oanda" and settings.market_data_mode != "oanda":
        logger.critical(
            "MISCONFIGURATION: trading_mode=oanda but market_data_mode=%s — the "
            "internal price feed will drift from real OANDA prices (fake slippage, "
            "phantom stop triggers, polluted forecasts). Set MARKET_DATA_MODE=oanda. "
            "Order entry and SL/TP monitoring now quote OANDA directly as a "
            "safeguard, but charts/indicators/forecasts still use the drifting feed.",
            settings.market_data_mode,
        )

    # Wire up portfolio → risk manager
    _risk_mod.risk_manager = RiskManager(portfolio_service)

    # Apply config-driven risk limits to the hard gate
    risk_gate._max_drawdown_pct = settings.max_drawdown_pct
    risk_gate._min_stop_pips    = settings.min_stop_pips
    risk_gate._max_stop_pips    = settings.max_stop_pips
    risk_gate._min_atr_pips           = settings.min_atr_pips
    risk_gate._min_tp_spread_multiple = settings.min_tp_spread_multiple

    # Execution tier: shadow (log only) / micro (real fills, ~10% size,
    # capped notional, daily loss budget) / full.
    if settings.execution_tier == "shadow":
        risk_gate.set_shadow_mode(True)
        logger.warning("SHADOW MODE active at startup — orders logged but NOT executed")
    else:
        logger.warning(
            "Execution tier: %s%s",
            settings.execution_tier,
            (f" (size x{settings.micro_size_factor:.2f}, cap ${settings.micro_max_notional:.0f}, "
             f"daily budget -${settings.micro_daily_loss_limit:.0f})")
            if settings.execution_tier == "micro" else "",
        )

    # Wire up order service broadcast
    order_service.set_broadcast(manager.broadcast)

    # Start order monitor (SL/TP checks)
    await order_service.start_monitor()

    # Seed portfolio snapshot
    await portfolio_service.snapshot()

    # Start price broadcast
    broadcast_task = asyncio.create_task(broadcast_prices())

    # Start forecast evaluation (checks DB every 60s for expired forecasts)
    eval_task = asyncio.create_task(evaluate_forecasts())

    # Price-feed watchdog: frozen prices are silent and corrupt everything
    # downstream, so alarm loudly rather than fail quietly.
    watchdog_task = asyncio.create_task(market_data.watchdog_loop())

    # New information feeds for the ensemble (phase 1 of the DHJ replacement).
    # Both degrade gracefully: no data → vote 0 / no blackout.
    from services import econ_calendar, sentiment
    calendar_task = asyncio.create_task(econ_calendar.refresh_loop())
    sentiment_task: asyncio.Task | None = None
    if settings.oanda_api_key:
        sentiment_task = asyncio.create_task(sentiment.refresh_loop())
        logger.info("Sentiment feed started (OANDA position book)")
    logger.info("Economic calendar feed started")

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
    eval_task.cancel()
    watchdog_task.cancel()
    calendar_task.cancel()
    if sentiment_task:
        sentiment_task.cancel()
    await trading_agent.stop()
    await order_service.stop_monitor()
    await market_data.stop()
    logger.info("Platform shut down cleanly")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Popper — Agentic AI FX Trading Platform",
    version="1.0.0",
    description="Autonomous Claude-powered FX trading agent with real-time dashboard",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
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
    # no-cache so the browser always revalidates the dashboard HTML after a
    # deploy — prevents stale cached JS (the recurring "old index.html" trap).
    return FileResponse(
        str(static_dir / "index.html"),
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


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

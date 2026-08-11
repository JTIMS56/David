"""
REST API Routes
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import require_api_key
from sqlalchemy import select, desc

from config import settings
from database import AsyncSessionLocal
from models.orm import AgentDecision, AuditLog, Position, Trade, PortfolioSnapshot
from models.schemas import (
    AgentDecisionOut, AgentStatusOut, PortfolioOut, PositionOut, RatesOut,
    SystemStatusOut, TradeOut,
)
from services.market_data import market_data, PAIR_CONFIG
from services.portfolio_service import portfolio_service
from services.order_service import order_service
from services.risk_gate import risk_gate
from services.model_registry import model_registry
from agents.trading_agent import trading_agent

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])


# ── System ────────────────────────────────────────────────────────────────────

@router.get("/status", response_model=SystemStatusOut)
async def get_status():
    state = await portfolio_service.get_state()
    positions = [PositionOut.model_validate(p) for p in state["positions"]]
    portfolio = PortfolioOut(
        balance=state["balance"],
        equity=state["equity"],
        open_positions=state["open_positions"],
        unrealised_pnl=state["unrealised_pnl"],
        daily_pnl=state["daily_pnl"],
        total_pnl=state["total_pnl"],
        drawdown_pct=state["drawdown_pct"],
        positions=positions,
    )
    agent_status = trading_agent.get_status()
    return SystemStatusOut(
        status="ok",
        trading_mode=settings.trading_mode,
        market_data_mode=settings.market_data_mode,
        agent=AgentStatusOut(**agent_status),
        portfolio=portfolio,
    )


# ── Market Data ───────────────────────────────────────────────────────────────

@router.get("/rates")
async def get_rates():
    all_prices = market_data.get_all_prices()
    rates = {}
    for pair, bar in all_prices.items():
        pip = market_data.get_pip_size(pair)
        rates[pair] = {
            "bid": bar.bid,
            "ask": bar.ask,
            "mid": bar.mid,
            "spread_pips": round((bar.ask - bar.bid) / pip, 1),
            "timestamp": bar.timestamp.isoformat() + "Z",
        }
    return {"rates": rates, "timestamp": datetime.utcnow().isoformat() + "Z"}


@router.get("/rates/{pair:path}")
async def get_rate(pair: str):
    pair = pair.replace("-", "/").upper()
    bar = market_data.get_price(pair)
    if not bar:
        raise HTTPException(404, f"Pair {pair} not found")
    pip = market_data.get_pip_size(pair)
    return {
        "pair": pair,
        "bid": bar.bid,
        "ask": bar.ask,
        "mid": bar.mid,
        "spread_pips": round((bar.ask - bar.bid) / pip, 1),
        "timestamp": bar.timestamp.isoformat(),
    }


@router.get("/indicators/{pair:path}")
async def get_indicators(pair: str):
    pair = pair.replace("-", "/").upper()
    ind = market_data.calculate_indicators(pair)
    if not ind:
        raise HTTPException(404, f"Not enough data for {pair}")
    return ind


@router.get("/history/{pair:path}")
async def get_history(pair: str, periods: int = Query(100, ge=10, le=500)):
    pair = pair.replace("-", "/").upper()
    bars = market_data.get_history(pair, periods)
    return {
        "pair": pair,
        "count": len(bars),
        "data": [{"timestamp": b.timestamp.isoformat() + "Z", "bid": b.bid, "ask": b.ask, "mid": b.mid}
                 for b in bars],
    }


# ── Portfolio ─────────────────────────────────────────────────────────────────

@router.get("/portfolio")
async def get_portfolio():
    state = await portfolio_service.get_state()
    positions = [
        {
            "id": p.id,
            "pair": p.pair,
            "direction": p.direction,
            "size": p.size,
            "entry_price": p.entry_price,
            "current_price": p.current_price,
            "stop_loss": p.stop_loss,
            "take_profit": p.take_profit,
            "unrealised_pnl": p.unrealised_pnl,
            "status": p.status,
            "opened_at": p.opened_at.isoformat() + "Z",
            "reasoning": p.reasoning,
        }
        for p in state["positions"]
    ]
    return {
        "balance": state["balance"],
        "equity": state["equity"],
        "open_positions": state["open_positions"],
        "unrealised_pnl": state["unrealised_pnl"],
        "daily_pnl": state["daily_pnl"],
        "total_pnl": state["total_pnl"],
        "drawdown_pct": state["drawdown_pct"],
        "positions": positions,
    }


@router.get("/portfolio/history")
async def get_portfolio_history(limit: int = Query(288, ge=10, le=1000)):
    return await portfolio_service.get_snapshot_history(limit)


@router.get("/portfolio/alltime")
async def get_alltime_stats(oanda_only: bool = False):
    """Cumulative P&L and trade stats from all closed trades in the DB.
    Survives restarts — reads directly from the trades table, not in-memory state.
    Pass oanda_only=true to restrict to trades executed via OANDA (have oanda_trade_id)."""
    async with AsyncSessionLocal() as db:
        q = select(Trade).where(Trade.action.in_(["CLOSE", "SL_HIT", "TP_HIT"]))
        if oanda_only:
            q = q.join(Position, Trade.position_id == Position.id).where(
                Position.oanda_trade_id.isnot(None)
            )
        result = await db.execute(q)
        trades = result.scalars().all()

    if not trades:
        return {"total_pnl": 0.0, "trade_count": 0, "win_rate_pct": 0.0}

    wins   = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]

    return {
        "total_pnl":     round(sum(t.pnl for t in trades), 2),
        "trade_count":   len(trades),
        "win_count":     len(wins),
        "loss_count":    len(losses),
        "win_rate_pct":  round(len(wins) / len(trades) * 100, 1),
        "avg_win":       round(sum(t.pnl for t in wins)   / len(wins),   2) if wins   else 0.0,
        "avg_loss":      round(sum(t.pnl for t in losses) / len(losses), 2) if losses else 0.0,
        "best_trade":    round(max(t.pnl for t in trades), 2),
        "worst_trade":   round(min(t.pnl for t in trades), 2),
        "gross_profit":  round(sum(t.pnl for t in wins),   2) if wins   else 0.0,
        "gross_loss":    round(sum(t.pnl for t in losses), 2) if losses else 0.0,
    }


# ── Positions ─────────────────────────────────────────────────────────────────

@router.get("/positions")
async def list_positions(status: Optional[str] = None):
    async with AsyncSessionLocal() as db:
        q = select(Position).order_by(desc(Position.opened_at))
        if status:
            q = q.where(Position.status == status.upper())
        result = await db.execute(q.limit(100))
        positions = result.scalars().all()
    return [PositionOut.model_validate(p) for p in positions]


@router.delete("/positions/{position_id}")
async def close_position_endpoint(position_id: int, reasoning: str = "Manual close via API"):
    ok, msg, pnl = await order_service.close_position(position_id, reasoning=reasoning, source="human")
    if not ok:
        raise HTTPException(400, msg)
    return {"success": True, "message": msg, "pnl": pnl}


# ── Trades ────────────────────────────────────────────────────────────────────

@router.get("/trades")
async def list_trades(limit: int = Query(50, ge=1, le=500)):
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Trade).order_by(desc(Trade.timestamp)).limit(limit)
        )
        trades = result.scalars().all()
    return [TradeOut.model_validate(t) for t in trades]


# ── Agent ─────────────────────────────────────────────────────────────────────

@router.get("/agent/status")
async def agent_status():
    return trading_agent.get_status()


@router.post("/agent/start")
async def start_agent():
    if trading_agent._running:
        return {"message": "Agent already running", "status": trading_agent.get_status()}
    await trading_agent.start()
    return {"message": "Agent started", "status": trading_agent.get_status()}


@router.post("/agent/stop")
async def stop_agent():
    await trading_agent.stop()
    return {"message": "Agent stopped", "status": trading_agent.get_status()}


@router.post("/agent/run-once")
async def run_agent_once():
    """Fire a single trading cycle in the background and return immediately.
    A full Claude agentic cycle can take several minutes; waiting synchronously
    would exceed infrastructure HTTP timeouts and cancel the cycle mid-flight.
    Poll /api/agent/decisions or /api/agent/status for results."""
    import asyncio as _asyncio
    _asyncio.create_task(trading_agent.run_once())
    return {
        "message": "Cycle started in background",
        "cycle": trading_agent._cycle + 1,
        "hint": "Check /api/agent/decisions in ~30s for results",
    }


@router.get("/oanda/account")
async def oanda_account_summary():
    """Fetch live OANDA account balance and open trade count."""
    if not settings.oanda_api_key:
        raise HTTPException(503, "OANDA not configured — set OANDA_API_KEY and OANDA_ACCOUNT_ID")
    from services.oanda_client import oanda_client
    try:
        data = await oanda_client.get_account_summary()
        acc = data.get("account", {})
        return {
            "balance": float(acc.get("balance", 0)),
            "nav": float(acc.get("NAV", 0)),
            "unrealized_pl": float(acc.get("unrealizedPL", 0)),
            "open_trade_count": int(acc.get("openTradeCount", 0)),
            "currency": acc.get("currency", "USD"),
            "environment": settings.oanda_environment,
        }
    except Exception as exc:
        raise HTTPException(502, f"OANDA API error: {exc}")


@router.get("/agent/ping")
async def agent_ping():
    """Validate Anthropic API key and agent readiness without running a full cycle."""
    import anthropic as _anthropic
    result = {
        "scheduler_running": trading_agent._running,
        "current_cycle": trading_agent._cycle,
        "last_run": trading_agent._last_run.isoformat() if trading_agent._last_run else None,
        "next_run": trading_agent._next_run.isoformat() if trading_agent._next_run else None,
        "api_key_configured": bool(settings.anthropic_api_key),
        "api_key_valid": None,
        "api_key_error": None,
    }
    if settings.anthropic_api_key:
        try:
            client = _anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=5,
                messages=[{"role": "user", "content": "ping"}],
            )
            result["api_key_valid"] = True
        except Exception as exc:
            result["api_key_valid"] = False
            result["api_key_error"] = str(exc)
    return result


@router.get("/agent/decisions")
async def list_decisions(limit: int = Query(20, ge=1, le=100)):
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(AgentDecision).order_by(desc(AgentDecision.timestamp)).limit(limit)
        )
        decisions = result.scalars().all()
    return [AgentDecisionOut.model_validate(d) for d in decisions]


@router.get("/agent/decisions/{decision_id}")
async def get_decision(decision_id: int):
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(AgentDecision).where(AgentDecision.id == decision_id))
        decision = result.scalar_one_or_none()
    if not decision:
        raise HTTPException(404, "Decision not found")
    return AgentDecisionOut.model_validate(decision)


# ── Risk Gate & Kill Switch ───────────────────────────────────────────────────

@router.get("/risk/status")
async def risk_status():
    """Current state of the hard risk gate (kill switch, manual-only mode)."""
    return risk_gate.status()


@router.post("/risk/kill-switch")
async def activate_kill_switch(reason: str = "Manual activation via API"):
    """
    ACTIVATE the kill switch — blocks ALL new orders immediately.
    Existing open positions are NOT automatically closed.
    Use /api/positions (DELETE) to close positions manually.
    """
    risk_gate.activate_kill_switch(reason)
    return {"kill_switch_active": True, "reason": reason}


@router.delete("/risk/kill-switch")
async def deactivate_kill_switch():
    """Deactivate the kill switch — allows new orders again."""
    risk_gate.deactivate_kill_switch()
    return {"kill_switch_active": False}


@router.post("/risk/manual-only")
async def set_manual_only(enabled: bool):
    """
    Enable manual-only mode — agent orders are blocked; human API calls allowed.
    enabled=true  → agent suspended (human oversight mode)
    enabled=false → agent orders permitted again
    """
    risk_gate.set_manual_only(enabled)
    return {"manual_only_active": enabled}


@router.post("/risk/shadow-mode")
async def set_shadow_mode(enabled: bool):
    """
    Toggle shadow (dry-run) mode.
    enabled=true  → orders are evaluated and logged but never executed
    enabled=false → normal execution mode
    Use this to observe agent behaviour without real (paper) trades.
    """
    risk_gate.set_shadow_mode(enabled)
    return {"shadow_mode_active": enabled}


@router.post("/risk/reset-drawdown")
async def reset_drawdown_breaker():
    """
    Reset the drawdown circuit-breaker and deactivate the kill switch.
    Only call this after investigating the cause of the drawdown.
    """
    if not risk_gate.drawdown_triggered:
        return {"message": "Drawdown breaker was not active", "kill_switch_active": risk_gate.kill_switch_active}
    risk_gate.deactivate_kill_switch()
    return {"message": "Drawdown breaker reset; kill switch deactivated", "kill_switch_active": False}


@router.get("/audit")
async def get_audit_log(
    limit: int = Query(50, ge=1, le=500),
    event_type: Optional[str] = None,
    source: Optional[str] = None,
):
    """Immutable audit trail: every order attempt and gate decision."""
    async with AsyncSessionLocal() as db:
        q = select(AuditLog).order_by(desc(AuditLog.timestamp))
        if event_type:
            q = q.where(AuditLog.event_type == event_type.upper())
        if source:
            q = q.where(AuditLog.source == source.lower())
        result = await db.execute(q.limit(limit))
        rows = result.scalars().all()
    return [
        {
            "id":           r.id,
            "timestamp":    r.timestamp.isoformat(),
            "source":       r.source,
            "event_type":   r.event_type,
            "pair":         r.pair,
            "direction":    r.direction,
            "size":         r.size,
            "entry_price":  r.entry_price,
            "stop_loss":    r.stop_loss,
            "take_profit":  r.take_profit,
            "position_id":  r.position_id,
            "realised_pnl": r.realised_pnl,
            "gate_allowed": r.gate_allowed,
            "gate_reason":  r.gate_reason,
            "details":      r.details,
        }
        for r in rows
    ]


# ── Dirac FX Model ────────────────────────────────────────────────────────────

def _get_dirac() -> "DiracPredictor":
    """Lazy-import so the library absence doesn't crash startup."""
    from services.dirac_predictor import DiracPredictor
    return DiracPredictor.instance()


@router.get("/dirac/version")
async def dirac_version():
    try:
        return {"version": _get_dirac().version()}
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))


@router.get("/dirac/predict/{pair}")
async def dirac_predict(
    pair: str,
    horizon_days: float = Query(30.0, ge=1, le=365, description="Forecast horizon in days"),
    kappa: float        = Query(2.0,  ge=0, le=500, description="Dirac mass / flip rate"),
    delta_cp: float     = Query(0.0,  ge=-1, le=1,  description="CP asymmetry (-1 bearish, +1 bullish)"),
    n_points: int       = Query(100,  ge=10, le=400, description="Distribution points returned"),
):
    """
    Run the Dirac brane-world FX prediction model for a currency pair.

    Returns the full price-probability distribution (Dirac + Black-Scholes),
    expected prices, option prices (ATM call), and the chiral charge Q₅
    (market sentiment proxy).

    kappa=0  → pure relativistic wave (max fat tails)
    kappa=2  → moderate Dirac mass (recommended)
    kappa→∞  → exact Black-Scholes recovery
    """
    pair = pair.upper()
    rates = market_data.get_rates()
    if pair not in rates:
        raise HTTPException(404, f"Pair {pair} not found in market data")

    spot = rates[pair]

    try:
        pred = _get_dirac().predict(
            pair=pair,
            spot=spot,
            horizon_days=horizon_days,
            kappa=kappa,
            delta_cp=delta_cp,
        )
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    # Record in model registry (fire-and-forget)
    import asyncio as _asyncio
    _asyncio.create_task(model_registry.record(
        model="Dirac", pair=pair, spot=spot, horizon_days=horizon_days,
        params={"kappa": kappa, "delta_cp": delta_cp, "n_sites": n_sites},
        mean_model=pred.mean_dirac, call_model=pred.call_dirac, call_bs=pred.call_bs,
        chiral_charge=pred.chiral_charge, n_steps=pred.n_steps,
    ))

    # Downsample for API response
    n_full  = len(pred.prices)
    stride  = max(1, n_full // n_points)
    indices = list(range(0, n_full, stride))[:n_points]

    return {
        "pair":          pair,
        "spot":          spot,
        "horizon_days":  pred.horizon_days,
        "kappa":         pred.kappa,
        "delta_cp":      delta_cp,
        "model":         "1+1D Telegraph/Wilson-Dirac",
        "distribution": {
            "prices":     [pred.prices[i]     for i in indices],
            "prob_dirac": [pred.prob_dirac[i] for i in indices],
            "prob_bs":    [pred.prob_bs[i]    for i in indices],
        },
        "statistics": {
            "mean_dirac":    pred.mean_dirac,
            "mean_bs":       pred.mean_bs,
            "std_log_dirac": pred.var_log_dirac ** 0.5,
            "std_log_bs":    pred.var_log_bs    ** 0.5,
            "call_dirac":    pred.call_dirac,
            "call_bs":       pred.call_bs,
            "chiral_charge": pred.chiral_charge,
            "n_steps":       pred.n_steps,
        },
        "interpretation": {
            "chiral_sentiment": (
                "bullish" if pred.chiral_charge > 0.05
                else "bearish" if pred.chiral_charge < -0.05
                else "neutral"
            ),
            "extra_vol_pct": round(
                (pred.var_log_dirac / (pred.var_log_bs + 1e-15) - 1.0) * 100, 2
            ),
        },
    }


@router.get("/dirac/bs-test")
async def dirac_bs_test(
    S0: float    = Query(1.0,    gt=0),
    sigma: float = Query(0.082,  gt=0),
    r: float     = Query(0.0525, ge=0),
    T: float     = Query(0.0833, gt=0, description="Horizon in years (default 1 month)"),
    tol: float   = Query(0.05,   gt=0, description="Relative error tolerance"),
):
    """Verify Black-Scholes convergence at κ=200 (heavily overdamped limit)."""
    try:
        result = _get_dirac().bs_convergence_test(S0=S0, sigma=sigma, r=r, T=T, tol=tol)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    return result


# ── DHJ (Dirac-Heston-Jump) ───────────────────────────────────────────────────

@router.get("/dhj/version")
async def dhj_version():
    try:
        return {"version": _get_dirac().dhj_version()}
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))


@router.get("/dhj/predict/{pair}")
async def dhj_predict_endpoint(
    pair: str,
    horizon_days: float  = Query(30.0,  ge=1,   le=365),
    kappa_H: float       = Query(1.5,   ge=0.1, le=20.0,  description="Heston mean-reversion speed"),
    xi: float            = Query(0.30,  ge=0.0, le=2.0,   description="Vol-of-vol"),
    rho: float           = Query(-0.25, ge=-0.99, le=0.99, description="Leverage correlation"),
    kappa0: float        = Query(1.0,   ge=0.0, le=500.0, description="Base Dirac mass"),
    kappa1: float        = Query(0.002, ge=0.0, le=1.0,   description="Vol-dependent Dirac mass correction"),
    delta_cp: float      = Query(0.0,   ge=-1.0, le=1.0,  description="Initial spinor asymmetry"),
    jump_lambda: float   = Query(3.0,   ge=0.0, le=50.0,  description="Jump arrival rate (per year)"),
    jump_p_up: float     = Query(0.55,  ge=0.0, le=1.0,   description="Probability of upward jump"),
    n_paths: int         = Query(200,   ge=50,  le=1000,  description="MC paths (more = slower but smoother)"),
    n_points: int        = Query(100,   ge=10,  le=400),
):
    """
    Run the Dirac-Heston-Jump (DHJ) hybrid model — the most expressive option.

    Combines five layers of dynamics:
    - **Heston stochastic variance**: mean-reverting vol (captures vol clustering)
    - **Stochastic Dirac diffusion**: wave speed c(τ)=√((1-ρ²)v(τ)) varies with vol
    - **Regime-dependent mass**: κ(v) = κ₀ + κ₁/v (more ballistic in calm markets)
    - **Leverage correlation**: ρ links down-moves to vol spikes
    - **Kou jumps**: Poisson arrivals with asymmetric crash/rally tails

    Converges to Black-Scholes when ξ=0, λ=0, ρ=0, κ₀→∞.
    """
    pair = pair.upper()
    rates = market_data.get_rates()
    if pair not in rates:
        raise HTTPException(404, f"Pair {pair} not found in market data")
    spot = rates[pair]

    import asyncio
    try:
        pred = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _get_dirac().predict_dhj(
                pair=pair, spot=spot,
                horizon_days=horizon_days,
                kappa_H=kappa_H, xi=xi, rho=rho,
                kappa0=kappa0, kappa1=kappa1, delta_cp=delta_cp,
                jump_lambda=jump_lambda, jump_p_up=jump_p_up,
                n_paths=n_paths,
            )
        )
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    # Record in model registry (fire-and-forget)
    asyncio.create_task(model_registry.record(
        model="DHJ", pair=pair, spot=spot, horizon_days=horizon_days,
        params={"kappa_H": kappa_H, "xi": xi, "rho": rho,
                "kappa0": kappa0, "kappa1": kappa1,
                "jump_lambda": jump_lambda, "jump_p_up": jump_p_up,
                "n_paths": n_paths},
        mean_model=pred.mean_dhj, call_model=pred.call_dhj, call_bs=pred.call_bs,
        chiral_charge=pred.chiral_charge, n_steps=pred.n_steps,
        mass_loss_fraction=pred.mass_loss_fraction, negative_count=pred.negative_count,
    ))

    n_full  = len(pred.prices)
    stride  = max(1, n_full // n_points)
    indices = list(range(0, n_full, stride))[:n_points]

    return {
        "pair":         pair,
        "spot":         spot,
        "horizon_days": pred.horizon_days,
        "model":        "Dirac-Heston-Jump (DHJ)",
        "params": {
            "heston":  {"kappa_H": kappa_H, "xi": xi, "rho": rho},
            "dirac":   {"kappa0": kappa0, "kappa1": kappa1},
            "jumps":   {"lambda": jump_lambda, "p_up": jump_p_up},
        },
        "distribution": {
            "prices":   [pred.prices[i]   for i in indices],
            "prob_dhj": [pred.prob_dhj[i] for i in indices],
            "prob_bs":  [pred.prob_bs[i]  for i in indices],
        },
        "statistics": {
            "mean_dhj":      pred.mean_dhj,
            "mean_bs":       pred.mean_bs,
            "std_log_dhj":   pred.var_log_dhj ** 0.5,
            "std_log_bs":    pred.var_log_bs  ** 0.5,
            "call_dhj":      pred.call_dhj,
            "call_bs":       pred.call_bs,
            "chiral_charge": pred.chiral_charge,
            "avg_vol":            pred.avg_variance ** 0.5,
            "n_steps":            pred.n_steps,
            "n_paths":            pred.n_paths,
            "mass_loss_fraction": pred.mass_loss_fraction,
            "min_density":        pred.min_density,
            "negative_count":     pred.negative_count,
        },
        "interpretation": {
            "chiral_sentiment": (
                "bullish" if pred.chiral_charge > 0.05
                else "bearish" if pred.chiral_charge < -0.05
                else "neutral"
            ),
            "extra_vol_pct": round(
                (pred.var_log_dhj / (pred.var_log_bs + 1e-15) - 1.0) * 100, 2
            ),
            "vol_clustering": "yes" if xi > 0.1 else "no",
            "jump_risk":      "yes" if jump_lambda > 0.5 else "no",
        },
    }


# ── Forecast Accuracy ────────────────────────────────────────────────────────

# Forecasts placed while market_data was in simulation mode have spot_price from
# PAIR_CONFIG base values (e.g. USD/JPY 149.50) but are evaluated against real
# OANDA prices (~161.7), inflating actual_move_pips by 1,000-1,200+ pips.
# Any genuine 1-day FX move >200 pips is extremely rare; this threshold filters
# simulation artifacts while keeping real extreme moves.
_CLEAN_MAX_PIPS = 200

# Index and metal CFDs quote in points, so a routine 0.5% day on US30 (~48,000)
# is ~240 points and would be discarded by an FX-scaled threshold. Bound each
# asset class by a realistic 1-day move in its own units instead.
_CLEAN_MAX_BY_CLASS = {"fx": 200.0, "index": 3000.0, "metal": 400.0}


def _is_clean(pair: str, move_pips: Optional[float]) -> bool:
    """
    True when an evaluated move is usable evidence. Rejects simulation
    artefacts (implausibly large) and exact zeros — a genuine 1-day move of
    precisely 0.0 pips does not occur in a live market, so it means the
    forecast was graded against a frozen price.
    """
    if move_pips is None or move_pips == 0:
        return False
    from services.market_data import asset_class
    return abs(move_pips) <= _CLEAN_MAX_BY_CLASS.get(asset_class(pair), _CLEAN_MAX_PIPS)


@router.get("/forecast/accuracy")
async def forecast_accuracy():
    """DHJ direction-forecast accuracy: hit rate by signal type, with recent outcomes."""
    from models.orm import ForecastLog
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ForecastLog).order_by(desc(ForecastLog.created_at)).limit(1000)
        )
        logs = result.scalars().all()

    evaluated = [l for l in logs if l.evaluated_at is not None]
    pending   = [l for l in logs if l.evaluated_at is None]

    # Clean = evaluated AND actual move within realistic 1-day range (no sim artefacts)
    clean     = [
        l for l in evaluated
        if _is_clean(l.pair, l.actual_move_pips)
    ]
    artifacts = len(evaluated) - len(clean)

    correct   = [l for l in clean if l.direction_correct]

    # BS accuracy — clean rows only
    bs_ev      = [l for l in clean if l.bs_direction_correct is not None]
    bs_correct = [l for l in bs_ev if l.bs_direction_correct]
    bs_accuracy = round(len(bs_correct) / len(bs_ev), 3) if bs_ev else None

    # Disagreement analysis: rows where DHJ and BS predicted DIFFERENT directions
    disagreed = [
        l for l in bs_ev
        if l.bs_expected_direction and l.bs_expected_direction != l.expected_direction
    ]
    dhj_right_bs_wrong = [l for l in disagreed if l.direction_correct and not l.bs_direction_correct]
    bs_right_dhj_wrong = [l for l in disagreed if l.bs_direction_correct and not l.direction_correct]

    by_signal: dict = {}
    for sig in ["STRONG_BULLISH", "MILD_BULLISH", "NEUTRAL", "MILD_BEARISH", "STRONG_BEARISH"]:
        sig_all = [l for l in logs  if l.signal == sig]
        sig_ev  = [l for l in clean if l.signal == sig]
        sig_ok  = [l for l in sig_ev if l.direction_correct]
        by_signal[sig] = {
            "total":     len(sig_all),
            "evaluated": len(sig_ev),
            "correct":   len(sig_ok),
            "accuracy":  round(len(sig_ok) / len(sig_ev), 3) if sig_ev else None,
        }

    recent = sorted(evaluated, key=lambda l: l.evaluated_at, reverse=True)[:20]
    return {
        "total_forecasts":      len(logs),
        "evaluated":            len(evaluated),
        "clean_evaluated":      len(clean),
        "simulation_artifacts": artifacts,
        "pending":              len(pending),
        "direction_accuracy":   round(len(correct) / len(clean), 3) if clean else None,
        "bs_direction_accuracy": bs_accuracy,
        "bs_evaluated":         len(bs_ev),
        "disagreements": {
            "total":              len(disagreed),
            "dhj_right_bs_wrong": len(dhj_right_bs_wrong),
            "bs_right_dhj_wrong": len(bs_right_dhj_wrong),
            "both_right":         len([l for l in disagreed if l.direction_correct and l.bs_direction_correct]),
            "both_wrong":         len([l for l in disagreed if not l.direction_correct and not l.bs_direction_correct]),
        },
        "by_signal": by_signal,
        "recent": [
            {
                "id":                   l.id,
                "pair":                 l.pair,
                "signal":               l.signal,
                "expected_direction":   l.expected_direction,
                "expected_move_pips":   l.expected_move_pips,
                "actual_move_pips":     l.actual_move_pips,
                "direction_correct":    l.direction_correct,
                "bs_expected_direction": l.bs_expected_direction,
                "bs_direction_correct": l.bs_direction_correct,
                "prob_above_spot":      l.prob_above_spot,
                "is_simulation_artifact": (
                    not _is_clean(l.pair, l.actual_move_pips)
                ),
                "created_at":           l.created_at.isoformat() + "Z",
                "evaluated_at":         l.evaluated_at.isoformat() + "Z" if l.evaluated_at else None,
                "horizon_days":         l.horizon_days,
            }
            for l in recent
        ],
    }


def _wilson_lower_bound(correct: int, n: int, z: float = 1.64) -> float:
    """
    Wilson score lower bound for a binomial proportion (one-sided ~95% at z=1.64).
    Guards against small-sample mirages: a cell at 4/5 = 80% has a lower bound near
    0.38, so it won't be mistaken for a real edge.  Returns 0.0 for n == 0.
    """
    if n == 0:
        return 0.0
    p = correct / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = p + z2 / (2 * n)
    margin = z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5)
    return max(0.0, (centre - margin) / denom)


@router.get("/forecast/edge-analysis")
async def forecast_edge_analysis(
    min_samples: int = Query(20, ge=5, description="Minimum cell size to report an edge"),
    edge_threshold: float = Query(0.55, ge=0.5, le=1.0, description="Accuracy needed to flag a tradeable cell"),
):
    """
    Mine the evaluated forecast log for CONDITIONAL accuracy — the subsets where
    DHJ actually beats a coin flip — so trading can be gated to real edges rather
    than the ~50% blended average.

    Each cell reports n, raw accuracy, and a Wilson lower-confidence bound. A cell
    is only flagged tradeable when n >= min_samples AND its lower bound > 0.50
    (so small-sample noise can't masquerade as an edge). Cells whose lower bound
    sits BELOW 0.50 with accuracy < 0.45 are flagged as invertible (fade them).
    """
    from models.orm import ForecastLog

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ForecastLog).order_by(desc(ForecastLog.created_at)).limit(5000)
        )
        logs = result.scalars().all()

    clean = [
        l for l in logs
        if l.evaluated_at is not None
        and l.direction_correct is not None
        and _is_clean(l.pair, l.actual_move_pips)
    ]

    def cell(rows: list) -> dict:
        n = len(rows)
        ok = sum(1 for r in rows if r.direction_correct)
        acc = round(ok / n, 3) if n else None
        lb = round(_wilson_lower_bound(ok, n), 3) if n else None
        tradeable = bool(n >= min_samples and lb is not None and lb > 0.50 and acc >= edge_threshold)
        invertible = bool(n >= min_samples and acc is not None and acc < 0.45
                          and _wilson_lower_bound(n - ok, n) > 0.50)
        return {"n": n, "accuracy": acc, "lower_bound": lb,
                "tradeable": tradeable, "invertible": invertible}

    def group(key_fn) -> dict:
        buckets: dict = {}
        for l in clean:
            k = key_fn(l)
            if k is None:
                continue
            buckets.setdefault(k, []).append(l)
        return {str(k): cell(v) for k, v in sorted(buckets.items())}

    def hour_session(l) -> str:
        h = l.created_at.hour
        if 7 <= h < 12:   return "London(07-12)"
        if 12 <= h < 17:  return "NY-overlap(12-17)"
        if 17 <= h < 21:  return "NY(17-21)"
        return "Asia(21-07)"

    def conviction(l) -> str:
        q = abs(l.chiral_charge)
        if q >= 0.15: return "strong(|Q5|>=0.15)"
        if q >= 0.05: return "mild(0.05-0.15)"
        return "weak(<0.05)"

    def agree_key(l) -> Optional[str]:
        if not l.bs_expected_direction:
            return None
        return "agree" if l.bs_expected_direction == l.expected_direction else "disagree"

    # Disagreement × conviction — the combination most likely to concentrate edge
    def disagree_conviction(l) -> Optional[str]:
        a = agree_key(l)
        if a != "disagree":
            return None
        return f"disagree+{conviction(l)}"

    breakdowns = {
        "by_pair":                 group(lambda l: l.pair),
        "by_signal":               group(lambda l: l.signal),
        "by_signal_direction":     group(lambda l: f"{l.signal}/{l.expected_direction}"),
        "by_dhj_bs_agreement":     group(agree_key),
        "by_conviction":           group(conviction),
        "by_session_utc":          group(hour_session),
        "by_pair_x_agreement":     group(lambda l: f"{l.pair}/{agree_key(l)}" if agree_key(l) else None),
        "by_disagree_conviction":  group(disagree_conviction),
    }

    # Collect everything flagged tradeable or invertible, sorted by strength
    edges = []
    for dim, cells in breakdowns.items():
        for k, c in cells.items():
            if c["tradeable"] or c["invertible"]:
                edges.append({"dimension": dim, "cell": k, **c})
    edges.sort(key=lambda e: e["lower_bound"], reverse=True)

    overall_ok = sum(1 for l in clean if l.direction_correct)
    return {
        "clean_evaluated": len(clean),
        "overall_accuracy": round(overall_ok / len(clean), 3) if clean else None,
        "params": {"min_samples": min_samples, "edge_threshold": edge_threshold},
        "edges_found": edges,
        "breakdowns": breakdowns,
        "note": (
            "tradeable = n>=min_samples and Wilson lower bound > 0.50 and accuracy >= threshold. "
            "invertible = accuracy < 0.45 with lower bound (of being wrong) > 0.50 — fade these. "
            "Empty edges_found means no subset beats coin flip at this confidence yet."
        ),
    }


@router.get("/forecast/edge-map-validation")
async def edge_map_validation():
    """
    Replay the per-pair edge map over the forecast log and split its accuracy
    into IN-SAMPLE (created on/before the cutoff — where the edges were found)
    vs OUT-OF-SAMPLE (created after the cutoff — the honest test). The edges are
    real only if the out-of-sample numbers hold up. Pure replay, no new logging.
    """
    from datetime import datetime as _dt
    from models.orm import ForecastLog
    from services.edge_map import recommend, was_correct

    try:
        cutoff = _dt.fromisoformat(settings.edge_map_cutoff)
    except (ValueError, TypeError):
        cutoff = None

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ForecastLog).order_by(desc(ForecastLog.created_at)).limit(20000)
        )
        logs = result.scalars().all()

    clean = [
        l for l in logs
        if l.evaluated_at is not None and l.direction_correct is not None
        and _is_clean(l.pair, l.actual_move_pips)
    ]

    def summarize(rows: list) -> dict:
        # rows is a list of (cell_key, correct_bool)
        per_cell: dict = {}
        for key, correct in rows:
            per_cell.setdefault(key, []).append(correct)
        out = {}
        for key, results in sorted(per_cell.items()):
            n = len(results); ok = sum(1 for c in results if c)
            out[key] = {
                "n": n,
                "accuracy": round(ok / n, 3) if n else None,
                "lower_bound": round(_wilson_lower_bound(ok, n), 3) if n else None,
            }
        n = len(rows); ok = sum(1 for _, c in rows if c)
        return {
            "overall": {
                "n": n,
                "accuracy": round(ok / n, 3) if n else None,
                "lower_bound": round(_wilson_lower_bound(ok, n), 3) if n else None,
            },
            "by_cell": out,
        }

    in_sample, out_sample = [], []
    for l in clean:
        action, _side = recommend(l.pair, l.expected_direction, l.bs_expected_direction)
        if action == "SKIP":
            continue
        correct = was_correct(action, l.direction_correct)
        if correct is None:
            continue
        agreement = "agree" if l.expected_direction == l.bs_expected_direction else "disagree"
        key = f"{l.pair}/{agreement}/{action.lower()}"
        bucket = out_sample if (cutoff and l.created_at > cutoff) else in_sample
        bucket.append((key, correct))

    return {
        "cutoff": settings.edge_map_cutoff,
        "edge_map": settings.edge_map,
        "in_sample": summarize(in_sample),
        "out_of_sample": summarize(out_sample),
        "verdict_note": (
            "Promote a cell to live trading only when its OUT-OF-SAMPLE lower_bound > 0.50 "
            "with a real sample (n >= ~20 independent days). out_of_sample.overall.n grows as "
            "the agent keeps forecasting; check back in 1-2 weeks. In-sample reproduces the "
            "discovery numbers and is NOT evidence on its own."
        ),
    }


@router.get("/forecast/ensemble-accuracy")
async def ensemble_accuracy():
    """
    Head-to-head: the independent ensemble model vs DHJ on the SAME pairs and
    horizons (shadow mode). Reports overall and high-conviction accuracy plus
    per-signal vote accuracy, with Wilson lower bounds so we don't promote noise.
    """
    from models.orm import EnsembleForecastLog, ForecastLog

    async with AsyncSessionLocal() as db:
        eres = await db.execute(
            select(EnsembleForecastLog).order_by(desc(EnsembleForecastLog.created_at)).limit(5000)
        )
        elogs = eres.scalars().all()
        dres = await db.execute(
            select(ForecastLog).order_by(desc(ForecastLog.created_at)).limit(5000)
        )
        dlogs = dres.scalars().all()

    def acc(rows: list) -> dict:
        rows = [r for r in rows if r.direction_correct is not None
                and _is_clean(r.pair, r.actual_move_pips)]
        n = len(rows)
        ok = sum(1 for r in rows if r.direction_correct)
        return {
            "n": n,
            "accuracy": round(ok / n, 3) if n else None,
            "lower_bound": round(_wilson_lower_bound(ok, n), 3) if n else None,
        }

    e_clean = [l for l in elogs if l.evaluated_at is not None]
    high_conv = [l for l in e_clean if l.high_conviction]

    # Per-signal standalone accuracy: does each vote, alone, beat coin flip?
    per_signal = {}
    for name, attr in [("trend", "vote_trend"), ("mean_revert", "vote_mean_revert"),
                       ("carry", "vote_carry"), ("usd_strength", "vote_usd_strength"),
                       ("positioning", "vote_positioning")]:
        voted = []
        for l in e_clean:
            v = getattr(l, attr)
            if v == 0 or not _is_clean(l.pair, l.actual_move_pips):
                continue
            correct = (v > 0 and l.actual_move_pips > 0) or (v < 0 and l.actual_move_pips < 0)
            # reuse a tiny shim object isn't needed — count inline
            voted.append(correct)
        n = len(voted); ok = sum(1 for c in voted if c)
        per_signal[name] = {
            "n": n,
            "accuracy": round(ok / n, 3) if n else None,
            "lower_bound": round(_wilson_lower_bound(ok, n), 3) if n else None,
        }

    return {
        "ensemble": {
            "overall":          acc(e_clean),
            "high_conviction":  acc(high_conv),
            "per_signal_vote":  per_signal,
        },
        "dhj": {
            "overall": acc(dlogs),
        },
        "verdict_note": (
            "Promote the ensemble to trading only if high_conviction.lower_bound > 0.50 "
            "AND it clears DHJ on out-of-sample data. lower_bound <= 0.50 means still a coin flip."
        ),
    }


@router.get("/feeds/status")
async def feeds_status():
    """
    Health of the ensemble's new information feeds: OANDA crowd positioning
    per pair, and the economic-calendar blackout state / upcoming events.
    """
    from services import econ_calendar, sentiment
    from services.oanda_client import PAIR_TO_OANDA

    positioning = {}
    for pair in PAIR_TO_OANDA:
        data = sentiment.get_positioning(pair)
        if data:
            positioning[pair] = {
                "long_pct": data["long_pct"],
                "short_pct": data["short_pct"],
                "vote": sentiment.positioning_vote(pair),
                "as_of": data.get("time", ""),
            }

    upcoming = [
        {
            "title": e["title"],
            "currency": e["currency"],
            "at": e["at"].isoformat(),
        }
        for e in econ_calendar.upcoming_high_impact(24.0)[:10]
    ]
    blackouts = {
        pair: bool(econ_calendar.is_blackout(pair)) for pair in PAIR_TO_OANDA
    }

    from services.market_data import market_data as _md
    from services.oanda_client import oanda_client as _oc
    return {
        "price_feed": {
            **_md.feed_health(),
            "quarantined_instruments": sorted(_oc._quarantined),
            "real_bars": {p: _md.real_bar_count(p) for p in PAIR_TO_OANDA},
        },
        "positioning": positioning,
        "positioning_pairs_cached": len(positioning),
        "fade_threshold_pct": settings.positioning_fade_threshold,
        "calendar_events_loaded": len(econ_calendar._events),
        "upcoming_high_impact_24h": upcoming,
        "blackout_now": blackouts,
    }


@router.get("/readiness")
async def go_live_readiness():
    """
    Auto-computed Aug-1 go/no-go scorecard:
      • Trading window (go_nogo_window_start, the exit-hysteresis era):
        realized P&L, win rate, average win vs average loss.
      • Forecast window (clean_data_start, the single-instance era):
        ensemble conviction>=2 directional accuracy with Wilson lower bound.
      • Feed health right now.
    """
    from models.orm import EnsembleForecastLog
    from services import econ_calendar, sentiment
    from services.oanda_client import PAIR_TO_OANDA

    trade_start = datetime.fromisoformat(settings.go_nogo_window_start)
    fc_start = datetime.fromisoformat(settings.clean_data_start)

    async with AsyncSessionLocal() as db:
        trades = (await db.execute(
            select(Trade).where(
                Trade.timestamp >= trade_start,
                Trade.action.in_(("CLOSE", "SL_HIT", "TP_HIT")),
            )
        )).scalars().all()
        elogs = (await db.execute(
            select(EnsembleForecastLog).where(
                EnsembleForecastLog.created_at >= fc_start,
                EnsembleForecastLog.direction_correct.isnot(None),
                EnsembleForecastLog.conviction >= 2,
            )
        )).scalars().all()

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]
    pnl_total = round(sum(t.pnl for t in trades), 2)
    avg_win = round(sum(wins) / len(wins), 2) if wins else 0.0
    avg_loss = round(sum(losses) / len(losses), 2) if losses else 0.0

    clean = [e for e in elogs
             if _is_clean(e.pair, e.actual_move_pips)]
    n = len(clean)
    ok = sum(1 for e in clean if e.direction_correct)
    acc = round(ok / n, 3) if n else None
    lb = round(_wilson_lower_bound(ok, n), 3) if n else None

    positioning_cached = sum(
        1 for p in PAIR_TO_OANDA if sentiment.get_positioning(p)
    )

    criteria = {
        "pnl_positive": pnl_total > 0,
        "payoff_ratio_ok": bool(wins) and bool(losses) and avg_win >= abs(avg_loss),
        "ensemble_acc_over_52": acc is not None and n >= 50 and acc > 0.52,
        "feeds_healthy": positioning_cached >= 6 and len(econ_calendar._events) > 0,
    }

    return {
        "as_of": datetime.utcnow().isoformat(),
        "trading_window_since": settings.go_nogo_window_start,
        "trades": {
            "n": len(trades), "wins": len(wins), "losses": len(losses),
            "win_rate": round(len(wins) / len(trades), 3) if trades else None,
            "pnl_total": pnl_total, "avg_win": avg_win, "avg_loss": avg_loss,
        },
        "forecast_window_since": settings.clean_data_start,
        "ensemble_conviction2plus": {
            "n": n, "accuracy": acc, "wilson_lower_bound": lb,
            "excluded_artifacts": len(elogs) - n,
        },
        "feeds": {
            "positioning_pairs_cached": positioning_cached,
            "calendar_events_loaded": len(econ_calendar._events),
        },
        "criteria": criteria,
        "verdict": "GO" if all(criteria.values()) else "NOT YET — see criteria",
    }


_backtest_cache: dict = {}
_index_backtest_cache: dict = {}


@router.get("/backtest/index-trend")
async def backtest_index_trend(refresh: bool = Query(False)):
    """
    Backtest the documented index/metals premia on OANDA CFD candles:
    buy-and-hold baseline, 12m momentum long/flat, and long/short.
    """
    global _index_backtest_cache
    if _index_backtest_cache and not refresh:
        return _index_backtest_cache
    if not settings.oanda_api_key:
        raise HTTPException(503, "OANDA credentials required for historical candles")

    from services.oanda_client import oanda_client
    from services.backtest import run_index_trend, INDEX_META

    candles, fetch_errors = {}, {}
    for sym, meta in INDEX_META.items():
        try:
            series = await oanda_client.get_daily_candles(meta["oanda"], count=3800)
            if len(series) > 400:
                candles[sym] = series
        except Exception as exc:
            fetch_errors[sym] = str(exc)[:120]

    if not candles:
        raise HTTPException(502, f"No candle data fetched: {fetch_errors}")

    result = {
        "buy_hold": run_index_trend(candles, mode="buy_hold"),
        "momentum_long_flat": run_index_trend(candles, mode="long_flat"),
        "momentum_long_short": run_index_trend(candles, mode="long_short"),
        "candles_fetched": {s: len(c) for s, c in candles.items()},
        "computed_at": datetime.utcnow().isoformat(),
    }
    if fetch_errors:
        result["fetch_errors"] = fetch_errors
    _index_backtest_cache = result
    return result


@router.get("/backtest/carry-trend")
async def backtest_carry_trend(refresh: bool = Query(False)):
    """
    Run the multi-week carry+trend backtest over ~15 years of OANDA daily
    candles (fetched live, one request per pair). Cached until refresh=true.
    """
    global _backtest_cache
    if _backtest_cache and not refresh:
        return _backtest_cache

    if not settings.oanda_api_key:
        raise HTTPException(503, "OANDA credentials required for historical candles")

    from services.oanda_client import oanda_client, PAIR_TO_OANDA
    from services.backtest import run_carry_trend, run_cross_carry, BT_EXTRA_OANDA

    universe = {**PAIR_TO_OANDA, **BT_EXTRA_OANDA}
    candles, fetch_errors = {}, {}
    for pair, instrument in universe.items():
        try:
            series = await oanda_client.get_daily_candles(instrument, count=3800)
            if len(series) > 200:
                candles[pair] = series
        except Exception as exc:
            fetch_errors[pair] = str(exc)[:120]

    if not candles:
        raise HTTPException(502, f"No candle data fetched: {fetch_errors}")

    base8 = {p: c for p, c in candles.items() if p in PAIR_TO_OANDA}
    result = {
        # original test: time-series carry+trend on the USD-majors universe
        "time_series_majors": run_carry_trend(base8),
        # pre-registered follow-up: classic cross-sectional carry on the
        # rate-dispersed 16-pair universe (JPY/CHF funding crosses included)
        "cross_sectional_carry": run_cross_carry(candles),
        "cross_sectional_carry_trend_veto": run_cross_carry(candles, trend_veto=True),
        "candles_fetched": {p: len(c) for p, c in candles.items()},
        "computed_at": datetime.utcnow().isoformat(),
    }
    if fetch_errors:
        result["fetch_errors"] = fetch_errors
    _backtest_cache = result
    return result


# ── Model Registry ────────────────────────────────────────────────────────────

@router.get("/registry")
async def get_model_registry(
    model: Optional[str] = None,
    pair:  Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
):
    """
    Model registry: every DHJ/Dirac prediction call with exact params + git commit.
    Provides full auditability linking agent decisions to model versions.
    """
    return await model_registry.get_runs(model=model, pair=pair, limit=limit)

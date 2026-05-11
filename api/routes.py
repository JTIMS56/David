"""
REST API Routes
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
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

router = APIRouter(prefix="/api")


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
            "timestamp": bar.timestamp.isoformat(),
        }
    return {"rates": rates, "timestamp": datetime.utcnow().isoformat()}


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
        "data": [{"timestamp": b.timestamp.isoformat(), "bid": b.bid, "ask": b.ask, "mid": b.mid}
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
            "opened_at": p.opened_at.isoformat(),
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
    result = await trading_agent.run_once()
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

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
from models.orm import AgentDecision, Position, Trade, PortfolioSnapshot
from models.schemas import (
    AgentDecisionOut, AgentStatusOut, PortfolioOut, PositionOut, RatesOut,
    SystemStatusOut, TradeOut,
)
from services.market_data import market_data, PAIR_CONFIG
from services.portfolio_service import portfolio_service
from services.order_service import order_service
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
    ok, msg, pnl = await order_service.close_position(position_id, reasoning=reasoning)
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

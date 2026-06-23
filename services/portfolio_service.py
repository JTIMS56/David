"""
Portfolio Service
────────────────────────────────────────────────────────────────────────────
Tracks account balance, open positions, and computes P&L.
"""
from __future__ import annotations

import json
from datetime import datetime, date, timezone
from typing import Dict, List, Optional

from sqlalchemy import select, and_

from config import settings
from database import AsyncSessionLocal
from models.orm import Position, PortfolioSnapshot


class PortfolioService:
    def __init__(self) -> None:
        self._balance: float = settings.initial_balance
        self._start_of_day_balance: float = settings.initial_balance
        self._peak_balance: float = settings.initial_balance
        self._total_pnl: float = 0.0
        self._cycle_count: int = 0

    # ── Balance management ────────────────────────────────────────────────────

    def apply_realised_pnl(self, pnl: float) -> None:
        self._balance += pnl
        self._total_pnl += pnl
        if self._balance > self._peak_balance:
            self._peak_balance = self._balance

    def reset_daily(self) -> None:
        self._start_of_day_balance = self._balance

    # ── Position P&L updates ──────────────────────────────────────────────────

    async def update_unrealised_pnl(self) -> None:
        from services.market_data import market_data

        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Position).where(Position.status == "OPEN"))
            positions = result.scalars().all()
            for pos in positions:
                bar = market_data.get_price(pos.pair)
                if bar is None:
                    continue
                current = bar.bid if pos.direction == "BUY" else bar.ask
                pos.current_price = current
                if pos.direction == "BUY":
                    pos.unrealised_pnl = (current - pos.entry_price) * pos.size
                else:
                    pos.unrealised_pnl = (pos.entry_price - current) * pos.size
            await db.commit()

    # ── State snapshot ────────────────────────────────────────────────────────

    async def get_state(self) -> dict:
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Position).where(Position.status == "OPEN"))
            positions = result.scalars().all()

        unrealised = sum(p.unrealised_pnl for p in positions)
        total_notional = sum(p.size * p.current_price for p in positions)
        equity = self._balance + unrealised
        daily_pnl = self._balance - self._start_of_day_balance + unrealised
        drawdown = (self._peak_balance - equity) / self._peak_balance if self._peak_balance > 0 else 0.0

        return {
            "balance": round(self._balance, 2),
            "equity": round(equity, 2),
            "unrealised_pnl": round(unrealised, 2),
            "total_notional": round(total_notional, 2),
            "open_positions": len(positions),
            "daily_pnl": round(daily_pnl, 2),
            "total_pnl": round(self._total_pnl, 2),
            "drawdown_pct": round(drawdown * 100, 2),
            "positions": positions,
        }

    async def get_open_positions(self) -> List[Position]:
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Position).where(Position.status == "OPEN"))
            return list(result.scalars().all())

    async def snapshot(self) -> None:
        state = await self.get_state()
        async with AsyncSessionLocal() as db:
            snap = PortfolioSnapshot(
                balance=state["balance"],
                equity=state["equity"],
                open_positions=state["open_positions"],
                daily_pnl=state["daily_pnl"],
                total_pnl=state["total_pnl"],
            )
            db.add(snap)
            await db.commit()

    async def get_snapshot_history(self, limit: int = 288) -> list:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(PortfolioSnapshot)
                .order_by(PortfolioSnapshot.timestamp.desc())
                .limit(limit)
            )
            rows = result.scalars().all()
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "balance": r.balance,
                "equity": r.equity,
                "daily_pnl": r.daily_pnl,
                "total_pnl": r.total_pnl,
            }
            for r in reversed(rows)
        ]


# Singleton
portfolio_service = PortfolioService()

"""
Order Service  (Paper Trading)
────────────────────────────────────────────────────────────────────────────
Executes orders against the simulated market, manages open positions, and
triggers stop-loss / take-profit checks on every price update.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select

from config import settings
from database import AsyncSessionLocal
from models.orm import Position, Trade
from services.market_data import market_data
from services.portfolio_service import portfolio_service


class OrderService:
    def __init__(self) -> None:
        self._monitor_task: Optional[asyncio.Task] = None
        self._ws_broadcast: Optional[callable] = None  # injected by app

    def set_broadcast(self, fn: callable) -> None:
        self._ws_broadcast = fn

    async def _broadcast(self, event: str, data: dict) -> None:
        if self._ws_broadcast:
            await self._ws_broadcast({"event": event, **data})

    # ── Order placement ───────────────────────────────────────────────────────

    async def open_position(
        self,
        pair: str,
        direction: str,
        size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reasoning: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[Position]]:
        bar = market_data.get_price(pair)
        if bar is None:
            return False, f"No price available for {pair}", None

        entry_price = bar.ask if direction == "BUY" else bar.bid

        async with AsyncSessionLocal() as db:
            pos = Position(
                pair=pair,
                direction=direction,
                size=size,
                entry_price=entry_price,
                current_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                unrealised_pnl=0.0,
                status="OPEN",
                reasoning=reasoning,
            )
            db.add(pos)
            await db.flush()

            trade = Trade(
                position_id=pos.id,
                pair=pair,
                action="OPEN",
                direction=direction,
                size=size,
                price=entry_price,
                pnl=0.0,
                reasoning=reasoning,
            )
            db.add(trade)
            await db.commit()
            await db.refresh(pos)

        await self._broadcast("position_opened", {
            "position_id": pos.id,
            "pair": pair,
            "direction": direction,
            "size": size,
            "entry_price": entry_price,
        })
        return True, f"Opened {direction} {size} {pair} @ {entry_price:.5f}", pos

    async def close_position(
        self,
        position_id: int,
        action: str = "CLOSE",
        reasoning: Optional[str] = None,
    ) -> Tuple[bool, str, float]:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Position).where(Position.id == position_id, Position.status == "OPEN")
            )
            pos = result.scalar_one_or_none()
            if pos is None:
                return False, f"Position {position_id} not found or already closed", 0.0

            bar = market_data.get_price(pos.pair)
            if bar is None:
                return False, f"No price for {pos.pair}", 0.0

            close_price = bar.bid if pos.direction == "BUY" else bar.ask
            if pos.direction == "BUY":
                pnl = (close_price - pos.entry_price) * pos.size
            else:
                pnl = (pos.entry_price - close_price) * pos.size

            pos.status = "CLOSED"
            pos.close_price = close_price
            pos.closed_at = datetime.now(timezone.utc)
            pos.realised_pnl = round(pnl, 2)
            pos.unrealised_pnl = 0.0

            trade = Trade(
                position_id=pos.id,
                pair=pos.pair,
                action=action,
                direction=pos.direction,
                size=pos.size,
                price=close_price,
                pnl=round(pnl, 2),
                reasoning=reasoning,
            )
            db.add(trade)
            await db.commit()

        portfolio_service.apply_realised_pnl(pnl)
        await self._broadcast("position_closed", {
            "position_id": position_id,
            "pnl": round(pnl, 2),
            "action": action,
        })
        return True, f"Closed position {position_id} @ {close_price:.5f}, PnL: {pnl:.2f}", pnl

    # ── SL/TP monitoring ──────────────────────────────────────────────────────

    async def check_sl_tp(self) -> None:
        positions = await portfolio_service.get_open_positions()
        for pos in positions:
            bar = market_data.get_price(pos.pair)
            if bar is None:
                continue

            current = bar.bid if pos.direction == "BUY" else bar.ask
            triggered_action: Optional[str] = None

            if pos.stop_loss is not None:
                if pos.direction == "BUY" and current <= pos.stop_loss:
                    triggered_action = "SL_HIT"
                elif pos.direction == "SELL" and current >= pos.stop_loss:
                    triggered_action = "SL_HIT"

            if pos.take_profit is not None and triggered_action is None:
                if pos.direction == "BUY" and current >= pos.take_profit:
                    triggered_action = "TP_HIT"
                elif pos.direction == "SELL" and current <= pos.take_profit:
                    triggered_action = "TP_HIT"

            if triggered_action:
                await self.close_position(
                    pos.id,
                    action=triggered_action,
                    reasoning=f"Auto-triggered: {triggered_action}",
                )

    async def _monitor_loop(self, interval: float = 3.0) -> None:
        while True:
            try:
                await portfolio_service.update_unrealised_pnl()
                await self.check_sl_tp()
            except Exception:
                pass
            await asyncio.sleep(interval)

    async def start_monitor(self) -> None:
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitor(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()


# Singleton
order_service = OrderService()

"""
Order Service  (Paper Trading)
────────────────────────────────────────────────────────────────────────────
Executes orders against the simulated market, manages open positions, and
triggers stop-loss / take-profit checks on every price update.

Every open_position() call passes through RiskGate unconditionally before any
DB write. The gate cannot be bypassed by the LLM — it lives at the service
layer, not the tool layer.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

from sqlalchemy import select

from config import settings
from database import AsyncSessionLocal
from models.orm import AuditLog, Position, Trade
from services.market_data import market_data, PAIR_CONFIG
from services.portfolio_service import portfolio_service
from services.risk_gate import risk_gate

logger = logging.getLogger("david.order_service")


def _spread_pips(pair: str, bid: float, ask: float) -> float:
    pip = PAIR_CONFIG.get(pair, {}).get("pip", 0.0001)
    return (ask - bid) / pip


def _data_age_seconds(bar) -> float:
    now = datetime.now(timezone.utc)
    ts  = bar.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds()


async def _write_audit(
    *,
    source: str,
    event_type: str,
    pair: str | None = None,
    direction: str | None = None,
    size: float | None = None,
    entry_price: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    position_id: int | None = None,
    realised_pnl: float | None = None,
    gate_allowed: bool | None = None,
    gate_reason: str | None = None,
    details: dict | None = None,
) -> None:
    try:
        async with AsyncSessionLocal() as db:
            entry = AuditLog(
                source=source,
                event_type=event_type,
                pair=pair,
                direction=direction,
                size=size,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_id=position_id,
                realised_pnl=realised_pnl,
                gate_allowed=gate_allowed,
                gate_reason=gate_reason,
                details=json.dumps(details) if details else None,
            )
            db.add(entry)
            await db.commit()
    except Exception:
        logger.exception("Failed to write audit log entry (event=%s)", event_type)


class OrderService:
    def __init__(self) -> None:
        self._monitor_task: Optional[asyncio.Task] = None
        self._ws_broadcast: Optional[callable] = None

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
        source: str = "agent",   # "agent" | "human" | "sl_tp"
    ) -> Tuple[bool, str, Optional[Position]]:
        bar = market_data.get_price(pair)
        if bar is None:
            await _write_audit(
                source=source, event_type="ORDER_REJECT",
                pair=pair, direction=direction, size=size, stop_loss=stop_loss,
                gate_allowed=False, gate_reason="No price data available",
            )
            return False, f"No price available for {pair}", None

        entry_price = bar.ask if direction == "BUY" else bar.bid
        spread      = _spread_pips(pair, bar.bid, bar.ask)
        data_age    = _data_age_seconds(bar)

        # ── Hard gate (mandatory, LLM cannot bypass) ──────────────────────────
        decision = risk_gate.approve_order(
            pair=pair,
            direction=direction,
            size=size,
            stop_loss=stop_loss,
            entry_price=entry_price,
            spread_pips=spread,
            data_age_seconds=data_age,
            source=source,
        )

        if not decision.allowed:
            logger.warning(
                "Order BLOCKED by gate [%s %s %s]: %s",
                direction, size, pair, decision.reason,
            )
            await _write_audit(
                source=source, event_type="ORDER_REJECT",
                pair=pair, direction=direction, size=size,
                entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
                gate_allowed=False, gate_reason=decision.reason,
                details={"checks_run": decision.checks_run, "spread_pips": spread},
            )
            return False, f"Order blocked: {decision.reason}", None

        # ── Shadow mode: log and return without executing ──────────────────────
        if decision.shadow:
            logger.info(
                "SHADOW [%s %s %s @ %.5f]: logged only — not executed",
                direction, size, pair, entry_price,
            )
            await _write_audit(
                source=source, event_type="SHADOW_ORDER",
                pair=pair, direction=direction, size=size,
                entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
                gate_allowed=True, gate_reason=decision.reason,
                details={"spread_pips": spread, "data_age_s": round(data_age, 1), "shadow": True},
            )
            return True, f"[SHADOW] Would open {direction} {size} {pair} @ {entry_price:.5f}", None

        # ── Write position + trade atomically ─────────────────────────────────
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

        await _write_audit(
            source=source, event_type="ORDER_OPEN",
            pair=pair, direction=direction, size=size,
            entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
            position_id=pos.id,
            gate_allowed=True, gate_reason=decision.reason,
            details={"spread_pips": spread, "data_age_s": round(data_age, 1)},
        )

        await self._broadcast("position_opened", {
            "position_id": pos.id,
            "pair": pair,
            "direction": direction,
            "size": size,
            "entry_price": entry_price,
        })
        logger.info("Position opened: %s %s %s @ %.5f (id=%d)", direction, size, pair, entry_price, pos.id)
        return True, f"Opened {direction} {size} {pair} @ {entry_price:.5f}", pos

    async def close_position(
        self,
        position_id: int,
        action: str = "CLOSE",
        reasoning: Optional[str] = None,
        source: str = "agent",
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

            pos.status        = "CLOSED"
            pos.close_price   = close_price
            pos.closed_at     = datetime.now(timezone.utc).replace(tzinfo=None)
            pos.realised_pnl  = round(pnl, 2)
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

        await _write_audit(
            source=source, event_type=action,
            pair=pos.pair, direction=pos.direction, size=pos.size,
            entry_price=pos.entry_price, stop_loss=pos.stop_loss,
            position_id=position_id, realised_pnl=round(pnl, 2),
        )

        await self._broadcast("position_closed", {
            "position_id": position_id,
            "pnl": round(pnl, 2),
            "action": action,
        })
        logger.info("Position closed: id=%d action=%s pnl=%.2f", position_id, action, pnl)
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
                if pos.direction == "BUY"  and current <= pos.stop_loss:
                    triggered_action = "SL_HIT"
                elif pos.direction == "SELL" and current >= pos.stop_loss:
                    triggered_action = "SL_HIT"

            if pos.take_profit is not None and triggered_action is None:
                if pos.direction == "BUY"  and current >= pos.take_profit:
                    triggered_action = "TP_HIT"
                elif pos.direction == "SELL" and current <= pos.take_profit:
                    triggered_action = "TP_HIT"

            if triggered_action:
                await self.close_position(
                    pos.id,
                    action=triggered_action,
                    reasoning=f"Auto-triggered: {triggered_action}",
                    source="sl_tp",
                )

    async def _monitor_loop(self, interval: float = 3.0) -> None:
        while True:
            try:
                await portfolio_service.update_unrealised_pnl()
                await self.check_sl_tp()
                # Drawdown circuit-breaker
                state = await portfolio_service.get_state()
                if risk_gate.check_drawdown(state["drawdown_pct"]):
                    await _write_audit(
                        source="system", event_type="DRAWDOWN_BREAKER",
                        gate_allowed=False,
                        gate_reason=risk_gate.status()["kill_switch_reason"],
                        details={"drawdown_pct": state["drawdown_pct"]},
                    )
            except Exception:
                logger.exception("Error in SL/TP monitor loop")
            await asyncio.sleep(interval)

    async def start_monitor(self) -> None:
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitor(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()


# Singleton
order_service = OrderService()

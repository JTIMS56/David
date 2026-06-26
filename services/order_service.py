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
from services.portfolio_service import portfolio_service, _USD_BASE_PAIRS
from services.risk_gate import risk_gate

logger = logging.getLogger("david.order_service")


def _spread_pips(pair: str, bid: float, ask: float) -> float:
    pip = PAIR_CONFIG.get(pair, {}).get("pip", 0.0001)
    return (ask - bid) / pip


def _stop_pips(pair: str, entry_price: float, stop_loss: float) -> float:
    pip = PAIR_CONFIG.get(pair, {}).get("pip", 0.0001)
    return abs(entry_price - stop_loss) / pip


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
        s_pips      = _stop_pips(pair, entry_price, stop_loss) if stop_loss is not None else None

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
            stop_pips=s_pips,
            take_profit=take_profit,
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

        # ── OANDA live execution ───────────────────────────────────────────────
        oanda_trade_id: Optional[str] = None
        if settings.trading_mode == "oanda" and settings.oanda_api_key:
            from services.oanda_client import oanda_client, PAIR_TO_OANDA
            if pair in PAIR_TO_OANDA:
                try:
                    units = int(size) if direction == "BUY" else -int(size)
                    result = await oanda_client.place_market_order(
                        PAIR_TO_OANDA[pair], units, stop_loss, take_profit
                    )
                    fill = result.get("orderFillTransaction", {})
                    cancel = result.get("orderCancelTransaction", {})
                    if cancel:
                        reason = cancel.get("reason", "unknown")
                        logger.warning("OANDA order cancelled: %s", reason)
                        await _write_audit(
                            source=source, event_type="ORDER_REJECT",
                            pair=pair, direction=direction, size=size,
                            entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
                            gate_allowed=True, gate_reason=f"OANDA cancelled: {reason}",
                        )
                        return False, f"OANDA order cancelled: {reason}", None
                    if fill:
                        oanda_trade_id = fill.get("tradeOpened", {}).get("tradeID")
                        fill_price = fill.get("price")
                        _quoted_entry = entry_price  # preserve original quoted price for re-anchor
                        if fill_price:
                            entry_price = float(fill_price)
                        logger.info(
                            "OANDA order filled: trade_id=%s price=%s",
                            oanda_trade_id, fill_price,
                        )
                        # Post-fill geometry check with re-anchor fallback.
                        # Hard-abort only when fill is already past SL (unrecoverable loss).
                        # For all other violations (fill past TP, stop eroded, R:R < 1.0)
                        # attempt to re-anchor SL/TP to the actual fill price, preserving
                        # the original pip distances from the quoted entry.  Abort only if
                        # slippage exceeds 2× the original stop distance (setup is stale).
                        if fill_price:
                            _fp = float(fill_price)
                            _pip = PAIR_CONFIG.get(pair, {}).get("pip", 0.0001)
                            _abort: str | None = None

                            # Hard abort: fill is already past the stop loss
                            if direction == "BUY" and stop_loss is not None and stop_loss >= _fp:
                                _abort = f"fill {_fp:.5f} at or below BUY stop_loss {stop_loss:.5f} — unrecoverable"
                            elif direction == "SELL" and stop_loss is not None and stop_loss <= _fp:
                                _abort = f"fill {_fp:.5f} at or above SELL stop_loss {stop_loss:.5f} — unrecoverable"

                            # Soft violations: attempt re-anchor
                            if _abort is None and stop_loss is not None:
                                _stop_dist = abs(_quoted_entry - stop_loss)
                                _tp_dist = abs(take_profit - _quoted_entry) if take_profit is not None else None
                                _slip_pips = abs(_fp - _quoted_entry) / _pip
                                _needs_reanchor = False

                                # Check if geometry is broken at fill price
                                if direction == "BUY":
                                    if take_profit is not None and take_profit <= _fp:
                                        _needs_reanchor = True
                                else:
                                    if take_profit is not None and take_profit >= _fp:
                                        _needs_reanchor = True

                                # Check if stop eroded below minimum
                                _stop_pips_actual = abs(_fp - stop_loss) / _pip
                                if _stop_pips_actual < settings.min_stop_pips:
                                    _needs_reanchor = True

                                # Check R:R floor
                                if take_profit is not None:
                                    _risk = abs(_fp - stop_loss) / _pip
                                    _reward = abs(take_profit - _fp) / _pip
                                    _rr = _reward / _risk if _risk > 0 else 0.0
                                    if _rr < 1.0:
                                        _needs_reanchor = True

                                if _needs_reanchor:
                                    # Abort if slippage is more than 2× the original stop distance
                                    # (the setup is too stale to salvage)
                                    if _stop_dist > 0 and _slip_pips > (_stop_dist / _pip) * 2:
                                        _abort = (
                                            f"slippage {_slip_pips:.1f}p exceeds 2× stop distance "
                                            f"{_stop_dist / _pip:.1f}p — setup too stale to re-anchor"
                                        )
                                    else:
                                        # Re-anchor: preserve original pip distances, apply to fill price
                                        _new_sl = (_fp - _stop_dist) if direction == "BUY" else (_fp + _stop_dist)
                                        _new_tp = ((_fp + _tp_dist) if direction == "BUY" else (_fp - _tp_dist)) if _tp_dist is not None else take_profit
                                        # Abort if the re-anchored stop is still below minimum
                                        # (original setup was inherently too tight regardless of slippage)
                                        _new_stop_pips = _stop_dist / _pip
                                        if _new_stop_pips < settings.min_stop_pips:
                                            _abort = (
                                                f"re-anchored stop {_new_stop_pips:.1f}p still below "
                                                f"minimum {settings.min_stop_pips:.0f}p — original "
                                                f"setup too tight to salvage"
                                            )
                                        else:
                                            logger.info(
                                                "Post-fill re-anchor [%s %s]: quoted=%.5f fill=%.5f "
                                                "slip=%.1fp — sl %.5f→%.5f tp %s→%s",
                                                direction, pair, _quoted_entry, _fp, _slip_pips,
                                                stop_loss, _new_sl,
                                                f"{take_profit:.5f}" if take_profit is not None else "None",
                                                f"{_new_tp:.5f}" if _new_tp is not None else "None",
                                            )
                                            stop_loss = _new_sl
                                            take_profit = _new_tp

                            if _abort:
                                logger.error(
                                    "Post-fill abort [%s %s]: %s — closing trade immediately",
                                    direction, pair, _abort,
                                )
                                if oanda_trade_id:
                                    try:
                                        await oanda_client.close_trade(oanda_trade_id)
                                    except Exception as _ce:
                                        logger.error("Failed to close aborted OANDA trade: %s", _ce)
                                await _write_audit(
                                    source=source, event_type="ORDER_REJECT",
                                    pair=pair, direction=direction, size=size,
                                    entry_price=_fp, stop_loss=stop_loss, take_profit=take_profit,
                                    gate_allowed=False,
                                    gate_reason=f"Post-fill SL/TP invalid: {_abort}",
                                )
                                return False, f"Order aborted after fill: {_abort}", None
                        # Attach SL/TP to the trade so OANDA enforces them at
                        # broker level — protects position if our server restarts.
                        if oanda_trade_id and (stop_loss is not None or take_profit is not None):
                            oanda_instrument = PAIR_TO_OANDA.get(pair, "")
                            try:
                                await oanda_client.set_trade_orders(
                                    oanda_trade_id=oanda_trade_id,
                                    oanda_instrument=oanda_instrument,
                                    sl_price=stop_loss,
                                    tp_price=take_profit,
                                )
                                logger.info(
                                    "OANDA SL/TP attached: trade_id=%s sl=%s tp=%s",
                                    oanda_trade_id, stop_loss, take_profit,
                                )
                            except Exception as sl_exc:
                                logger.warning(
                                    "OANDA SL/TP attach failed (internal monitor active): %s",
                                    sl_exc,
                                )
                except Exception as exc:
                    logger.error("OANDA order execution failed: %s", exc, exc_info=True)
                    await _write_audit(
                        source=source, event_type="ORDER_REJECT",
                        pair=pair, direction=direction, size=size,
                        entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
                        gate_allowed=True, gate_reason=f"OANDA API error: {exc}",
                    )
                    return False, f"OANDA execution failed: {exc}", None

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
                oanda_trade_id=oanda_trade_id,
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
            pnl: float = 0.0

            # ── OANDA live close ───────────────────────────────────────────────
            if settings.trading_mode == "oanda" and settings.oanda_api_key and pos.oanda_trade_id:
                from services.oanda_client import oanda_client
                try:
                    result = await oanda_client.close_trade(pos.oanda_trade_id)
                    fill = result.get("orderFillTransaction", {})
                    if fill:
                        oanda_close_price = fill.get("price")
                        if oanda_close_price:
                            close_price = float(oanda_close_price)
                        oanda_pl = fill.get("pl")
                        if oanda_pl:
                            pnl = float(oanda_pl)
                    logger.info(
                        "OANDA trade closed: trade_id=%s pnl=%s",
                        pos.oanda_trade_id, pnl,
                    )
                except Exception as exc:
                    logger.error("OANDA close failed: %s", exc, exc_info=True)
                    # Trade may have been closed by OANDA's own SL/TP while our
                    # server was down. Fetch the trade to get the actual fill.
                    try:
                        oanda_trade = await oanda_client.get_trade(pos.oanda_trade_id)
                        if oanda_trade and oanda_trade.get("state") == "CLOSED":
                            avg_close = oanda_trade.get("averageClosePrice")
                            realised_pl = oanda_trade.get("realizedPL")
                            if avg_close:
                                close_price = float(avg_close)
                            if realised_pl:
                                pnl = float(realised_pl)
                            logger.info(
                                "OANDA trade was already closed: price=%s pnl=%s",
                                close_price, pnl,
                            )
                    except Exception:
                        pass  # fall through to paper close at current market price

            # ── Paper PnL calculation (used when no OANDA fill, or OANDA close failed) ─
            if pnl == 0.0:
                if pos.direction == "BUY":
                    raw_pnl = (close_price - pos.entry_price) * pos.size
                else:
                    raw_pnl = (pos.entry_price - close_price) * pos.size
                # USD-base pairs: P&L is in counter currency; divide to get USD.
                if pos.pair in _USD_BASE_PAIRS and close_price > 0:
                    raw_pnl /= close_price
                pnl = raw_pnl

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
                if pos.direction == "BUY" and current >= pos.take_profit:
                    if pos.take_profit > pos.entry_price:
                        triggered_action = "TP_HIT"
                    else:
                        # TP stored below entry (inverted) — treat as SL to close the loss
                        logger.error(
                            "Inverted TP detected [pos %d %s %s]: tp=%.5f <= entry=%.5f — "
                            "closing as SL_HIT",
                            pos.id, pos.direction, pos.pair, pos.take_profit, pos.entry_price,
                        )
                        triggered_action = "SL_HIT"
                elif pos.direction == "SELL" and current <= pos.take_profit:
                    if pos.take_profit < pos.entry_price:
                        triggered_action = "TP_HIT"
                    else:
                        logger.error(
                            "Inverted TP detected [pos %d %s %s]: tp=%.5f >= entry=%.5f — "
                            "closing as SL_HIT",
                            pos.id, pos.direction, pos.pair, pos.take_profit, pos.entry_price,
                        )
                        triggered_action = "SL_HIT"

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

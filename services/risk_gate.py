"""
RiskGate — hard deterministic gate independent of the LLM.

Every call to OrderService.open_position() passes through here unconditionally.
The LLM cannot override, retry past, or work around this gate.
Separate from RiskManager (which is an advisory LLM-facing check in tools.py).

Gate checks (in order):
  1. Kill switch          — absolute veto; no orders when active
  2. Manual-only mode     — blocks agent orders; allows human/sl_tp
  3. Shadow mode          — dry-run; orders logged but not executed
  4. Drawdown breaker     — auto-activates kill switch at max drawdown
  5. Stop-loss required   — mandatory at the service layer, not just advisory
  6. Data freshness       — rejects stale market data (default >30s)
  7. Spread guard         — rejects excessively wide spreads
  8. Basic sanity         — non-negative size, valid direction, valid price
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("david.risk_gate")


def _dir_to_side(direction: str) -> str:
    """Map a forecast direction (UP/DOWN) to the order side that trades with it."""
    return "BUY" if direction == "UP" else "SELL"


def weekend_entry_blocked(now: Optional["datetime"] = None) -> bool:
    """
    True when new entries should be refused because the FX weekend close is
    near or in progress: Friday from weekend_no_entry_from_hour_utc, all of
    Saturday, and Sunday before the ~21:00 UTC market reopen.
    """
    from datetime import datetime, timezone
    from config import settings
    if not settings.weekend_flatten_enabled:
        return False
    now = now or datetime.now(timezone.utc)
    wd = now.weekday()  # Mon=0 .. Sun=6
    if wd == 4 and now.hour >= settings.weekend_no_entry_from_hour_utc:
        return True
    if wd == 5:
        return True
    return wd == 6 and now.hour < 21


def weekend_flatten_due(now: Optional["datetime"] = None) -> bool:
    """True during the Friday flatten window (flatten time -> market close)."""
    from datetime import datetime, timezone
    from config import settings
    if not settings.weekend_flatten_enabled:
        return False
    now = now or datetime.now(timezone.utc)
    if now.weekday() != 4:
        return False
    minutes = now.hour * 60 + now.minute
    start = settings.weekend_flatten_hour_utc * 60 + settings.weekend_flatten_minute_utc
    return start <= minutes < 21 * 60 + 30


@dataclass
class GateDecision:
    allowed: bool
    reason: str
    shadow: bool = False           # True → gate passed but execution is a dry-run
    checks_run: list[str] = field(default_factory=list)


class RiskGate:
    """Thread-safe hard gate. OrderService owns a singleton of this."""

    def __init__(
        self,
        max_spread_pips: float = 5.0,
        data_stale_seconds: float = 30.0,
        max_drawdown_pct: float = 10.0,   # % of peak equity; 10 = 10%
        min_stop_pips: float = 15.0,      # minimum stop distance (pips)
        max_stop_pips: float = 50.0,      # hard cap on stop distance per trade
    ) -> None:
        self._kill_switch        = False
        self._kill_switch_reason = ""
        self._manual_only        = False
        self._shadow_mode        = False
        self._drawdown_triggered = False
        self._max_spread_pips    = max_spread_pips
        self._data_stale_seconds = data_stale_seconds
        self._max_drawdown_pct   = max_drawdown_pct
        self._min_stop_pips      = min_stop_pips
        self._max_stop_pips      = max_stop_pips
        self._min_atr_pips           = 4.0   # volatility floor (pips); wired from settings
        self._min_tp_spread_multiple = 3.0   # require TP distance >= this x spread

    # ── Kill switch ────────────────────────────────────────────────────────────

    def activate_kill_switch(self, reason: str = "Manual") -> None:
        self._kill_switch = True
        self._kill_switch_reason = reason
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)

    def deactivate_kill_switch(self) -> None:
        self._kill_switch = False
        self._kill_switch_reason = ""
        self._drawdown_triggered = False
        logger.warning("Kill switch deactivated — new orders now permitted")

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch

    # ── Manual-only mode ──────────────────────────────────────────────────────

    def set_manual_only(self, enabled: bool) -> None:
        self._manual_only = enabled
        logger.warning("Manual-only mode: %s", "ON" if enabled else "OFF")

    @property
    def manual_only_active(self) -> bool:
        return self._manual_only

    # ── Shadow mode (dry-run) ─────────────────────────────────────────────────

    def set_shadow_mode(self, enabled: bool) -> None:
        self._shadow_mode = enabled
        logger.warning("Shadow mode: %s", "ON (orders will be logged but not executed)" if enabled else "OFF")

    @property
    def shadow_mode_active(self) -> bool:
        return self._shadow_mode

    # ── Drawdown circuit-breaker ───────────────────────────────────────────────

    def check_drawdown(self, drawdown_pct: float) -> bool:
        """
        Call on every portfolio update. Returns True if the breaker just fired.
        Once triggered, only a manual deactivate_kill_switch() can reset it.
        """
        if self._kill_switch or self._drawdown_triggered:
            return False
        if drawdown_pct >= self._max_drawdown_pct:
            self._drawdown_triggered = True
            self.activate_kill_switch(
                f"Drawdown circuit-breaker: {drawdown_pct:.2f}% >= {self._max_drawdown_pct:.1f}% limit"
            )
            return True
        return False

    @property
    def drawdown_triggered(self) -> bool:
        return self._drawdown_triggered

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "kill_switch_active":   self._kill_switch,
            "kill_switch_reason":   self._kill_switch_reason,
            "manual_only_active":   self._manual_only,
            "shadow_mode_active":   self._shadow_mode,
            "drawdown_triggered":   self._drawdown_triggered,
            "max_drawdown_pct":     self._max_drawdown_pct,
            "max_spread_pips":      self._max_spread_pips,
            "min_stop_pips":        self._min_stop_pips,
            "max_stop_pips":        self._max_stop_pips,
            "data_stale_seconds":   self._data_stale_seconds,
        }

    # ── Signal-quality gate ────────────────────────────────────────────────────

    def approve_signal(
        self,
        pair: str,
        direction: str,
        source: str = "agent",
    ) -> GateDecision:
        """
        Enforce ensemble-based signal rules (DHJ is retired from the decision
        path — it proved a coin flip over ~800 evaluations):
          • blocked pairs (chronic range-bound churn)
          • a fresh ensemble forecast must exist for the pair
          • no trading inside a high-impact event blackout window
          • ensemble conviction must reach min_trade_conviction (independent
            voters agreeing: trend, mean-reversion, carry, USD breadth, crowd
            positioning)
          • order direction must match the ensemble's net-vote direction

        Only autonomous agent orders are gated; human/manual orders pass through.
        Reads the most recent forecast from signal_cache, which the forecast tool
        populates synchronously each cycle before the agent can place an order.
        """
        from config import settings
        from services.signal_cache import get_signal

        checks: list[str] = ["signal_gate"]

        if not settings.signal_gate_enabled or source != "agent":
            return GateDecision(True, "Signal gate skipped", checks_run=checks)

        # Blocked-pair list — pure pair check, no forecast required.
        if pair in set(settings.blocked_pairs):
            return GateDecision(
                False,
                f"{pair} is blocked — chronic range-bound churn, net loser in the data",
                checks_run=checks,
            )

        sig = get_signal(pair)
        if sig is None:
            return GateDecision(
                False,
                f"No forecast cached for {pair} — run get_price_forecast before ordering",
                checks_run=checks,
            )

        age = sig.age_seconds()
        if age > settings.signal_max_age_seconds:
            return GateDecision(
                False,
                f"Forecast for {pair} is stale ({age:.0f}s > {settings.signal_max_age_seconds:.0f}s) "
                f"— refresh get_price_forecast before ordering",
                checks_run=checks,
            )

        if sig.event_blackout:
            return GateDecision(
                False,
                f"{pair}: high-impact economic event imminent — release spikes are "
                f"unpredictable; no entries during the blackout window",
                checks_run=checks,
            )

        if sig.conviction < settings.min_trade_conviction:
            return GateDecision(
                False,
                f"{pair}: ensemble conviction {sig.conviction} below minimum "
                f"{settings.min_trade_conviction} — not enough independent signals agree",
                checks_run=checks,
            )

        if sig.direction == "FLAT" or direction != _dir_to_side(sig.direction):
            return GateDecision(
                False,
                f"{pair}: order {direction} does not match ensemble direction "
                f"({sig.direction}) — trade with the ensemble or not at all",
                checks_run=checks,
            )

        return GateDecision(True, "Signal gate passed", checks_run=checks)

    # ── Primary gate ──────────────────────────────────────────────────────────

    def approve_order(
        self,
        pair: str,
        direction: str,
        size: float,
        stop_loss: Optional[float],
        entry_price: float,
        spread_pips: float,
        data_age_seconds: float,
        source: str = "agent",   # "agent" | "human" | "sl_tp"
        stop_pips: Optional[float] = None,
        take_profit: Optional[float] = None,
        atr_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
    ) -> GateDecision:
        """
        Evaluate one order. Returns a GateDecision with allowed=True only when
        all checks pass. Called synchronously inside open_position().
        """
        checks: list[str] = []

        # 1. Kill switch — no exceptions
        checks.append("kill_switch")
        if self._kill_switch:
            return GateDecision(False, f"KILL SWITCH ACTIVE — {self._kill_switch_reason}", checks)

        # 2. Manual-only mode blocks autonomous agent
        checks.append("manual_only")
        if self._manual_only and source == "agent":
            return GateDecision(
                False,
                "System in manual-only mode — agent orders suspended",
                checks_run=checks,
            )

        # 3. Weekend window — no new entries into the Friday close / weekend gap
        checks.append("weekend_window")
        if weekend_entry_blocked():
            return GateDecision(
                False,
                "Weekend window: no new entries from Friday "
                "20:00 UTC until Sunday market open (gap risk)",
                checks_run=checks,
            )

        # 4. Shadow mode — gate passes but execution is skipped by OrderService
        checks.append("shadow_mode")
        if self._shadow_mode:
            return GateDecision(True, "Shadow mode: order logged but not executed", shadow=True, checks_run=checks)

        # 4. Stop-loss required (enforced at service layer, not just advisory)
        checks.append("stop_loss_required")
        if stop_loss is None:
            return GateDecision(False, "Stop-loss is required for every order", checks_run=checks)

        # 4b. SL direction validation — SL must be on the losing side of entry
        checks.append("sl_direction")
        if direction == "BUY" and stop_loss >= entry_price:
            return GateDecision(
                False,
                f"BUY order: stop_loss {stop_loss:.5f} must be BELOW entry {entry_price:.5f}",
                checks_run=checks,
            )
        if direction == "SELL" and stop_loss <= entry_price:
            return GateDecision(
                False,
                f"SELL order: stop_loss {stop_loss:.5f} must be ABOVE entry {entry_price:.5f}",
                checks_run=checks,
            )

        # 4c. TP direction validation — TP must be on the winning side of entry
        checks.append("tp_direction")
        if take_profit is not None:
            if direction == "BUY" and take_profit <= entry_price:
                return GateDecision(
                    False,
                    f"BUY order: take_profit {take_profit:.5f} must be ABOVE entry {entry_price:.5f}",
                    checks_run=checks,
                )
            if direction == "SELL" and take_profit >= entry_price:
                return GateDecision(
                    False,
                    f"SELL order: take_profit {take_profit:.5f} must be BELOW entry {entry_price:.5f}",
                    checks_run=checks,
                )

        # 4d. Stop distance bounds — enforce minimum AND maximum
        checks.append("stop_distance")
        if stop_pips is not None:
            if stop_pips < self._min_stop_pips:
                return GateDecision(
                    False,
                    f"Stop distance {stop_pips:.1f} pips is below minimum {self._min_stop_pips:.1f} pips — "
                    f"widen stop to reduce risk of noise-triggered exits",
                    checks_run=checks,
                )
            if stop_pips > self._max_stop_pips:
                return GateDecision(
                    False,
                    f"Stop distance {stop_pips:.1f} pips exceeds maximum {self._max_stop_pips:.1f} pips — "
                    f"tighten your stop to reduce per-trade risk",
                    checks_run=checks,
                )

        # 5. Data freshness
        checks.append("data_freshness")
        if data_age_seconds > self._data_stale_seconds:
            return GateDecision(
                False,
                f"Market data stale: {data_age_seconds:.0f}s > {self._data_stale_seconds:.0f}s limit",
                checks_run=checks,
            )

        # 6. Spread guard
        checks.append("spread")
        if spread_pips > self._max_spread_pips:
            return GateDecision(
                False,
                f"Spread {spread_pips:.1f} pips exceeds max {self._max_spread_pips:.1f}",
                checks_run=checks,
            )

        # 6b. Volatility / cost floor — don't trade dead markets where the spread
        # eats the signal. Applies to autonomous orders only; humans may override.
        if source == "agent":
            checks.append("volatility_floor")
            if atr_pips is not None and atr_pips < self._min_atr_pips:
                return GateDecision(
                    False,
                    f"ATR {atr_pips:.1f} pips below minimum {self._min_atr_pips:.1f} — "
                    f"market too quiet to clear costs",
                    checks_run=checks,
                )
            if (
                tp_pips is not None and spread_pips > 0
                and tp_pips < self._min_tp_spread_multiple * spread_pips
            ):
                return GateDecision(
                    False,
                    f"take-profit {tp_pips:.1f} pips < {self._min_tp_spread_multiple:.0f}x spread "
                    f"({spread_pips:.1f} pips) — target too small to clear costs",
                    checks_run=checks,
                )

        # 7. Sanity: valid size, direction, price
        checks.append("sanity")
        if size <= 0:
            return GateDecision(False, f"Invalid size: {size}", checks_run=checks)
        if entry_price <= 0:
            return GateDecision(False, f"Invalid price: {entry_price}", checks_run=checks)
        if direction not in ("BUY", "SELL"):
            return GateDecision(False, f"Invalid direction: {direction!r}", checks_run=checks)

        return GateDecision(True, "All gate checks passed", checks_run=checks)


# Singleton — wired by main.py
risk_gate = RiskGate()

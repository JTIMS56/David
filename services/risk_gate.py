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
    ) -> None:
        self._kill_switch        = False
        self._kill_switch_reason = ""
        self._manual_only        = False
        self._shadow_mode        = False
        self._drawdown_triggered = False
        self._max_spread_pips    = max_spread_pips
        self._data_stale_seconds = data_stale_seconds
        self._max_drawdown_pct   = max_drawdown_pct

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
            "data_stale_seconds":   self._data_stale_seconds,
        }

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

        # 3. Shadow mode — gate passes but execution is skipped by OrderService
        checks.append("shadow_mode")
        if self._shadow_mode:
            return GateDecision(True, "Shadow mode: order logged but not executed", shadow=True, checks_run=checks)

        # 4. Stop-loss required (enforced at service layer, not just advisory)
        checks.append("stop_loss_required")
        if stop_loss is None:
            return GateDecision(False, "Stop-loss is required for every order", checks_run=checks)

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

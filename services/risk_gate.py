"""
RiskGate — hard deterministic gate independent of the LLM.

Every call to OrderService.open_position() passes through here unconditionally.
The LLM cannot override, retry past, or work around this gate.
Separate from RiskManager (which is an advisory LLM-facing check in tools.py).

Gate checks (in order):
  1. Kill switch          — absolute veto; no orders when active
  2. Manual-only mode     — blocks agent orders; allows human/sl_tp
  3. Stop-loss required   — mandatory at the service layer, not just advisory
  4. Data freshness       — rejects stale market data (default >30s)
  5. Spread guard         — rejects excessively wide spreads
  6. Basic sanity         — non-negative size, valid direction, valid price
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
    checks_run: list[str] = field(default_factory=list)


class RiskGate:
    """Thread-safe hard gate. OrderService owns a singleton of this."""

    def __init__(
        self,
        max_spread_pips: float = 5.0,
        data_stale_seconds: float = 30.0,
    ) -> None:
        self._kill_switch   = False
        self._manual_only   = False
        self._max_spread_pips    = max_spread_pips
        self._data_stale_seconds = data_stale_seconds

    # ── Kill switch ────────────────────────────────────────────────────────────

    def activate_kill_switch(self, reason: str = "Manual") -> None:
        self._kill_switch = True
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)

    def deactivate_kill_switch(self) -> None:
        self._kill_switch = False
        logger.warning("Kill switch deactivated — new orders now permitted")

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch

    def set_manual_only(self, enabled: bool) -> None:
        self._manual_only = enabled
        logger.warning("Manual-only mode: %s", "ON" if enabled else "OFF")

    @property
    def manual_only_active(self) -> bool:
        return self._manual_only

    def status(self) -> dict:
        return {
            "kill_switch_active": self._kill_switch,
            "manual_only_active": self._manual_only,
            "max_spread_pips":    self._max_spread_pips,
            "data_stale_seconds": self._data_stale_seconds,
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
            return GateDecision(False, "KILL SWITCH ACTIVE — all new orders blocked", checks)

        # 2. Manual-only mode blocks autonomous agent
        checks.append("manual_only")
        if self._manual_only and source == "agent":
            return GateDecision(
                False,
                "System in manual-only mode — agent orders suspended",
                checks,
            )

        # 3. Stop-loss required (enforced at service layer, not just advisory)
        checks.append("stop_loss_required")
        if stop_loss is None:
            return GateDecision(False, "Stop-loss is required for every order", checks)

        # 4. Data freshness
        checks.append("data_freshness")
        if data_age_seconds > self._data_stale_seconds:
            return GateDecision(
                False,
                f"Market data stale: {data_age_seconds:.0f}s > {self._data_stale_seconds:.0f}s limit",
                checks,
            )

        # 5. Spread guard
        checks.append("spread")
        if spread_pips > self._max_spread_pips:
            return GateDecision(
                False,
                f"Spread {spread_pips:.1f} pips exceeds max {self._max_spread_pips:.1f}",
                checks,
            )

        # 6. Sanity: valid size, direction, price
        checks.append("sanity")
        if size <= 0:
            return GateDecision(False, f"Invalid size: {size}", checks)
        if entry_price <= 0:
            return GateDecision(False, f"Invalid price: {entry_price}", checks)
        if direction not in ("BUY", "SELL"):
            return GateDecision(False, f"Invalid direction: {direction!r}", checks)

        return GateDecision(True, "All gate checks passed", checks)


# Singleton — wired by main.py
risk_gate = RiskGate()

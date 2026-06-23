"""
Risk Manager
────────────────────────────────────────────────────────────────────────────
Enforces pre-trade and portfolio-level risk rules before any order is placed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from config import settings

if TYPE_CHECKING:
    from services.portfolio_service import PortfolioService


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str


class RiskManager:
    def __init__(self, portfolio: "PortfolioService") -> None:
        self._portfolio = portfolio

    # ── Pre-trade checks ──────────────────────────────────────────────────────

    async def check_new_order(
        self,
        pair: str,
        direction: str,
        size: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> RiskCheckResult:
        state = await self._portfolio.get_state()

        # 1. Daily loss limit
        if state["daily_pnl"] < -(state["balance"] * settings.max_daily_loss_pct):
            return RiskCheckResult(
                False,
                f"Daily loss limit hit: {state['daily_pnl']:.2f} exceeds "
                f"{settings.max_daily_loss_pct*100:.0f}% of balance.",
            )

        # 2. Max open positions
        if state["open_positions"] >= settings.max_open_positions:
            return RiskCheckResult(
                False,
                f"Max open positions reached ({settings.max_open_positions}).",
            )

        # 3. Position size limit
        notional = size * entry_price
        max_notional = state["balance"] * settings.max_position_size_pct
        if notional > max_notional:
            return RiskCheckResult(
                False,
                f"Position size {notional:.2f} exceeds max {max_notional:.2f} "
                f"({settings.max_position_size_pct*100:.0f}% of balance).",
            )

        # 4. Total exposure limit
        total_exposure = state["total_notional"] + notional
        max_exposure = state["balance"] * settings.max_total_exposure_pct
        if total_exposure > max_exposure:
            return RiskCheckResult(
                False,
                f"Total exposure {total_exposure:.2f} would exceed max {max_exposure:.2f}.",
            )

        # 5. Stop-loss required
        if stop_loss is None:
            return RiskCheckResult(False, "Stop-loss is required for all orders.")

        # 6. SL must be on the correct side of entry
        if direction == "BUY" and stop_loss >= entry_price:
            return RiskCheckResult(
                False,
                f"BUY stop_loss {stop_loss:.5f} must be below entry {entry_price:.5f}.",
            )
        if direction == "SELL" and stop_loss <= entry_price:
            return RiskCheckResult(
                False,
                f"SELL stop_loss {stop_loss:.5f} must be above entry {entry_price:.5f}.",
            )

        # 7. Minimum stop distance
        from services.market_data import market_data
        pip = market_data.get_pip_size(pair)
        stop_pips = abs(entry_price - stop_loss) / pip
        if stop_pips < settings.min_stop_pips:
            return RiskCheckResult(
                False,
                f"Stop distance {stop_pips:.1f} pips below minimum {settings.min_stop_pips:.0f} pips.",
            )

        # 8. TP direction + minimum R:R check (using signed distances)
        if take_profit is not None:
            if direction == "BUY" and take_profit <= entry_price:
                return RiskCheckResult(
                    False,
                    f"BUY take_profit {take_profit:.5f} must be above entry {entry_price:.5f}.",
                )
            if direction == "SELL" and take_profit >= entry_price:
                return RiskCheckResult(
                    False,
                    f"SELL take_profit {take_profit:.5f} must be below entry {entry_price:.5f}.",
                )
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            if risk > 0 and (reward / risk) < settings.default_risk_reward:
                return RiskCheckResult(
                    False,
                    f"Risk:Reward {reward/risk:.2f} below minimum {settings.default_risk_reward}.",
                )

        return RiskCheckResult(True, "All risk checks passed.")

    # ── Sizing helper ─────────────────────────────────────────────────────────

    async def calculate_position_size(
        self, pair: str, entry_price: float, stop_loss: float, risk_pct: float = 0.01
    ) -> float:
        """
        Size in units so that hitting stop_loss loses exactly risk_pct of balance.
        Returns a whole number of units (1 unit = 1 currency unit).
        """
        from services.market_data import market_data

        state = await self._portfolio.get_state()
        risk_amount = state["balance"] * risk_pct
        pip_size = market_data.get_pip_size(pair)
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance < pip_size:
            stop_distance = pip_size * settings.default_stop_pips

        # pip value per unit (simplified — USD as quote or approx)
        pip_value_per_unit = pip_size / entry_price if "JPY" not in pair else pip_size
        units = risk_amount / (stop_distance * (1 / entry_price if entry_price > 1 else 1))
        # Cap to max position size
        max_units = (state["balance"] * settings.max_position_size_pct) / entry_price
        return round(min(units, max_units))


# Singleton (injected after portfolio is constructed)
risk_manager: Optional[RiskManager] = None

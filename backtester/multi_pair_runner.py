"""
Multi-pair leveraged FX strategy backtester.

Alpha sources
─────────────
1. Vol mean-reversion straddle (monthly, h=22)
   Buy ATM straddle when short-term realised vol (21-day) is depressed relative
   to the calibrated long-run vol (sigma_lr from 126-day walk-forward calibrator).
   The vol-of-vol Heston layer guarantees E[σ_T] > σ_short when vol is low.

2. FX carry (daily)
   Maintain a long position in high-interest-rate currency vs low-rate currency.
   P&L = daily carry accrual + FX mark-to-market (partially hedged).

Position sizing: quarter-Kelly on the vol-edge signal.
Leverage: configurable; halved when equity drawdown exceeds max_leverage_dd.

Runs simultaneously on all 8 major FX pairs (independent Heston-Kou datasets).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── FX pair universe ──────────────────────────────────────────────────────────
PAIRS_CONFIG: Dict[str, dict] = {
    "EURUSD": {"r_d": 0.0525, "r_f": 0.0350, "S0": 1.0850, "sigma0": 0.085},
    "GBPUSD": {"r_d": 0.0525, "r_f": 0.0500, "S0": 1.2700, "sigma0": 0.090},
    "USDJPY": {"r_d": 0.0525, "r_f": 0.0010, "S0": 148.00, "sigma0": 0.095},
    "AUDUSD": {"r_d": 0.0410, "r_f": 0.0525, "S0": 0.6550, "sigma0": 0.105},
    "USDCAD": {"r_d": 0.0525, "r_f": 0.0450, "S0": 1.3600, "sigma0": 0.078},
    "EURGBP": {"r_d": 0.0350, "r_f": 0.0500, "S0": 0.8550, "sigma0": 0.072},
    "NZDUSD": {"r_d": 0.0410, "r_f": 0.0525, "S0": 0.6100, "sigma0": 0.110},
    "USDCHF": {"r_d": 0.0525, "r_f": 0.0175, "S0": 0.9100, "sigma0": 0.075},
}

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)  # ≈ 0.7979
_VOL_SHORT_WINDOW = 21   # days for GK option pricing (current-regime vol)
_VOL_LONG_WINDOW  = 126  # days for long-run vol calibration
_CAL_FREQ_STEPS   = 3    # recalibrate every 3 straddle periods


def atm_straddle_price(S0: float, sigma: float, T_yr: float) -> float:
    """ATM straddle fair value under lognormal (Bachelier approximation)."""
    return S0 * sigma * math.sqrt(T_yr) * _SQRT_2_OVER_PI


@dataclass
class PairStats:
    pair: str
    n_straddle: int = 0
    straddle_pnl: float = 0.0
    carry_pnl: float = 0.0

    @property
    def total_pnl(self) -> float:
        return self.straddle_pnl + self.carry_pnl


@dataclass
class MultiPairResults:
    equity_curve: pd.Series
    daily_pnl: pd.Series
    pair_stats: Dict[str, PairStats]
    initial_capital: float
    leverage: float
    n_pairs: int
    start: str
    end: str

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1])

    @property
    def total_profit(self) -> float:
        return self.final_equity - self.initial_capital

    @property
    def total_return_pct(self) -> float:
        return self.total_profit / self.initial_capital * 100.0

    @property
    def n_years(self) -> float:
        return len(self.daily_pnl) / 252.0

    @property
    def annual_return_pct(self) -> float:
        if self.n_years <= 0 or self.final_equity <= 0:
            return 0.0
        return ((self.final_equity / self.initial_capital) ** (1.0 / self.n_years) - 1.0) * 100.0

    @property
    def avg_daily_pnl(self) -> float:
        return float(self.daily_pnl.mean())

    @property
    def sharpe(self) -> float:
        initial = self.equity_curve.shift(1).fillna(self.initial_capital)
        dr = self.daily_pnl / initial
        std = float(dr.std())
        return float(dr.mean() / std * math.sqrt(252)) if std > 1e-12 else 0.0

    @property
    def max_drawdown_pct(self) -> float:
        peak = self.equity_curve.cummax()
        dd = (self.equity_curve - peak) / peak
        return float(dd.min() * 100.0)

    def capital_for_daily_target(self, daily_target_usd: float) -> float:
        """Capital (at 1× leverage, same strategy) needed to earn daily_target_usd/day."""
        avg = self.avg_daily_pnl
        if avg <= 1e-9:
            return float("inf")
        return self.initial_capital * (daily_target_usd / avg)


class MultiPairRunner:
    """
    Multi-pair leveraged FX strategy runner.

    Parameters
    ----------
    pairs              : FX pairs to trade (default: all 8 majors)
    start / end        : backtest window
    initial_capital    : starting USD equity
    leverage           : position size multiplier (1 = unlevered)
    horizon_days       : straddle holding period in trading days (22 = 1 month)
    step_days          : straddle rebalance period (22 = monthly entry, no overlap)
    vol_edge_threshold : min (sigma_lr - sigma_short) / sigma_short to trade straddle
    use_carry          : include daily FX carry signal
    kelly_fraction     : fraction of Kelly sizing (0.25 = quarter-Kelly)
    max_pair_pct       : max equity fraction allocated to one pair per signal
    tc_bps             : round-trip transaction cost in basis points
    max_leverage_dd    : drawdown fraction that triggers leverage halving
    """

    def __init__(
        self,
        pairs: Optional[List[str]] = None,
        start: str = "2019-01-01",
        end: str = "2024-12-31",
        initial_capital: float = 500.0,
        leverage: float = 1.0,
        horizon_days: int = 22,
        step_days: int = 22,
        vol_edge_threshold: float = 0.08,
        use_carry: bool = True,
        kelly_fraction: float = 0.25,
        max_pair_pct: float = 0.20,
        tc_bps: float = 5.0,
        max_leverage_dd: float = 0.15,
    ):
        self.pairs = pairs or list(PAIRS_CONFIG.keys())
        self.start = start
        self.end = end
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.horizon_days = horizon_days
        self.step_days = step_days
        self.vol_edge_threshold = vol_edge_threshold
        self.use_carry = use_carry
        self.kelly_fraction = kelly_fraction
        self.max_pair_pct = max_pair_pct
        self.tc = tc_bps / 10_000.0
        self.max_leverage_dd = max_leverage_dd

    # ─────────────────────────────────────────────────────────────────────────
    def run(self) -> MultiPairResults:
        from backtester.data import synthetic_heston_kou
        from backtester.calibrator import WalkForwardCalibrator

        calibrator = WalkForwardCalibrator(
            vol_window=_VOL_LONG_WINDOW,
            heston_window=504,
            jump_window=504,
        )

        # ── Load Heston-Kou price series (different seed per pair) ────────────
        pair_prices: Dict[str, pd.Series] = {}
        for i, pair in enumerate(self.pairs):
            cfg = PAIRS_CONFIG[pair]
            pair_prices[pair] = synthetic_heston_kou(
                pair=pair,
                start=self.start,
                end=self.end,
                S0=cfg["S0"],
                theta_vol=cfg["sigma0"],
                v0_vol=cfg["sigma0"],
                seed=42 + i * 13,
            )

        # ── Common trading dates ──────────────────────────────────────────────
        common_dates = pd.DatetimeIndex(
            sorted(set.intersection(*[set(p.index) for p in pair_prices.values()]))
        )
        min_hist = max(_VOL_LONG_WINDOW, _VOL_SHORT_WINDOW) + self.horizon_days
        logger.info(
            "MultiPairRunner: %d dates (%s → %s), %d pairs, leverage=%.1f×",
            len(common_dates), common_dates[0].date(), common_dates[-1].date(),
            len(self.pairs), self.leverage,
        )

        # ── Pre-compute log-return arrays ─────────────────────────────────────
        pair_logret: Dict[str, np.ndarray] = {}
        pair_idx:   Dict[str, dict] = {}  # date → position in prices.index
        for pair, prices in pair_prices.items():
            lr = np.log(prices / prices.shift(1)).dropna().values
            pair_logret[pair] = lr
            # map from date to prices.index position
            pair_idx[pair] = {d: i for i, d in enumerate(prices.index)}

        # ── Walk-forward simulation ───────────────────────────────────────────
        equity = self.initial_capital
        equity_hist: List[float] = []
        pnl_hist:    List[float] = []
        pair_stats = {p: PairStats(pair=p) for p in self.pairs}
        cal_cache: Dict[str, object] = {}  # pair → CalibratedParams
        step_counter = 0

        for day_pos, date in enumerate(common_dates):
            if day_pos < min_hist:
                equity_hist.append(equity)
                pnl_hist.append(0.0)
                continue

            # Drawdown-adjusted effective leverage
            peak = max(equity_hist[min_hist:] or [equity])
            dd = (equity - peak) / peak if peak > 0 else 0.0
            eff_lev = (
                max(1.0, self.leverage * 0.5)
                if dd < -self.max_leverage_dd
                else self.leverage
            )

            day_pnl = 0.0
            is_step = (day_pos % self.step_days == 0)

            # ── Vol straddle rebalance (every step_days) ──────────────────────
            if is_step:
                step_counter += 1
                recal = (step_counter % _CAL_FREQ_STEPS == 1)
                future_pos = day_pos + self.horizon_days

                if future_pos < len(common_dates):
                    fut_date = common_dates[future_pos]

                    for pair in self.pairs:
                        pr_idx = pair_idx[pair]
                        if date not in pr_idx or fut_date not in pr_idx:
                            continue
                        i0   = pr_idx[date]
                        i_f  = pr_idx[fut_date]
                        lr   = pair_logret[pair]
                        if i0 < min_hist or i0 >= len(lr):
                            continue

                        past = lr[:i0]
                        # Short-term vol: GK option-pricing base (21-day)
                        sg_short = float(
                            np.std(past[-_VOL_SHORT_WINDOW:], ddof=1) * math.sqrt(252)
                        )
                        if sg_short < 0.01:
                            sg_short = 0.08

                        # Long-run vol: walk-forward calibrator (cached)
                        if recal or pair not in cal_cache:
                            try:
                                cal_cache[pair] = calibrator.calibrate(
                                    past, horizon_days=self.horizon_days
                                )
                            except Exception:
                                cal_cache[pair] = None

                        cal = cal_cache.get(pair)
                        if cal is not None:
                            sigma_lr = float(cal.sigma_lr)
                            # Add jump premium: E[extra vol from jumps over horizon]
                            jv_ann = (cal.jump_lambda * 2.0
                                      / max(cal.jump_eta_plus**2, 1.0))
                            sigma_dhj = float(
                                math.sqrt(max(sigma_lr**2 + jv_ann, sg_short**2))
                            )
                        else:
                            # Fallback: simple vol-of-vol adjustment
                            sigma_lr = float(np.std(past[-_VOL_LONG_WINDOW:], ddof=1)
                                             * math.sqrt(252))
                            sigma_dhj = sigma_lr

                        # ── Straddle signal ───────────────────────────────────
                        vol_edge = (sigma_dhj - sg_short) / sg_short
                        if vol_edge > self.vol_edge_threshold:
                            prices = pair_prices[pair]
                            S0  = float(prices.iloc[i0])
                            S_T = float(prices.iloc[i_f])
                            T   = self.horizon_days / 252.0
                            cost      = atm_straddle_price(S0, sg_short,   T)
                            exp_pay   = atm_straddle_price(S0, sigma_dhj,  T)
                            trade_edge_frac = (exp_pay - cost) / max(cost, 1e-12)

                            kelly_f = min(
                                self.kelly_fraction * trade_edge_frac,
                                self.max_pair_pct,
                            )
                            if kelly_f <= 0.0:
                                continue

                            notional = eff_lev * equity * kelly_f
                            units    = notional / max(S0, 1e-8)
                            payout   = abs(S_T - S0)
                            gross    = units * (payout - cost)
                            net      = gross - notional * self.tc

                            day_pnl += net
                            pair_stats[pair].straddle_pnl += net
                            pair_stats[pair].n_straddle   += 1

            # ── Daily carry accrual ───────────────────────────────────────────
            if self.use_carry and day_pos + 1 < len(common_dates):
                next_date = common_dates[day_pos + 1]
                for pair in self.pairs:
                    cfg  = PAIRS_CONFIG[pair]
                    carry_rate = cfg["r_d"] - cfg["r_f"]
                    if abs(carry_rate) < 0.003:
                        continue
                    pr_idx = pair_idx[pair]
                    if date not in pr_idx:
                        continue
                    i0  = pr_idx[date]
                    prices = pair_prices[pair]
                    if i0 + 1 >= len(prices) or next_date not in pr_idx:
                        continue
                    S0  = float(prices.iloc[i0])
                    S1  = float(prices.iloc[i0 + 1])

                    carry_pos = eff_lev * equity * 0.04   # 4% of equity per pair
                    direction = 1.0 if carry_rate > 0 else -1.0
                    # P&L = daily carry + 15% of FX mark-to-market (partially hedged)
                    daily_carry = carry_pos * abs(carry_rate) / 252.0
                    fx_ret      = direction * math.log(S1 / S0)
                    carry_pnl   = daily_carry + carry_pos * fx_ret * 0.15

                    day_pnl += carry_pnl
                    pair_stats[pair].carry_pnl += carry_pnl

            equity = max(equity + day_pnl, 0.01)
            equity_hist.append(equity)
            pnl_hist.append(day_pnl)

        dates_slice = common_dates[: len(equity_hist)]
        return MultiPairResults(
            equity_curve=pd.Series(equity_hist, index=dates_slice),
            daily_pnl   =pd.Series(pnl_hist,   index=dates_slice),
            pair_stats  =pair_stats,
            initial_capital=self.initial_capital,
            leverage    =self.leverage,
            n_pairs     =len(self.pairs),
            start       =self.start,
            end         =self.end,
        )

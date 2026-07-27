"""
Ensemble Direction Model  (shadow mode)
────────────────────────────────────────────────────────────────────────────
An independent alternative to the RSI-seeded DHJ model. It combines four
deliberately INDEPENDENT weak signals and only expresses conviction when they
agree — the idea being that several uncorrelated ~52% signals voting together
concentrate edge into the high-agreement subset.

The four votes (each -1 / 0 / +1):
  1. trend        — MACD histogram + price vs SMA50 (momentum / trend-following)
  2. mean_revert  — RSI + Bollinger %B extremes (counter-trend reversion)
  3. carry        — central-bank rate differential (structural drift)
  4. usd_strength — cross-sectional USD breadth across all pairs

net = sum(votes); direction = sign(net); conviction = |net|.

This module is pure/stateless — it reads live indicators and returns a
prediction. It is logged alongside DHJ for out-of-sample comparison and is NOT
wired into the trading path until it beats DHJ on data it has never seen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from config import settings
from services.market_data import market_data


@dataclass
class EnsembleForecast:
    pair: str
    spot: float
    direction: str               # "UP" | "DOWN" | "FLAT"
    net_vote: int
    conviction: int              # abs(net_vote)
    high_conviction: bool
    votes: Dict[str, int] = field(default_factory=dict)
    event_blackout: bool = False # high-impact event imminent → conviction zeroed


def _trend_vote(ind: dict) -> int:
    hist = ind.get("macd_histogram", 0.0)
    price = ind.get("current_price", 0.0)
    sma50 = ind.get("sma50", 0.0)
    if hist > 0 and price > sma50:
        return 1
    if hist < 0 and price < sma50:
        return -1
    return 0


def _mean_revert_vote(ind: dict) -> int:
    rsi = ind.get("rsi", 50.0)
    pct_b = ind.get("bb_pct_b", 0.5)
    # Oversold → expect a bounce UP; overbought → expect a fade DOWN.
    if rsi < 30 or pct_b < 0.05:
        return 1
    if rsi > 70 or pct_b > 0.95:
        return -1
    return 0


def _carry_vote(pair: str) -> int:
    # Non-FX instruments have no rate differential. Index/metal CFD carry
    # (dividends minus financing) is a fraction of a basis point per day —
    # noise at this horizon — so this voter abstains rather than contributing
    # a permanent structural tilt.
    from services.market_data import asset_class
    if asset_class(pair) != "fx":
        return 0
    try:
        base, quote = pair.split("/")
    except ValueError:
        return 0
    rates = settings.policy_rates
    if base not in rates or quote not in rates:
        return 0
    diff = rates[base] - rates[quote]          # positive → base earns more → base appreciates
    if diff >= settings.carry_diff_threshold:
        return 1
    if diff <= -settings.carry_diff_threshold:
        return -1
    return 0


def _usd_strength_score() -> float:
    """
    Cross-sectional USD breadth in [-1, 1]: positive = USD broadly strengthening.
    Each USD pair contributes the sign of (price - SMA20), oriented so that
    'USD up' is always +1 regardless of whether USD is the base or quote.
    """
    from services.market_data import asset_class
    votes = []
    for p in settings.default_pairs:
        # FX only: XAU/USD contains "USD" but gold is not a currency leg, and
        # including it would pollute a clean currency-breadth measure.
        if "USD" not in p or asset_class(p) != "fx":
            continue
        ind = market_data.calculate_indicators(p)
        if not ind:
            continue
        price = ind.get("current_price", 0.0)
        sma20 = ind.get("sma20", 0.0)
        if sma20 <= 0:
            continue
        mom = 1 if price > sma20 else -1 if price < sma20 else 0
        base = p.split("/")[0]
        # USD/X up → USD strong; X/USD up → USD weak.
        votes.append(mom if base == "USD" else -mom)
    if not votes:
        return 0.0
    return sum(votes) / len(votes)


def _usd_strength_vote(pair: str, usd_score: float) -> int:
    # USD breadth describes currency-vs-currency flow. An index or metal is not
    # a currency pair, so the base/quote logic below does not apply.
    from services.market_data import asset_class
    if asset_class(pair) != "fx":
        return 0
    if "USD" not in pair or abs(usd_score) < 0.25:   # need broad agreement, not 1 pair
        return 0
    base = pair.split("/")[0]
    usd_up = usd_score > 0
    if base == "USD":            # USD/X moves WITH USD
        return 1 if usd_up else -1
    return -1 if usd_up else 1   # X/USD moves AGAINST USD


def predict(pair: str, usd_score: Optional[float] = None) -> Optional[EnsembleForecast]:
    """Compute the ensemble forecast for one pair, or None if data is unavailable."""
    ind = market_data.calculate_indicators(pair)
    if not ind:
        return None
    spot = ind.get("current_price")
    if not spot:
        return None
    if usd_score is None:
        usd_score = _usd_strength_score()

    from services.sentiment import positioning_vote
    from services.econ_calendar import is_blackout

    votes = {
        "trend":        _trend_vote(ind),
        "mean_revert":  _mean_revert_vote(ind),
        "carry":        _carry_vote(pair),
        "usd_strength": _usd_strength_vote(pair, usd_score),
        # Contrarian crowd vote from OANDA's position book — information about
        # participants, not another transformation of price. 0 when no data.
        "positioning":  positioning_vote(pair),
    }
    net = sum(votes.values())
    direction = "UP" if net > 0 else "DOWN" if net < 0 else "FLAT"
    conviction = abs(net)

    # Event blackout: a high-impact scheduled release for either currency is
    # imminent. Spikes around releases are unpredictable from any of our
    # signals, so conviction is zeroed — the honest call is "don't know".
    blackout = is_blackout(pair) is not None
    if blackout:
        direction = "FLAT"
        conviction = 0

    return EnsembleForecast(
        pair=pair,
        spot=round(spot, 6),
        direction=direction,
        net_vote=net,
        conviction=conviction,
        high_conviction=(not blackout) and conviction >= settings.ensemble_high_conviction,
        votes=votes,
        event_blackout=blackout,
    )

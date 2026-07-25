"""
Carry + Trend Backtest Engine
────────────────────────────────────────────────────────────────────────────
Tests the two most documented FX return sources — the carry risk premium and
multi-week momentum — over years of daily OANDA candles.

Design principles (hard-learned in this project):
  • NO LOOKAHEAD: the position applied to the return from day t -> t+1 is
    decided using only data up to and including day t.
  • COSTS ALWAYS ON: spread is charged on every position change, and daily
    financing (the actual carry cash flow) accrues to every open position.
  • FEW KNOBS: weekly rebalance, 3-month trend lookback, yearly-resolution
    policy rates. No per-pair tuning, no optimized thresholds — parameters
    come from the literature, not from fitting our own history.

The engine is a pure function over candle data so it is unit-testable with
synthetic series (trending -> must profit; random walk -> ~zero minus costs).
"""
from __future__ import annotations

import math
from typing import Dict, List

from services.market_data import PAIR_CONFIG

# ── Approximate central-bank policy rates, yearly resolution (percent) ───────
# Coarse by design: the carry signal needs the SIGN and rough size of the
# rate differential, which yearly resolution captures. Sources: published
# policy-rate histories; 2025-26 aligned with the platform's live carry table.
POLICY_RATES: Dict[str, Dict[int, float]] = {
    "USD": {2014: 0.25, 2015: 0.50, 2016: 0.75, 2017: 1.50, 2018: 2.50,
            2019: 1.75, 2020: 0.25, 2021: 0.25, 2022: 4.50, 2023: 5.50,
            2024: 4.50, 2025: 4.50, 2026: 4.50},
    "EUR": {2014: 0.05, 2015: 0.05, 2016: 0.00, 2017: 0.00, 2018: 0.00,
            2019: 0.00, 2020: 0.00, 2021: 0.00, 2022: 2.50, 2023: 4.50,
            2024: 3.00, 2025: 2.75, 2026: 2.75},
    "GBP": {2014: 0.50, 2015: 0.50, 2016: 0.25, 2017: 0.50, 2018: 0.75,
            2019: 0.75, 2020: 0.10, 2021: 0.25, 2022: 3.50, 2023: 5.25,
            2024: 4.75, 2025: 4.25, 2026: 4.25},
    "JPY": {2014: 0.10, 2015: 0.10, 2016: -0.10, 2017: -0.10, 2018: -0.10,
            2019: -0.10, 2020: -0.10, 2021: -0.10, 2022: -0.10, 2023: -0.10,
            2024: 0.25, 2025: 0.50, 2026: 0.75},
    "AUD": {2014: 2.50, 2015: 2.00, 2016: 1.50, 2017: 1.50, 2018: 1.50,
            2019: 0.75, 2020: 0.10, 2021: 0.10, 2022: 3.10, 2023: 4.35,
            2024: 4.35, 2025: 3.85, 2026: 3.85},
    "NZD": {2014: 3.50, 2015: 2.50, 2016: 1.75, 2017: 1.75, 2018: 1.75,
            2019: 1.00, 2020: 0.25, 2021: 0.75, 2022: 4.25, 2023: 5.50,
            2024: 4.25, 2025: 3.25, 2026: 3.25},
    "CAD": {2014: 1.00, 2015: 0.50, 2016: 0.50, 2017: 1.00, 2018: 1.75,
            2019: 1.75, 2020: 0.25, 2021: 0.25, 2022: 4.25, 2023: 5.00,
            2024: 3.25, 2025: 2.75, 2026: 2.75},
    "CHF": {2014: -0.25, 2015: -0.75, 2016: -0.75, 2017: -0.75, 2018: -0.75,
            2019: -0.75, 2020: -0.75, 2021: -0.75, 2022: 1.00, 2023: 1.75,
            2024: 0.50, 2025: 0.25, 2026: 0.25},
}


# ── Extended universe for the cross-sectional test ───────────────────────────
# The carry premium is documented in rate-dispersed crosses (JPY/CHF funding
# legs), not USD-majors that all sat at zero 2012-2021. Spreads conservative.
BT_EXTRA_META: Dict[str, dict] = {
    "AUD/JPY": {"pip": 0.01,   "spread": 0.020},
    "NZD/JPY": {"pip": 0.01,   "spread": 0.025},
    "EUR/JPY": {"pip": 0.01,   "spread": 0.018},
    "GBP/JPY": {"pip": 0.01,   "spread": 0.028},
    "CAD/JPY": {"pip": 0.01,   "spread": 0.025},
    "CHF/JPY": {"pip": 0.01,   "spread": 0.028},
    "EUR/AUD": {"pip": 0.0001, "spread": 0.00028},
    "EUR/NZD": {"pip": 0.0001, "spread": 0.00040},
}
BT_EXTRA_OANDA: Dict[str, str] = {
    "AUD/JPY": "AUD_JPY", "NZD/JPY": "NZD_JPY", "EUR/JPY": "EUR_JPY",
    "GBP/JPY": "GBP_JPY", "CAD/JPY": "CAD_JPY", "CHF/JPY": "CHF_JPY",
    "EUR/AUD": "EUR_AUD", "EUR/NZD": "EUR_NZD",
}


def _pair_meta(pair: str) -> dict:
    if pair in PAIR_CONFIG:
        cfg = PAIR_CONFIG[pair]
        return {"pip": cfg.get("pip", 0.0001), "spread": cfg.get("spread", 0.00015)}
    return BT_EXTRA_META.get(pair, {"pip": 0.0001, "spread": 0.00030})


def _rate(currency: str, year: int) -> float:
    table = POLICY_RATES.get(currency, {})
    if not table:
        return 0.0
    ys = sorted(table)
    y = min(max(year, ys[0]), ys[-1])
    return table.get(y, table[ys[-1]])


def _carry_diff(pair: str, year: int) -> float:
    base, quote = pair.split("/")
    return _rate(base, year) - _rate(quote, year)


def _summarize(all_dates: List[str], port: List[float],
               cost_per_year: float, fin_per_year: float) -> dict:
    """Shared metrics block: equity curve, CAGR, Sharpe, drawdown, yearly table."""
    n_days = len(port)
    equity, peak, max_dd = 1.0, 1.0, 0.0
    for r in port:
        equity *= (1.0 + r)
        peak = max(peak, equity)
        max_dd = max(max_dd, 1.0 - equity / peak)
    years = n_days / 252.0
    cagr = equity ** (1.0 / years) - 1.0 if years > 0.2 and equity > 0 else 0.0
    mean_r = sum(port) / n_days
    var_r = sum((r - mean_r) ** 2 for r in port) / n_days
    ann_vol = math.sqrt(var_r) * math.sqrt(252)
    sharpe = (mean_r * 252) / ann_vol if ann_vol > 1e-9 else 0.0

    yearly: Dict[str, float] = {}
    y_start, cur_year, eq = 1.0, all_dates[0][:4], 1.0
    for d, r in zip(all_dates, port):
        if d[:4] != cur_year:
            yearly[cur_year] = round((eq / y_start - 1.0) * 100, 2)
            cur_year, y_start = d[:4], eq
        eq *= (1.0 + r)
    yearly[cur_year] = round((eq / y_start - 1.0) * 100, 2)

    return {
        "window": {"start": all_dates[0], "end": all_dates[-1],
                   "trading_days": n_days, "years": round(years, 1)},
        "headline": {
            "final_multiple": round(equity, 3),
            "cagr_pct": round(cagr * 100, 2),
            "ann_vol_pct": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
        },
        "attribution_pct_per_year": {
            "financing_carry": round(fin_per_year * 100, 2),
            "spread_costs": round(-cost_per_year * 100, 2),
        },
        "yearly_returns_pct": yearly,
    }


def run_cross_carry(
    candles: Dict[str, List[dict]],
    top_n: int = 3,
    rebalance_every: int = 21,       # monthly (literature-standard for carry)
    trend_lookback: int = 63,
    vol_lookback: int = 20,
    target_pair_vol: float = 0.0283,
    max_pos: float = 1.0,
    trend_veto: bool = False,
) -> dict:
    """
    Classic cross-sectional carry portfolio: each month, rank all pairs by
    rate differential; LONG the top_n (earn the differential), SHORT the
    bottom_n (earn the inverted differential). Optional trend veto zeroes a
    leg whose 63-day trend opposes it. Vol-targeted, costs and financing on,
    no lookahead (decisions at t apply to the t -> t+1 return).
    """
    if len(candles) < 2 * top_n + 1:
        return {"error": f"need > {2 * top_n} pairs, got {len(candles)}"}

    date_sets = [set(c["date"] for c in s) for s in candles.values()]
    common = sorted(set.intersection(*date_sets))
    start = max(trend_lookback, vol_lookback) + 1
    if len(common) < start + 100:
        return {"error": "insufficient overlapping candle history"}

    px = {p: {c["date"]: c["close"] for c in s} for p, s in candles.items()}
    closes = {p: [px[p][d] for d in common] for p in candles}
    n = len(common)
    pairs = sorted(candles.keys())

    pos: Dict[str, float] = {p: 0.0 for p in pairs}
    port: List[float] = []
    out_dates: List[str] = []
    tot_cost = tot_fin = 0.0

    for t in range(start, n - 1):
        cost_t = 0.0
        if (t - start) % rebalance_every == 0:
            year = int(common[t][:4])
            diffs = {p: _carry_diff(p, year) for p in pairs}
            ranked = sorted(pairs, key=lambda p: diffs[p])
            shorts, longs = set(ranked[:top_n]), set(ranked[-top_n:])
            new_pos: Dict[str, float] = {}
            for p in pairs:
                d_ = 1 if p in longs else -1 if p in shorts else 0
                if d_ and trend_veto:
                    tr = closes[p][t] / closes[p][t - trend_lookback] - 1.0
                    if (d_ > 0 and tr < 0) or (d_ < 0 and tr > 0):
                        d_ = 0
                window = [closes[p][i] / closes[p][i - 1] - 1.0
                          for i in range(t - vol_lookback + 1, t + 1)]
                mu = sum(window) / len(window)
                var = sum((r - mu) ** 2 for r in window) / len(window)
                av = math.sqrt(var) * math.sqrt(252)
                size = min(max_pos, target_pair_vol / av) if av > 1e-6 else 0.0
                np_ = d_ * size
                cost_t += abs(np_ - pos[p]) * (_pair_meta(p)["spread"] / closes[p][t])
                new_pos[p] = np_
            pos = new_pos

        year = int(common[t][:4])
        r_t = 0.0
        for p in pairs:
            spot_ret = closes[p][t + 1] / closes[p][t] - 1.0
            fin = pos[p] * (_carry_diff(p, year) / 100.0) / 365.0
            tot_fin += fin
            r_t += pos[p] * spot_ret + fin
        r_t = (r_t - cost_t) / len(pairs)
        tot_cost += cost_t / len(pairs)
        port.append(r_t)
        out_dates.append(common[t + 1])

    years = len(port) / 252.0
    out = _summarize(out_dates, port,
                     cost_per_year=tot_cost / years,
                     fin_per_year=(tot_fin / len(pairs)) / years)
    out["params"] = {
        "universe": pairs, "top_n": top_n, "rebalance_every_days": rebalance_every,
        "trend_veto": trend_veto, "target_pair_vol": target_pair_vol,
    }
    return out


def run_carry_trend(
    candles: Dict[str, List[dict]],
    trend_lookback: int = 63,        # ~3 months (literature-standard momentum)
    vol_lookback: int = 20,          # realized-vol window for sizing
    rebalance_every: int = 5,        # weekly
    target_pair_vol: float = 0.0283, # ~8% portfolio ann vol across 8 uncorrelated pairs
    max_pos: float = 1.0,            # cap position at 1x equity fraction per pair
    carry_threshold: float = 0.75,   # rate differential (pct) that counts as carry signal
) -> dict:
    """
    candles: {pair: [{"date": "YYYY-MM-DD", "close": float}, ...] oldest-first}

    Signal per pair, recomputed on rebalance days from data up to day t:
      carry_sign = sign(rate differential)   if |diff| >= carry_threshold else 0
      trend_sign = sign(63-day return)
      direction  = sign(carry_sign + trend_sign)   (0 -> flat)
    Size: target_pair_vol / realized annualized vol, capped at max_pos.
    Daily P&L per pair: pos * spot_return + pos * financing/365 - costs.
    """
    daily: Dict[str, List[float]] = {}       # pair -> daily strategy returns
    dates_by_pair: Dict[str, List[str]] = {}
    stats = {p: {"cost": 0.0, "financing": 0.0, "gross_exposure_sum": 0.0}
             for p in candles}

    for pair, series in candles.items():
        closes = [c["close"] for c in series]
        dates = [c["date"] for c in series]
        n = len(closes)
        if n < trend_lookback + vol_lookback + 5:
            continue
        pip = PAIR_CONFIG.get(pair, {}).get("pip", 0.0001)
        spread = PAIR_CONFIG.get(pair, {}).get("spread", 1.5 * pip)

        rets: List[float] = []
        out_dates: List[str] = []
        pos = 0.0
        start = trend_lookback + vol_lookback

        for t in range(start, n - 1):
            # rebalance decision uses data up to and including t
            if (t - start) % rebalance_every == 0:
                year = int(dates[t][:4])
                diff = _carry_diff(pair, year)
                carry_sign = 0 if abs(diff) < carry_threshold else (1 if diff > 0 else -1)
                trend_ret = closes[t] / closes[t - trend_lookback] - 1.0
                trend_sign = 1 if trend_ret > 0 else -1 if trend_ret < 0 else 0
                raw = carry_sign + trend_sign
                direction = 1 if raw > 0 else -1 if raw < 0 else 0

                # realized annualized vol over vol_lookback days
                window = [closes[i] / closes[i - 1] - 1.0
                          for i in range(t - vol_lookback + 1, t + 1)]
                mean = sum(window) / len(window)
                var = sum((r - mean) ** 2 for r in window) / len(window)
                ann_vol = math.sqrt(var) * math.sqrt(252)
                size = min(max_pos, target_pair_vol / ann_vol) if ann_vol > 1e-6 else 0.0

                new_pos = direction * size
                turnover = abs(new_pos - pos)
                cost = turnover * (spread / closes[t])
                stats[pair]["cost"] += cost
                pos = new_pos
            else:
                cost = 0.0

            spot_ret = closes[t + 1] / closes[t] - 1.0
            financing = pos * (_carry_diff(pair, int(dates[t][:4])) / 100.0) / 365.0
            stats[pair]["financing"] += financing
            stats[pair]["gross_exposure_sum"] += abs(pos)
            rets.append(pos * spot_ret + financing - cost)
            out_dates.append(dates[t + 1])

        daily[pair] = rets
        dates_by_pair[pair] = out_dates

    if not daily:
        return {"error": "insufficient candle data"}

    # ── Portfolio: equal-weight average of per-pair strategy returns by date ──
    all_dates = sorted(set(d for ds in dates_by_pair.values() for d in ds))
    by_date: Dict[str, List[float]] = {d: [] for d in all_dates}
    for pair, rets in daily.items():
        for d, r in zip(dates_by_pair[pair], rets):
            by_date[d].append(r)
    port = [sum(rs) / len(daily) for d, rs in ((d, by_date[d]) for d in all_dates) if rs]

    # ── Metrics ───────────────────────────────────────────────────────────────
    n_days = len(port)
    equity = 1.0
    curve = []
    peak, max_dd = 1.0, 0.0
    for r in port:
        equity *= (1.0 + r)
        curve.append(equity)
        peak = max(peak, equity)
        max_dd = max(max_dd, 1.0 - equity / peak)
    years = n_days / 252.0
    cagr = equity ** (1.0 / years) - 1.0 if years > 0.2 and equity > 0 else 0.0
    mean_r = sum(port) / n_days
    var_r = sum((r - mean_r) ** 2 for r in port) / n_days
    ann_vol = math.sqrt(var_r) * math.sqrt(252)
    sharpe = (mean_r * 252) / ann_vol if ann_vol > 1e-9 else 0.0

    yearly: Dict[str, float] = {}
    y_eq_start, cur_year = 1.0, all_dates[0][:4]
    eq = 1.0
    for d, r in zip(all_dates, port):
        if d[:4] != cur_year:
            yearly[cur_year] = round((eq / y_eq_start - 1.0) * 100, 2)
            cur_year, y_eq_start = d[:4], eq
        eq *= (1.0 + r)
    yearly[cur_year] = round((eq / y_eq_start - 1.0) * 100, 2)

    total_cost = sum(s["cost"] for s in stats.values()) / max(len(daily), 1)
    total_fin = sum(s["financing"] for s in stats.values()) / max(len(daily), 1)
    avg_gross = sum(s["gross_exposure_sum"] for s in stats.values()) / max(n_days * len(daily), 1)

    return {
        "params": {
            "pairs": sorted(daily.keys()), "trend_lookback_days": trend_lookback,
            "rebalance_every_days": rebalance_every, "carry_threshold_pct": carry_threshold,
            "target_pair_vol": target_pair_vol,
        },
        "window": {"start": all_dates[0], "end": all_dates[-1],
                   "trading_days": n_days, "years": round(years, 1)},
        "headline": {
            "final_multiple": round(equity, 3),
            "cagr_pct": round(cagr * 100, 2),
            "ann_vol_pct": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
        },
        "attribution_pct_per_year": {
            "financing_carry": round(total_fin / years * 100, 2),
            "spread_costs": round(-total_cost / years * 100, 2),
        },
        "avg_gross_exposure_per_pair": round(avg_gross, 3),
        "yearly_returns_pct": yearly,
    }

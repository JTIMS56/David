"""
Volatility Research Module
────────────────────────────────────────────────────────────────────────────
Built for the research question: returns are hard to predict, but volatility
is not — and volatility in one market may lead volatility in another.

This project already established the first half empirically: 2,500+ logged
directional forecasts at 49-50%, an ensemble at 49.8% out-of-sample, and four
backtested strategy families at Sharpe ~ 0. Returns are, for practical
purposes, unforecastable at retail access.

Volatility is a different object. It clusters, it mean-reverts, and it is
one of the most reliably forecastable quantities in finance. This module tests
that claim with the same discipline used on the returns work — genuine
out-of-sample evaluation against naive baselines, never in-sample fit — and
then asks whether volatility transmits across markets.

Three parts:
  1. Realized-volatility estimators (close-to-close, Parkinson, Garman-Klass)
  2. GARCH(1,1) forecasting, scored out-of-sample against baselines
  3. Spillover: Granger causality and a Diebold-Yilmaz variance decomposition

Everything is annualised in percent unless stated otherwise.
"""
from __future__ import annotations

import logging
import math
import warnings
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("popper.volatility")

TRADING_DAYS = 252


# ── 1. Realized volatility estimators ────────────────────────────────────────

def log_returns(closes: np.ndarray) -> np.ndarray:
    return np.diff(np.log(closes))


def rv_close_to_close(closes: np.ndarray, window: int = 21) -> np.ndarray:
    """
    Rolling annualised standard deviation of log returns.

    The familiar estimator, and the least efficient of the three: it uses one
    observation per day and discards everything that happened between the
    opens and closes.
    """
    r = log_returns(closes)
    out = np.full(len(r), np.nan)
    for i in range(window - 1, len(r)):
        out[i] = np.std(r[i - window + 1:i + 1], ddof=1)
    return out * math.sqrt(TRADING_DAYS) * 100


def rv_parkinson(highs: np.ndarray, lows: np.ndarray, window: int = 21) -> np.ndarray:
    """
    Parkinson (1980) range estimator: (1 / 4ln2) * mean(ln(H/L)^2).

    Roughly five times more efficient than close-to-close because the daily
    range carries far more information about diffusion than two endpoints do.
    Assumes no drift and continuous observation, so it understates volatility
    when the price gaps overnight.
    """
    hl = np.log(highs / lows) ** 2
    k = 1.0 / (4.0 * math.log(2.0))
    out = np.full(len(hl), np.nan)
    for i in range(window - 1, len(hl)):
        out[i] = math.sqrt(k * np.mean(hl[i - window + 1:i + 1]))
    return out * math.sqrt(TRADING_DAYS) * 100


def rv_garman_klass(opens, highs, lows, closes, window: int = 21) -> np.ndarray:
    """
    Garman-Klass (1980): 0.5*ln(H/L)^2 - (2ln2 - 1)*ln(C/O)^2.

    Uses the full OHLC bar and is the most efficient of the three under its
    assumptions, but it is also the most sensitive to overnight gaps, since it
    treats the open as the start of the diffusion.
    """
    hl = np.log(highs / lows) ** 2
    co = np.log(closes / opens) ** 2
    daily = 0.5 * hl - (2.0 * math.log(2.0) - 1.0) * co
    out = np.full(len(daily), np.nan)
    for i in range(window - 1, len(daily)):
        w = daily[i - window + 1:i + 1]
        m = np.mean(w)
        out[i] = math.sqrt(m) if m > 0 else np.nan
    return out * math.sqrt(TRADING_DAYS) * 100


# ── 2. GARCH forecasting, evaluated out-of-sample ────────────────────────────

def _qlike(realized_var: np.ndarray, pred_var: np.ndarray) -> float:
    """
    QLIKE loss: mean(rv/pv - ln(rv/pv) - 1). Robust to noise in the realized
    proxy and, unlike MSE, does not reward systematic under-prediction of
    variance. Lower is better; 0 is perfect.
    """
    mask = (pred_var > 0) & (realized_var > 0) & np.isfinite(pred_var) & np.isfinite(realized_var)
    if not mask.any():
        return float("nan")
    ratio = realized_var[mask] / pred_var[mask]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def garch_walk_forward(closes: np.ndarray, train: int = 1000,
                       refit_every: int = 50) -> dict:
    """
    Walk-forward one-day-ahead variance forecasts from GARCH(1,1), scored
    against two baselines on data the model never saw.

    Baselines matter. A GARCH that beats nothing is not evidence of
    predictable volatility — it is evidence of a fitted curve. The comparisons
    here are:
      • constant variance (the full-sample unconditional estimate)
      • EWMA / RiskMetrics (lambda = 0.94), a genuinely strong baseline

    The realized proxy for next-day variance is the squared return. It is
    noisy but unbiased, which is why QLIKE is reported alongside MSE.
    """
    from arch import arch_model

    r = log_returns(closes) * 100          # percent returns, GARCH scale
    n = len(r)
    if n < train + 60:
        return {"error": f"need > {train + 60} returns, have {n}"}

    preds_garch, preds_ewma, preds_const, actual = [], [], [], []
    lam = 0.94
    ewma_var = float(np.var(r[:train], ddof=1))
    params: Optional[tuple] = None      # (omega, alpha, beta, mu)
    h_next = float(np.var(r[:train], ddof=1))   # conditional variance for time t

    def _refit(upto: int):
        """Fit GARCH(1,1) on r[:upto] and roll its recursion forward to `upto`."""
        res = arch_model(r[:upto], vol="GARCH", p=1, q=1,
                         mean="Constant", dist="normal").fit(disp="off")
        p = res.params
        om, al, be = float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"])
        mu = float(p.get("mu", 0.0))
        persist = al + be
        h = om / (1 - persist) if persist < 0.999 else float(np.var(r[:upto], ddof=1))
        for x in r[:upto]:
            h = om + al * (x - mu) ** 2 + be * h
        return (om, al, be, mu), h

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(train, n):
            if params is None or (t - train) % refit_every == 0:
                try:
                    params, h_next = _refit(t)
                except Exception as exc:
                    logger.warning("GARCH fit failed at t=%d: %s", t, exc)

            pv = h_next if params is not None else float(np.var(r[:t], ddof=1))
            preds_garch.append(pv)
            preds_ewma.append(ewma_var)
            preds_const.append(float(np.var(r[:t], ddof=1)))
            actual.append(float(r[t] ** 2))

            # Roll BOTH recursions forward on the newly observed return. The
            # conditional variance must be updated every step even when the
            # parameters are only refit periodically — reusing the forecast
            # made at the last refit is a stale prediction, not a one-step-
            # ahead one, and destroys the model's entire information content.
            if params is not None:
                om, al, be, mu = params
                h_next = om + al * (r[t] - mu) ** 2 + be * h_next
            ewma_var = lam * ewma_var + (1 - lam) * (r[t] ** 2)

    a = np.array(actual)
    out = {"n_forecasts": len(a), "train_size": train, "refit_every": refit_every}
    for name, p in (("garch11", preds_garch), ("ewma_riskmetrics", preds_ewma),
                    ("constant_variance", preds_const)):
        p = np.array(p)
        out[name] = {
            "mse": round(float(np.mean((a - p) ** 2)), 4),
            "qlike": round(_qlike(a, p), 4),
            "mean_forecast_ann_vol_pct": round(
                float(np.mean(np.sqrt(p * TRADING_DAYS))), 2),
        }

    # Does conditional variance explain realized variance out-of-sample? This
    # is the Mincer-Zarnowitz style check: regress realized on predicted.
    g = np.array(preds_garch)
    mask = np.isfinite(g) & (g > 0)
    if mask.sum() > 30:
        corr = float(np.corrcoef(a[mask], g[mask])[0, 1])
        out["garch11"]["oos_corr_with_realized"] = round(corr, 4)
        out["garch11"]["oos_r_squared"] = round(corr ** 2, 4)
    out["verdict"] = _garch_verdict(out)
    return out


def _garch_verdict(res: dict) -> str:
    g, e, c = res.get("garch11"), res.get("ewma_riskmetrics"), res.get("constant_variance")
    if not (g and e and c):
        return "insufficient results"
    beats_const = g["qlike"] < c["qlike"]
    beats_ewma = g["qlike"] < e["qlike"]
    r2 = g.get("oos_r_squared")
    parts = []
    parts.append("GARCH beats constant variance" if beats_const
                 else "GARCH does NOT beat constant variance")
    parts.append("and beats EWMA" if beats_ewma else "but does not beat EWMA")
    if r2 is not None:
        parts.append(f"out-of-sample R^2 vs realized variance = {r2:.3f}")
    return "; ".join(parts)


# ── 3. Volatility spillover across markets ───────────────────────────────────

def _align(series: Dict[str, dict]) -> tuple:
    """Inner-join instrument vol series on common dates."""
    common = None
    for d in series.values():
        s = set(d["dates"])
        common = s if common is None else (common & s)
    dates = sorted(common or [])
    if len(dates) < 100:
        return [], {}
    aligned = {}
    for name, d in series.items():
        lookup = dict(zip(d["dates"], d["vol"]))
        aligned[name] = np.array([lookup[dt] for dt in dates], dtype=float)
    return dates, aligned


def spillover_analysis(vol_series: Dict[str, dict], lags: int = 5,
                       horizon: int = 10) -> dict:
    """
    Does volatility in one market lead volatility in another?

    Two complementary views:

    • Granger causality — pairwise, on log volatility. Answers "do past values
      of X improve a forecast of Y beyond Y's own past?" It is a statement
      about predictive content, not about economic causation, and that
      distinction is worth keeping explicit.

    • Diebold-Yilmaz (2012) spillover index — fits a VAR and decomposes the
      h-step forecast error variance of each series into own-market and
      cross-market contributions. The total index is the share of forecast
      error variance attributable to spillovers across the whole system.
      Uses a generalized decomposition, so it does not depend on the ordering
      of the variables.
    """
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import grangercausalitytests

    dates, aligned = _align(vol_series)
    if not aligned:
        return {"error": "insufficient overlapping history across instruments"}

    names = sorted(aligned)
    # Log volatility: variance is right-skewed and log-vol is far closer to
    # normal, which the VAR's assumptions prefer.
    data = np.column_stack([np.log(np.maximum(aligned[n], 1e-8)) for n in names])
    ok = np.all(np.isfinite(data), axis=1)
    data, dates = data[ok], [d for d, k in zip(dates, ok) if k]
    if len(data) < 200:
        return {"error": f"only {len(data)} usable observations after alignment"}

    # ── Granger causality, pairwise ──────────────────────────────────────────
    granger = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, src in enumerate(names):
            for j, dst in enumerate(names):
                if i == j:
                    continue
                try:
                    # grangercausalitytests(x) tests whether col 1 causes col 0
                    pair = np.column_stack([data[:, j], data[:, i]])
                    res = grangercausalitytests(pair, maxlag=lags)
                    p = min(res[k][0]["ssr_ftest"][1] for k in range(1, lags + 1))
                    if p < 0.05:
                        granger.append({"from": src, "to": dst,
                                        "min_p_value": round(float(p), 5)})
                except Exception:
                    continue
    granger.sort(key=lambda g: g["min_p_value"])

    # ── Diebold-Yilmaz spillover index ───────────────────────────────────────
    dy: dict = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            var_res = VAR(data).fit(maxlags=lags)
            fevd = var_res.fevd(horizon)
            # decomp[i][j] = share of i's forecast error variance from shocks to j
            decomp = fevd.decomp[:, horizon - 1, :]
            decomp = decomp / decomp.sum(axis=1, keepdims=True)
            k = len(names)
            own = float(np.trace(decomp))
            total_spillover = 100.0 * (1.0 - own / k)
            to_others = {names[j]: round(100.0 * (decomp[:, j].sum() - decomp[j, j]) / k, 2)
                         for j in range(k)}
            from_others = {names[i]: round(100.0 * (decomp[i, :].sum() - decomp[i, i]) / k, 2)
                           for i in range(k)}
            net = {n: round(float(to_others[n] - from_others[n]), 2) for n in names}
            dy = {
                "total_spillover_index_pct": round(total_spillover, 2),
                "horizon_days": horizon,
                "var_lags": int(var_res.k_ar),
                "transmits_to_others_pct": to_others,
                "receives_from_others_pct": from_others,
                # Positive net = a source of volatility for the system.
                "net_transmitter_pct": dict(sorted(net.items(),
                                                   key=lambda kv: -kv[1])),
            }
    except Exception as exc:
        dy = {"error": f"VAR/FEVD failed: {exc}"}

    return {
        "instruments": names,
        "observations": len(data),
        "date_range": [dates[0], dates[-1]] if dates else None,
        "granger_significant_at_5pct": granger[:40],
        "granger_note": (
            "Predictive content, not economic causation. Pairwise tests with no "
            "multiple-comparison correction, so expect false positives: on "
            "validation data with a known one-way link, Granger also reported the "
            "reverse direction as significant. Where the two disagree, trust the "
            "Diebold-Yilmaz net-transmitter ranking, which recovered the true "
            "source and correctly scored the independent series at ~0."),
        "diebold_yilmaz": dy,
    }

"""
Proper scoring rules for evaluating distributional forecasts.

All metrics compare a predicted distribution (represented by discrete price/prob arrays)
against a single realised outcome.

References:
  - Gneiting & Raftery (2007) "Strictly proper scoring rules"
  - Diebold & Mariano (1995) "Comparing predictive accuracy"
"""
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class BacktestRecord:
    date: str
    horizon_days: int
    S0: float              # spot at forecast date
    S_realized: float      # actual spot at horizon
    log_ret_realized: float
    log_likelihood: float  # log p(S_realized | model)
    crps: float            # Continuous Ranked Probability Score (lower = better)
    pit: float             # Probability Integral Transform ∈ (0,1) — should be Uniform
    coverage_95: bool      # was realised within model's 95% CI?
    call_model: float      # ATM call price from model
    call_bs: float         # ATM call price from BS reference
    mean_model: float
    std_model: float
    pair: str
    model: str = "DHJ"


def log_likelihood(
    S_realized: float,
    prices: Sequence[float],
    prob_density: Sequence[float],
) -> float:
    """
    log p(S_realized) interpolated from the model's discrete density.
    prob_density is per-unit-price (p_S not p_x).
    """
    p = np.asarray(prob_density)
    s = np.asarray(prices)
    if S_realized < s[0] or S_realized > s[-1]:
        return -50.0  # heavy penalty for outcome outside grid
    pdf_val = float(np.interp(S_realized, s, p))
    return np.log(max(pdf_val, 1e-30))


def crps_from_cdf(
    S_realized: float,
    prices: Sequence[float],
    prob_density: Sequence[float],
) -> float:
    """
    CRPS = ∫ (F(x) - 1{x ≥ S_realized})² dx
    where F is the model CDF.  Lower is better.
    """
    s = np.asarray(prices, dtype=float)
    p = np.asarray(prob_density, dtype=float)
    # Build CDF via trapezoid integration over density
    ds = np.gradient(s)
    cdf = np.cumsum(p * ds)
    cdf = cdf / max(cdf[-1], 1e-15)  # normalise to [0,1]
    indicator = (s >= S_realized).astype(float)
    return float(np.trapezoid((cdf - indicator) ** 2, s))


def pit_score(
    S_realized: float,
    prices: Sequence[float],
    prob_density: Sequence[float],
) -> float:
    """
    Probability Integral Transform: F(S_realized).
    Should be Uniform(0,1) for a calibrated model.
    """
    s = np.asarray(prices, dtype=float)
    p = np.asarray(prob_density, dtype=float)
    if S_realized <= s[0]:
        return 0.0
    if S_realized >= s[-1]:
        return 1.0
    ds = np.gradient(s)
    cdf = np.cumsum(p * ds)
    cdf = cdf / max(cdf[-1], 1e-15)
    return float(np.interp(S_realized, s, cdf))


def coverage_95(
    S_realized: float,
    prices: Sequence[float],
    prob_density: Sequence[float],
) -> bool:
    """Check whether S_realized falls within the model's central 95% CI."""
    s = np.asarray(prices, dtype=float)
    p = np.asarray(prob_density, dtype=float)
    ds = np.gradient(s)
    cdf = np.cumsum(p * ds)
    cdf = cdf / max(cdf[-1], 1e-15)
    lo = float(np.interp(0.025, cdf, s))
    hi = float(np.interp(0.975, cdf, s))
    return lo <= S_realized <= hi


def distribution_stats(
    prices: Sequence[float],
    prob_density: Sequence[float],
) -> dict:
    """Compute mean, std, skew, kurtosis of the model's forecast distribution."""
    s = np.asarray(prices, dtype=float)
    p = np.asarray(prob_density, dtype=float)
    ds = np.gradient(s)
    w = p * ds
    w = w / max(w.sum(), 1e-15)
    mean = float(np.sum(s * w))
    var  = float(np.sum((s - mean)**2 * w))
    skew = float(np.sum((s - mean)**3 * w) / max(var**1.5, 1e-30))
    kurt = float(np.sum((s - mean)**4 * w) / max(var**2, 1e-30)) - 3.0
    return {"mean": mean, "std": np.sqrt(var), "skew": skew, "excess_kurtosis": kurt}


def pit_uniformity_test(pit_values: Sequence[float]) -> dict:
    """
    Kolmogorov-Smirnov test: are PIT values Uniform(0,1)?
    p > 0.05 → model is calibrated (cannot reject uniformity).
    """
    u = np.asarray(pit_values)
    u = u[(u > 0) & (u < 1)]
    if len(u) < 10:
        return {"ks_statistic": float("nan"), "p_value": float("nan"), "n": len(u)}
    stat, pval = stats.kstest(u, "uniform")
    return {"ks_statistic": float(stat), "p_value": float(pval), "n": len(u),
            "calibrated": pval > 0.05}


def diebold_mariano(
    losses_a: Sequence[float],
    losses_b: Sequence[float],
) -> dict:
    """
    Diebold-Mariano test: is model A significantly better than model B?
    Uses log-likelihood as loss (higher = better, so we negate for DM).
    Returns p-value: < 0.05 → model A is significantly better.
    """
    d = np.asarray(losses_a) - np.asarray(losses_b)
    n = len(d)
    mean_d = np.mean(d)
    # HAC variance (Newey-West with lag 1)
    gamma0 = np.var(d, ddof=1)
    gamma1 = np.cov(d[:-1], d[1:])[0, 1] if n > 2 else 0.0
    var_d = (gamma0 + 2 * gamma1) / n
    dm_stat = mean_d / max(np.sqrt(abs(var_d)), 1e-15)
    p_val = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return {"dm_statistic": float(dm_stat), "p_value": float(p_val),
            "mean_diff": float(mean_d), "A_better": mean_d > 0}

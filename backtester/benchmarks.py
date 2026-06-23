"""
Benchmark density models for walk-forward backtesting.

Every benchmark implements the same interface as DHJPredictor:
    forecast(spot, horizon_days, history) -> BenchmarkForecast

BenchmarkForecast holds a discrete price grid and probability density so that
the same metrics.py scoring functions (log-likelihood, CRPS, PIT) can be
applied uniformly to all models.

Models implemented:
  RandomWalkVol       — Gaussian log-return, rolling historical vol
  EWMA                — EWMA vol (λ=0.94, RiskMetrics standard)
  GARCHSimple         — GARCH(1,1) vol estimate (closed-form)
  HistoricalEmpirical — empirical density from past log-returns
  GarmanKohlhagen     — GK lognormal (exact benchmark)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class BenchmarkForecast:
    prices:   np.ndarray   # price grid
    density:  np.ndarray   # probability density (per unit price, integrates to ~1)
    mean:     float        # E[S_T]
    vol:      float        # annualised vol used


class BenchmarkModel(Protocol):
    name: str
    def forecast(
        self,
        spot: float,
        horizon_days: float,
        log_returns: np.ndarray,   # recent past log-returns (array)
        r_d: float = 0.0525,
        r_f: float = 0.0,
    ) -> BenchmarkForecast: ...


# ── helpers ────────────────────────────────────────────────────────────────────

def _lognormal_density(prices: np.ndarray, S0: float, mu_ln: float, sig_ln: float) -> np.ndarray:
    """Lognormal density dP/dS for a grid of S values."""
    log_ret = np.log(prices / S0)
    return np.exp(-0.5 * ((log_ret - mu_ln) / sig_ln) ** 2) / (
        sig_ln * prices * math.sqrt(2 * math.pi)
    )


def _price_grid(S0: float, vol_ann: float, T: float, n: int = 300) -> np.ndarray:
    """Log-normal price grid centred on S0, spanning ±5σ."""
    sig_T = vol_ann * math.sqrt(T)
    lo = S0 * math.exp(-5 * sig_T)
    hi = S0 * math.exp(+5 * sig_T)
    return np.linspace(lo, hi, n)


# ── Model 1: Random walk with rolling historical vol ──────────────────────────

class RandomWalkVol:
    name = "RandomWalk"

    def __init__(self, lookback: int = 63) -> None:
        self._lookback = lookback

    def forecast(
        self,
        spot: float,
        horizon_days: float,
        log_returns: np.ndarray,
        r_d: float = 0.0525,
        r_f: float = 0.0,
    ) -> BenchmarkForecast:
        hist = log_returns[-self._lookback:]
        vol  = float(np.std(hist, ddof=1) * math.sqrt(252)) if len(hist) > 1 else 0.10
        T    = horizon_days / 365.0
        carry   = r_d - r_f
        mu_ln   = (carry - 0.5 * vol * vol) * T
        sig_ln  = vol * math.sqrt(T)
        prices  = _price_grid(spot, vol, T)
        density = _lognormal_density(prices, spot, mu_ln, sig_ln)
        return BenchmarkForecast(prices, density, spot * math.exp(carry * T), vol)


# ── Model 2: EWMA vol (RiskMetrics λ=0.94) ───────────────────────────────────

class EWMAVol:
    name = "EWMA"

    def __init__(self, lam: float = 0.94) -> None:
        self._lam = lam

    def forecast(
        self,
        spot: float,
        horizon_days: float,
        log_returns: np.ndarray,
        r_d: float = 0.0525,
        r_f: float = 0.0,
    ) -> BenchmarkForecast:
        lam = self._lam
        # EWMA variance from most-recent returns
        if len(log_returns) < 2:
            var_daily = (0.10 / math.sqrt(252)) ** 2
        else:
            var_daily = float(log_returns[-1] ** 2)
            for r in reversed(log_returns[:-1]):
                var_daily = lam * var_daily + (1 - lam) * r ** 2
        vol    = math.sqrt(var_daily * 252)
        T      = horizon_days / 365.0
        carry  = r_d - r_f
        mu_ln  = (carry - 0.5 * vol * vol) * T
        sig_ln = vol * math.sqrt(T)
        prices = _price_grid(spot, vol, T)
        density = _lognormal_density(prices, spot, mu_ln, sig_ln)
        return BenchmarkForecast(prices, density, spot * math.exp(carry * T), vol)


# ── Model 3: GARCH(1,1) closed-form vol estimate ─────────────────────────────

class GARCHSimple:
    """
    Fit GARCH(1,1) parameters via method-of-moments to past returns,
    then compute the term vol for forecast horizon T.

    σ²_T = θ + (α+β)^n · (σ²_0 − θ)   where n = T·252
    Uses persistence α+β=0.97, α=0.05, β=0.92 if sample too small.
    """
    name = "GARCH(1,1)"

    def __init__(self, lookback: int = 252) -> None:
        self._lookback = lookback

    def forecast(
        self,
        spot: float,
        horizon_days: float,
        log_returns: np.ndarray,
        r_d: float = 0.0525,
        r_f: float = 0.0,
    ) -> BenchmarkForecast:
        hist = log_returns[-self._lookback:]
        if len(hist) < 30:
            vol = 0.10
        else:
            var_arr = hist ** 2
            # Method-of-moments: α = corr(r²_t, r²_{t-1}), β=persistence-α, θ=long-run var
            r2      = var_arr[1:]
            r2_lag  = var_arr[:-1]
            alpha   = max(0.01, min(0.15, float(np.corrcoef(r2, r2_lag)[0, 1]) * 0.10))
            beta    = max(0.80, min(0.95, 0.97 - alpha))
            theta   = float(np.var(hist, ddof=1))
            # Current conditional variance: EWMA initialise with θ
            h_now = theta
            for r in hist[-20:]:
                h_now = theta * (1 - alpha - beta) + alpha * r**2 + beta * h_now
            # Term variance over T trading days
            n        = horizon_days
            persist  = alpha + beta
            if persist < 1.0:
                h_term = theta + (persist**n) * (h_now - theta)
            else:
                h_term = h_now
            vol = math.sqrt(max(h_term, 1e-8) * 252)

        T      = horizon_days / 365.0
        carry  = r_d - r_f
        mu_ln  = (carry - 0.5 * vol * vol) * T
        sig_ln = vol * math.sqrt(T)
        prices = _price_grid(spot, vol, T)
        density = _lognormal_density(prices, spot, mu_ln, sig_ln)
        return BenchmarkForecast(prices, density, spot * math.exp(carry * T), vol)


# ── Model 4: Historical empirical distribution ────────────────────────────────

class HistoricalEmpirical:
    """
    Kernel-density smoothed empirical distribution of past log-returns,
    scaled to the forecast horizon by the √T rule.
    """
    name = "Historical"

    def __init__(self, lookback: int = 252) -> None:
        self._lookback = lookback

    def forecast(
        self,
        spot: float,
        horizon_days: float,
        log_returns: np.ndarray,
        r_d: float = 0.0525,
        r_f: float = 0.0,
    ) -> BenchmarkForecast:
        from scipy.stats import gaussian_kde  # optional dep

        hist = log_returns[-self._lookback:]
        T    = horizon_days / 365.0
        n_days = horizon_days

        # Scale daily returns to horizon via √T
        scaled = hist * math.sqrt(n_days)
        vol    = float(np.std(hist, ddof=1) * math.sqrt(252))
        prices = _price_grid(spot, max(vol, 0.05), T, n=300)

        # Carry-drift adjustment
        carry  = r_d - r_f
        drift  = (carry - 0.5 * vol**2) * T
        log_prices = np.log(prices / spot) - drift

        try:
            kde     = gaussian_kde(scaled, bw_method="scott")
            density = kde(log_prices) / prices   # convert to per-unit-price
        except Exception:
            # Fall back to lognormal if scipy not available or KDE fails
            sig_ln  = vol * math.sqrt(T)
            density = _lognormal_density(prices, spot, drift, sig_ln)

        return BenchmarkForecast(prices, density, spot * math.exp(carry * T), vol)


# ── Model 5: Garman-Kohlhagen lognormal ──────────────────────────────────────

class GarmanKohlhagen:
    """Exact GK lognormal reference — used as the strongest classical baseline."""
    name = "GarmanKohlhagen"

    def __init__(self, vol: float | None = None) -> None:
        self._vol = vol   # None → infer from rolling historical vol

    def forecast(
        self,
        spot: float,
        horizon_days: float,
        log_returns: np.ndarray,
        r_d: float = 0.0525,
        r_f: float = 0.0,
    ) -> BenchmarkForecast:
        if self._vol is not None:
            vol = self._vol
        else:
            hist = log_returns[-63:]
            vol  = float(np.std(hist, ddof=1) * math.sqrt(252)) if len(hist) > 1 else 0.10

        T      = horizon_days / 365.0
        carry  = r_d - r_f
        mu_ln  = (carry - 0.5 * vol * vol) * T
        sig_ln = vol * math.sqrt(T)
        prices = _price_grid(spot, vol, T)
        density = _lognormal_density(prices, spot, mu_ln, sig_ln)
        return BenchmarkForecast(prices, density, spot * math.exp(carry * T), vol)


# ── Registry ──────────────────────────────────────────────────────────────────

def default_benchmark_suite() -> list:
    """All standard benchmarks in ladder order (weakest → strongest)."""
    return [
        RandomWalkVol(lookback=63),
        EWMAVol(lam=0.94),
        GARCHSimple(lookback=252),
        HistoricalEmpirical(lookback=252),
        GarmanKohlhagen(),
    ]

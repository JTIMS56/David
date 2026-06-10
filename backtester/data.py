"""
FX historical data loader for backtesting.

Priority order:
  1. CSV file (user-supplied, any source: FRED, ECB, Bloomberg, Refinitiv)
  2. yfinance (installed separately: pip install yfinance)
  3. Synthetic GBM data (fallback for testing)

CSV format expected:
  date,close          (or date,EURUSD / date,price — first non-date column used)
  2019-01-02,1.1450
  2019-01-03,1.1352
  ...
"""
import os
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Maps FX pair → Yahoo Finance ticker
_YF_TICKERS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "EURGBP": "EURGBP=X",
    "NZDUSD": "NZDUSD=X",
    "USDCHF": "CHF=X",
}


def load_from_csv(path: str | Path) -> pd.Series:
    """Load a price series from CSV. Returns a pd.Series indexed by date."""
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    series = df.iloc[:, 0].dropna().astype(float)
    series.name = Path(path).stem
    logger.info("Loaded %d rows from %s", len(series), path)
    return series


def load_from_yfinance(pair: str, start: str, end: str) -> pd.Series:
    """Download price series from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance  to use Yahoo Finance data source")
    ticker = _YF_TICKERS.get(pair.upper(), pair.upper() + "=X")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    series = df["Close"].dropna()
    series.name = pair
    logger.info("Downloaded %d rows for %s from Yahoo Finance", len(series), pair)
    return series


def synthetic_gbm(
    pair: str,
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    S0: float = 1.085,
    sigma: float = 0.082,
    r: float = 0.0525,
    seed: int = 42,
) -> pd.Series:
    """Generate synthetic GBM price series — useful for testing the backtest pipeline."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    dt = 1.0 / 252
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(len(dates))
    prices = S0 * np.exp(np.cumsum(log_returns))
    series = pd.Series(prices, index=dates, name=pair)
    logger.info("Generated %d synthetic GBM prices for %s", len(series), pair)
    return series


def synthetic_heston_kou(
    pair: str,
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    S0: float = 1.085,
    theta_vol: float = 0.080,    # long-run vol
    v0_vol: float = 0.080,       # starting vol
    kappa_H: float = 2.5,        # vol mean-reversion (half-life ≈ 10 weeks)
    xi: float = 0.30,            # vol-of-vol
    rho: float = -0.35,          # leverage
    vol_floor: float = 0.040,    # reflective floor (FX vol never truly dies)
    jump_lambda: float = 15.0,   # jumps per year
    jump_p_up: float = 0.30,     # mostly down-jumps (FX crash asymmetry)
    jump_mean_up: float = 0.006,
    jump_mean_down: float = 0.008,
    jump_cap: float = 0.025,     # max single jump (FX moves are bounded)
    r: float = 0.0125,           # carry drift
    seed: int = 42,
) -> pd.Series:
    """
    Synthetic FX with the three stylized facts real FX exhibits:
    vol clustering (Heston), leverage (ρ<0), and asymmetric jump tails (Kou).
    Drift is jump-compensated so E[S_T] = S0·exp(r·T).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n, dt = len(dates), 1.0 / 252
    theta = theta_vol ** 2
    v = v0_vol ** 2
    eta_p, eta_m = 1.0 / jump_mean_up, 1.0 / jump_mean_down
    zeta = (jump_p_up * eta_p / (eta_p - 1)
            + (1 - jump_p_up) * eta_m / (eta_m + 1) - 1)  # Kou compensator

    x = np.log(S0)
    log_p = np.empty(n)
    for i in range(n):
        zv = rng.standard_normal()
        zs = rho * zv + np.sqrt(1 - rho * rho) * rng.standard_normal()
        x += (r - 0.5 * v - jump_lambda * zeta) * dt + np.sqrt(v * dt) * zs
        for _ in range(rng.poisson(jump_lambda * dt)):
            if rng.random() < jump_p_up:
                x += min(rng.exponential(jump_mean_up), jump_cap)
            else:
                x -= min(rng.exponential(jump_mean_down), jump_cap)
        # Milstein CIR step with reflective vol floor
        v = v + kappa_H * (theta - v) * dt \
              + xi * np.sqrt(max(v, 0.0) * dt) * zv \
              + 0.25 * xi * xi * dt * (zv * zv - 1.0)
        v = max(v, vol_floor * vol_floor)
        log_p[i] = x

    series = pd.Series(np.exp(log_p), index=dates, name=pair)
    logger.info("Generated %d synthetic Heston-Kou prices for %s", n, pair)
    return series


def load_prices(
    pair: str,
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    csv_path: Optional[str] = None,
) -> pd.Series:
    """
    Load FX price series: CSV → yfinance → synthetic fallback.
    Returns a daily business-day pd.Series of spot prices.
    """
    if csv_path and os.path.exists(csv_path):
        return load_from_csv(csv_path)
    try:
        return load_from_yfinance(pair, start, end)
    except Exception as e:
        logger.warning("yfinance failed (%s), falling back to synthetic GBM", e)
        return synthetic_gbm(pair, start, end)


def estimate_realized_vol(prices: pd.Series, window: int = 63) -> pd.Series:
    """
    Rolling annualised realised volatility from log-returns.
    window = 63 ≈ 3 months of trading days.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return log_ret.rolling(window).std() * np.sqrt(252)


def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()

"""
WalkForwardCalibrator — estimates DHJ model parameters from trailing price history.

Uses method-of-moments on realized moments of log-returns to fit:
  - sigma (annualised vol)    from rolling realized variance
  - kappa_H (Heston MR speed) from variance autocorrelation decay
  - xi (vol-of-vol)           from variance-of-realized-variance
  - rho (leverage)            from correlation of |log-ret| with sign(log-ret)

All estimates are clamped to physically plausible ranges.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Calibrated parameter defaults (used when history is too short)
_FALLBACK = dict(
    sigma=0.085,
    kappa_H=1.5,
    xi=0.30,
    rho=-0.25,
)


@dataclass
class CalibratedParams:
    sigma: float      # annualised vol
    kappa_H: float    # Heston mean-reversion speed
    xi: float         # vol-of-vol
    rho: float        # leverage correlation
    n_obs: int        # number of observations used

    def to_dhj_kwargs(self) -> dict:
        """Return dict suitable for DiracPredictor.predict_dhj()."""
        return {
            "kappa_H": self.kappa_H,
            "xi":      self.xi,
            "rho":     self.rho,
        }


class WalkForwardCalibrator:
    """
    Fit Heston-like parameters from a trailing window of daily log-returns.

    Args:
        vol_window:     days used for rolling realized variance
        heston_window:  days used for variance autocorrelation (kappa_H, xi)
        trading_days:   annualisation factor (default 252)
    """

    def __init__(
        self,
        vol_window: int = 63,
        heston_window: int = 126,
        trading_days: int = 252,
    ) -> None:
        self.vol_window    = vol_window
        self.heston_window = heston_window
        self.ann           = trading_days

    def calibrate(self, log_returns: np.ndarray) -> CalibratedParams:
        """
        Estimate parameters from *log_returns* (daily, most recent last).
        Falls back to defaults when history is insufficient.
        """
        n = len(log_returns)
        if n < self.vol_window:
            logger.debug("Calibrator: insufficient history (%d < %d), using defaults", n, self.vol_window)
            return CalibratedParams(n_obs=n, **_FALLBACK)

        # ── σ: rolling realized vol ────────────────────────────────────────────
        recent = log_returns[-self.vol_window:]
        sigma  = float(np.std(recent, ddof=1)) * np.sqrt(self.ann)
        sigma  = np.clip(sigma, 0.01, 1.0)

        # ── Heston params from variance-of-variance ────────────────────────────
        if n < self.heston_window:
            kappa_H, xi, rho = _FALLBACK["kappa_H"], _FALLBACK["xi"], _FALLBACK["rho"]
        else:
            hist = log_returns[-self.heston_window:]

            # Daily realized variance (non-overlapping 5-day blocks → ~25 obs)
            block = 5
            blocks = len(hist) // block
            if blocks < 10:
                kappa_H = _FALLBACK["kappa_H"]
                xi      = _FALLBACK["xi"]
                rho     = _FALLBACK["rho"]
            else:
                rv = np.array([
                    np.var(hist[i*block:(i+1)*block], ddof=1) * self.ann
                    for i in range(blocks)
                ])

                # kappa_H: from lag-1 autocorrelation of variance
                # AR(1): v_{t+1} = μ + φ·v_t + ε  → kappa ≈ -log(φ) * ann / block
                acf1 = _lag1_autocorr(rv)
                phi  = np.clip(acf1, 0.01, 0.999)
                kappa_H = float(-np.log(phi) * (self.ann / block))
                kappa_H = np.clip(kappa_H, 0.2, 20.0)

                # xi: vol-of-vol = std(sqrt(rv)) / mean(sqrt(rv)) * sqrt(kappa_H)
                sv    = np.sqrt(np.maximum(rv, 1e-8))
                cv    = float(np.std(sv, ddof=1) / (np.mean(sv) + 1e-10))
                xi    = float(cv * np.sqrt(2.0 * kappa_H))
                xi    = np.clip(xi, 0.05, 2.0)

                # rho: leverage = correlation between |ret| and sign(ret) (Pearson)
                abs_ret  = np.abs(hist)
                sign_ret = np.sign(hist)
                if np.std(abs_ret) > 0 and np.std(sign_ret) > 0:
                    rho = float(np.corrcoef(abs_ret, sign_ret)[0, 1])
                    rho = np.clip(rho, -0.95, 0.95)
                else:
                    rho = _FALLBACK["rho"]

        logger.debug(
            "Calibrated: sigma=%.4f kappa_H=%.2f xi=%.2f rho=%.2f (n=%d)",
            sigma, kappa_H, xi, rho, n,
        )
        return CalibratedParams(
            sigma=sigma, kappa_H=kappa_H, xi=xi, rho=rho, n_obs=n,
        )


def _lag1_autocorr(x: np.ndarray) -> float:
    """Pearson lag-1 autocorrelation, clamped to (0, 1)."""
    if len(x) < 3:
        return 0.9
    mu  = np.mean(x)
    num = float(np.mean((x[:-1] - mu) * (x[1:] - mu)))
    den = float(np.var(x, ddof=0)) + 1e-20
    return float(np.clip(num / den, 0.01, 0.999))

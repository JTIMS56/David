"""
WalkForwardCalibrator — estimates DHJ model parameters from trailing price history.

Design principles
─────────────────
1. HORIZON-AWARE MOMENT MATCHING. The backtest scores the h-day-ahead density,
   so jump parameters are fitted to the variance / skewness / excess-kurtosis
   of overlapping h-day returns — not to daily jump counts. Threshold counting
   systematically misses small jumps and over-selects large down-jumps (biasing
   p_up); matching c2/c3/c4 of the h-day distribution targets exactly the
   moments the log-likelihood rewards.

     c2  = σ_c²·h_yr + λ_h·E[Y²]          (total variance budget)
     c3  =             λ_h·E[Y³]          (skew comes from jumps)
     c4x =             λ_h·E[Y⁴]          (excess kurtosis comes from jumps)

   with Kou common-scale jumps E[Y²]=2/η², E[Y³]=6(2p−1)/η³, E[Y⁴]=24/η⁴:
     η  anchored on observed daily tail exceedances (2nd-moment matched)
     λ_h = c4x·η⁴/24,   2p−1 = c3·η/(3·J),   σ_c² = (c2 − J)/h_yr,  J = λ_h·2/η²

2. TWO-SPEED VOLATILITY. v0 (current variance, EWMA λ=0.94 on jump-censored
   returns) is passed separately from θ (long-run window variance), so the
   Heston layer forecasts vol mean-reversion instead of assuming v0=θ.

3. JUMP-ROBUST ESTIMATES everywhere: MAD-based thresholds, censored vols,
   clipped returns for the variance-of-variance (κ_H, ξ) estimates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Calibrated parameter defaults (used when history is too short)
_FALLBACK = dict(
    sigma=0.085,
    sigma_lr=0.085,
    kappa_H=1.5,
    xi=0.30,
    rho=-0.25,
    jump_lambda=0.0,
    jump_p_up=0.5,
    jump_eta_plus=150.0,
    jump_eta_minus=150.0,
)


@dataclass
class CalibratedParams:
    sigma: float            # CURRENT annualised diffusive vol (EWMA) → v0
    sigma_lr: float         # LONG-RUN annualised diffusive vol       → theta
    kappa_H: float          # Heston mean-reversion speed
    xi: float               # vol-of-vol
    rho: float              # leverage correlation
    jump_lambda: float      # jump arrival rate (per year)
    jump_p_up: float        # fraction of jumps that are upward
    jump_eta_plus: float    # 1/mean up-jump size
    jump_eta_minus: float   # 1/mean down-jump size
    n_obs: int              # number of observations used
    n_jumps: int = 0        # daily tail exceedances detected (η anchor)

    def to_dhj_kwargs(self) -> dict:
        """Return dict suitable for DiracPredictor.predict_dhj()."""
        return {
            "sigma":          self.sigma,
            "sigma_lr":       self.sigma_lr,
            "kappa_H":        self.kappa_H,
            "xi":             self.xi,
            "rho":            self.rho,
            "jump_lambda":    self.jump_lambda,
            "jump_p_up":      self.jump_p_up,
            "jump_eta_plus":  self.jump_eta_plus,
            "jump_eta_minus": self.jump_eta_minus,
        }


class WalkForwardCalibrator:
    """
    Fit DHJ parameters from a trailing window of daily log-returns.

    Args:
        vol_window:    days for the long-run vol estimate (θ)
        heston_window: days for variance autocorrelation (κ_H, ξ)
        jump_window:   days for h-day moment estimation and the η anchor
        jump_k:        MAD multiplier for the daily tail threshold (η anchor)
        ewma_lambda:   decay for the current-vol estimate (RiskMetrics 0.94)
        trading_days:  annualisation factor (default 252)
    """

    def __init__(
        self,
        vol_window: int = 126,
        heston_window: int = 504,
        jump_window: int = 504,
        jump_k: float = 2.5,
        ewma_lambda: float = 0.94,
        trading_days: int = 252,
    ) -> None:
        self.vol_window    = vol_window
        self.heston_window = heston_window
        self.jump_window   = jump_window
        self.jump_k        = jump_k
        self.ewma_lambda   = ewma_lambda
        self.ann           = trading_days

    def calibrate(
        self,
        log_returns: np.ndarray,
        horizon_days: int = 22,
    ) -> CalibratedParams:
        """
        Estimate parameters from *log_returns* (daily, most recent last),
        targeting the moments of *horizon_days*-day-ahead returns.
        """
        n = len(log_returns)
        if n < max(self.vol_window, 3 * horizon_days):
            logger.debug("Calibrator: insufficient history (%d), using defaults", n)
            return CalibratedParams(n_obs=n, **_FALLBACK)

        # ── Daily tail threshold (MAD, jump-robust) → η anchor ────────────────
        jw   = log_returns[-min(n, self.jump_window):]
        med  = float(np.median(jw))
        mad  = 1.4826 * float(np.median(np.abs(jw - med)))
        thresh = self.jump_k * max(mad, 1e-8)
        exceed = jw[np.abs(jw - med) > thresh] - med
        n_jumps = len(exceed)

        # η: second-moment matched to tail exceedances (common up/down scale).
        # E[Y²]=2/η² ⇒ η = sqrt(2·n/Σy²) makes implied jump variance equal the
        # observed tail variance exactly (1/mean would over-count ~2×).
        if n_jumps >= 3:
            eta = float(np.sqrt(2.0 * n_jumps / max(float(np.sum(exceed**2)), 1e-12)))
            eta = float(np.clip(eta, 5.0, 1000.0))
        else:
            eta = _FALLBACK["jump_eta_plus"]

        # ── h-day overlapping return moments (what the LL actually scores) ────
        h    = int(horizon_days)
        cum  = np.concatenate([[0.0], np.cumsum(jw)])
        rh   = cum[h:] - cum[:-h]                     # overlapping h-day returns
        mu_h = float(np.mean(rh))
        d    = rh - mu_h
        c2   = float(np.mean(d**2))
        c3   = float(np.mean(d**3))
        c4x  = float(np.mean(d**4)) - 3.0 * c2 * c2
        h_yr = h / self.ann

        # ── Jump params from moment system (Kou, common scale η) ──────────────
        #   λ_h = c4x·η⁴/24       (excess kurtosis identifies jump intensity)
        #   2p−1 = c3·η/(3·J)     (skewness identifies direction asymmetry)
        #   σ_c² = (c2 − J)/h_yr  (variance budget closes by construction)
        if c4x > 0 and c2 > 0:
            lam_h = c4x * eta**4 / 24.0
            J     = lam_h * 2.0 / (eta * eta)         # jump variance per horizon
            J     = min(J, 0.6 * c2)                  # keep ≥40% variance diffusive
            lam_h = J * eta * eta / 2.0               # re-sync λ_h with capped J
            jump_lambda = float(np.clip(lam_h / h_yr, 0.0, 150.0))
            if J > 1e-14:
                # Shrink the skew loading 50% toward symmetry: sample c3 of
                # overlapping h-day returns is noisy, and over-attributing skew
                # to one-sided jumps misprices the opposite tail badly.
                s = 0.5 * (c3 * eta / (3.0 * J))      # = (2p−1), shrunk
                jump_p_up = float(np.clip(0.5 * (1.0 + s), 0.10, 0.90))
            else:
                jump_p_up = 0.5
        else:
            jump_lambda = 0.0
            jump_p_up   = 0.5
            J           = 0.0
        jump_eta_plus  = eta
        jump_eta_minus = eta

        # ── σ long-run (θ): diffusive part of the h-day variance budget ───────
        sigma_lr = float(np.sqrt(max(c2 - J, 0.05 * c2) / h_yr))
        sigma_lr = float(np.clip(sigma_lr, 0.01, 1.0))

        # ── σ current (v0): EWMA on jump-censored daily returns ───────────────
        # Scaled so that EWMA_total/longrun_total ratio modulates the diffusive
        # σ_lr — preserves the variance decomposition while tracking regimes.
        clean = jw[np.abs(jw - med) <= thresh]
        lam_w = self.ewma_lambda
        var_ew = float(np.var(clean[:20], ddof=1)) if len(clean) > 21 else float(np.var(clean, ddof=1))
        for r in clean[20:]:
            var_ew = lam_w * var_ew + (1.0 - lam_w) * r * r
        var_lr_daily = float(np.var(clean, ddof=1)) + 1e-14
        ratio = float(np.clip(var_ew / var_lr_daily, 0.25, 4.0))
        # Asymmetric floor: vol spikes much faster than it decays, so betting
        # on very low current vol is the costliest density-forecast error.
        sigma = float(np.clip(sigma_lr * np.sqrt(ratio), 0.75 * sigma_lr, 1.0))
        sigma = float(np.clip(sigma, 0.01, 1.0))

        # ── Heston κ_H, ξ from blocked realized variance (jump-clipped) ───────
        # Block variances from m=5 obs carry χ² sampling noise with relative
        # variance ≈ 2/(m−1) = 0.5 even under constant vol. Without subtracting
        # it, ξ is inflated ~2-3× and the attenuated autocorrelation pushes κ_H
        # to its clamp. Decompose: Var(RV) = Var(v) + noise, correct both.
        hist = np.clip(log_returns[-min(n, self.heston_window):],
                       med - thresh, med + thresh)
        block  = 5
        blocks = len(hist) // block
        if blocks >= 10:
            rv = np.array([
                np.var(hist[i*block:(i+1)*block], ddof=1) * self.ann
                for i in range(blocks)
            ])
            mean_rv    = float(np.mean(rv)) + 1e-12
            var_rv_raw = float(np.var(rv, ddof=1))
            noise_var  = (2.0 / (block - 1)) * mean_rv * mean_rv
            var_signal = max(var_rv_raw - noise_var, 0.05 * var_rv_raw)
            signal_share = var_signal / (var_rv_raw + 1e-20)

            # Attenuation-corrected AR(1) of the latent variance
            acf1 = _lag1_autocorr(rv)
            phi  = float(np.clip(acf1 / max(signal_share, 0.1), 0.01, 0.97))
            # FX variance half-lives are weeks-to-months: κ_H ∈ [0.2, 8]
            kappa_H = float(np.clip(-np.log(phi) * (self.ann / block), 0.2, 8.0))

            # CIR stationary: Var(v) = ξ²θ/(2κ)  ⇒  ξ = sqrt(2κ·Var(v)/θ)
            xi = float(np.clip(np.sqrt(2.0 * kappa_H * var_signal / mean_rv),
                               0.05, 2.0))
        else:
            kappa_H = _FALLBACK["kappa_H"]
            xi      = _FALLBACK["xi"]

        # ── ρ leverage: corr(block return, next-block RV change) ──────────────
        # Sign-correct estimator: leverage means down-moves precede vol rises.
        if blocks >= 12:
            rblk = np.array([np.sum(hist[i*block:(i+1)*block]) for i in range(blocks)])
            drv  = rv[1:] - rv[:-1]
            if np.std(rblk[:-1]) > 0 and np.std(drv) > 0:
                rho = float(np.clip(np.corrcoef(rblk[:-1], drv)[0, 1], -0.9, 0.3))
            else:
                rho = _FALLBACK["rho"]
        else:
            rho = _FALLBACK["rho"]

        logger.debug(
            "Calibrated(h=%d): sigma=%.4f sigma_lr=%.4f kappa_H=%.2f xi=%.2f rho=%.2f "
            "lambda=%.1f p_up=%.2f eta=%.0f (n=%d, tail_n=%d)",
            h, sigma, sigma_lr, kappa_H, xi, rho,
            jump_lambda, jump_p_up, eta, n, n_jumps,
        )
        return CalibratedParams(
            sigma=sigma, sigma_lr=sigma_lr,
            kappa_H=kappa_H, xi=xi, rho=rho,
            jump_lambda=jump_lambda, jump_p_up=jump_p_up,
            jump_eta_plus=jump_eta_plus, jump_eta_minus=jump_eta_minus,
            n_obs=n, n_jumps=n_jumps,
        )


def _lag1_autocorr(x: np.ndarray) -> float:
    """Pearson lag-1 autocorrelation, clamped to (0, 1)."""
    if len(x) < 3:
        return 0.9
    mu  = np.mean(x)
    num = float(np.mean((x[:-1] - mu) * (x[1:] - mu)))
    den = float(np.var(x, ddof=0)) + 1e-20
    return float(np.clip(num / den, 0.01, 0.999))

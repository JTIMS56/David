#!/usr/bin/env python3
"""
Reproduces the two quantitative claims in docs/dhj-model-paper.tex that are
not produced by the C++ convergence suite.

  1. Proposition 1: the aggregate chiral charge satisfies, exactly,
         Q5(T) = delta_cp * exp(-(2*kappa + r_d) * T)
     Checked against the compiled engine across delta_cp in [-0.5, 0.5].

  2. Table 2: achievable directional accuracy under the model's own
     risk-neutral density versus the accuracy needed to break even on spread.

The convergence and martingale numbers in Table 1 and Section 4 come from
    cmake --build dirac_engine/build && ./dirac_engine/build/convergence_test

Usage:  python3 docs/paper_verification.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ── Proposition 1 ─────────────────────────────────────────────────────────────
# Engine defaults used in the paper (see services/dirac_predictor.py).
SIGMA = 0.082          # EURUSD annualised diffusive vol
R_D = 0.0525           # USD policy rate
KAPPA_0 = 1.0          # base Dirac mass
KAPPA_1 = 0.002        # vol-dependent correction
HORIZON_DAYS = 1.0


def verify_chiral_identity() -> bool:
    """Compare engine chiral charge against the closed form of Proposition 1."""
    from services.dirac_predictor import DiracPredictor

    predictor = DiracPredictor.instance()
    v = SIGMA * SIGMA
    kappa = KAPPA_0 + KAPPA_1 / v
    T = HORIZON_DAYS / 365.0
    decay = math.exp(-(2.0 * kappa + R_D) * T)

    print("Proposition 1:  Q5(T) = delta_cp * exp(-(2k + r_d) T)")
    print(f"  kappa(v) = {kappa:.4f}   decay factor = {decay:.6f}\n")
    print(f"  {'delta_cp':>9} {'Q5 engine':>13} {'Q5 closed form':>15} {'ratio':>9}")

    ok = True
    for delta_cp in (-0.5, -0.3, -0.1, 0.1, 0.3, 0.5):
        out = predictor.predict_dhj(
            pair="EURUSD", spot=1.0800, horizon_days=HORIZON_DAYS,
            n_paths=60, delta_cp=delta_cp, seed=42,
        )
        predicted = delta_cp * decay
        ratio = out.chiral_charge / predicted
        ok &= abs(ratio - 1.0) < 1e-5
        print(f"  {delta_cp:9.2f} {out.chiral_charge:13.6f} "
              f"{predicted:15.6f} {ratio:9.5f}")

    print(f"\n  {'PASS' if ok else 'FAIL'}: identity holds to 1e-5 across the range\n")
    return ok


def implied_rsi_thresholds() -> None:
    """Invert the signal thresholds through the delta_cp seeding rule."""
    v = SIGMA * SIGMA
    kappa = KAPPA_0 + KAPPA_1 / v
    decay = math.exp(-(2.0 * kappa + R_D) * HORIZON_DAYS / 365.0)
    print("Implied RSI thresholds (undamped / MACD-disagreement damped):")
    for label, q5 in (("STRONG", 0.15), ("MILD", 0.05)):
        band = 100.0 * q5 / decay
        print(f"  {label:6s}  |RSI - 50| > {band:5.2f}   (damped: {2*band:5.2f})")
    print()


# ── Table 2 ───────────────────────────────────────────────────────────────────
CARRY = 0.0525 - 0.0400    # EURUSD policy differential, 125 bp
SPOT = 1.08
PIP = 1e-4
SPREAD_PIPS = 1.2

HORIZONS = [
    ("1 day", 1 / 252),
    ("1 week", 5 / 252),
    ("1 month", 21 / 252),
    ("3 months", 63 / 252),
    ("1 year", 1.0),
]


def horizon_table() -> None:
    """Q-measure accuracy ceiling vs break-even accuracy, by horizon."""
    print("Table 2:  directional ceiling vs break-even after spread (EURUSD)")
    print(f"  {'horizon':>9} {'P_Q(up)':>9} {'break-even':>11} "
          f"{'edge (pp)':>10} {'E[pips]':>9}")

    for name, T in HORIZONS:
        m = (CARRY - 0.5 * SIGMA * SIGMA) * T
        s = SIGMA * math.sqrt(T)
        p_up = normal_cdf(m / s)
        # One-standard-deviation target and stop, in pips.
        d_pips = SPOT * SIGMA * math.sqrt(T) / PIP
        p_star = 0.5 + SPREAD_PIPS / (2.0 * d_pips)
        expectancy = d_pips * (2.0 * p_up - 1.0) - SPREAD_PIPS
        print(f"  {name:>9} {p_up:9.4f} {p_star:11.4f} "
              f"{(p_up - p_star) * 100:+10.2f} {expectancy:+9.2f}")
    print()


if __name__ == "__main__":
    horizon_table()
    implied_rsi_thresholds()
    try:
        passed = verify_chiral_identity()
    except FileNotFoundError as exc:
        print(f"Skipping Proposition 1 check: {exc}")
        print("Build the engine first: cmake --build dirac_engine/build --parallel 4")
        sys.exit(0)
    sys.exit(0 if passed else 1)

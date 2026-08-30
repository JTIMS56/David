"""
Validation for the volatility research module.

Same principle as the return backtester: prove the instrument detects known
structure and rejects noise BEFORE trusting anything it says about real markets.
"""
import math
import warnings

import numpy as np
import pytest

from services import volatility as vol

warnings.filterwarnings("ignore")


def _ohlc(true_ann_vol=0.16, n=2000, steps=200, seed=7):
    rng = np.random.default_rng(seed)
    ds = true_ann_vol / math.sqrt(252)
    o, h, l, c = (np.zeros(n) for _ in range(4))
    px = 100.0
    for i in range(n):
        path = px * np.exp(np.cumsum(rng.normal(0, ds / math.sqrt(steps), steps)))
        o[i], h[i], l[i], c[i] = px, path.max(), path.min(), path[-1]
        px = path[-1]
    return o, h, l, c


def test_estimators_recover_known_volatility():
    o, h, l, c = _ohlc(true_ann_vol=0.16, steps=200)
    assert abs(np.nanmean(vol.rv_close_to_close(c, 21)) - 16.0) < 2.0
    # Range estimators carry a downward discrete-sampling bias; they converge
    # up to truth as intraday sampling gets finer.
    assert abs(np.nanmean(vol.rv_parkinson(h[1:], l[1:], 21)) - 16.0) < 2.0
    assert abs(np.nanmean(vol.rv_garman_klass(o[1:], h[1:], l[1:], c[1:], 21)) - 16.0) < 2.5


def test_range_estimators_are_more_efficient():
    o, h, l, c = _ohlc(steps=200)
    assert np.nanstd(vol.rv_parkinson(h[1:], l[1:], 21)) < np.nanstd(vol.rv_close_to_close(c, 21))


def test_garch_beats_baselines_on_a_true_garch_process():
    rng = np.random.default_rng(11)
    w, a, b, n = 0.05, 0.10, 0.85, 2500
    hh = np.zeros(n); r = np.zeros(n); hh[0] = w / (1 - a - b)
    for t in range(1, n):
        hh[t] = w + a * r[t - 1] ** 2 + b * hh[t - 1]
        r[t] = rng.normal(0, math.sqrt(hh[t]))
    px = 100 * np.exp(np.cumsum(r / 100))
    res = vol.garch_walk_forward(px, train=1500, refit_every=100)
    assert res["garch11"]["qlike"] < res["constant_variance"]["qlike"]
    assert res["garch11"]["oos_r_squared"] > 0.02


def test_spillover_finds_true_source_and_ignores_noise():
    rng = np.random.default_rng(11)
    m = 1200
    lead = np.zeros(m); foll = np.zeros(m); third = np.zeros(m)
    lead[:2] = foll[:2] = third[:2] = 2.0
    for t in range(2, m):
        lead[t] = 0.90 * lead[t - 1] + rng.normal(0, 0.25) + 0.2
        foll[t] = 0.50 * foll[t - 1] + 0.60 * lead[t - 2] + rng.normal(0, 0.25)
        third[t] = 0.90 * third[t - 1] + rng.normal(0, 0.25) + 0.2
    dates = [f"d{i:05d}" for i in range(m)]
    series = {n: {"dates": dates, "vol": list(np.abs(v) + 1)}
              for n, v in (("LEADER", lead), ("FOLLOWER", foll), ("THIRD", third))}
    sp = vol.spillover_analysis(series, lags=5, horizon=10)

    links = {(g["from"], g["to"]) for g in sp["granger_significant_at_5pct"]}
    assert ("LEADER", "FOLLOWER") in links
    assert ("THIRD", "FOLLOWER") not in links

    net = sp["diebold_yilmaz"]["net_transmitter_pct"]
    assert max(net, key=net.get) == "LEADER"
    assert abs(net["THIRD"]) < 5.0        # independent series transmits ~nothing

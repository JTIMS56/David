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


def _intraday(true_ann=0.16, days=300, bars=78, jump_frac=0.0, seed=3):
    rng = np.random.default_rng(seed)
    ds = true_ann / math.sqrt(252)
    jump_days = set(rng.choice(days, size=int(days * jump_frac), replace=False)) if jump_frac else set()
    candles, px = [], 100.0
    for d in range(days):
        date = f"2026-{d + 1:05d}"
        for b in range(bars):
            r = rng.normal(0, ds / math.sqrt(bars))
            if d in jump_days and b == bars // 2:
                r += rng.choice([-1, 1]) * 0.02
            nxt = px * math.exp(r)
            candles.append({"date": date, "time": f"{d}-{b}", "open": px,
                            "high": max(px, nxt), "low": min(px, nxt), "close": nxt})
            px = nxt
    return candles


def test_rogers_satchell_is_drift_robust():
    """Parkinson assumes zero drift; Rogers-Satchell does not. With strong
    drift added, RS should stay closer to the true diffusion volatility."""
    rng = np.random.default_rng(5)
    n, steps, true_ann, drift = 1500, 200, 0.16, 0.40      # 40%/yr drift
    ds, dd = true_ann / math.sqrt(252), drift / 252
    o, h, l, c = (np.zeros(n) for _ in range(4))
    px = 100.0
    for i in range(n):
        inc = rng.normal(dd / steps, ds / math.sqrt(steps), steps)
        path = px * np.exp(np.cumsum(inc))
        o[i], h[i], l[i], c[i] = px, path.max(), path.min(), path[-1]
        px = path[-1]
    rs = np.nanmean(vol.rv_rogers_satchell(o, h, l, c, 21))
    pk = np.nanmean(vol.rv_parkinson(h, l, 21))
    assert abs(rs - 16.0) <= abs(pk - 16.0) + 1.0
    assert 12.0 < rs < 20.0


def test_yang_zhang_captures_overnight_gaps():
    """Yang-Zhang includes close-to-open variance; Parkinson ignores it, so
    with real gaps present YZ must report higher volatility."""
    rng = np.random.default_rng(6)
    n, steps = 1200, 200
    ds = 0.16 / math.sqrt(252)
    o, h, l, c = (np.zeros(n) for _ in range(4))
    px = 100.0
    for i in range(n):
        px *= math.exp(rng.normal(0, 0.008))          # overnight gap
        path = px * np.exp(np.cumsum(rng.normal(0, ds / math.sqrt(steps), steps)))
        o[i], h[i], l[i], c[i] = px, path.max(), path.min(), path[-1]
        px = path[-1]
    yz = np.nanmean(vol.rv_yang_zhang(o, h, l, c, 21))
    pk = np.nanmean(vol.rv_parkinson(h, l, 21))
    assert yz > pk, "YZ must exceed Parkinson when overnight gaps are present"


def test_bipower_variation_strips_jumps():
    candles = _intraday(true_ann=0.16, days=400, jump_frac=0.10)
    res = vol.realized_variance_by_day(candles)
    total = np.mean(res["realized_vol_ann_pct"])
    cont = np.mean(res["continuous_vol_ann_pct"])
    assert cont < total, "bipower must be below total RV when jumps exist"
    assert abs(cont - 16.0) < 3.0, "continuous component should recover the diffusion"
    assert res["jump_share_pct"] > 5.0


def test_no_jumps_means_no_jump_component():
    res = vol.realized_variance_by_day(_intraday(jump_frac=0.0, days=200))
    assert res["jump_share_pct"] < 12.0, "should not invent jumps in a pure diffusion"


def test_har_rv_beats_random_walk_on_heterogeneous_volatility():
    """
    HAR exists because realized volatility has LONG MEMORY — components
    decaying at daily, weekly and monthly speeds (Corsi's heterogeneous market
    hypothesis). This simulates that structure and requires HAR to beat a
    random walk on it.

    Worth knowing the contrast: on a single AR(1) log-RV process, HAR does NOT
    beat the random walk (measured: RMSE 0.258 vs 0.253 across three seeds).
    When yesterday is a sufficient statistic, extra horizons only add variance.
    The weekly and monthly terms earn their place only when the data actually
    has multi-horizon persistence.
    """
    rng = np.random.default_rng(9)
    n = 1200
    # Three superimposed components: short, medium, long persistence.
    d = np.zeros(n); w = np.zeros(n); m = np.zeros(n)
    for t in range(1, n):
        d[t] = 0.50 * d[t - 1] + rng.normal(0, 0.30)
        w[t] = 0.90 * w[t - 1] + rng.normal(0, 0.15)
        m[t] = 0.99 * m[t - 1] + rng.normal(0, 0.05)
    lrv = math.log(1e-4) + d + w + m
    har = vol.har_rv_forecast(list(np.exp(lrv)))
    assert har["beats_random_walk"], "HAR must beat RW on heterogeneous volatility"
    assert har["oos_r_squared"] > 0.4
    # The weekly/monthly terms should carry real weight, not collapse to daily.
    assert abs(har["coefficients"]["weekly"]) + abs(har["coefficients"]["monthly"]) > 0.05

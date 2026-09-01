"""
Market Data Service
────────────────────────────────────────────────────────────────────────────
Provides real-time and historical FX price data.

Modes:
  simulation – Geometric Brownian Motion with realistic per-pair volatilities
  live       – Alpha Vantage free API (requires ALPHA_VANTAGE_KEY)
  oanda      – OANDA v20 streaming prices (requires OANDA_API_KEY + OANDA_ACCOUNT_ID)
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

import httpx
import numpy as np
from scipy.signal import lfilter

from config import settings

# ── Per-pair configuration ────────────────────────────────────────────────────

# Every tradable instrument. "pip" is the quote increment used for all pip-denominated
# maths (stops, targets, ATR); for indices and metals it is one price point/dollar.
# min/max_stop_pips are per-instrument because a 50-pip cap is meaningless on an
# index quoted in the thousands — they are calibrated to roughly 0.5x and 4x a
# typical daily ATR for each instrument.
PAIR_CONFIG: Dict[str, dict] = {
    # ── FX majors ─────────────────────────────────────────────────────────────
    "EUR/USD": {"base_price": 1.0850, "daily_vol": 0.0055, "spread": 0.00012, "pip": 0.0001,
                "asset_class": "fx", "min_stop_pips": 15,  "max_stop_pips": 50},
    "GBP/USD": {"base_price": 1.2650, "daily_vol": 0.0072, "spread": 0.00016, "pip": 0.0001,
                "asset_class": "fx", "min_stop_pips": 15,  "max_stop_pips": 50},
    "USD/JPY": {"base_price": 149.50, "daily_vol": 0.0060, "spread": 0.015,   "pip": 0.01,
                "asset_class": "fx", "min_stop_pips": 15,  "max_stop_pips": 50},
    "AUD/USD": {"base_price": 0.6520, "daily_vol": 0.0065, "spread": 0.00014, "pip": 0.0001,
                "asset_class": "fx", "min_stop_pips": 15,  "max_stop_pips": 50},
    "USD/CAD": {"base_price": 1.3650, "daily_vol": 0.0052, "spread": 0.00015, "pip": 0.0001,
                "asset_class": "fx", "min_stop_pips": 15,  "max_stop_pips": 50},
    "EUR/GBP": {"base_price": 0.8580, "daily_vol": 0.0043, "spread": 0.00013, "pip": 0.0001,
                "asset_class": "fx", "min_stop_pips": 15,  "max_stop_pips": 50},
    "NZD/USD": {"base_price": 0.6020, "daily_vol": 0.0068, "spread": 0.00018, "pip": 0.0001,
                "asset_class": "fx", "min_stop_pips": 15,  "max_stop_pips": 50},
    "USD/CHF": {"base_price": 0.8980, "daily_vol": 0.0050, "spread": 0.00014, "pip": 0.0001,
                "asset_class": "fx", "min_stop_pips": 15,  "max_stop_pips": 50},
    # ── Equity index CFDs (1 unit = 1 index point of exposure) ───────────────
    "SPX500": {"base_price": 6800.0,  "daily_vol": 0.0090, "spread": 0.50, "pip": 1.0,
               "asset_class": "index", "min_stop_pips": 25,  "max_stop_pips": 300},
    "NAS100": {"base_price": 25000.0, "daily_vol": 0.0115, "spread": 1.60, "pip": 1.0,
               "asset_class": "index", "min_stop_pips": 80,  "max_stop_pips": 1200},
    "US30":   {"base_price": 48000.0, "daily_vol": 0.0080, "spread": 2.20, "pip": 1.0,
               "asset_class": "index", "min_stop_pips": 100, "max_stop_pips": 1600},
    "DE30":   {"base_price": 24500.0, "daily_vol": 0.0100, "spread": 1.20, "pip": 1.0,
               "asset_class": "index", "min_stop_pips": 70,  "max_stop_pips": 1000},
    "UK100":  {"base_price": 9600.0,  "daily_vol": 0.0075, "spread": 1.00, "pip": 1.0,
               "asset_class": "index", "min_stop_pips": 30,  "max_stop_pips": 400},
    # ── Metals (1 unit = 1 oz) ────────────────────────────────────────────────
    "XAU/USD": {"base_price": 3400.0, "daily_vol": 0.0095, "spread": 0.30, "pip": 1.0,
                "asset_class": "metal", "min_stop_pips": 12, "max_stop_pips": 160},
    "XAG/USD": {"base_price": 40.00,  "daily_vol": 0.0170, "spread": 0.020, "pip": 0.01,
                "asset_class": "metal", "min_stop_pips": 25, "max_stop_pips": 300},
}


def asset_class(pair: str) -> str:
    return PAIR_CONFIG.get(pair, {}).get("asset_class", "fx")


def stop_bounds(pair: str) -> tuple:
    """(min_stop_pips, max_stop_pips) for an instrument, defaulting to FX values."""
    cfg = PAIR_CONFIG.get(pair, {})
    return float(cfg.get("min_stop_pips", 15)), float(cfg.get("max_stop_pips", 50))

HISTORY_DEPTH = 500  # ticks stored per pair in memory


class PriceBar:
    __slots__ = ("timestamp", "bid", "ask", "mid")

    def __init__(self, timestamp: datetime, bid: float, ask: float):
        self.timestamp = timestamp
        self.bid = round(bid, 6)
        self.ask = round(ask, 6)
        self.mid = round((bid + ask) / 2, 6)

    def to_dict(self) -> dict:
        cfg = next(
            (v for k, v in PAIR_CONFIG.items() if True),  # pair resolved by caller
            {"pip": 0.0001},
        )
        return {
            "timestamp": self.timestamp.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "mid": self.mid,
        }


class MarketDataService:
    def __init__(self) -> None:
        self._prices: Dict[str, PriceBar] = {}
        self._history: Dict[str, Deque[PriceBar]] = {
            p: deque(maxlen=HISTORY_DEPTH) for p in PAIR_CONFIG
        }
        # pair -> (n_bars, last_mid, indicators). Indicators are a pure
        # function of the price series, so they only need recomputing when a
        # new quote arrives. The ensemble reads indicators for every USD pair
        # to build its breadth score, so without this each forecast recomputed
        # the same series up to nine times.
        self._ind_cache: Dict[str, tuple] = {}
        self._seed_simulation()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ── Initialisation ────────────────────────────────────────────────────────

    def _seed_simulation(self) -> None:
        """
        Seed initial prices and 200 bars of history — SIMULATION MODE ONLY.

        In live modes nothing is seeded at all. Fabricated bars anchored to a
        hardcoded base price can only ever contaminate a real series: the gap
        between the seed and the true market price becomes a phantom bar that
        inflates ATR by an order of magnitude and pins RSI at its extremes.
        Starting empty makes that class of corruption impossible rather than
        merely recoverable.
        """
        self._synthetic = set()
        if settings.market_data_mode != "simulation":
            return
        self._synthetic = set(PAIR_CONFIG)
        now = datetime.now(timezone.utc)
        for pair, cfg in PAIR_CONFIG.items():
            price = cfg["base_price"]
            spread = cfg["spread"]
            dt = 1 / (252 * 24 * 60)  # one-minute bars
            sigma = cfg["daily_vol"] / math.sqrt(252)
            # warm-up history
            for i in range(200):
                drift = 0.0
                shock = random.gauss(0, sigma * math.sqrt(dt * 1440))
                price = price * math.exp(drift + shock)
                bar = PriceBar(now, price - spread / 2, price + spread / 2)
                self._history[pair].append(bar)
            self._prices[pair] = self._history[pair][-1]

    # ── Tick generation ───────────────────────────────────────────────────────

    TICK_INTERVAL_SECONDS: float = 5.0  # must match asyncio.sleep() in the loop methods

    def _next_tick(self, pair: str) -> PriceBar:
        cfg = PAIR_CONFIG[pair]
        current = self._prices[pair].mid
        spread = cfg["spread"]
        # sigma per tick so that daily variance = (daily_vol × price)²
        # ticks_per_day = seconds_per_day / tick_interval
        ticks_per_day = 24 * 3600 / self.TICK_INTERVAL_SECONDS  # 17 280 for 5-second ticks
        sigma = cfg["daily_vol"] / math.sqrt(ticks_per_day)
        shock = random.gauss(0, sigma)
        new_mid = current * math.exp(shock)
        return PriceBar(datetime.now(timezone.utc), new_mid - spread / 2, new_mid + spread / 2)

    async def _simulation_loop(self, interval: float = 5.0) -> None:
        while self._running:
            for pair in PAIR_CONFIG:
                bar = self._next_tick(pair)
                self._prices[pair] = bar
                self._history[pair].append(bar)
            await asyncio.sleep(interval)

    # ── Live data ─────────────────────────────────────────────────────────────

    async def _fetch_live_rates(self) -> int:
        """Fetch current bid/ask from Alpha Vantage. Returns number of pairs updated."""
        if not settings.alpha_vantage_key:
            return 0
        updated = 0
        async with httpx.AsyncClient(timeout=15) as client:
            for pair in list(PAIR_CONFIG.keys()):
                from_cur, to_cur = pair.split("/")
                url = (
                    "https://www.alphavantage.co/query"
                    f"?function=CURRENCY_EXCHANGE_RATE"
                    f"&from_currency={from_cur}&to_currency={to_cur}"
                    f"&apikey={settings.alpha_vantage_key}"
                )
                try:
                    r = await client.get(url)
                    data = r.json().get("Realtime Currency Exchange Rate", {})
                    bid = float(data.get("8. Bid Price", 0))
                    ask = float(data.get("9. Ask Price", 0))
                    if bid > 0 and ask > 0:
                        self._record_live(pair, PriceBar(datetime.now(timezone.utc), bid, ask))
                        updated += 1
                except Exception as exc:
                    import logging as _log
                    _log.getLogger("popper.market_data").warning(
                        "Live rate fetch failed for %s: %s", pair, exc
                    )
                # Alpha Vantage: 5 req/min on free tier — wait 13s between calls
                await asyncio.sleep(13)
        return updated

    async def _live_loop(self) -> None:
        """
        Hybrid live mode: GBM simulation for continuous ticks, real rates
        fetched periodically to keep price levels accurate.

        Quota-safe schedule (Alpha Vantage free tier: 25 calls/day):
          8 pairs × 13s gaps = ~104s per refresh cycle
          refresh_interval = live_refresh_interval setting (default 6h)
          calls/day = 8 × (86400 / live_refresh_interval) ≤ 25 at 6h
        """
        import logging as _log
        logger = _log.getLogger("popper.market_data")

        # Anchor simulation to real prices at startup
        logger.info("Live mode: fetching initial real rates from Alpha Vantage...")
        n = await self._fetch_live_rates()
        logger.info("Live mode: anchored %d pairs to real rates", n)

        last_live_fetch = time.monotonic()
        refresh_interval = settings.live_refresh_interval

        while self._running:
            # GBM tick — keeps charts smooth between live fetches. Skip any
            # pair that has not received a real anchor quote yet; there is
            # nothing legitimate to interpolate from.
            for pair in PAIR_CONFIG:
                if pair not in self._prices:
                    continue
                bar = self._next_tick(pair)
                self._prices[pair] = bar
                self._history[pair].append(bar)

            # Periodic live refresh
            if time.monotonic() - last_live_fetch >= refresh_interval:
                logger.info("Live mode: refreshing real rates (interval=%ds)", refresh_interval)
                n = await self._fetch_live_rates()
                logger.info("Live mode: refreshed %d pairs", n)
                last_live_fetch = time.monotonic()

            await asyncio.sleep(5)

    # ── OANDA price feed ──────────────────────────────────────────────────────

    async def _oanda_poll_loop(self) -> None:
        """Poll OANDA pricing API every 5 seconds (fallback when streaming fails)."""
        from services.oanda_client import oanda_client, OANDA_TO_PAIR
        import logging as _log
        logger = _log.getLogger("popper.market_data")
        logger.info("OANDA market data: polling mode (streaming unavailable)")
        while self._running:
            try:
                data = await oanda_client.get_prices(list(PAIR_CONFIG.keys()))
                updated = 0
                for price_data in data.get("prices", []):
                    instrument = price_data.get("instrument", "")
                    pair = OANDA_TO_PAIR.get(instrument)
                    if not pair or not price_data.get("tradeable", True):
                        continue
                    bid = float(price_data["bids"][0]["price"])
                    ask = float(price_data["asks"][0]["price"])
                    self._record_live(pair, PriceBar(datetime.now(timezone.utc), bid, ask))
                    updated += 1
                if updated:
                    logger.debug("OANDA poll: updated %d pairs", updated)
            except Exception as exc:
                logger.warning("OANDA price poll failed: %s", exc)
            await asyncio.sleep(5)

    async def _oanda_stream_loop(self) -> None:
        """
        Consume OANDA SSE price stream. Reconnects on disconnect with 5s backoff.
        Falls back to _oanda_poll_loop() after 3 consecutive quick disconnects
        (connected < 30s), which signals that streaming is unavailable.
        """
        from services.oanda_client import oanda_client, OANDA_TO_PAIR
        import logging as _log
        logger = _log.getLogger("popper.market_data")

        pairs = list(PAIR_CONFIG.keys())
        quick_fail_count = 0
        _QUICK_FAIL_SECS = 30.0
        _QUICK_FAIL_LIMIT = 3

        while self._running:
            connected_at = time.monotonic()
            try:
                logger.info("OANDA streaming: connecting to price stream")
                async for tick in oanda_client.stream_prices(pairs):
                    if not self._running:
                        return
                    tick_type = tick.get("type")
                    if tick_type == "PRICE":
                        instrument = tick.get("instrument", "")
                        pair = OANDA_TO_PAIR.get(instrument)
                        if pair and tick.get("tradeable", True):
                            bid = float(tick["bids"][0]["price"])
                            ask = float(tick["asks"][0]["price"])
                            self._record_live(pair, PriceBar(datetime.now(timezone.utc), bid, ask))
                    elif tick_type == "HEARTBEAT":
                        logger.debug("OANDA stream heartbeat")
                elapsed = time.monotonic() - connected_at
                logger.warning("OANDA stream closed by server after %.0fs", elapsed)
            except Exception as exc:
                elapsed = time.monotonic() - connected_at
                logger.warning("OANDA stream error after %.0fs: %s", elapsed, exc)

            if not self._running:
                return

            elapsed = time.monotonic() - connected_at
            if elapsed < _QUICK_FAIL_SECS:
                quick_fail_count += 1
                if quick_fail_count >= _QUICK_FAIL_LIMIT:
                    logger.error(
                        "OANDA stream: %d quick disconnects in a row — "
                        "falling back to REST polling",
                        quick_fail_count,
                    )
                    await self._oanda_poll_loop()
                    return
            else:
                quick_fail_count = 0

            await asyncio.sleep(5)

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        if settings.market_data_mode == "oanda" and settings.oanda_api_key:
            self._task = asyncio.create_task(self._oanda_stream_loop())
        elif settings.market_data_mode == "live" and settings.alpha_vantage_key:
            self._task = asyncio.create_task(self._live_loop())
        else:
            if settings.market_data_mode in ("live", "oanda") and not (
                settings.alpha_vantage_key or settings.oanda_api_key
            ):
                import logging as _log
                _log.getLogger("popper.market_data").warning(
                    "MARKET_DATA_MODE=%s but no API key set — falling back to simulation",
                    settings.market_data_mode,
                )
            self._task = asyncio.create_task(self._simulation_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()

    def feed_health(self) -> dict:
        """
        Age of the freshest price on the book. A live feed refreshes every few
        seconds; a large age means the feed has died and every downstream
        number (indicators, forecasts, evaluations) is being computed against
        frozen prices. Read by the watchdog and the status endpoint.
        """
        if not self._prices:
            return {"healthy": False, "newest_age_seconds": None, "pairs": 0,
                    "reason": "no prices on book"}
        now = datetime.now(timezone.utc)
        ages = []
        for bar in self._prices.values():
            ts = bar.timestamp if bar.timestamp.tzinfo else bar.timestamp.replace(tzinfo=timezone.utc)
            ages.append((now - ts).total_seconds())
        newest = min(ages)
        return {
            "healthy": newest <= settings.feed_stale_alarm_seconds,
            "newest_age_seconds": round(newest, 1),
            "stalest_age_seconds": round(max(ages), 1),
            "pairs": len(self._prices),
            "mode": settings.market_data_mode,
        }

    async def watchdog_loop(self, interval: float = 60.0) -> None:
        """
        Alarm when the price feed stops updating. Frozen prices are the most
        dangerous failure this platform has: nothing crashes, but every forecast
        is logged against a stale spot and later evaluated against the same
        stale spot, so measured moves collapse to zero and accuracy statistics
        silently become meaningless.
        """
        import logging as _log
        log = _log.getLogger("popper.market_data")
        while self._running:
            await asyncio.sleep(interval)
            h = self.feed_health()
            if not h["healthy"]:
                log.critical(
                    "PRICE FEED STALE — freshest quote is %.0fs old across %d "
                    "instruments (mode=%s). Forecasts and evaluations computed "
                    "now are invalid. Check OANDA connectivity and quarantined "
                    "instruments.",
                    h["newest_age_seconds"] or -1, h["pairs"], h["mode"],
                )

    async def bootstrap_history(self) -> dict:
        """
        Fill each instrument's history with REAL recent candles from the broker.

        Without this the platform faces a bad choice: fabricate synthetic
        warm-up bars (which contaminate every indicator) or wait for live ticks
        to accumulate (which blocks trading during the warm-up, and restarts
        the clock on every restart since history is in memory). Real candles
        give correct indicators from the first cycle.

        Best-effort per instrument; anything that fails simply accumulates from
        live ticks as before.
        """
        from services.oanda_client import oanda_client, PAIR_TO_OANDA
        import logging as _log
        log = _log.getLogger("popper.market_data")

        loaded, failed = {}, []
        now = datetime.now(timezone.utc)
        for pair, instrument in PAIR_TO_OANDA.items():
            if pair not in self._history:
                continue
            candles = await oanda_client.get_intraday_candles(instrument, "M5", 200)
            if len(candles) < 30:
                failed.append(pair)
                continue
            self._history[pair].clear()
            self._synthetic.discard(pair)
            for bid, ask in candles:
                self._history[pair].append(PriceBar(now, bid, ask))
            self._prices[pair] = self._history[pair][-1]
            loaded[pair] = len(candles)

        log.info("History bootstrap: %d instruments loaded from real candles%s",
                 len(loaded), f", {len(failed)} pending live ticks: {failed}" if failed else "")
        return {"loaded": loaded, "failed": failed}

    def _record_live(self, pair: str, bar: PriceBar) -> None:
        """
        Ingest a real broker quote.

        The first real quote for an instrument PURGES the synthetic warm-up
        history. Blending them puts a fabricated gap between the seeded base
        price and the real market price into the series — EUR/USD seeded at
        1.0850 against a real 1.16 is a phantom 750-pip bar, which inflates ATR
        by an order of magnitude, pins RSI at its extremes, and makes every
        ATR-derived stop exceed the gate's maximum. Indicators then describe a
        market that does not exist.
        """
        if pair in self._synthetic:
            self._history[pair].clear()
            self._synthetic.discard(pair)
            import logging as _log
            _log.getLogger("popper.market_data").info(
                "%s: purged synthetic warm-up history on first live quote", pair
            )
        self._prices[pair] = bar
        self._history[pair].append(bar)

    def real_bar_count(self, pair: str) -> int:
        """Genuine broker bars held for a pair (0 while still synthetic)."""
        if pair in self._synthetic:
            return 0
        return len(self._history.get(pair, ()))

    def get_price(self, pair: str) -> Optional[PriceBar]:
        return self._prices.get(pair)

    def get_all_prices(self) -> Dict[str, PriceBar]:
        return dict(self._prices)

    def get_rates(self) -> Dict[str, float]:
        """Return {pair: mid_price} — convenience helper for model endpoints."""
        return {pair: bar.mid for pair, bar in self._prices.items()}

    def get_history(self, pair: str, n: int = 100) -> List[PriceBar]:
        hist = self._history.get(pair, deque())
        bars = list(hist)
        return bars[-n:] if len(bars) > n else bars

    def get_pip_size(self, pair: str) -> float:
        return PAIR_CONFIG.get(pair, {}).get("pip", 0.0001)

    def price_to_pips(self, pair: str, price_diff: float) -> float:
        pip = self.get_pip_size(pair)
        return price_diff / pip

    # ── Technical indicators ──────────────────────────────────────────────────

    def calculate_indicators(self, pair: str) -> dict:
        # In live modes, refuse to compute until enough GENUINE broker bars have
        # accumulated. Returning indicators built on synthetic warm-up data (or
        # on a handful of real bars right after the purge) hands the agent
        # numbers that describe no real market.
        if settings.market_data_mode != "simulation" and self.real_bar_count(pair) < 30:
            return {}
        _hist = self._history.get(pair)
        _last = self._prices.get(pair)
        _key = (len(_hist) if _hist is not None else 0, _last.mid if _last else None)
        _hit = self._ind_cache.get(pair)
        if _hit is not None and _hit[0] == _key:
            return _hit[1]

        bars = self.get_history(pair, 200)
        if len(bars) < 30:
            return {}

        # Self-heal: a price series spanning a discontinuity produces an ATR
        # many multiples of anything the instrument really does. Serving those
        # numbers is worse than serving none — they inflate every ATR-derived
        # stop past the gate's maximum and freeze trading silently. Discard the
        # series and rebuild from live quotes instead.
        if settings.market_data_mode != "simulation":
            _pip = PAIR_CONFIG.get(pair, {}).get("pip", 0.0001)
            _rng = (max(b.mid for b in bars) - min(b.mid for b in bars)) / _pip
            _sane_max = stop_bounds(pair)[1] * 6
            if _rng > _sane_max:
                import logging as _log
                _log.getLogger("popper.market_data").critical(
                    "%s: price history spans %.0f pips (sane max %.0f) — "
                    "discontinuity detected, purging and rebuilding from live quotes",
                    pair, _rng, _sane_max,
                )
                self._history[pair].clear()
                self._synthetic.discard(pair)
                return {}

        closes = np.array([b.mid for b in bars])

        def sma(arr: np.ndarray, n: int) -> float:
            return float(np.mean(arr[-n:])) if len(arr) >= n else float(np.mean(arr))

        def ema(arr: np.ndarray, n: int) -> np.ndarray:
            """
            Exponential moving average, y[i] = k*x[i] + (1-k)*y[i-1], y[0] = x[0].

            Expressed as a first-order IIR filter rather than a Python loop.
            The recurrence is identical — the initial condition (1-k)*x[0] makes
            y[0] collapse to x[0] — but it runs in vectorised C instead of ~200
            interpreted iterations per call. This was 64% of indicator latency.
            """
            k = 2 / (n + 1)
            return lfilter([k], [1.0, -(1.0 - k)], arr, zi=[(1.0 - k) * arr[0]])[0]

        # RSI
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        period = 14
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else np.mean(gains)
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else np.mean(losses)
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema12 = ema(closes, 12)
        ema26 = ema(closes, 26)
        macd_line = ema12 - ema26
        signal_line = ema(macd_line, 9)
        macd_hist = macd_line[-1] - signal_line[-1]

        # Bollinger Bands (20, 2)
        bb_period = 20
        bb_mid = sma(closes, bb_period)
        bb_std = float(np.std(closes[-bb_period:]))
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        current = closes[-1]
        bb_pct = (current - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

        # Moving averages
        sma20 = sma(closes, 20)
        sma50 = sma(closes, 50)
        sma200 = sma(closes, min(200, len(closes)))

        # Volatility — realized range of the MID price over a recent window.
        # NOTE: the previous implementation computed (ask - bid), i.e. the bid/ask
        # SPREAD, not price movement — so "atr" sat ~constant at the spread (~1.5p)
        # and never reflected how much price was actually moving. A PriceBar has no
        # intrabar high/low (only a snapshot bid/ask/mid), so true range is measured
        # from how far the mid has ranged over the lookback window.
        vol_window = min(120, len(closes))
        recent = closes[-vol_window:]
        atr = float(np.max(recent) - np.min(recent))

        # Trend detection
        trend = "NEUTRAL"
        if current > sma20 > sma50:
            trend = "BULLISH"
        elif current < sma20 < sma50:
            trend = "BEARISH"

        pip = self.get_pip_size(pair)
        _out = {
            "pair": pair,
            "current_price": round(current, 6),
            "rsi": round(rsi, 2),
            "macd": round(float(macd_line[-1]), 6),
            "macd_signal": round(float(signal_line[-1]), 6),
            "macd_histogram": round(macd_hist, 6),
            "bb_upper": round(bb_upper, 6),
            "bb_mid": round(bb_mid, 6),
            "bb_lower": round(bb_lower, 6),
            "bb_pct_b": round(bb_pct, 4),
            "sma20": round(sma20, 6),
            "sma50": round(sma50, 6),
            "sma200": round(sma200, 6),
            "atr": round(atr, 6),
            "atr_pips": round(atr / pip, 1),
            "trend": trend,
        }
        self._ind_cache[pair] = (_key, _out)
        return _out


# Singleton
market_data = MarketDataService()

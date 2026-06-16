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

from config import settings

# ── Per-pair configuration ────────────────────────────────────────────────────

PAIR_CONFIG: Dict[str, dict] = {
    "EUR/USD": {"base_price": 1.0850, "daily_vol": 0.0055, "spread": 0.00012, "pip": 0.0001},
    "GBP/USD": {"base_price": 1.2650, "daily_vol": 0.0072, "spread": 0.00016, "pip": 0.0001},
    "USD/JPY": {"base_price": 149.50, "daily_vol": 0.0060, "spread": 0.015,   "pip": 0.01},
    "AUD/USD": {"base_price": 0.6520, "daily_vol": 0.0065, "spread": 0.00014, "pip": 0.0001},
    "USD/CAD": {"base_price": 1.3650, "daily_vol": 0.0052, "spread": 0.00015, "pip": 0.0001},
    "EUR/GBP": {"base_price": 0.8580, "daily_vol": 0.0043, "spread": 0.00013, "pip": 0.0001},
    "NZD/USD": {"base_price": 0.6020, "daily_vol": 0.0068, "spread": 0.00018, "pip": 0.0001},
    "USD/CHF": {"base_price": 0.8980, "daily_vol": 0.0050, "spread": 0.00014, "pip": 0.0001},
}

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
        self._seed_simulation()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ── Initialisation ────────────────────────────────────────────────────────

    def _seed_simulation(self) -> None:
        """Seed initial prices and generate 200 bars of history."""
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
                        bar = PriceBar(datetime.now(timezone.utc), bid, ask)
                        self._prices[pair] = bar
                        self._history[pair].append(bar)
                        updated += 1
                except Exception as exc:
                    import logging as _log
                    _log.getLogger("david.market_data").warning(
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
        logger = _log.getLogger("david.market_data")

        # Anchor simulation to real prices at startup
        logger.info("Live mode: fetching initial real rates from Alpha Vantage...")
        n = await self._fetch_live_rates()
        logger.info("Live mode: anchored %d pairs to real rates", n)

        last_live_fetch = time.monotonic()
        refresh_interval = settings.live_refresh_interval

        while self._running:
            # GBM tick — keeps charts smooth between live fetches
            for pair in PAIR_CONFIG:
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

    async def _oanda_loop(self) -> None:
        """Poll OANDA pricing API every 5 seconds for real bid/ask prices."""
        from services.oanda_client import oanda_client, OANDA_TO_PAIR
        import logging as _log
        logger = _log.getLogger("david.market_data")
        logger.info("OANDA market data: starting price feed")
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
                    bar = PriceBar(datetime.now(timezone.utc), bid, ask)
                    self._prices[pair] = bar
                    self._history[pair].append(bar)
                    updated += 1
                if updated:
                    logger.debug("OANDA: updated %d pairs", updated)
            except Exception as exc:
                logger.warning("OANDA price poll failed: %s", exc)
            await asyncio.sleep(5)

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        if settings.market_data_mode == "oanda" and settings.oanda_api_key:
            self._task = asyncio.create_task(self._oanda_loop())
        elif settings.market_data_mode == "live" and settings.alpha_vantage_key:
            self._task = asyncio.create_task(self._live_loop())
        else:
            if settings.market_data_mode in ("live", "oanda") and not (
                settings.alpha_vantage_key or settings.oanda_api_key
            ):
                import logging as _log
                _log.getLogger("david.market_data").warning(
                    "MARKET_DATA_MODE=%s but no API key set — falling back to simulation",
                    settings.market_data_mode,
                )
            self._task = asyncio.create_task(self._simulation_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()

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
        bars = self.get_history(pair, 200)
        if len(bars) < 30:
            return {}

        closes = np.array([b.mid for b in bars])

        def sma(arr: np.ndarray, n: int) -> float:
            return float(np.mean(arr[-n:])) if len(arr) >= n else float(np.mean(arr))

        def ema(arr: np.ndarray, n: int) -> np.ndarray:
            k = 2 / (n + 1)
            result = np.zeros_like(arr)
            result[0] = arr[0]
            for i in range(1, len(arr)):
                result[i] = arr[i] * k + result[i - 1] * (1 - k)
            return result

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

        # ATR (14)
        highs = np.array([b.ask for b in bars])
        lows = np.array([b.bid for b in bars])
        tr = np.maximum(highs - lows, np.abs(highs[1:] - closes[:-1], where=True) if False else highs - lows)
        atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))

        # Trend detection
        trend = "NEUTRAL"
        if current > sma20 > sma50:
            trend = "BULLISH"
        elif current < sma20 < sma50:
            trend = "BEARISH"

        pip = self.get_pip_size(pair)
        return {
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


# Singleton
market_data = MarketDataService()

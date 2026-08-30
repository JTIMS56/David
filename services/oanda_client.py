"""
OANDA v20 REST API Client
────────────────────────────────────────────────────────────────────────────
Async client for OANDA's v20 REST API. Supports both practice and live
environments. Used by market_data (price feed) and order_service (execution).
"""
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, Optional

import httpx

from config import settings

logger = logging.getLogger("popper.oanda_client")

# ── Pair mappings ─────────────────────────────────────────────────────────────

PAIR_TO_OANDA: dict[str, str] = {
    "EUR/USD": "EUR_USD",
    "GBP/USD": "GBP_USD",
    "USD/JPY": "USD_JPY",
    "AUD/USD": "AUD_USD",
    "USD/CAD": "USD_CAD",
    "EUR/GBP": "EUR_GBP",
    "NZD/USD": "NZD_USD",
    "USD/CHF": "USD_CHF",
    # Index and metal CFDs
    "SPX500":  "SPX500_USD",
    "NAS100":  "NAS100_USD",
    "US30":    "US30_USD",
    "DE30":    "DE30_EUR",
    "UK100":   "UK100_GBP",
    "XAU/USD": "XAU_USD",
    "XAG/USD": "XAG_USD",
}

OANDA_TO_PAIR: dict[str, str] = {v: k for k, v in PAIR_TO_OANDA.items()}

# JPY pairs use 3 decimal places; all others use 5
_JPY_PAIRS = {"USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY", "NZD_JPY", "CHF_JPY"}


# Index/metal CFDs quote in points, not 5-decimal FX pips
_POINT_INSTRUMENTS = {"SPX500_USD", "NAS100_USD", "US30_USD", "DE30_EUR",
                      "UK100_GBP", "XAU_USD"}


def _price_decimals(oanda_instrument: str) -> int:
    """Return the number of decimal places to use for a given OANDA instrument."""
    if oanda_instrument in _POINT_INSTRUMENTS:
        return 2
    return 3 if oanda_instrument in _JPY_PAIRS else 5


def _fmt_price(price: float, oanda_instrument: str) -> str:
    """Format a price value for OANDA order fields."""
    return f"{price:.{_price_decimals(oanda_instrument)}f}"


class OandaClient:
    """Async OANDA v20 REST client. Thread-safe singleton — share across tasks."""

    def _setup(self) -> tuple[str, dict]:
        """Build base URL and auth headers from current settings (lazy so env loads first)."""
        if settings.oanda_environment == "live":
            base = "https://api-fxtrade.oanda.com"
        else:
            base = "https://api-fxpractice.oanda.com"
        headers = {
            "Authorization": f"Bearer {settings.oanda_api_key}",
            "Content-Type": "application/json",
        }
        return base, headers

    # ── Price feed ────────────────────────────────────────────────────────────

    # Instruments OANDA has rejected for this account. A single unknown
    # instrument makes the pricing endpoint 400 the ENTIRE request, so one bad
    # symbol would otherwise freeze every price including FX. Quarantined
    # symbols are skipped by both the REST and streaming paths.
    _quarantined: set = set()

    async def _probe_instruments(self, instruments: list[str]) -> set:
        """Return the subset OANDA rejects, by pricing each one individually."""
        base, headers = self._setup()
        url = f"{base}/v3/accounts/{settings.oanda_account_id}/pricing"
        bad = set()
        async with httpx.AsyncClient(timeout=10.0) as client:
            for inst in instruments:
                try:
                    r = await client.get(url, headers=headers,
                                         params={"instruments": inst})
                    if r.status_code != 200:
                        bad.add(inst)
                except Exception:
                    bad.add(inst)
        return bad

    async def get_prices(self, pairs: list[str]) -> dict:
        """
        Fetch current bid/ask prices for a list of standard pairs (e.g. "EUR/USD").

        GET /v3/accounts/{id}/pricing?instruments=EUR_USD,...

        Returns the raw OANDA response dict with a "prices" list, each item
        containing: instrument, bids, asks, tradeable, status.

        Self-healing: OANDA 400s the whole request if any single instrument is
        unknown to the account. On failure this probes each symbol, quarantines
        the offenders, and retries with the survivors, so one bad symbol can
        never take the price feed down.
        """
        if not settings.oanda_api_key or not settings.oanda_account_id:
            raise RuntimeError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be set")

        wanted = [PAIR_TO_OANDA[p] for p in pairs
                  if p in PAIR_TO_OANDA and PAIR_TO_OANDA[p] not in self._quarantined]
        instruments = ",".join(wanted)
        if not instruments:
            return {"prices": []}

        base, headers = self._setup()
        url = f"{base}/v3/accounts/{settings.oanda_account_id}/pricing"
        params = {"instruments": instruments}

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code == 200:
                return resp.json()
            # 400 usually means one unknown instrument poisoned the batch.
            # Find the offenders, quarantine them, and serve the rest.
            logger.error(
                "OANDA pricing returned %s for %d instruments — probing for "
                "unsupported symbols", resp.status_code, len(wanted),
            )
            bad = await self._probe_instruments(wanted)
            if not bad:
                resp.raise_for_status()
                return resp.json()
            self._quarantined |= bad
            logger.critical(
                "QUARANTINED unsupported OANDA instruments: %s — price feed "
                "continues with the remaining %d. Remove them from PAIR_CONFIG "
                "or correct the symbol names.",
                ", ".join(sorted(bad)), len(wanted) - len(bad),
            )
            survivors = [i for i in wanted if i not in bad]
            if not survivors:
                return {"prices": []}
            resp2 = await client.get(url, headers=headers,
                                     params={"instruments": ",".join(survivors)})
            resp2.raise_for_status()
            return resp2.json()

    # ── Sentiment: aggregate client positioning ───────────────────────────────

    async def get_position_book(self, oanda_instrument: str) -> Optional[dict]:
        """
        Fetch OANDA's aggregate client position book for an instrument and
        reduce it to overall long/short percentages.

        GET /v3/instruments/{instrument}/positionBook

        Each bucket carries longCountPercent / shortCountPercent as a share of
        ALL open positions, so summing across buckets yields total crowd
        positioning. OANDA refreshes this data ~every 20 minutes.

        Returns {"long_pct": float, "short_pct": float, "time": str} or None
        when unavailable (no credentials, unsupported instrument, API error).
        """
        if not settings.oanda_api_key:
            return None

        base, headers = self._setup()
        url = f"{base}/v3/instruments/{oanda_instrument}/positionBook"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                book = resp.json().get("positionBook", {})
        except Exception as exc:
            logger.warning("Position book fetch failed for %s: %s", oanda_instrument, exc)
            return None

        buckets = book.get("buckets", [])
        if not buckets:
            return None
        long_pct = sum(float(b.get("longCountPercent", 0.0)) for b in buckets)
        short_pct = sum(float(b.get("shortCountPercent", 0.0)) for b in buckets)
        return {
            "long_pct": round(long_pct, 1),
            "short_pct": round(short_pct, 1),
            "time": book.get("time", ""),
        }

    # ── Historical candles (for backtesting) ─────────────────────────────────

    async def get_intraday_candles(self, oanda_instrument: str,
                                   granularity: str = "M5", count: int = 200) -> list:
        """
        Fetch recent intraday candles as (bid, ask) pairs for history bootstrap.

        GET /v3/instruments/{instrument}/candles?granularity=M5&price=BA&count=N

        Lets the platform start with REAL market history immediately instead of
        either fabricating synthetic bars (which corrupts indicators) or waiting
        for live ticks to accumulate (which blocks trading for the warm-up).
        Returns [] on any failure — the caller falls back to live accumulation.
        """
        if not settings.oanda_api_key:
            return []
        base, headers = self._setup()
        url = f"{base}/v3/instruments/{oanda_instrument}/candles"
        params = {"granularity": granularity, "price": "BA", "count": str(min(count, 500))}
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(url, headers=headers, params=params)
                if resp.status_code != 200:
                    logger.warning("Candle bootstrap %s: HTTP %s", oanda_instrument, resp.status_code)
                    return []
                out = []
                for c in resp.json().get("candles", []):
                    if not c.get("complete"):
                        continue
                    try:
                        out.append((float(c["bid"]["c"]), float(c["ask"]["c"])))
                    except (KeyError, TypeError, ValueError):
                        continue
                return out
        except Exception as exc:
            logger.warning("Candle bootstrap failed for %s: %s", oanda_instrument, exc)
            return []

    async def get_daily_candles(self, oanda_instrument: str, count: int = 3800) -> list:
        """
        Fetch up to `count` daily mid-price candles (OANDA max 5000/request).

        GET /v3/instruments/{instrument}/candles?granularity=D&price=M&count=N

        Returns [{"date", "open", "high", "low", "close"}, ...] oldest-first,
        complete candles only. High and low are required by the range-based
        volatility estimators (Parkinson, Garman-Klass), which are far more
        efficient than close-to-close for a given sample size.
        Raises on missing credentials or API errors.
        """
        if not settings.oanda_api_key:
            raise RuntimeError("OANDA_API_KEY must be set for historical candles")
        base, headers = self._setup()
        url = f"{base}/v3/instruments/{oanda_instrument}/candles"
        params = {"granularity": "D", "price": "M", "count": str(min(count, 5000))}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            out = []
            for c in resp.json().get("candles", []):
                if c.get("complete"):
                    m = c["mid"]
                    out.append({
                        "date": c["time"][:10],
                        "open": float(m["o"]),
                        "high": float(m["h"]),
                        "low": float(m["l"]),
                        "close": float(m["c"]),
                    })
            return out

    # ── Order execution ───────────────────────────────────────────────────────

    async def place_market_order(
        self,
        oanda_instrument: str,
        units: int,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
    ) -> dict:
        """
        Place a market order on OANDA.

        Args:
            oanda_instrument: OANDA instrument code, e.g. "EUR_USD"
            units: positive = BUY, negative = SELL (must be non-zero integer)
            sl_price: ignored — SL/TP are managed by our internal monitor to
                      avoid OANDA rejections when price moves between quote and fill
            tp_price: ignored — see sl_price note above

        Returns the raw OANDA response dict which may contain:
          - "orderFillTransaction" (success) with keys: price, tradeOpened, pl
          - "orderCancelTransaction" (rejected) with key: reason
        """
        if not settings.oanda_api_key or not settings.oanda_account_id:
            raise RuntimeError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be set")

        # SL/TP are intentionally omitted from the OANDA order — attaching them
        # causes rejections when price moves between the agent's quote and fill.
        # Our check_sl_tp monitor closes positions within 3s of hitting SL/TP.
        order_body: dict = {
            "type": "MARKET",
            "instrument": oanda_instrument,
            "units": str(units),
            "timeInForce": "FOK",
        }

        body = {"order": order_body}
        base, headers = self._setup()
        url = f"{base}/v3/accounts/{settings.oanda_account_id}/orders"

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code not in (200, 201):
                raise RuntimeError(
                    f"OANDA place_market_order failed [{resp.status_code}]: {resp.text}"
                )
            return resp.json()

    async def close_trade(self, oanda_trade_id: str) -> dict:
        """
        Close an open OANDA trade by its trade ID.

        PUT /v3/accounts/{id}/trades/{trade_id}/close  {"units": "ALL"}

        Returns the raw OANDA response with "orderFillTransaction" containing
        "price" and "pl" (realized P&L in account currency).
        """
        if not settings.oanda_api_key or not settings.oanda_account_id:
            raise RuntimeError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be set")

        base, headers = self._setup()
        url = (
            f"{base}/v3/accounts/{settings.oanda_account_id}"
            f"/trades/{oanda_trade_id}/close"
        )
        body = {"units": "ALL"}

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.put(url, headers=headers, json=body)
            if resp.status_code not in (200, 201):
                raise RuntimeError(
                    f"OANDA close_trade failed [{resp.status_code}]: {resp.text}"
                )
            return resp.json()

    # ── Account info ──────────────────────────────────────────────────────────

    async def get_account_summary(self) -> dict:
        """
        Fetch OANDA account summary.

        GET /v3/accounts/{id}/summary

        Returns dict with "account" sub-dict containing: balance, NAV,
        unrealizedPL, openTradeCount, currency, etc.
        """
        if not settings.oanda_api_key or not settings.oanda_account_id:
            raise RuntimeError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be set")

        base, headers = self._setup()
        url = f"{base}/v3/accounts/{settings.oanda_account_id}/summary"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def set_trade_orders(
        self,
        oanda_trade_id: str,
        oanda_instrument: str,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
    ) -> dict:
        """
        Attach stop-loss and/or take-profit to an existing OANDA trade.

        Called immediately after a market order fills so broker-level protection
        is active even if our server restarts.

        PUT /v3/accounts/{id}/trades/{tradeID}/orders
        """
        if not settings.oanda_api_key or not settings.oanda_account_id:
            raise RuntimeError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be set")

        body: dict = {}
        if tp_price is not None:
            body["takeProfit"] = {
                "price": _fmt_price(tp_price, oanda_instrument),
                "timeInForce": "GTC",
            }
        if sl_price is not None:
            body["stopLoss"] = {
                "price": _fmt_price(sl_price, oanda_instrument),
                "timeInForce": "GTC",
            }

        if not body:
            return {}

        base, headers = self._setup()
        url = f"{base}/v3/accounts/{settings.oanda_account_id}/trades/{oanda_trade_id}/orders"

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.put(url, headers=headers, json=body)
            if resp.status_code not in (200, 201):
                raise RuntimeError(
                    f"OANDA set_trade_orders failed [{resp.status_code}]: {resp.text}"
                )
            return resp.json()

    async def get_trade(self, oanda_trade_id: str) -> Optional[dict]:
        """
        Fetch a single OANDA trade by ID regardless of state.

        Returns the raw trade dict, or None if not found.
        Used to reconcile our DB when OANDA closes a trade independently
        (e.g. its own SL/TP fires while our server is restarting).
        """
        if not settings.oanda_api_key or not settings.oanda_account_id:
            return None

        base, headers = self._setup()
        url = f"{base}/v3/accounts/{settings.oanda_account_id}/trades/{oanda_trade_id}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json().get("trade")

    async def stream_prices(self, pairs: list[str]) -> AsyncGenerator[dict, None]:
        """
        Stream real-time prices via OANDA's SSE pricing stream.

        Yields parsed JSON dicts: {"type": "PRICE", "instrument": ..., "bids": [...], "asks": [...], ...}
        or {"type": "HEARTBEAT", "time": ...} every ~5s.

        Uses the streaming domain (stream-fxpractice / stream-fxtrade), not the
        REST domain. Read timeout is 15s — if no bytes arrive in 15s the server
        is considered dead and an httpx.ReadTimeout is raised so the caller reconnects.
        """
        if not settings.oanda_api_key or not settings.oanda_account_id:
            raise RuntimeError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be set")

        # Same poisoning risk as the REST endpoint: one unknown instrument makes
        # the whole stream 400. Skip anything already quarantined.
        instruments = ",".join(
            PAIR_TO_OANDA[p] for p in pairs
            if p in PAIR_TO_OANDA and PAIR_TO_OANDA[p] not in self._quarantined
        )
        if not instruments:
            return

        if settings.oanda_environment == "live":
            stream_base = "https://stream-fxtrade.oanda.com"
        else:
            stream_base = "https://stream-fxpractice.oanda.com"

        _, headers = self._setup()
        url = f"{stream_base}/v3/accounts/{settings.oanda_account_id}/pricing/stream"
        params = {"instruments": instruments}
        timeout = httpx.Timeout(connect=10.0, read=15.0, write=None, pool=None)

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("GET", url, headers=headers, params=params) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        yield json.loads(line)

    async def get_open_trades(self) -> list:
        """
        Fetch all open trades on the OANDA account.

        GET /v3/accounts/{id}/openTrades

        Returns list of trade objects (raw OANDA dicts).
        """
        if not settings.oanda_api_key or not settings.oanda_account_id:
            raise RuntimeError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be set")

        base, headers = self._setup()
        url = f"{base}/v3/accounts/{settings.oanda_account_id}/openTrades"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data.get("trades", [])


# Singleton — import and use directly
oanda_client = OandaClient()

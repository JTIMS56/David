"""
OANDA v20 REST API Client
────────────────────────────────────────────────────────────────────────────
Async client for OANDA's v20 REST API. Supports both practice and live
environments. Used by market_data (price feed) and order_service (execution).
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger("david.oanda_client")

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
}

OANDA_TO_PAIR: dict[str, str] = {v: k for k, v in PAIR_TO_OANDA.items()}

# JPY pairs use 3 decimal places; all others use 5
_JPY_PAIRS = {"USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY", "NZD_JPY", "CHF_JPY"}


def _price_decimals(oanda_instrument: str) -> int:
    """Return the number of decimal places to use for a given OANDA instrument."""
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

    async def get_prices(self, pairs: list[str]) -> dict:
        """
        Fetch current bid/ask prices for a list of standard pairs (e.g. "EUR/USD").

        GET /v3/accounts/{id}/pricing?instruments=EUR_USD,...

        Returns the raw OANDA response dict with a "prices" list, each item
        containing: instrument, bids, asks, tradeable, status.
        Raises on non-2xx responses.
        """
        if not settings.oanda_api_key or not settings.oanda_account_id:
            raise RuntimeError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be set")

        instruments = ",".join(
            PAIR_TO_OANDA[p] for p in pairs if p in PAIR_TO_OANDA
        )
        if not instruments:
            return {"prices": []}

        base, headers = self._setup()
        url = f"{base}/v3/accounts/{settings.oanda_account_id}/pricing"
        params = {"instruments": instruments}

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            return resp.json()

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

"""
Crowd Sentiment Service  (OANDA position book)
────────────────────────────────────────────────────────────────────────────
Maintains an in-memory cache of OANDA's aggregate client positioning per pair,
refreshed by a background task (OANDA updates the book ~every 20 minutes).

This is genuinely NEW information — it describes market participants, not a
transformation of the price series. The documented (modest) edge is
contrarian: when the retail crowd is heavily long, fade them.

The ensemble model reads the cache synchronously via positioning_vote().
When no data is available (simulation mode, no credentials, API failure) the
vote is 0 — the ensemble simply loses one voter, never blocks.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from config import settings

logger = logging.getLogger("david.sentiment")

# pair -> {"long_pct": float, "short_pct": float, "time": str, "fetched_at": datetime}
_cache: Dict[str, dict] = {}


def get_positioning(pair: str) -> Optional[dict]:
    """Latest cached position-book summary for a pair, or None."""
    return _cache.get(pair)


def positioning_vote(pair: str) -> int:
    """
    Contrarian crowd vote: -1 (SELL) when the crowd is crowded long,
    +1 (BUY) when crowded short, else 0. Threshold from settings.
    """
    data = _cache.get(pair)
    if not data:
        return 0
    threshold = settings.positioning_fade_threshold
    if data["long_pct"] >= threshold:
        return -1   # crowd long → fade → SELL
    if data["short_pct"] >= threshold:
        return 1    # crowd short → fade → BUY
    return 0


async def refresh_loop() -> None:
    """Background task: refresh the position book for all pairs periodically."""
    from services.oanda_client import oanda_client, PAIR_TO_OANDA

    # slight initial delay so startup isn't blocked on 8 HTTP calls
    await asyncio.sleep(10)
    while True:
        for pair, instrument in PAIR_TO_OANDA.items():
            try:
                book = await oanda_client.get_position_book(instrument)
                if book:
                    book["fetched_at"] = datetime.now(timezone.utc)
                    _cache[pair] = book
            except Exception as exc:
                logger.warning("Sentiment refresh failed for %s: %s", pair, exc)
            await asyncio.sleep(1)   # gentle pacing between instruments
        logger.info(
            "Sentiment refreshed for %d/%d pairs", len(_cache), len(PAIR_TO_OANDA)
        )
        await asyncio.sleep(settings.position_book_refresh_seconds)

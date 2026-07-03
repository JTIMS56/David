"""
Economic Calendar Service  (event blackout)
────────────────────────────────────────────────────────────────────────────
Fetches the free ForexFactory weekly calendar feed and answers one question:
"is a high-impact event for either currency of this pair imminent?"

Rationale: scheduled releases (rate decisions, CPI, NFP) produce spikes that
are unpredictable from any of our signals. The cheapest expectancy gain is to
simply not hold or open positions into them. This service provides the
blackout flag; the ensemble records it and zeroes conviction during the window.

Feed: https://nfs.faireconomy.media/ff_calendar_thisweek.json
  [{"title": "...", "country": "USD", "date": "2026-07-03T08:30:00-04:00",
    "impact": "High", ...}, ...]

All failures degrade gracefully to "no blackout" (with a logged warning) —
a missing calendar must never halt the platform.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import httpx

from config import settings

logger = logging.getLogger("david.econ_calendar")

# Parsed events: [{"title": str, "currency": str, "at": datetime(utc), "impact": str}]
_events: List[dict] = []
_last_fetch: Optional[datetime] = None


def _parse_events(raw: list) -> List[dict]:
    out: List[dict] = []
    for item in raw:
        try:
            when = datetime.fromisoformat(item["date"])
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            out.append({
                "title": item.get("title", ""),
                "currency": (item.get("country") or "").upper(),
                "at": when.astimezone(timezone.utc),
                "impact": item.get("impact", ""),
            })
        except Exception:
            continue
    return out


async def fetch_calendar() -> int:
    """Fetch and cache this week's calendar. Returns number of events parsed."""
    global _events, _last_fetch
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(settings.calendar_url)
            resp.raise_for_status()
            _events = _parse_events(resp.json())
            _last_fetch = datetime.now(timezone.utc)
            logger.info("Economic calendar loaded: %d events", len(_events))
            return len(_events)
    except Exception as exc:
        logger.warning("Calendar fetch failed (blackouts disabled until next try): %s", exc)
        return 0


def is_blackout(pair: str, now: Optional[datetime] = None) -> Optional[dict]:
    """
    Return the blocking event dict if a high-impact event for either currency
    of `pair` is within the blackout window, else None.

    Window: [event - pre_minutes, event + post_minutes].
    """
    try:
        base, quote = pair.split("/")
    except ValueError:
        return None
    now = now or datetime.now(timezone.utc)
    pre = timedelta(minutes=settings.event_blackout_pre_minutes)
    post = timedelta(minutes=settings.event_blackout_post_minutes)
    for ev in _events:
        if ev["impact"] != "High" or ev["currency"] not in (base, quote):
            continue
        if ev["at"] - pre <= now <= ev["at"] + post:
            return ev
    return None


def upcoming_high_impact(hours: float = 24.0, now: Optional[datetime] = None) -> List[dict]:
    """High-impact events in the next `hours`, soonest first (for status/UI)."""
    now = now or datetime.now(timezone.utc)
    horizon = now + timedelta(hours=hours)
    hits = [e for e in _events if e["impact"] == "High" and now <= e["at"] <= horizon]
    return sorted(hits, key=lambda e: e["at"])


async def refresh_loop() -> None:
    """Background task: keep the weekly calendar fresh."""
    await asyncio.sleep(5)
    while True:
        await fetch_calendar()
        await asyncio.sleep(settings.calendar_refresh_seconds)

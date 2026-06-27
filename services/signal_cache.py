"""
Signal Cache
────────────────────────────────────────────────────────────────────────────
Holds the most recent DHJ forecast per pair so the hard pre-trade gate can
enforce signal-based rules (disagreement-only, no MILD_BULLISH) that the LLM
cannot bypass.

The forecast tool (get_price_forecast) writes here synchronously the moment a
forecast is computed; the order gate reads here when the agent tries to open a
position.  No external dependencies so both the tool layer and the gate layer
can import it without circular-import risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional


@dataclass(frozen=True)
class CachedSignal:
    pair: str
    signal: str           # STRONG_BULLISH | MILD_BULLISH | NEUTRAL | MILD_BEARISH | STRONG_BEARISH
    dhj_direction: str    # "UP" | "DOWN"
    bs_direction: str     # "UP" | "DOWN"
    at: datetime          # UTC timestamp the forecast was computed

    def age_seconds(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        ts = self.at if self.at.tzinfo else self.at.replace(tzinfo=timezone.utc)
        return (now - ts).total_seconds()


_cache: Dict[str, CachedSignal] = {}


def put_signal(pair: str, signal: str, dhj_direction: str, bs_direction: str) -> None:
    _cache[pair] = CachedSignal(
        pair=pair,
        signal=signal,
        dhj_direction=dhj_direction,
        bs_direction=bs_direction,
        at=datetime.now(timezone.utc),
    )


def get_signal(pair: str) -> Optional[CachedSignal]:
    return _cache.get(pair)

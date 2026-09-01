"""
Signal Cache
────────────────────────────────────────────────────────────────────────────
Holds the most recent ENSEMBLE forecast per pair so the hard pre-trade gate
can enforce signal rules the LLM cannot bypass.

The forecast tool (get_price_forecast) writes here synchronously the moment a
forecast is computed; the order gate reads here when the agent tries to open a
position.  No external dependencies so both the tool layer and the gate layer
can import it without circular-import risk.

History: this used to cache DHJ signal labels and DHJ/BS directions. DHJ was
retired from the decision path after ~800 evaluated forecasts showed ~50%
accuracy with non-stationary conditional slices; it now runs only as a silent
background benchmark. The gate keys off the independent ensemble instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional


@dataclass(frozen=True)
class CachedSignal:
    pair: str
    direction: str        # "UP" | "DOWN" | "FLAT"  (ensemble net-vote direction)
    conviction: int       # abs(net vote) across the ensemble's independent voters
    event_blackout: bool  # high-impact scheduled event imminent for either currency
    at: datetime          # UTC timestamp the forecast was computed

    def age_seconds(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        ts = self.at if self.at.tzinfo else self.at.replace(tzinfo=timezone.utc)
        return (now - ts).total_seconds()


_cache: Dict[str, CachedSignal] = {}


def put_signal(pair: str, direction: str, conviction: int, event_blackout: bool) -> None:
    _cache[pair] = CachedSignal(
        pair=pair,
        direction=direction,
        conviction=conviction,
        event_blackout=event_blackout,
        at=datetime.now(timezone.utc),
    )


def get_signal(pair: str) -> Optional[CachedSignal]:
    return _cache.get(pair)

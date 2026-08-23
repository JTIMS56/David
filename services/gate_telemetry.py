"""
Gate Telemetry
────────────────────────────────────────────────────────────────────────────
Answers "why aren't we trading?" with a distribution instead of an anecdote.

Every forecast records its conviction and, if it was blocked, why. Over a few
hundred evaluations that produces the number that actually matters: what
fraction of instrument-evaluations clear the gate, and which rule is doing the
blocking.

Without this, a quiet market and an impossibly strict gate look identical from
the outside — both just produce cycle after cycle of "no setups". They are very
different problems and deserve different fixes.
"""
from __future__ import annotations

import time
from collections import Counter, deque
from typing import Deque, Optional

# (timestamp, pair, conviction, passed, reason)
_events: Deque[tuple] = deque(maxlen=5000)


def record(pair: str, conviction: int, passed: bool, reason: Optional[str] = None) -> None:
    _events.append((time.time(), pair, int(conviction), bool(passed), reason or ""))


def _bucket(reason: str) -> str:
    """Collapse gate messages into stable categories."""
    r = reason.lower()
    if not r:
        return "passed"
    if "conviction" in r:
        return "conviction_below_minimum"
    if "flat" in r:
        return "direction_flat"
    if "atr" in r or "quiet" in r:
        return "volatility_floor"
    if "blackout" in r or "economic event" in r:
        return "event_blackout"
    if "weekend" in r:
        return "weekend_window"
    if "blocked" in r and "churn" in r:
        return "blocked_pair"
    if "stale" in r or "no forecast" in r:
        return "stale_forecast"
    if "spread" in r:
        return "spread_cost_floor"
    if "exposure" in r or "position size" in r or "max" in r:
        return "risk_limits"
    return "other"


def summary(window_hours: float = 24.0) -> dict:
    cutoff = time.time() - window_hours * 3600
    rows = [e for e in _events if e[0] >= cutoff]
    if not rows:
        return {"window_hours": window_hours, "evaluations": 0,
                "note": "no forecasts recorded in this window yet"}

    conv = Counter(r[2] for r in rows)
    passed = sum(1 for r in rows if r[3])
    reasons = Counter(_bucket(r[4]) for r in rows if not r[3])
    per_pair = Counter(r[1] for r in rows if r[3])

    return {
        "window_hours": window_hours,
        "evaluations": len(rows),
        "passed": passed,
        "pass_rate_pct": round(100.0 * passed / len(rows), 2),
        # How often the ensemble actually reaches each conviction level. This is
        # the number that says whether the >=2 threshold is reachable at all.
        "conviction_distribution": {str(k): conv[k] for k in sorted(conv)},
        "conviction_ge_2_pct": round(
            100.0 * sum(v for k, v in conv.items() if k >= 2) / len(rows), 2),
        "block_reasons": dict(reasons.most_common()),
        "passing_instruments": dict(per_pair.most_common(10)),
    }

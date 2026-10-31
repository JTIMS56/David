"""
Per-pair Edge Map  (shadow validation)
────────────────────────────────────────────────────────────────────────────
Encodes the conditional edges discovered in the forecast log: for certain
pairs, whether to trade WITH or FADE the DHJ call depending on whether DHJ and
Black-Scholes agreed. These were found in-sample and are being validated
out-of-sample before they touch the trading path.

The map is deterministic in (pair, dhj_direction, bs_direction), so the
validation harness can replay it over historical ForecastLog rows and split
accuracy into in-sample vs out-of-sample without any new logging.
"""
from __future__ import annotations

from typing import Optional, Tuple

from config import settings


def _opposite(direction: str) -> str:
    return "UP" if direction == "DOWN" else "DOWN"


def recommend(
    pair: str,
    dhj_direction: Optional[str],
    bs_direction: Optional[str],
) -> Tuple[str, Optional[str]]:
    """
    Return (action, side) where action is "WITH" | "FADE" | "SKIP" and side is
    "BUY" | "SELL" | None. SKIP means the edge map has no rule for this cell.
    """
    if not dhj_direction or not bs_direction:
        return ("SKIP", None)
    rules = settings.edge_map.get(pair)
    if not rules:
        return ("SKIP", None)
    agreement = "agree" if dhj_direction == bs_direction else "disagree"
    action = rules.get(agreement)
    if action not in ("with", "fade"):
        return ("SKIP", None)
    target = dhj_direction if action == "with" else _opposite(dhj_direction)
    side = "BUY" if target == "UP" else "SELL"
    return (action.upper(), side)


def was_correct(action: str, dhj_direction_correct: Optional[bool]) -> Optional[bool]:
    """
    Given the edge-map action and whether DHJ's raw direction call was correct,
    return whether the edge-map recommendation was correct. WITH tracks DHJ;
    FADE is the inverse. Returns None when not evaluable.
    """
    if dhj_direction_correct is None:
        return None
    if action == "WITH":
        return dhj_direction_correct
    if action == "FADE":
        return not dhj_direction_correct
    return None

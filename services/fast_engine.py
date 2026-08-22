"""
Medium-Frequency Execution Engine
────────────────────────────────────────────────────────────────────────────
A deterministic sub-second decision loop that takes the LLM out of the hot
path entirely.

The agent cycle takes 30-120 SECONDS per decision because it is an LLM
round-trip. That is fine for supervision and analysis, and useless for
execution timing. This engine runs the same ensemble forecast and the same
hard risk gate as pure computation — no network, no model call — at a
configurable interval measured in milliseconds.

    LLM agent      30-120 s   → supervision, narrative, position review
    fast engine    ~0.1-1 s   → detection, gating, order submission

Nothing here bypasses a safety control. Every order still goes through
order_service.open_position(), which applies the signal gate, volatility
floor, exposure caps, event blackouts, weekend rules and execution tiers.
The engine adds its own rate limits on top, because a loop running four
times a second against a persistent signal would otherwise submit the same
trade repeatedly.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from config import settings

logger = logging.getLogger("popper.fast_engine")


class LatencyStats:
    """Rolling latency percentiles for one pipeline stage, in microseconds."""

    __slots__ = ("name", "_samples")

    def __init__(self, name: str, window: int = 2000) -> None:
        self.name = name
        self._samples: Deque[float] = deque(maxlen=window)

    def record(self, microseconds: float) -> None:
        self._samples.append(microseconds)

    def snapshot(self) -> dict:
        if not self._samples:
            return {"stage": self.name, "n": 0}
        ordered = sorted(self._samples)
        n = len(ordered)

        def pct(p: float) -> float:
            return round(ordered[min(n - 1, int(n * p))], 1)

        return {
            "stage": self.name,
            "n": n,
            "p50_us": pct(0.50),
            "p95_us": pct(0.95),
            "p99_us": pct(0.99),
            "max_us": round(ordered[-1], 1),
        }


class FastEngine:
    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._stats: Dict[str, LatencyStats] = {
            k: LatencyStats(k) for k in
            ("usd_breadth", "forecast_per_instrument", "gate_check", "full_cycle", "order_submit")
        }
        self._last_entry: Dict[str, float] = defaultdict(float)   # pair -> monotonic ts
        self._recent_entries: Deque[float] = deque(maxlen=64)
        self._cycles = 0
        self._orders_attempted = 0
        self._orders_filled = 0
        self._last_skip_reason: Dict[str, str] = {}

    # ── Rate limiting ────────────────────────────────────────────────────────

    def _rate_limited(self, pair: str, now: float) -> Optional[str]:
        """
        Guard against a fast loop hammering a persistent signal. Returns a
        reason string when the instrument must be skipped, else None.
        """
        since = now - self._last_entry[pair]
        if since < settings.fast_engine_pair_cooldown_s:
            return f"cooldown {settings.fast_engine_pair_cooldown_s - since:.0f}s"
        recent = [t for t in self._recent_entries if now - t < 60.0]
        if len(recent) >= settings.fast_engine_max_entries_per_min:
            return "global entry rate limit"
        return None

    # ── One deterministic pass over the whole universe ───────────────────────

    async def _cycle(self) -> None:
        from services.market_data import market_data
        from services.ensemble_model import predict, _usd_strength_score
        from services.risk_gate import risk_gate
        from services.portfolio_service import portfolio_service
        from services.order_service import order_service

        t_cycle = time.perf_counter_ns()

        # USD breadth is cross-sectional: compute once per cycle and pass it in,
        # rather than letting every forecast rebuild it.
        t = time.perf_counter_ns()
        usd_score = _usd_strength_score()
        self._stats["usd_breadth"].record((time.perf_counter_ns() - t) / 1000)

        open_pairs = {p.pair for p in await portfolio_service.get_open_positions()}
        now = time.monotonic()
        candidates = []

        for pair in settings.default_pairs:
            if pair in open_pairs:
                continue
            t = time.perf_counter_ns()
            fc = predict(pair, usd_score=usd_score)
            self._stats["forecast_per_instrument"].record((time.perf_counter_ns() - t) / 1000)
            if fc is None or fc.direction == "FLAT":
                continue

            direction = "BUY" if fc.direction == "UP" else "SELL"
            t = time.perf_counter_ns()
            from services.signal_cache import put_signal
            put_signal(pair, fc.direction, fc.conviction, fc.event_blackout)
            decision = risk_gate.approve_signal(pair=pair, direction=direction, source="agent")
            self._stats["gate_check"].record((time.perf_counter_ns() - t) / 1000)
            if not decision.allowed:
                self._last_skip_reason[pair] = decision.reason
                continue

            limited = self._rate_limited(pair, now)
            if limited:
                self._last_skip_reason[pair] = limited
                continue
            candidates.append((fc.conviction, pair, direction, fc))

        # Highest conviction first, capped per cycle.
        candidates.sort(reverse=True, key=lambda c: c[0])
        for conviction, pair, direction, fc in candidates[:settings.fast_engine_max_new_per_cycle]:
            await self._submit(pair, direction, fc, order_service, market_data)

        self._stats["full_cycle"].record((time.perf_counter_ns() - t_cycle) / 1000)
        self._cycles += 1

    async def _submit(self, pair, direction, fc, order_service, market_data) -> None:
        """Size and submit one order. All hard risk checks live downstream."""
        import math
        from services.market_data import asset_class, stop_bounds

        ind = market_data.calculate_indicators(pair)
        if not ind:
            return
        min_stop, max_stop = stop_bounds(pair)
        stop_pips = max(ind["atr_pips"] * 1.5, min_stop * 1.33)
        if stop_pips > max_stop:
            self._last_skip_reason[pair] = f"stop {stop_pips:.0f}p exceeds {max_stop:.0f}p"
            return

        from services.portfolio_service import portfolio_service
        state = await portfolio_service.get_state()
        # Conviction-scaled sizing, same rule the agent follows.
        max_notional = (state["balance"] * settings.max_position_size_pct
                        * (1.0 if fc.conviction >= 3 else 0.75))

        price = ind["current_price"]
        cls = asset_class(pair)
        if cls == "fx":
            size = float(max_notional) if pair.startswith("USD/") else float(int(max_notional / price))
        else:
            size = math.floor(max_notional / price * 10) / 10
        if size <= 0:
            return

        self._orders_attempted += 1
        t = time.perf_counter_ns()
        ok, msg, pos = await order_service.open_position(
            pair=pair, direction=direction, size=size,
            reasoning=(f"fast-engine: conviction {fc.conviction} {fc.direction}, "
                       f"votes {fc.votes}"),
            source="agent",
            stop_pips_req=round(stop_pips, 1),
            tp_pips_req=round(stop_pips * 1.6, 1),
        )
        self._stats["order_submit"].record((time.perf_counter_ns() - t) / 1000)
        if ok and pos is not None:
            self._orders_filled += 1
            now = time.monotonic()
            self._last_entry[pair] = now
            self._recent_entries.append(now)
            logger.info("FAST ENTRY %s %s %.1f (conviction %d) — %s",
                        direction, pair, size, fc.conviction, msg[:70])
        else:
            self._last_skip_reason[pair] = msg[:120]

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        interval = settings.fast_engine_interval_ms / 1000.0
        logger.info("Fast engine started — %.0fms interval, deterministic path only",
                    settings.fast_engine_interval_ms)
        while self._running:
            try:
                from agents.trading_agent import trading_agent
                if trading_agent._is_market_open():
                    await self._cycle()
            except Exception:
                logger.exception("Fast engine cycle failed")
            await asyncio.sleep(interval)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    def status(self) -> dict:
        return {
            "running": self._running,
            "interval_ms": settings.fast_engine_interval_ms,
            "cycles": self._cycles,
            "orders_attempted": self._orders_attempted,
            "orders_filled": self._orders_filled,
            "latency": [s.snapshot() for s in self._stats.values()],
            "rate_limits": {
                "pair_cooldown_s": settings.fast_engine_pair_cooldown_s,
                "max_entries_per_min": settings.fast_engine_max_entries_per_min,
                "max_new_per_cycle": settings.fast_engine_max_new_per_cycle,
            },
            "last_skip_reason": dict(list(self._last_skip_reason.items())[:15]),
        }


fast_engine = FastEngine()

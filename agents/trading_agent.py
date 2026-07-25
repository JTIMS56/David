"""
Popper — Autonomous AI FX Trading Agent
────────────────────────────────────────────────────────────────────────────
Powered by Claude. Runs an agentic trading loop:
  1. Scans all pairs for signals
  2. Reviews current portfolio & risk metrics
  3. Makes trading decisions via tool use
  4. Logs all decisions with full reasoning
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import anthropic
from sqlalchemy import desc, select

from config import settings
from database import AsyncSessionLocal
from models.orm import AgentDecision, CycleLock
from agents.tools import TOOL_DEFINITIONS, handle_tool_call
from services.portfolio_service import portfolio_service

logger = logging.getLogger("popper.agent")

SYSTEM_PROMPT = """\
You are Popper, an autonomous AI FX (foreign exchange) trading agent. Your mandate is \
to generate consistent risk-adjusted returns by trading major currency pairs in the \
foreign exchange market using a disciplined, rules-based approach.

## Capabilities
You have access to the following tools:
- scan_all_pairs           → Quick overview of all pairs and their signals
- get_fx_rates             → Current bid/ask prices for any pairs
- get_technical_indicators → RSI, MACD, Bollinger Bands, SMAs, ATR for a pair
- get_price_history        → Raw recent price data
- get_portfolio_status     → Account balance, equity, open positions, P&L
- get_risk_metrics         → Exposure, daily loss, position limits
- get_price_forecast       → Ensemble direction forecast (votes, conviction, signal_gate)
- place_order              → Open a new BUY or SELL position (size in base-currency UNITS, not lots)
- close_position           → Close an existing position

## Trading Philosophy
1. **Capital preservation first.** Never risk more than 1% of balance per trade.
2. **Trend following with confirmation.** Look for alignment between:
   - Price above/below key SMAs (20, 50)
   - RSI not in extreme overbought/oversold (unless reverting)
   - MACD histogram in the direction of the trend
3. **Mean reversion on extremes.** When BB %B < 0.10 (oversold) or > 0.90 (overbought)
   AND RSI confirms, consider counter-trend fades.
4. **Never average losers.** If a position is going against you, let the stop-loss work.
5. **Let winners run.** Use trailing targets, don't exit profitable trades too early.

## Risk Rules (NON-NEGOTIABLE)
- Always include stop_loss when placing an order — never trade without one.
- Minimum stop distance: 15 pips. Maximum stop distance: 50 pips.
- Minimum Risk:Reward ratio: 1.5 (TP must be at least 1.5× the stop distance).
- Maximum position size: 5% of balance in notional terms.
- Maximum concurrent open positions: {max_positions}.
- Do NOT trade if daily loss already exceeds {max_daily_loss_pct}% of balance.

## Position Sizing (IMPORTANT — size is in base-currency UNITS, not lots)
All sizes must be whole numbers of base-currency units. The formula depends on which currency
is base (the left side of the pair):

  USD-COUNTER pairs (USD is on the RIGHT: EUR/USD, GBP/USD, AUD/USD, NZD/USD, EUR/GBP):
    1 unit = 1 unit of the base currency ≠ $1
    size = int(max_position_notional / entry_price)
    Examples at $100,000 balance (max_position_notional = $5,000):
      EUR/USD @ 1.15 → size = int(5000 / 1.15) ≈ 4,350 units
      GBP/USD @ 1.32 → size = int(5000 / 1.32) ≈ 3,788 units

  USD-BASE pairs (USD is on the LEFT: USD/JPY, USD/CAD, USD/CHF):
    1 unit = 1 USD — do NOT divide by entry price
    size = max_position_notional  (the entry price is in foreign currency, irrelevant to USD notional)
    Examples at $100,000 balance:
      USD/JPY → size = 5,000 units  (NOT 5000/161 = 31 — that is wrong by 160×)
      USD/CAD → size = 5,000 units  (NOT 5000/1.41 = 3,546)
      USD/CHF → size = 5,000 units  (NOT 5000/0.90 = 5,556)

Correct sizing workflow:
  1. Get max_position_notional from get_risk_metrics (= 5% of balance).
  2. Compute max_units:
       - Pair starts with "USD/" (USD/JPY, USD/CAD, USD/CHF): size = max_position_notional
       - All other pairs: size = int(max_position_notional / entry_price)
  3. Apply the conviction-based size adjustment if applicable.
  4. Round to nearest whole number.
  5. Never pass fractional units (e.g., 3.7) — that is a lot, not units.

## CRITICAL: Stop-Loss and Take-Profit Placement
SL and TP must always be on opposite sides of the entry price:

  BUY  @ 1.34000 → stop_loss = 1.33750 (BELOW entry, -25 pips)
                  → take_profit = 1.34375 (ABOVE entry, +37.5 pips = 1.5× RR)

  SELL @ 1.34000 → stop_loss = 1.34250 (ABOVE entry, +25 pips)
                  → take_profit = 1.33625 (BELOW entry, -37.5 pips = 1.5× RR)

A BUY with stop_loss ABOVE entry, or TP BELOW entry, will be rejected by the risk gate.
A SELL with stop_loss BELOW entry, or TP ABOVE entry, will be rejected by the risk gate.

## Stop-Distance Computation (apply EXACTLY in this order every time)
1. Get atr_pips from get_technical_indicators output.
2. raw_stop_pips = atr_pips × 1.5
3. stop_pips = max(raw_stop_pips, 20)   ← ALWAYS floor to 20 (well above the 15-pip minimum)
4. If stop_pips > 50: skip the pair — too volatile for our limits. Do NOT try to force a trade.
5. Place the order in PIPS — do NOT compute absolute SL/TP prices:
     place_order(pair, direction, size, stop_pips=<stop_pips>,
                 take_profit_pips=<stop_pips × 1.6>, reasoning=...)
   The server anchors your distances to the real entry quote at execution
   time, so price movement between your data fetch and the order can never
   invalidate the geometry. There is no need to call get_fx_rates first.

Example (EUR/GBP BUY, ATR=4 pips):
  raw_stop_pips = 4 × 1.5 = 6 → floor to 20
  place_order(..., stop_pips=20, take_profit_pips=32)

(Absolute stop_loss/take_profit prices are still accepted for manual cases,
but pips are strictly better for you: same geometry, zero drift risk.)

## Order Rejection Protocol
If place_order returns success=false, read the message field EXACTLY:
- "below minimum 15 pips" → your stop is too tight; recompute using max(ATR×1.5, 20) pips
- "exceeds maximum 50 pips" → your stop is too wide; skip this pair (don't force it to 50 pips)
- Any SL/TP side error → you passed absolute prices that drifted; re-place ONCE using stop_pips/take_profit_pips instead.

HARD RETRY LIMIT: at most ONE corrected attempt per pair per cycle — two
rejections means SKIP the pair, no exceptions, no third attempt. Retrying
the same geometry against a moving market wastes the cycle and never wins.

Make ONE corrected attempt using the exact fix described. If still rejected, SKIP this pair \
entirely. Do NOT try a third time or vary parameters at random — move on to the next pair. \
Exhausting your tool budget on retries is worse than missing a trade.

**Post-fill aborts**: OANDA can fill an order 10–30 pips away from the quoted price during
volatile conditions (e.g. news, extreme RSI). If the actual fill price degrades the realized
R:R below 1.0, the order is automatically aborted and you will receive:
  {{"success": false, "message": "Order aborted after fill: post-fill R:R ..."}}
This is NOT a risk-gate rejection — the order executed and was then closed immediately.
Treat it the same as a SKIP: do not retry, move to the next pair.

## The Ensemble Forecast (your decision engine)
get_price_forecast runs an ensemble of FIVE INDEPENDENT voters. Each votes
-1 (down) / 0 (abstain) / +1 (up); direction is the sign of the net vote and
conviction is its magnitude:
- **trend**: MACD histogram + price vs SMA50 (momentum)
- **mean_revert**: RSI / Bollinger extremes (counter-trend reversion)
- **carry**: central-bank rate differential (structural drift)
- **usd_strength**: cross-pair USD breadth (is USD moving as a bloc?)
- **positioning**: OANDA's aggregate client position book — CONTRARIAN: when the
  retail crowd is heavily one-sided, this votes to fade them. This is information
  about market participants, not another chart indicator.

Key outputs:
- **direction**: UP / DOWN / FLAT (FLAT = voters cancel out → never tradeable)
- **conviction**: how many net votes agree. 2+ is required to trade; 3+ is strong.
- **votes**: the per-voter breakdown — cite it in your reasoning.
- **event_blackout**: true = a high-impact economic release (rate decision, CPI,
  NFP) for either currency is imminent. Conviction is zeroed because release
  spikes are unpredictable. Never fight this; there is nothing to predict there.

Note: the legacy DHJ physics model has been RETIRED from decision-making after
800+ evaluated forecasts showed coin-flip accuracy. It no longer appears in your
data. Do not reference DHJ, chiral charge, or DHJ/BS disagreement in decisions.

## The Hard Signal Gate (server-enforced — you CANNOT bypass it)
Every forecast returns a **signal_gate** object. READ IT and obey it:
- signal_gate.tradeable = true  → this pair passes; trade in signal_gate.trade_direction.
- signal_gate.tradeable = false → SKIP this pair; signal_gate.reason says why.

A trade is accepted only when ALL of these hold (the gate enforces them; orders
that fail are rejected at the server):
1. You requested get_price_forecast for the pair THIS cycle (forecast must be fresh).
2. Ensemble conviction >= 2 — at least two net independent votes agreeing.
3. Your order direction MATCHES the ensemble direction (BUY if UP, SELL if DOWN).
   Never trade against the ensemble; FLAT means no trade exists.
4. No event blackout is active for the pair.
5. The pair is not EUR/GBP (chronic range-bound churn).

Trust signal_gate over your own re-derivation. If it says tradeable, that is a
valid setup — take it. If it says blocked, move on without retrying.

## Execution tiers
get_risk_metrics reports the active execution_tier. Behavior by tier:
- **shadow**: place_order returns "[SHADOW] Would open..." — logged for
  validation, NOT executed, no position appears. Intended behavior, not an
  error: do not retry or treat the missing position as a discrepancy.
- **micro** (current): orders EXECUTE for real, but the server scales agent
  orders to ~10% of requested size, capped at ~$500 notional, with a hard
  daily loss budget. Positions and P&L will look small — that is INTENTIONAL
  (gathering live fill data while the strategy validates). Size your orders
  normally and let the server scale; NEVER inflate requested size to
  compensate. If blocked for "daily loss budget exhausted", stop opening
  positions for the rest of the day.
- **full**: normal sizing (only after a signal validates out-of-sample).

## Position sizing (AFTER the gate passes — advisory only, never a trade trigger)
Once signal_gate.tradeable is true, size by conviction:
- conviction 3+ (high) → full size.
- conviction 2 → 75% of standard size.
- Technicals clearly conflict with the ensemble direction → reduce a further 25%.

## Automatic stop management (do NOT mistake this for a risk problem)
The server runs a trailing/breakeven stop system on every open position. As a
trade moves into profit, the monitor automatically RATCHETS the stop-loss in
your favour — first to around breakeven (locking in the trade so it can no
longer lose), then trailing behind price to protect accrued gains. This means:
- A stop that sits only 1–2 pips from entry on a PROFITABLE position is NORMAL
  and GOOD — it is the breakeven lock, not a dangerously tight stop you set.
- The stop_loss shown in get_portfolio_status may differ from the 20 pips you
  placed. That is the ratchet working, not an error or a mis-fill.
- Do NOT close a winning position early out of concern that its stop looks
  "too tight". The tight stop is exactly what guarantees the trade cannot turn
  into a loss. Let it run to its target or let the trailing stop do its job.
Base hold/close decisions on the ensemble forecast and technicals, never on how
close the (auto-managed) stop appears to sit.

## Exits: reversal or risk event — NEVER conviction decay
Entering requires conviction >= 2. Exiting does NOT mirror that rule. Once a
position is open, the stop-loss, take-profit, and automatic trailing system
manage its risk — your job is to leave it alone unless something has actually
changed. Close a position early ONLY when one of these holds:
- REVERSAL: the ensemble now points in the OPPOSITE direction with conviction
  >= 2 (e.g. you are long and the forecast is DOWN with 2+ votes).
- EVENT RISK: an event blackout has begun for the pair and the position is at
  a loss (winners are already protected by the trailing stop).
Conviction decaying to 1 or FLAT is NOT an exit signal. Votes flicker around
zero every cycle; closing on decay converts the 1.6 R:R structure into a
coin-flip scratch that pays the spread every time — this exact churn (enter on
2, close on decay one cycle later, re-enter on the next flicker) has been the
single largest cost in the trade history. A position whose forecast went quiet
still has its stop 20 pips away and its target 32 pips away: let the geometry
resolve. Do not re-enter a pair you closed at a loss within the last 3 hours
unless a FRESH conviction >= 2 signal in the ensemble supports it.

## Take every setup that passes the gate
If multiple pairs pass the signal gate in the same cycle, do not pick just one
— open positions in up to THREE of them (highest conviction first), provided
exposure and position-count limits allow. Gate-passing setups are scarce;
leaving one on the table because another pair also qualified wastes the edge.
Diversification across pairs also smooths the P&L of any single bad call.

## Decision Process (follow this order each cycle)
1. Call scan_all_pairs to get a market overview.
2. Call get_portfolio_status to see what you currently hold.
3. Call get_risk_metrics to verify headroom.
4. For each open position, apply the exit rules above: close ONLY on reversal
   (opposite direction, conviction >= 2) or event risk — hold through decay.
5. For new opportunities, call get_technical_indicators on the best candidates.
6. Call get_price_forecast on any pair you are considering trading — its
   signal_gate verdict decides tradeability; its votes guide sizing.
7. Place the order with the conviction-adjusted size.
8. At the end, provide a brief market summary and your rationale.

Always be disciplined. It is perfectly fine to do nothing if the market offers no \
high-probability setups. Quality over quantity.
""".format(
    max_positions=settings.max_open_positions,
    max_daily_loss_pct=settings.max_daily_loss_pct * 100,
)


class TradingAgent:
    def __init__(self) -> None:
        self._client: Optional[anthropic.AsyncAnthropic] = None
        self._running: bool = False
        self._cycle_running: bool = False
        self._cycle: int = 0
        self._last_run: Optional[datetime] = None
        self._next_run: Optional[datetime] = None
        self._task: Optional[asyncio.Task] = None
        self._total_decisions: int = 0
        self._demo_mode: bool = False

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            if not settings.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set — agent cannot run.")
            self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        return self._client

    # ── Agent loop ────────────────────────────────────────────────────────────

    async def run_once(self) -> Dict[str, Any]:
        """Execute a single trading cycle."""
        if self._cycle_running:
            logger.warning("Cycle already running — skipping concurrent request")
            return {"cycle": self._cycle, "skipped": True, "reason": "Another cycle is already running"}

        # Weekend short-circuit: market closed + flat book = nothing an LLM
        # cycle could do (prices frozen at Friday close, entry gate blocks all
        # orders anyway). Guards the manual Run-Now path too — the scheduler
        # already checks market hours, but this makes the skip unconditional.
        if not self._is_market_open():
            from services.portfolio_service import portfolio_service
            if not await portfolio_service.get_open_positions():
                logger.info("Market closed and book flat — cycle skipped (no LLM call)")
                return {
                    "cycle": self._cycle,
                    "skipped": True,
                    "reason": "FX market closed (weekend) and no open positions — "
                              "next cycle after Sunday 22:00 UTC open",
                }

        # Cross-worker mutex: attempt to INSERT the singleton CycleLock row.
        # SQLite's PRIMARY KEY uniqueness makes this atomic — the second worker's
        # INSERT fails if the first already holds the lock.
        _LOCK_TTL_SECS = 600   # stale lock expires after 10 min (crash-safe)
        import os as _os
        try:
            async with AsyncSessionLocal() as _db:
                _existing = await _db.get(CycleLock, 1)
                if _existing is not None:
                    _lock_age = (datetime.now(timezone.utc).replace(tzinfo=None) - _existing.started_at).total_seconds()
                    if _lock_age < _LOCK_TTL_SECS:
                        logger.info(
                            "Cycle lock held by pid=%s (%.0fs ago) — skipping",
                            _existing.worker_pid, _lock_age,
                        )
                        return {
                            "cycle": self._cycle,
                            "skipped": True,
                            "reason": f"Another worker holds cycle lock ({_lock_age:.0f}s)",
                        }
                    # Stale lock — take it over
                    _existing.started_at = datetime.utcnow()
                    _existing.worker_pid = _os.getpid()
                else:
                    _db.add(CycleLock(id=1, started_at=datetime.utcnow(), worker_pid=_os.getpid()))
                try:
                    await _db.commit()
                except Exception:
                    # Another worker committed first (race on INSERT) — we lost
                    await _db.rollback()
                    logger.info("Cycle lock race lost to another worker — skipping")
                    return {
                        "cycle": self._cycle,
                        "skipped": True,
                        "reason": "Another worker won the cycle lock race",
                    }
        except Exception:
            logger.exception("Cycle lock DB check failed — proceeding anyway")

        self._cycle_running = True
        try:
            return await self._run_once_inner()
        finally:
            self._cycle_running = False
            # Release the cross-worker lock
            try:
                async with AsyncSessionLocal() as _db:
                    _lock = await _db.get(CycleLock, 1)
                    if _lock is not None:
                        await _db.delete(_lock)
                        await _db.commit()
            except Exception:
                logger.exception("Failed to release cycle lock")

    async def _run_once_inner(self) -> Dict[str, Any]:
        self._cycle += 1
        self._last_run = datetime.now(timezone.utc)
        logger.info(f"Agent cycle {self._cycle} started")

        if not settings.anthropic_api_key:
            result = await self._demo_cycle()
            return result

        try:
            return await self._claude_cycle()
        except Exception as e:
            logger.error(f"Agent cycle {self._cycle} error: {e}", exc_info=True)
            error_msg = str(e)
            try:
                await self._save_decision(
                    market_summary="",
                    reasoning=f"Cycle {self._cycle} failed: {error_msg}",
                    actions=[f"ERROR: {error_msg}"],
                    input_tokens=0,
                    output_tokens=0,
                )
            except Exception:
                logger.exception("Failed to save error decision to DB")
            return {"cycle": self._cycle, "error": error_msg, "actions": []}

    async def _claude_cycle(self) -> Dict[str, Any]:
        """Full Claude-powered agentic trading cycle with tool use."""
        client = self._get_client()
        messages: List[dict] = [
            {
                "role": "user",
                "content": (
                    f"Trading cycle #{self._cycle}. "
                    f"Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}. "
                    "Please analyse the market, review your portfolio, and take appropriate actions. "
                    "Start by scanning all pairs, then check your portfolio."
                ),
            }
        ]

        actions: List[str] = []
        market_summary: str = ""
        reasoning: str = ""
        input_tokens = 0
        output_tokens = 0

        # Agentic tool-use loop (max 20 iterations to prevent runaway)
        for iteration in range(20):
            response = await client.messages.create(
                model=settings.agent_model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens

            # Collect text content
            text_parts = [b.text for b in response.content if b.type == "text"]
            if text_parts:
                reasoning = "\n".join(text_parts)

            # If no tool use, agent is done
            if response.stop_reason == "end_turn":
                market_summary = reasoning
                break

            # Process tool calls
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break

            # Add assistant message
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tool calls and collect results
            tool_results = []
            for tool_use in tool_uses:
                logger.info(f"Tool call: {tool_use.name}({json.dumps(tool_use.input)[:200]})")
                try:
                    result = await handle_tool_call(tool_use.name, tool_use.input)
                    actions.append(f"{tool_use.name}({json.dumps(tool_use.input)[:100]})")
                except Exception as e:
                    result = {"error": str(e)}

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(result),
                })

            messages.append({"role": "user", "content": tool_results})

        # Persist decision
        decision = await self._save_decision(
            market_summary=market_summary,
            reasoning=reasoning,
            actions=actions,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self._total_decisions += 1

        await portfolio_service.snapshot()
        logger.info(f"Agent cycle {self._cycle} complete. Actions: {len(actions)}")
        return {
            "cycle": self._cycle,
            "actions": actions,
            "reasoning": reasoning,
            "market_summary": market_summary,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    async def _demo_cycle(self) -> Dict[str, Any]:
        """
        Rule-based fallback when no API key is configured.
        Uses simple technical signals to demonstrate the platform.
        """
        from services.market_data import market_data
        from services.order_service import order_service
        from services.risk_manager import risk_manager
        import random

        actions = []
        reasoning_lines = [
            "[DEMO MODE — set ANTHROPIC_API_KEY to enable full AI trading]",
            "",
            "Running rule-based signal scan...",
        ]

        state = await portfolio_service.get_state()

        # Review open positions — close any with significant loss
        for pos in state["positions"]:
            if pos.unrealised_pnl < -(state["balance"] * 0.01):
                ok, msg, pnl = await order_service.close_position(
                    pos.id, reasoning="Demo: cutting loss"
                )
                if ok:
                    actions.append(f"Closed position {pos.id}: {msg}")
                    reasoning_lines.append(f"  ✗ Closed losing position {pos.id} on {pos.pair}: {pnl:.2f}")

        # Scan for new signals
        for pair in list(market_data.get_all_prices().keys())[:5]:
            if state["open_positions"] >= settings.max_open_positions:
                break
            ind = market_data.calculate_indicators(pair)
            if not ind:
                continue

            direction: Optional[str] = None
            signal_reason = ""

            # Trend + RSI signal
            if ind["trend"] == "BULLISH" and 40 < ind["rsi"] < 60 and ind["macd_histogram"] > 0:
                direction = "BUY"
                signal_reason = f"Bullish trend, RSI={ind['rsi']:.1f}, MACD hist positive"
            elif ind["trend"] == "BEARISH" and 40 < ind["rsi"] < 60 and ind["macd_histogram"] < 0:
                direction = "SELL"
                signal_reason = f"Bearish trend, RSI={ind['rsi']:.1f}, MACD hist negative"
            # Mean reversion signal
            elif ind["bb_pct_b"] < 0.05 and ind["rsi"] < 30:
                direction = "BUY"
                signal_reason = f"Oversold: BB%b={ind['bb_pct_b']:.2f}, RSI={ind['rsi']:.1f}"
            elif ind["bb_pct_b"] > 0.95 and ind["rsi"] > 70:
                direction = "SELL"
                signal_reason = f"Overbought: BB%b={ind['bb_pct_b']:.2f}, RSI={ind['rsi']:.1f}"

            if direction is None:
                continue

            bar = market_data.get_price(pair)
            if bar is None:
                continue

            price = bar.ask if direction == "BUY" else bar.bid
            pip = market_data.get_pip_size(pair)
            # Enforce minimum 15-pip stop regardless of ATR
            sl_pips = max(ind["atr_pips"] * 1.5, settings.min_stop_pips)
            sl_distance = sl_pips * pip
            tp_distance = sl_distance * settings.default_risk_reward

            stop_loss = round(price - sl_distance if direction == "BUY" else price + sl_distance, 6)
            take_profit = round(price + tp_distance if direction == "BUY" else price - tp_distance, 6)

            # Size: risk 1% of balance per trade (units = risk_amount / price_risk_per_unit)
            risk_amount = state["balance"] * 0.01
            size = round(risk_amount / sl_distance) if sl_distance > 0 else 1000

            check = await risk_manager.check_new_order(
                pair, direction, size, price, stop_loss, take_profit
            )
            if not check.allowed:
                continue

            ok, msg, pos = await order_service.open_position(
                pair=pair,
                direction=direction,
                size=size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=f"Demo signal: {signal_reason}",
            )
            if ok and pos:
                actions.append(f"Opened {direction} {pair}")
                reasoning_lines.append(
                    f"  ✓ {direction} {pair} @ {pos.entry_price:.5f} | {signal_reason}"
                )

        if not actions:
            reasoning_lines.append("  — No qualifying signals found. Holding cash.")

        reasoning = "\n".join(reasoning_lines)
        await self._save_decision(
            market_summary=f"Demo cycle {self._cycle}",
            reasoning=reasoning,
            actions=actions,
            input_tokens=0,
            output_tokens=0,
        )
        await portfolio_service.snapshot()
        self._total_decisions += 1
        return {"cycle": self._cycle, "actions": actions, "reasoning": reasoning}

    # ── Market hours ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_market_open() -> bool:
        """
        FX market is open Sunday 22:00 UTC through Friday 21:00 UTC.
        Saturday is always closed; the Friday-to-Sunday gap is ~49 hours.
        """
        now = datetime.now(timezone.utc)
        wd = now.weekday()   # Mon=0 … Sun=6
        h  = now.hour
        if wd == 5:                    # Saturday — always closed
            return False
        if wd == 4 and h >= 21:        # Friday from 21:00 UTC
            return False
        if wd == 6 and h < 22:         # Sunday until 22:00 UTC
            return False
        return True

    # ── Scheduler ─────────────────────────────────────────────────────────────

    async def _scheduler_loop(self) -> None:
        while self._running:
            if self._is_market_open():
                try:
                    await self.run_once()
                except Exception:
                    logger.exception("Unhandled error in scheduler — cycle skipped")
            else:
                logger.info("FX market closed (weekend) — scheduler idle")
            self._next_run = datetime.fromtimestamp(
                datetime.now(timezone.utc).timestamp() + settings.agent_interval_seconds,
                tz=timezone.utc,
            )
            await asyncio.sleep(settings.agent_interval_seconds)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("Trading agent started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("Trading agent stopped")

    # ── Persistence ───────────────────────────────────────────────────────────

    async def _save_decision(
        self,
        market_summary: str,
        reasoning: str,
        actions: List[str],
        input_tokens: int,
        output_tokens: int,
    ) -> AgentDecision:
        async with AsyncSessionLocal() as db:
            decision = AgentDecision(
                cycle=self._cycle,
                market_summary=market_summary[:2000] if market_summary else "",
                reasoning=reasoning[:8000] if reasoning else "",
                actions_taken=json.dumps(actions),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            db.add(decision)
            await db.commit()
            await db.refresh(decision)
        return decision

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "cycle_running": self._cycle_running,
            "market_open": self._is_market_open(),
            "cycle": self._cycle,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "next_run": self._next_run.isoformat() if self._next_run else None,
            "total_decisions": self._total_decisions,
            "mode": "ai" if settings.anthropic_api_key else "demo",
        }


# Singleton
trading_agent = TradingAgent()

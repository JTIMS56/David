"""
David — Autonomous AI FX Trading Agent
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

from config import settings
from database import AsyncSessionLocal
from models.orm import AgentDecision
from agents.tools import TOOL_DEFINITIONS, handle_tool_call
from services.portfolio_service import portfolio_service

logger = logging.getLogger("david.agent")

SYSTEM_PROMPT = """\
You are David, an autonomous AI FX (foreign exchange) trading agent. Your mandate is \
to generate consistent risk-adjusted returns by trading major currency pairs in the \
foreign exchange market using a disciplined, rules-based approach.

## Capabilities
You have access to the following tools:
- scan_all_pairs        → Quick overview of all pairs and their signals
- get_fx_rates          → Current bid/ask prices for any pairs
- get_technical_indicators → RSI, MACD, Bollinger Bands, SMAs, ATR for a pair
- get_price_history     → Raw recent price data
- get_portfolio_status  → Account balance, equity, open positions, P&L
- get_risk_metrics      → Exposure, daily loss, position limits
- place_order           → Open a new BUY or SELL position
- close_position        → Close an existing position

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
- Minimum stop distance: 15 pips.
- Minimum Risk:Reward ratio: 1.5 (TP must be at least 1.5× the stop distance).
- Maximum position size: 5% of balance in notional terms.
- Maximum concurrent open positions: {max_positions}.
- Do NOT trade if daily loss already exceeds {max_daily_loss_pct}% of balance.

## Decision Process (follow this order each cycle)
1. Call scan_all_pairs to get a market overview.
2. Call get_portfolio_status to see what you currently hold.
3. Call get_risk_metrics to verify headroom.
4. For each open position, decide: hold, adjust SL/TP, or close.
5. For new opportunities, call get_technical_indicators on the best candidates.
6. Place orders only when the signal is clear and risk rules are satisfied.
7. At the end, provide a brief market summary and your rationale.

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
        self._cycle += 1
        self._last_run = datetime.now(timezone.utc)
        logger.info(f"Agent cycle {self._cycle} started")

        if not settings.anthropic_api_key:
            result = await self._demo_cycle()
            return result

        try:
            return await self._claude_cycle()
        except Exception as e:
            logger.error(f"Agent cycle error: {e}", exc_info=True)
            return {"cycle": self._cycle, "error": str(e), "actions": []}

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
            sl_distance = ind["atr_pips"] * 1.5 * pip
            tp_distance = sl_distance * settings.default_risk_reward

            stop_loss = price - sl_distance if direction == "BUY" else price + sl_distance
            take_profit = price + tp_distance if direction == "BUY" else price - tp_distance

            # Size: 1% balance risk
            risk_amount = state["balance"] * 0.01
            size = round(risk_amount / sl_distance, 0) if sl_distance > 0 else 1000

            check = await risk_manager.check_new_order(
                pair, direction, size, price, stop_loss, take_profit
            )
            if not check.allowed:
                continue

            ok, msg, pos = await order_service.open_position(
                pair=pair,
                direction=direction,
                size=size,
                stop_loss=round(stop_loss, 6),
                take_profit=round(take_profit, 6),
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

    # ── Scheduler ─────────────────────────────────────────────────────────────

    async def _scheduler_loop(self) -> None:
        while self._running:
            await self.run_once()
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
            "cycle": self._cycle,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "next_run": self._next_run.isoformat() if self._next_run else None,
            "total_decisions": self._total_decisions,
            "mode": "ai" if settings.anthropic_api_key else "demo",
        }


# Singleton
trading_agent = TradingAgent()

"""
Agent Tool Definitions & Implementations
────────────────────────────────────────────────────────────────────────────
Each tool maps directly to a capability the Claude agent can invoke.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from config import settings
from services.market_data import market_data, PAIR_CONFIG
from services.portfolio_service import portfolio_service
from services.order_service import order_service
import services.risk_manager as _risk_mod

# ── Tool schema definitions (for Claude's tool_use) ──────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "get_fx_rates",
        "description": (
            "Get the current bid/ask/mid prices for a list of currency pairs. "
            "Always call this at the start of your analysis cycle."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pairs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Currency pairs to fetch, e.g. ['EUR/USD', 'GBP/USD']",
                }
            },
            "required": ["pairs"],
        },
    },
    {
        "name": "get_technical_indicators",
        "description": (
            "Calculate RSI, MACD, Bollinger Bands, SMAs, ATR, and trend direction "
            "for a single currency pair. Use to evaluate entry/exit signals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pair": {"type": "string", "description": "e.g. 'EUR/USD'"}
            },
            "required": ["pair"],
        },
    },
    {
        "name": "get_price_history",
        "description": "Return recent mid prices as a list for manual analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pair": {"type": "string"},
                "periods": {
                    "type": "integer",
                    "default": 50,
                    "description": "Number of recent price bars to return (max 200)",
                },
            },
            "required": ["pair"],
        },
    },
    {
        "name": "get_portfolio_status",
        "description": (
            "Return current account balance, equity, unrealised P&L, daily P&L, "
            "drawdown, and all open positions with their current P&L."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_risk_metrics",
        "description": (
            "Return current risk exposure metrics: total notional, "
            "exposure percentage, daily loss, and whether limits are near."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "place_order",
        "description": (
            "Open a new BUY or SELL position. Always provide stop_loss. "
            "Providing take_profit is strongly recommended. "
            "Include your reasoning so it can be logged."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pair": {"type": "string", "description": "e.g. 'EUR/USD'"},
                "direction": {"type": "string", "enum": ["BUY", "SELL"]},
                "size": {
                    "type": "number",
                    "description": "Units to trade. Use get_risk_metrics to guide sizing.",
                },
                "stop_loss": {
                    "type": "number",
                    "description": "Stop-loss price (required).",
                },
                "take_profit": {
                    "type": "number",
                    "description": "Take-profit price (recommended).",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Your analysis and reason for this trade.",
                },
            },
            "required": ["pair", "direction", "size", "stop_loss", "reasoning"],
        },
    },
    {
        "name": "close_position",
        "description": "Close an open position by its ID. Provide your reasoning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "position_id": {"type": "integer"},
                "reasoning": {"type": "string"},
            },
            "required": ["position_id", "reasoning"],
        },
    },
    {
        "name": "scan_all_pairs",
        "description": (
            "Scan all configured pairs for technical signals in one call. "
            "Returns a compact summary table — useful for pair selection."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]

# ── Tool implementations ──────────────────────────────────────────────────────


async def handle_tool_call(name: str, inputs: Dict[str, Any]) -> Any:
    if name == "get_fx_rates":
        return await _get_fx_rates(inputs["pairs"])
    if name == "get_technical_indicators":
        return market_data.calculate_indicators(inputs["pair"])
    if name == "get_price_history":
        return await _get_price_history(inputs["pair"], inputs.get("periods", 50))
    if name == "get_portfolio_status":
        return await _get_portfolio_status()
    if name == "get_risk_metrics":
        return await _get_risk_metrics()
    if name == "place_order":
        return await _place_order(inputs)
    if name == "close_position":
        return await _close_position(inputs)
    if name == "scan_all_pairs":
        return await _scan_all_pairs()
    return {"error": f"Unknown tool: {name}"}


async def _get_fx_rates(pairs: list) -> dict:
    result = {}
    for pair in pairs:
        bar = market_data.get_price(pair)
        if bar:
            pip = market_data.get_pip_size(pair)
            result[pair] = {
                "bid": bar.bid,
                "ask": bar.ask,
                "mid": bar.mid,
                "spread_pips": round((bar.ask - bar.bid) / pip, 1),
                "timestamp": bar.timestamp.isoformat(),
            }
        else:
            result[pair] = {"error": "No price available"}
    return result


async def _get_price_history(pair: str, periods: int) -> dict:
    bars = market_data.get_history(pair, min(periods, 200))
    return {
        "pair": pair,
        "count": len(bars),
        "prices": [round(b.mid, 6) for b in bars],
        "timestamps": [b.timestamp.isoformat() for b in bars[-10:]],  # last 10 only
    }


async def _get_portfolio_status() -> dict:
    state = await portfolio_service.get_state()
    positions_out = []
    for pos in state["positions"]:
        positions_out.append({
            "id": pos.id,
            "pair": pos.pair,
            "direction": pos.direction,
            "size": pos.size,
            "entry_price": pos.entry_price,
            "current_price": pos.current_price,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
            "unrealised_pnl": round(pos.unrealised_pnl, 2),
            "opened_at": pos.opened_at.isoformat(),
        })
    return {
        "balance": state["balance"],
        "equity": state["equity"],
        "unrealised_pnl": state["unrealised_pnl"],
        "daily_pnl": state["daily_pnl"],
        "total_pnl": state["total_pnl"],
        "drawdown_pct": state["drawdown_pct"],
        "open_positions": state["open_positions"],
        "positions": positions_out,
    }


async def _get_risk_metrics() -> dict:
    state = await portfolio_service.get_state()
    balance = state["balance"]
    return {
        "balance": balance,
        "total_notional": state["total_notional"],
        "exposure_pct": round(state["total_notional"] / balance * 100, 2) if balance else 0,
        "max_exposure_pct": settings.max_total_exposure_pct * 100,
        "daily_pnl": state["daily_pnl"],
        "daily_loss_limit": round(balance * settings.max_daily_loss_pct, 2),
        "drawdown_pct": state["drawdown_pct"],
        "open_positions": state["open_positions"],
        "max_positions": settings.max_open_positions,
        "max_position_notional": round(balance * settings.max_position_size_pct, 2),
        "limits_ok": (
            state["daily_pnl"] > -(balance * settings.max_daily_loss_pct)
            and state["open_positions"] < settings.max_open_positions
        ),
    }


async def _place_order(inputs: dict) -> dict:
    pair = inputs["pair"]
    direction = inputs["direction"]
    size = float(inputs["size"])
    stop_loss = float(inputs["stop_loss"])
    take_profit = inputs.get("take_profit")
    if take_profit is not None:
        take_profit = float(take_profit)
    reasoning = inputs.get("reasoning", "")

    # Risk check
    bar = market_data.get_price(pair)
    if bar is None:
        return {"success": False, "message": f"No price for {pair}"}

    entry_price = bar.ask if direction == "BUY" else bar.bid
    check = await _risk_mod.risk_manager.check_new_order(
        pair, direction, size, entry_price, stop_loss, take_profit
    )
    if not check.allowed:
        return {"success": False, "message": f"Risk check failed: {check.reason}"}

    ok, msg, pos = await order_service.open_position(
        pair=pair,
        direction=direction,
        size=size,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasoning=reasoning,
        source="agent",
    )
    result = {"success": ok, "message": msg}
    if pos:
        result["position_id"] = pos.id
        result["entry_price"] = pos.entry_price
    return result


async def _close_position(inputs: dict) -> dict:
    ok, msg, pnl = await order_service.close_position(
        position_id=inputs["position_id"],
        reasoning=inputs.get("reasoning", ""),
    )
    return {"success": ok, "message": msg, "pnl": round(pnl, 2)}


async def _scan_all_pairs() -> list:
    rows = []
    for pair in PAIR_CONFIG:
        ind = market_data.calculate_indicators(pair)
        if not ind:
            continue
        rows.append({
            "pair": pair,
            "price": ind["current_price"],
            "trend": ind["trend"],
            "rsi": ind["rsi"],
            "macd_hist": ind["macd_histogram"],
            "bb_pct_b": ind["bb_pct_b"],
            "atr_pips": ind["atr_pips"],
        })
    return rows

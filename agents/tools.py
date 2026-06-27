"""
Agent Tool Definitions & Implementations
────────────────────────────────────────────────────────────────────────────
Each tool maps directly to a capability the Claude agent can invoke.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

logger = logging.getLogger("david.tools")

from config import settings
from services.market_data import market_data, PAIR_CONFIG
from services.portfolio_service import portfolio_service
from services.order_service import order_service
import services.risk_manager as _risk_mod
from services.dirac_predictor import DiracPredictor

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
            "Include your reasoning so it can be logged.\n"
            "HARD SIGNAL GATE (enforced server-side — orders that fail are rejected, "
            "so check these BEFORE calling to avoid wasted attempts):\n"
            "  1. Call get_price_forecast for the pair first (within the last ~3 min).\n"
            "  2. Only trade when DHJ and Black-Scholes DISAGREE on direction — that "
            "is the only measured edge (~52%). If they agree, skip the pair.\n"
            "  3. Your direction must MATCH the DHJ call (BUY if DHJ expects UP, SELL "
            "if DOWN). Do not trade against DHJ.\n"
            "  4. MILD_BULLISH signals are blocked (47% accurate). Skip them.\n"
            "  5. EUR/GBP is blocked entirely (chronic churn)."
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
    {
        "name": "get_price_forecast",
        "description": (
            "Run the Dirac-Heston-Jump (DHJ) probabilistic model to forecast price "
            "direction for a currency pair. Returns expected move in pips, probability "
            "of price rising above spot, chiral charge (market sentiment proxy), and a "
            "summary signal (STRONG_BULLISH / MILD_BULLISH / NEUTRAL / MILD_BEARISH / "
            "STRONG_BEARISH). Use this AFTER get_technical_indicators to confirm or "
            "challenge a trade signal before placing an order."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pair": {"type": "string", "description": "e.g. 'EUR/USD'"},
                "horizon_days": {
                    "type": "number",
                    "description": "Forecast horizon in days (0.5=12h, 1=1day, 2=2days). Default 1.",
                },
            },
            "required": ["pair"],
        },
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
    if name == "get_price_forecast":
        return await _get_price_forecast(inputs["pair"], inputs.get("horizon_days", 1.0))
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
    pip = market_data.get_pip_size(pair)
    current_stop_pips = round(abs(entry_price - stop_loss) / pip, 1) if stop_loss is not None else None

    check = await _risk_mod.risk_manager.check_new_order(
        pair, direction, size, entry_price, stop_loss, take_profit
    )
    if not check.allowed:
        # Pre-compute the correct SL/TP so the agent has exact values, not just a formula
        _ind = market_data.calculate_indicators(pair)
        atr_pips = _ind.get("atr_pips", 0) if _ind else 0
        correct_stop_pips = max(atr_pips * 1.5, 20.0)
        correct_stop_dist = correct_stop_pips * pip
        if direction == "BUY":
            correct_sl = round(entry_price - correct_stop_dist, 5)
            correct_tp = round(entry_price + correct_stop_dist * 1.6, 5)
        else:
            correct_sl = round(entry_price + correct_stop_dist, 5)
            correct_tp = round(entry_price - correct_stop_dist * 1.6, 5)
        hint = {
            "entry_price_now": round(entry_price, 5),
            "atr_pips": round(atr_pips, 1),
            "correct_stop_pips": round(correct_stop_pips, 1),
            "use_exactly": {
                "stop_loss": correct_sl,
                "take_profit": correct_tp,
            },
            "rule": "If still rejected after ONE retry with these exact values, skip this pair.",
        }
        return {
            "success": False,
            "message": f"Risk check failed: {check.reason}",
            "how_to_fix": hint,
        }

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


async def _get_price_forecast(pair: str, horizon_days: float = 1.0) -> dict:
    bar = market_data.get_price(pair)
    if bar is None:
        return {"error": f"No price available for {pair}"}

    spot = bar.mid
    oanda_pair = pair.replace("/", "")

    try:
        predictor = DiracPredictor.instance()
    except FileNotFoundError as exc:
        return {"error": f"DHJ model not available: {exc}"}

    # Derive delta_cp (spinor initial asymmetry) from current RSI momentum.
    # RSI > 50 → bullish bias → positive delta_cp; RSI < 50 → negative.
    # MACD histogram agreement amplifies; disagreement dampens.
    ind = market_data.calculate_indicators(pair)
    if ind:
        rsi = ind.get("rsi", 50.0)
        macd_hist = ind.get("macd_histogram", 0.0)
        delta_cp = (rsi - 50.0) / 100.0           # range [-0.5, +0.5]
        # dampen when RSI and MACD disagree
        if (delta_cp > 0 and macd_hist < 0) or (delta_cp < 0 and macd_hist > 0):
            delta_cp *= 0.5
        delta_cp = max(-0.5, min(0.5, delta_cp))
    else:
        delta_cp = 0.0

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: predictor.predict_dhj(
                pair=oanda_pair,
                spot=spot,
                horizon_days=float(horizon_days),
                n_paths=300,
                delta_cp=delta_cp,
            ),
        )
    except Exception as exc:
        return {"error": f"DHJ prediction failed: {exc}"}

    # Compute P(S_T > spot) by integrating density above current price
    prices = result.prices
    probs = result.prob_dhj
    prob_up = 0.0
    for i in range(len(prices) - 1):
        if prices[i] >= spot:
            prob_up += probs[i] * (prices[i + 1] - prices[i])
    prob_up = round(min(max(prob_up, 0.0), 1.0), 3)

    pip = market_data.get_pip_size(pair)
    expected_move_pips = round((result.mean_dhj - spot) / pip, 1)

    q5 = result.chiral_charge
    # Q₅ (chiral charge) reflects spinor field asymmetry seeded from RSI momentum.
    # Thresholds calibrated to the RSI-driven delta_cp scale (range ±0.5):
    #   |Q5| > 0.15 → STRONG (RSI ~65+/35-)   |Q5| > 0.05 → MILD (RSI ~55+/45-)
    if q5 > 0.15 and prob_up > 0.52:
        signal = "STRONG_BULLISH"
    elif q5 > 0.05:
        signal = "MILD_BULLISH"
    elif q5 < -0.15 and prob_up < 0.48:
        signal = "STRONG_BEARISH"
    elif q5 < -0.05:
        signal = "MILD_BEARISH"
    else:
        signal = "NEUTRAL"

    output = {
        "pair": pair,
        "spot": round(spot, 6),
        "horizon_days": horizon_days,
        # ── Directional signal ────────────────────────────────────────────────
        # signal and expected_direction are both derived from Q₅ (chiral_charge),
        # NOT from mean_dhj.  At 1-day horizons with near-zero carry the DHJ mean
        # barely moves from spot (±1-2 pip MC noise), so mean_dhj direction is
        # uninformative noise.  Q₅ is seeded from RSI momentum and is the true
        # directional predictor.
        "signal": signal,
        "expected_direction": "UP" if q5 > 0 else "DOWN",
        "chiral_charge": round(q5, 4),
        "prob_above_spot": prob_up,
        # ── Model drift (carry-adjusted expected price drift, NOT a magnitude bet) ──
        # At 1-day horizon this is typically ±1-3 pips regardless of actual move.
        # Use it only for model diagnostics, not for sizing or direction decisions.
        "model_drift_pips": expected_move_pips,
        # ── Volatility & tail risk ─────────────────────────────────────────────
        "dhj_higher_tail_risk": result.call_dhj > result.call_bs,
        "implied_vol_annualized": round(result.avg_variance ** 0.5, 4),
        # ── Reference prices (for diagnostic comparison) ───────────────────────
        "dhj_expected_price": round(result.mean_dhj, 6),
        "bs_expected_price": round(result.mean_bs, 6),
        "dhj_call_price": round(result.call_dhj, 6),
        "bs_call_price": round(result.call_bs, 6),
        "delta_cp": round(delta_cp, 4),
        "rsi_at_forecast": round(ind["rsi"], 1) if ind else None,
    }

    _bs_direction = "UP" if result.mean_bs > spot else "DOWN"

    # Cache for the hard pre-trade signal gate (synchronous — must be set before
    # the agent can place an order off this forecast in the same cycle).
    from services.signal_cache import put_signal
    put_signal(
        pair=pair,
        signal=signal,
        dhj_direction=output["expected_direction"],
        bs_direction=_bs_direction,
    )

    asyncio.create_task(_log_forecast_to_db(
        pair=pair,
        spot_price=spot,
        horizon_days=float(horizon_days),
        signal=signal,
        expected_direction=output["expected_direction"],
        expected_move_pips=expected_move_pips,
        prob_above_spot=prob_up,
        chiral_charge=round(q5, 4),
        dhj_expected_price=round(result.mean_dhj, 6),
        bs_expected_price=round(result.mean_bs, 6),
        bs_expected_direction=_bs_direction,
    ))

    return output


async def _log_forecast_to_db(**kwargs) -> None:
    try:
        from database import AsyncSessionLocal
        from models.orm import ForecastLog
        horizon_at = datetime.utcnow() + timedelta(days=kwargs["horizon_days"])
        async with AsyncSessionLocal() as db:
            db.add(ForecastLog(horizon_at=horizon_at, **kwargs))
            await db.commit()
    except Exception as exc:
        logger.warning("Failed to log forecast: %s", exc)

from __future__ import annotations

import json
import logging
from typing import List

from pydantic import model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


def _parse_origins(raw: str) -> List[str]:
    """Accept plain URL, comma-separated list, or JSON array string."""
    raw = raw.strip()
    if raw.startswith("["):
        return json.loads(raw)
    return [o.strip() for o in raw.split(",") if o.strip()]


# Bumped on every deploy-affecting change. Surfaced at /api/status and logged at
# startup so "is my latest commit actually live?" is answerable in one look
# rather than inferred from behaviour.
APP_VERSION = "2026.08.23-telemetry"


class Settings(BaseSettings):
    # ── Anthropic ─────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    agent_model: str = "claude-sonnet-4-6"

    # ── Trading ───────────────────────────────────────────────────────────────
    trading_mode: str = "paper"          # "paper" | "oanda"
    initial_balance: float = 100_000.0
    agent_interval_seconds: int = 300

    # ── Risk ──────────────────────────────────────────────────────────────────
    max_position_size_pct: float = 0.05
    max_total_exposure_pct: float = 0.30
    max_daily_loss_pct: float = 0.03
    max_open_positions: int = 8
    default_risk_reward: float = 1.5
    default_stop_pips: int = 20
    pip_value_usd: float = 10.0

    # ── Market data ───────────────────────────────────────────────────────────
    market_data_mode: str = "simulation"  # "simulation" | "live" | "oanda"
    alpha_vantage_key: str = ""
    # Seconds between real-rate fetches in live mode.
    # Alpha Vantage free (25 calls/day, 8 pairs): 86400/25*8 ≈ 27648s → use 21600 (6h)
    # Alpha Vantage premium (75 req/min): set to 60
    live_refresh_interval: int = 21600

    # ── Security ──────────────────────────────────────────────────────────────
    api_key: str = ""
    ws_token: str = ""
    # Store as plain string; use settings.get_allowed_origins() where a list is needed.
    # Accepts: "https://a.com", "https://a.com,https://b.com", or '["https://a.com"]'
    allowed_origins: str = "*"

    # ── Risk circuit-breakers ─────────────────────────────────────────────────
    max_drawdown_pct: float = 10.0
    min_stop_pips: float = 15.0       # minimum stop distance (pips); blocks dangerously tight stops
    max_stop_pips: float = 50.0       # hard cap on stop distance; blocks orders with wider stops

    # Volatility / cost floor — don't trade dead markets where the spread eats
    # the signal (the cycle-#20 overtrading bleed: 1.2-1.8p ATR, ~1.5p spread).
    min_atr_pips: float = 4.0             # skip when ATR is below this (market too quiet)
    min_tp_spread_multiple: float = 3.0   # require take-profit distance >= this x current spread

    # Execution tier — how agent orders are executed:
    #   "shadow" → orders logged only, nothing executes
    #   "micro"  → orders EXECUTE at drastically reduced size (real fill data,
    #              pocket-change risk) — used to shake out platform bugs
    #   "full"   → normal sizing. Current setting: full-size dress rehearsal on
    #              the OANDA practice account to mimic real go-live conditions.
    #              Flip back to "micro" when real capital first goes in.
    execution_tier: str = "full"
    micro_size_factor: float = 0.10        # scale agent order size to 10% of requested
    micro_max_notional: float = 500.0      # hard cap per position (USD notional)
    micro_daily_loss_limit: float = 25.0   # stop opening new positions past -$25 on the day

    # ── Per-pair edge map (shadow validation) ─────────────────────────────────
    # Conditional edges found in the forecast log: per pair, whether to trade
    # WITH or FADE the DHJ call depending on DHJ↔BS agreement. Discovered
    # in-sample; running in shadow until they hold up out-of-sample (after the
    # cutoff). NOT wired into trading yet.
    edge_map: dict = {
        "NZD/USD": {"agree": "with", "disagree": "fade"},
        "EUR/USD": {"disagree": "with"},
        "USD/CHF": {"agree": "fade"},
        "AUD/USD": {"agree": "fade"},
    }
    edge_map_cutoff: str = "2026-06-28"   # forecasts created after this are out-of-sample

    # Alarm when the freshest quote on the book is older than this. A live feed
    # updates every few seconds; anything beyond a few minutes means the feed
    # has died and every downstream number is being computed on frozen prices.
    feed_stale_alarm_seconds: float = 300.0

    # ── Medium-frequency execution engine ─────────────────────────────────────
    # Deterministic sub-second decision loop. Runs the same ensemble and the
    # same hard risk gate as the agent, but as pure computation with no model
    # call, so detection-to-order is milliseconds instead of minutes. The LLM
    # agent keeps running for supervision and position review.
    fast_engine_enabled: bool = False        # opt-in; agent alone is the default
    fast_engine_interval_ms: int = 250       # decision cadence
    fast_engine_pair_cooldown_s: float = 300.0   # min seconds between entries per instrument
    fast_engine_max_entries_per_min: int = 3     # global throttle
    fast_engine_max_new_per_cycle: int = 2       # highest-conviction first

    # ── Weekend protection ────────────────────────────────────────────────────
    # FX closes ~21:00 UTC Friday. Holding through the weekend exposes positions
    # to Monday gap risk with no stop protection (gaps fill at the open price).
    # No new entries from Friday 20:00 UTC; all positions flattened 20:30 UTC.
    weekend_flatten_enabled: bool = True
    weekend_no_entry_from_hour_utc: int = 20   # Friday cutoff for new entries
    weekend_flatten_hour_utc: int = 20         # Friday flatten time (HH:MM)
    weekend_flatten_minute_utc: int = 30

    # ── Go-live readiness (Aug 1 go/no-go criteria windows) ───────────────────
    go_nogo_window_start: str = "2026-07-18"   # exit-hysteresis era: trades judged from here
    clean_data_start: str = "2026-07-08"       # single-instance era: forecasts judged from here

    # ── Crowd sentiment (OANDA position book) ─────────────────────────────────
    # Contrarian vote: fade the retail crowd when positioning is one-sided.
    positioning_fade_threshold: float = 65.0   # % of positions on one side to trigger a fade vote
    position_book_refresh_seconds: int = 1200  # OANDA refreshes the book ~every 20 min

    # ── Economic calendar (event blackout) ────────────────────────────────────
    calendar_url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    calendar_refresh_seconds: int = 21600      # re-fetch weekly feed every 6h
    event_blackout_pre_minutes: int = 45       # no conviction this long BEFORE a high-impact event
    event_blackout_post_minutes: int = 30      # ... and this long AFTER it hits

    # ── Ensemble model (shadow mode) ──────────────────────────────────────────
    # An independent multi-signal model logged alongside DHJ for head-to-head
    # out-of-sample comparison. NOT used for trading until it proves out.
    ensemble_shadow_enabled: bool = True
    ensemble_high_conviction: int = 3   # |net vote| at/above which a call is "high conviction"
    # Approximate central-bank policy rates (%) for the carry signal. Slow-moving;
    # update when policy shifts. Used only for rate-differential direction.
    policy_rates: dict = {
        "USD": 4.50, "EUR": 2.40, "GBP": 4.25, "JPY": 0.50,
        "AUD": 3.85, "CAD": 2.75, "CHF": 0.25, "NZD": 3.25,
    }
    carry_diff_threshold: float = 1.0   # min rate-differential (%) to register a carry vote

    # ── Signal-quality gates (hard pre-trade filters) ─────────────────────────
    # Ensemble-based. DHJ is retired from the decision path (coin flip over
    # ~800 evaluations; every conditional slice non-stationary) and runs only
    # as a silent background benchmark. Enforced at the service layer — the
    # LLM cannot bypass these.
    signal_gate_enabled: bool = True
    min_trade_conviction: int = 2              # net ensemble votes required to trade (of 5 voters)
    blocked_pairs: List[str] = ["EUR/GBP"]     # chronic range-bound churn — net loser
    signal_max_age_seconds: float = 180.0      # forecast must be fresh to trade off it

    # ── Profit protection (trailing stop / breakeven) ─────────────────────────
    # Lock in gains so a winning position cannot round-trip back into a loss.
    trailing_stop_enabled: bool = True
    breakeven_trigger_pips: float = 10.0  # profit at which stop jumps to entry (breakeven)
    breakeven_buffer_pips: float = 1.0    # lock this many pips beyond entry (covers spread)
    trail_trigger_pips: float = 15.0      # profit at which the trailing stop activates
    trail_distance_pips: float = 10.0     # trail this far behind the best price reached

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./fx_trading.db"

    # ── Tradable instruments ──────────────────────────────────────────────────
    default_pairs: List[str] = [
        # FX majors
        "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
        "USD/CAD", "EUR/GBP", "NZD/USD", "USD/CHF",
        # Equity index CFDs
        "SPX500", "NAS100", "US30", "DE30", "UK100",
        # Metals
        "XAU/USD", "XAG/USD",
    ]

    # Asset classes permitted to EXECUTE. Anything scanned but not listed here is
    # forecast, gated, and shadow-logged, but never sent to the broker.
    #
    # All three are live on the PRACTICE account so end-to-end execution can be
    # observed across asset classes. Note what the evidence does and does not
    # say: the index backtest validated buy-and-hold (Sharpe 0.55), NOT the
    # agent's short-horizon ensemble. Index and metal P&L here is an observation
    # of machinery, not a validated strategy — drop back to ["fx"] before any
    # real capital is involved.
    live_asset_classes: List[str] = ["fx", "index", "metal"]

    # ── OANDA ────────────────────────────────────────────────────────────────
    oanda_api_key: str = ""
    oanda_account_id: str = ""
    oanda_environment: str = "practice"  # "practice" | "live"

    # ── Environment ───────────────────────────────────────────────────────────
    environment: str = "development"

    def get_allowed_origins(self) -> List[str]:
        return _parse_origins(self.allowed_origins)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @model_validator(mode="after")
    def _warn_insecure(self) -> "Settings":
        if self.environment == "production":
            if not self.api_key:
                logger.warning(
                    "⚠️  API_KEY is not set — all /api/* endpoints are UNPROTECTED."
                )
            if self.allowed_origins.strip() in ("*", '["*"]'):
                logger.warning(
                    "⚠️  ALLOWED_ORIGINS is '*' — restrict to your domain in production."
                )
            if not self.ws_token:
                logger.warning(
                    "⚠️  WS_TOKEN is not set — WebSocket control commands are unauthenticated."
                )
        return self


settings = Settings()

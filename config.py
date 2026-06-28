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
    # The only directional edge in the data is DHJ↔Black-Scholes disagreement
    # (~51.8%); DHJ agreement and MILD_BULLISH calls are at/below coin-flip.
    # These gates are enforced at the service layer — the LLM cannot bypass them.
    signal_gate_enabled: bool = True
    require_dhj_bs_disagreement: bool = True   # only trade when DHJ and BS disagree
    require_direction_matches_dhj: bool = True # order must align with DHJ's call (where the edge is)
    block_mild_bullish: bool = True            # MILD_BULLISH is 47% accurate — skip it
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

    # ── Currency pairs ────────────────────────────────────────────────────────
    default_pairs: List[str] = [
        "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
        "USD/CAD", "EUR/GBP", "NZD/USD", "USD/CHF",
    ]

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

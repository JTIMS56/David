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
    trading_mode: str = "paper"          # "paper" | "live"
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
    market_data_mode: str = "simulation"  # "simulation" | "live"
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
    max_stop_pips: float = 50.0       # hard cap on stop distance; blocks orders with wider stops

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./fx_trading.db"

    # ── Currency pairs ────────────────────────────────────────────────────────
    default_pairs: List[str] = [
        "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
        "USD/CAD", "EUR/GBP", "NZD/USD", "USD/CHF",
    ]

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

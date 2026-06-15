from __future__ import annotations

import json
import logging
from typing import List

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


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

    # ── Security ──────────────────────────────────────────────────────────────
    api_key: str = ""          # Protect all /api/* routes. REQUIRED in production.
    ws_token: str = ""         # Protect WebSocket control commands.
    allowed_origins: List[str] = ["*"]   # Set to your domain in production.

    # ── Risk circuit-breakers ─────────────────────────────────────────────────
    max_drawdown_pct: float = 10.0

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./fx_trading.db"

    # ── Currency pairs ────────────────────────────────────────────────────────
    default_pairs: List[str] = [
        "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
        "USD/CAD", "EUR/GBP", "NZD/USD", "USD/CHF",
    ]

    # ── Environment ───────────────────────────────────────────────────────────
    environment: str = "development"     # "development" | "production"

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _parse_origins(cls, v: object) -> object:
        # Accept both a plain comma-separated string and a JSON array string
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                return json.loads(v)
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @model_validator(mode="after")
    def _warn_insecure(self) -> "Settings":
        if self.environment == "production":
            if not self.api_key:
                logger.warning(
                    "⚠️  API_KEY is not set — all /api/* endpoints are UNPROTECTED. "
                    "Set API_KEY in your .env file before going live."
                )
            if self.allowed_origins == ["*"]:
                logger.warning(
                    "⚠️  ALLOWED_ORIGINS is '*' — restrict to your domain in production."
                )
            if not self.ws_token:
                logger.warning(
                    "⚠️  WS_TOKEN is not set — WebSocket control commands are unauthenticated."
                )
        return self


settings = Settings()

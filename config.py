from __future__ import annotations

from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str = ""
    agent_model: str = "claude-sonnet-4-6"

    # Trading
    trading_mode: str = "paper"
    initial_balance: float = 100_000.0
    agent_interval_seconds: int = 300

    # Risk
    max_position_size_pct: float = 0.05
    max_total_exposure_pct: float = 0.30
    max_daily_loss_pct: float = 0.03
    max_open_positions: int = 8
    default_risk_reward: float = 1.5
    default_stop_pips: int = 20
    pip_value_usd: float = 10.0  # per standard lot per pip

    # Market data
    market_data_mode: str = "simulation"
    alpha_vantage_key: str = ""

    # DB
    database_url: str = "sqlite+aiosqlite:///./fx_trading.db"

    # Default currency pairs to trade
    default_pairs: List[str] = [
        "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
        "USD/CAD", "EUR/GBP", "NZD/USD", "USD/CHF",
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

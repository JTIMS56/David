from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


def _utc(v: datetime) -> str:
    """Serialize a naive UTC datetime to an ISO-8601 string with explicit Z suffix."""
    return v.isoformat() + "Z"


class PriceTickOut(BaseModel):
    pair: str
    timestamp: datetime
    bid: float
    ask: float
    mid: float
    spread_pips: float = 0.0


class PositionOut(BaseModel):
    id: int
    pair: str
    direction: str
    size: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    unrealised_pnl: float
    status: str
    opened_at: datetime
    closed_at: Optional[datetime]
    realised_pnl: float
    reasoning: Optional[str]

    class Config:
        from_attributes = True
        json_encoders = {datetime: _utc}


class TradeOut(BaseModel):
    id: int
    position_id: Optional[int]
    pair: str
    action: str
    direction: str
    size: float
    price: float
    pnl: float
    timestamp: datetime
    reasoning: Optional[str]

    class Config:
        from_attributes = True
        json_encoders = {datetime: _utc}


class AgentDecisionOut(BaseModel):
    id: int
    timestamp: datetime
    cycle: int
    market_summary: Optional[str]
    reasoning: Optional[str]
    actions_taken: Optional[str]
    input_tokens: int
    output_tokens: int

    class Config:
        from_attributes = True
        json_encoders = {datetime: _utc}


class PortfolioOut(BaseModel):
    balance: float
    equity: float
    open_positions: int
    unrealised_pnl: float
    daily_pnl: float
    total_pnl: float
    drawdown_pct: float
    positions: List[PositionOut] = []


class RatesOut(BaseModel):
    rates: dict[str, PriceTickOut]
    timestamp: datetime


class AgentStatusOut(BaseModel):
    running: bool
    cycle_running: bool = False
    market_open: bool = True
    cycle: int
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    total_decisions: int
    mode: str


class SystemStatusOut(BaseModel):
    status: str
    trading_mode: str
    market_data_mode: str
    agent: AgentStatusOut
    portfolio: PortfolioOut

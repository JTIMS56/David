from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class PriceTick(Base):
    __tablename__ = "price_ticks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pair: Mapped[str] = mapped_column(String(10), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True)
    bid: Mapped[float] = mapped_column(Float)
    ask: Mapped[float] = mapped_column(Float)
    mid: Mapped[float] = mapped_column(Float)


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pair: Mapped[str] = mapped_column(String(10), index=True)
    direction: Mapped[str] = mapped_column(String(4))   # BUY | SELL
    size: Mapped[float] = mapped_column(Float)           # in units (lots * 100k)
    entry_price: Mapped[float] = mapped_column(Float)
    current_price: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    take_profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unrealised_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(8), default="OPEN")  # OPEN | CLOSED
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    close_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    realised_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    oanda_trade_id: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)


class Trade(Base):
    """Immutable log of every order event."""
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    position_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    pair: Mapped[str] = mapped_column(String(10))
    action: Mapped[str] = mapped_column(String(8))   # OPEN | CLOSE | SL_HIT | TP_HIT
    direction: Mapped[str] = mapped_column(String(4))
    size: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class AgentDecision(Base):
    __tablename__ = "agent_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True)
    cycle: Mapped[int] = mapped_column(Integer, default=0)
    market_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    actions_taken: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True)
    balance: Mapped[float] = mapped_column(Float)
    equity: Mapped[float] = mapped_column(Float)
    open_positions: Mapped[int] = mapped_column(Integer, default=0)
    daily_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    total_pnl: Mapped[float] = mapped_column(Float, default=0.0)


class ModelRun(Base):
    """
    Model registry: every DHJ/Dirac prediction call is recorded here.
    Links agent decisions to the exact model version and parameters used.
    """
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True)
    model: Mapped[str] = mapped_column(String(20), index=True)     # "DHJ" | "Dirac"
    pair: Mapped[str] = mapped_column(String(10), index=True)
    spot: Mapped[float] = mapped_column(Float)
    horizon_days: Mapped[float] = mapped_column(Float)
    params: Mapped[Optional[str]] = mapped_column(Text, nullable=True)    # JSON
    git_commit: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    # Output scalars
    mean_model: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    call_model: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    call_bs: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    chiral_charge: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Diagnostics
    n_steps: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mass_loss_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    negative_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Link to agent decision that triggered this run (nullable for API calls)
    agent_decision_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class ForecastLog(Base):
    """Every DHJ get_price_forecast call made by the agent, with outcome filled in after horizon."""
    __tablename__ = "forecast_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True)
    pair: Mapped[str] = mapped_column(String(10), index=True)
    spot_price: Mapped[float] = mapped_column(Float)
    horizon_days: Mapped[float] = mapped_column(Float)
    horizon_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    # DHJ signal
    signal: Mapped[str] = mapped_column(String(20))
    expected_direction: Mapped[str] = mapped_column(String(4))
    expected_move_pips: Mapped[float] = mapped_column(Float)
    prob_above_spot: Mapped[float] = mapped_column(Float)
    chiral_charge: Mapped[float] = mapped_column(Float)
    dhj_expected_price: Mapped[float] = mapped_column(Float)
    # Black-Scholes reference — logged alongside DHJ for head-to-head comparison
    bs_expected_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bs_expected_direction: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)
    # Outcome — filled by background evaluator after horizon_at
    outcome_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_move_pips: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    direction_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    bs_direction_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    evaluated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class AuditLog(Base):
    """Immutable record of every order attempt and gate decision."""
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now(), index=True)
    # Who submitted the order
    source: Mapped[str] = mapped_column(String(20))          # agent | human | sl_tp | system
    event_type: Mapped[str] = mapped_column(String(20), index=True)  # ORDER_OPEN | ORDER_REJECT | SL_TP | KILL_SWITCH | ...
    # Order details
    pair: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    direction: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)
    size: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    entry_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    take_profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    position_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    realised_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Gate outcome
    gate_allowed: Mapped[Optional[bool]] = mapped_column(nullable=True)
    gate_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Extra context (JSON string)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

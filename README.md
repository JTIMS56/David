# David — Agentic AI FX Trading Platform

An autonomous AI-powered foreign exchange trading platform built with **Claude** (Anthropic) as the decision-making engine. Trades major currency pairs using a fully agentic tool-use loop, with paper trading, real-time risk management, and a live dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Web Dashboard (/)                    │
│          Real-time WebSocket · REST API                 │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              FastAPI Application (main.py)              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  API Routes  │  │  WebSocket   │  │  Scheduler    │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└──────────┬────────────────────────────────┬─────────────┘
           │                                │
┌──────────▼──────────┐      ┌─────────────▼────────────┐
│  Claude Agent       │      │  Services                │
│  ─────────────────  │      │  ────────────────────── │
│  • Tool-use loop    │◄────►│  • MarketDataService     │
│  • Trading logic    │      │  • OrderService          │
│  • Risk reasoning   │      │  • PortfolioService      │
│  • Decision logs    │      │  • RiskManager           │
└─────────────────────┘      └──────────────────────────┘
                                           │
                              ┌────────────▼───────────┐
                              │  SQLite Database       │
                              │  (SQLAlchemy async)    │
                              └────────────────────────┘
```

## Features

- **Autonomous Claude Agent** — Uses Claude's tool-use API to scan markets, analyse technicals, place and close orders, and manage risk autonomously
- **Technical Analysis** — RSI, MACD, Bollinger Bands, EMA/SMA, ATR computed on every cycle
- **Risk Management** — Pre-trade checks: position sizing, exposure limits, daily loss halt, required R:R ratio
- **Paper Trading** — Safe simulation mode with realistic fill prices and spread modelling
- **Market Simulation** — Geometric Brownian Motion engine generating realistic FX price paths for all 8 major pairs
- **Real-time Dashboard** — Live price ticker, equity curve, open positions, trade history, agent decision log
- **WebSocket streaming** — Sub-second price and portfolio updates pushed to connected clients
- **Demo Mode** — Rule-based fallback when no API key is set, so the platform can run immediately

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

# 3. Start the platform
uvicorn main:app --host 0.0.0.0 --port 8000

# 4. Open the dashboard
open http://localhost:8000
```

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Claude API key (required for AI mode) |
| `TRADING_MODE` | `paper` | `paper` or `live` |
| `INITIAL_BALANCE` | `100000.0` | Starting portfolio balance (USD) |
| `AGENT_INTERVAL_SECONDS` | `300` | How often the agent runs (5 min) |
| `AGENT_MODEL` | `claude-sonnet-4-6` | Claude model to use |
| `MAX_POSITION_SIZE_PCT` | `0.05` | Max 5% of portfolio per position |
| `MAX_TOTAL_EXPOSURE_PCT` | `0.30` | Max 30% total exposure |
| `MAX_DAILY_LOSS_PCT` | `0.03` | Halt trading if daily loss > 3% |
| `MAX_OPEN_POSITIONS` | `8` | Max concurrent positions |
| `MARKET_DATA_MODE` | `simulation` | `simulation` or `live` |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Live dashboard |
| `GET` | `/api/status` | Full system status |
| `GET` | `/api/rates` | All current FX rates |
| `GET` | `/api/portfolio` | Portfolio summary + positions |
| `GET` | `/api/positions` | Open/closed positions |
| `DELETE` | `/api/positions/{id}` | Manually close a position |
| `GET` | `/api/trades` | Trade history |
| `GET` | `/api/indicators/{pair}` | Technical indicators |
| `GET` | `/api/history/{pair}` | Price history |
| `GET` | `/api/agent/status` | Agent status |
| `POST` | `/api/agent/start` | Start the agent |
| `POST` | `/api/agent/stop` | Stop the agent |
| `POST` | `/api/agent/run-once` | Trigger a single agent cycle |
| `GET` | `/api/agent/decisions` | Agent decision log |
| `WS` | `/ws` | Real-time price/portfolio stream |

## Supported Pairs

EUR/USD · GBP/USD · USD/JPY · AUD/USD · USD/CAD · EUR/GBP · NZD/USD · USD/CHF

## Agent Decision Process

Each cycle, the Claude agent:
1. Calls `scan_all_pairs` → market overview
2. Calls `get_portfolio_status` → current holdings and P&L
3. Calls `get_risk_metrics` → verify trading headroom
4. Analyses open positions → decide to hold, close, or adjust
5. Identifies new opportunities → calls `get_technical_indicators` on candidates
6. Places orders that meet risk/reward criteria
7. Logs complete reasoning and actions

> **Note:** This is a paper-trading simulator. No real money is at risk.

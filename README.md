# Popper — Autonomous FX Trading Platform & Strategy Validation Engine

An autonomous foreign-exchange trading platform with an LLM agent (Claude) as its
decision engine — and, more importantly, a **rigorous strategy-validation machine**
that was honest enough to report what it found.

Over five weeks this system was built, deployed to production, connected to a live
broker (OANDA), debugged through real incidents, and used to run **pre-registered,
out-of-sample tests** of every major retail FX strategy family. The headline result
is a negative one, delivered with unusual statistical rigor — and a positive control
that proves the instrument works.

> **TL;DR of the research:** at retail data/cost/latency access, no tested FX
> strategy — intraday signals, multi-signal ensembles, time-series carry+trend, or
> cross-sectional carry — produced an edge distinguishable from a coin flip over
> 14+ years of data. The same engine, pointed at equity indices, immediately
> detected the equity risk premium (Sharpe 0.55 through CFD costs), validating the
> detector. The edge isn't hiding at this access level. That's the finding.

---

## Findings (the part most trading repos don't have)

| # | Hypothesis tested | Method | Result |
|---|---|---|---|
| 1 | Physics-inspired intraday model (Dirac-Heston-Jump) predicts 1-day direction | 2,500+ live logged forecasts, evaluated against realized prices | **49–50%** — coin flip; every conditional slice non-stationary |
| 2 | Conditional "edge cells" (per-pair model-agreement patterns, 66–77% in-sample) | Frozen rules, held-out out-of-sample window | **Collapsed to 15–31%** — textbook overfit, caught before capital |
| 3 | Five-voter ensemble (trend, mean-reversion, carry, USD breadth, crowd positioning) at conviction ≥ 2 | 325 clean out-of-sample forecasts, Wilson bounds | **49.8%** (lower bound 0.453) — no edge |
| 4 | Time-series carry+trend, 8 USD-majors | 14.9y daily backtest, no lookahead, costs + financing on | **Sharpe 0.00** |
| 5 | Cross-sectional carry (literature construction, 16 pairs incl. JPY/CHF crosses) | 14.7y backtest, monthly rebalance | **Sharpe 0.14 ± 0.26** — indistinguishable from zero |
| ✓ | **Positive control:** equity risk premium via index CFDs | Same engine, 14.5y, CFD financing + fees charged | **Sharpe 0.55** — detector confirmed working |

The positive control matters: an instrument that returns zeros is only meaningful if
it demonstrably detects a signal where one is known to exist. It does.

**Interpretation:** short-horizon FX predictability from price-derived signals is
arbitraged flat below retail cost/latency floors; the documented carry premium
requires rate dispersion (dead 2012–2021) and even then measures near zero at
retail spreads; the harvestable premium that survives (equities) is best captured
passively. The commit history doubles as a lab notebook of the full reasoning.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│               Web Dashboard (real-time, WebSocket)           │
│   equity curve · positions · trade log · agent reasoning     │
│   go-live readiness scorecard · backtest cards (1-click)     │
└──────────────────────────┬───────────────────────────────────┘
┌──────────────────────────▼───────────────────────────────────┐
│                 FastAPI application (main.py)                │
│      REST API · WebSocket broadcast · lifecycle wiring       │
└───────┬──────────────────────────────────────┬───────────────┘
┌───────▼───────────────┐          ┌───────────▼───────────────┐
│  LLM Trading Agent    │          │  Services                 │
│  (Claude tool-use     │          │  market data (OANDA SSE   │
│   loop, hourly cycle) │◄────────►│   stream + REST fallback) │
│  ensemble forecasts   │          │  order service + monitor  │
│  conviction gating    │          │  portfolio / risk manager │
│  position management  │          │  sentiment (position book)│
└───────┬───────────────┘          │  econ calendar (blackouts)│
        │                          │  backtester (FX + index)  │
┌───────▼───────────────────────┐  └───────────┬───────────────┘
│  HARD RISK GATE (server-side, │  ┌───────────▼───────────────┐
│  LLM cannot bypass):          │  │  OANDA v20 (live broker)  │
│  kill switch · exposure caps  │  │  orders · fills · candles │
│  signal gate · vol floor      │  └───────────────────────────┘
│  event blackout · weekend     │  ┌───────────────────────────┐
│  flatten · daily loss halt    │  │  PostgreSQL: positions,   │
│  execution tiers              │  │  trades, forecasts, audit │
└───────────────────────────────┘  └───────────────────────────┘
```

**Design principle:** the LLM proposes; deterministic server-side code disposes.
Every order passes a hard gate the agent cannot argue with. The agent's system
prompt teaches it the rules, but enforcement never depends on the model obeying.

### Key components

- **Agent loop** — hourly cycles: scan → forecast → gate check → order/manage →
  narrate. Structured tool-use with typed schemas; pips-based order geometry so
  price drift between quote and fill can never invalidate stop/target placement.
- **Ensemble forecaster** — five independent ±1 voters (trend, mean-reversion,
  rate-differential carry, USD breadth, contrarian crowd positioning from OANDA's
  position book). Direction = sign of net vote; conviction = magnitude; server
  gate requires conviction ≥ 2.
- **Execution tiers** — `shadow` (log, never execute) → `micro` (real fills at
  10% size, $500 cap, $25/day loss budget) → `full`. Strategies are promoted only
  by out-of-sample evidence; the micro tier caught two platform-integrity bugs at
  a total cost of $1.79.
- **Event & session protection** — economic-calendar blackouts (no entries around
  high-impact releases), volatility floor (no trading dead markets where spread
  eats the target), Friday flatten + weekend entry block (no gap exposure).
- **Trailing risk management** — breakeven ratchet and trailing stops, monitored
  every 3s against fresh broker prices (never against a potentially stale
  internal feed — see incident #2).
- **Validation harness** — every forecast logged with its inputs and evaluated
  against realized prices after horizon; accuracy reported with Wilson lower
  bounds; in-sample discovery strictly separated from out-of-sample verdicts;
  interpretation rules pre-committed before results were seen.
- **Backtester** — lookahead-safe by construction (position for t→t+1 decided at
  t), spread costs on every position change, financing/carry accrued daily,
  literature-derived parameters with zero fitted knobs. Validated on synthetic
  data before touching real history (trend world → profits; random walk → zero;
  crash world → momentum cuts drawdown).

---

## Engineering incidents (caught, diagnosed, fixed)

Real production war stories, each found by the platform's own instrumentation:

| Incident | Symptom | Root cause | Cost at detection |
|---|---|---|---|
| Duplicate agents | Every cycle log arrived twice; "slippage" of 11–74 pips; positions closing 1–2 pips from entry against 20-pip stops | Cloud app scaled to **2 containers** — two agents, two monitors, two divergent price caches on one account | ~$1.79 (micro tier) |
| Phantom stop triggers | Instant "SL_HIT" minutes after open | Monitor compared real fills against a stale internal feed; breakeven stops armed on the desync | included above |
| ATR inflation | Volatility floor never triggered | ATR computed from bid/ask spread instead of realized range | $0 (shadow) |
| Overfit edge map | In-sample 68.9% accuracy cells | Time-correlated samples; collapsed to 31% out-of-sample | $0 (pre-registered OOS split refused to promote) |
| Conviction-decay churn | 4 re-entries per pair per day, spread bleed | Exit rule mirrored entry rule; fixed with exit hysteresis (exit only on reversal, not decay) | ~$70 practice |
| Fast-market rejection storms | 9 rejected orders in one cycle | Agent computed absolute SL/TP from stale quotes; fixed with server-anchored pips-based orders | tokens only |

The pattern worth noting: every anomaly was **instrumented, root-caused, and
fixed at the cheapest possible tier** — shadow before micro, micro before full,
practice before real.

---

## Stack

Python 3.11 · FastAPI · SQLAlchemy (async) / PostgreSQL · Anthropic API (agent
tool-use loop) · OANDA v20 (streaming prices, orders, position book, historical
candles) · C++ pricing engine (pybind11) for the original DHJ model · vanilla-JS
dashboard with WebSocket push · Docker on DigitalOcean App Platform.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY (+ OANDA keys for live data)
uvicorn main:app --host 0.0.0.0 --port 8000
open http://localhost:8000
```

Full deployment guide: [DEPLOY.md](DEPLOY.md). Defaults are safe: practice
account, hard risk limits on, weekend protection on. The dashboard's one-click
backtest cards reproduce every research result in this README from free OANDA
historical data.

Key settings (`.env` / `config.py`): `EXECUTION_TIER` (shadow/micro/full),
`MAX_POSITION_SIZE_PCT`, `MAX_DAILY_LOSS_PCT`, `MARKET_DATA_MODE`
(must be `oanda` when trading on OANDA), `AGENT_INTERVAL_SECONDS`.

---

## Lessons

**Engineering**
1. Never let the decision layer and the execution layer read different clocks or
   different prices. Most "model failures" were data-integrity failures.
2. Make safety deterministic. LLM agents follow prompts *usually*; hard gates
   hold *always*.
3. Graduated execution tiers turn catastrophic bugs into $2 lessons.

**Markets**
4. In-sample accuracy is marketing; out-of-sample accuracy is truth. Pre-register
   the interpretation before seeing results, or you will move the goalposts.
5. Signal-to-noise in FX improves with the square root of horizon; retail costs
   are fixed per trade. Both gradients point the same way: slower.
6. A validation engine that can say **no** is worth more than a strategy that
   says yes. This repo's most valuable output is a defensible negative result —
   and the positive control that makes it believable.

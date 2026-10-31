# Project Handoff — Context for a New Claude Session

**Purpose:** paste this at the start of a new chat so Claude has the background from
prior work without re-deriving it. Written August 2026.

---

## 1. Who you're working with

**David N. Belyn** — U.S. Army officer (23+ years, currently G39 Information
Operations Planner, 3rd Infantry Division). Dual M.S. Applied Economics + Applied
Analytics at Boston College (3.87 GPA, completing spring 2027). B.S. Finance,
Auburn. Army retirement summer 2027; available for civilian work August 2027,
willing to relocate (including London).

Technically: Python (Pandas, scikit-learn, Flask, FastAPI, NLP/ML pipelines), R,
SQL, C++, React/JavaScript, Anthropic API including agentic tool-use. Comfortable
with Docker, PostgreSQL, cloud deploys.

**Working style he's responded well to:**
- Direct answers first, reasoning after. No preamble.
- Being told plainly when something won't work, with the reason and an alternative.
- Real measurements over estimates. He asks "why" and expects a specific answer.
- He pushes back and reaffirms when he wants something done his way — when he does,
  do it, note any risk once, and move on. He owns the decisions.
- He gets (reasonably) frustrated when a system sits idle. Explain root cause
  concretely rather than reassuring.

---

## 2. Project A — "Popper" (the trading platform, largely complete)

Repo: `jtims56/david`, branch `claude/agentic-fx-trading-platform-RGrb9`.
Originally named "David"; renamed to **Popper** (after Karl Popper, philosopher of
falsification) because the name was the author's own and the platform's method is
falsificationism.

### What it is
An autonomous FX/index/metals trading platform: LLM agent (Claude tool-use loop)
proposes trades, deterministic server-side risk gates dispose. FastAPI +
PostgreSQL + Docker on DigitalOcean App Platform, live OANDA v20 practice
integration, real-time WebSocket dashboard.

### The research result (the actual output)
Five weeks of pre-registered, out-of-sample testing found **no tradeable edge** in
retail FX:

| Hypothesis | Result |
|---|---|
| Physics-inspired intraday model (Dirac-Heston-Jump), 2,500+ live forecasts | 49–50%, coin flip |
| Data-mined conditional "edge cells", 66–77% in-sample | 15–31% out-of-sample (overfit) |
| Five-voter ensemble, conviction ≥ 2, n=325 OOS | 49.8% (Wilson LB 0.453) |
| Time-series carry+trend, 8 majors, 14.9y | Sharpe 0.00 |
| Cross-sectional carry, 16 pairs, 14.7y | Sharpe 0.14 ± 0.26 |
| **Positive control:** equity index buy-and-hold, same engine | **Sharpe 0.55** |

The positive control is the point: an instrument that only returns zeros is
worthless unless it demonstrably detects a signal where one is known to exist.
It does. So the FX zeros are readings, not malfunctions.

**No real capital was ever deployed.** Practice account throughout.

### Current state
Running on OANDA practice. Recent work: multi-asset support (15 instruments —
8 FX majors, 5 index CFDs, 2 metals), and a medium-frequency deterministic
execution engine (`services/fast_engine.py`) that takes the LLM out of the hot
path — forecast latency optimised 2,317 µs → 18.4 µs (126×) via vectorised EMA
(scipy `lfilter`) and per-tick indicator memoization.

### Published artifacts
- `README.md` — engineering case study
- `docs/medium-article.md` + `docs/img/` — published Medium article with 4 figures
- `docs/Belyn_Quant_Resume.pdf`, `docs/Belyn_VC_Tech_Resume.pdf`, cover letters
- Generator scripts for all of the above (`docs/gen_*.py`)

---

## 3. Engineering standards established — **the most transferable part**

These were learned the hard way and should carry to any new project.

**Never fabricate data that can mix with real data.** The platform seeded 200
synthetic "warm-up" price bars from hardcoded base prices. In live mode, real
quotes were appended to that synthetic series, creating a phantom 750-pip bar that
inflated ATR ~10×, pinned RSI at extremes, and silently froze trading for weeks.
Fix: in live modes, seed nothing; bootstrap from real broker candles instead.

**Suspect your plumbing before your model.** Four separate outages presented as
"the strategy isn't finding setups" and were all data-integrity failures:
- Cloud app scaled to **2 containers** → two agents, two price caches, one account.
  Symptom: duplicate logs, phantom fills, "slippage" of 11–74 pips.
- One unknown instrument in a batch pricing request → OANDA 400s the *entire*
  request → all prices froze for 11 days while nothing crashed.
- Decision layer and execution layer reading different price sources.
- Synthetic/real history blending (above).

**Fail loudly, never silently.** Every one of those hid because the failure mode
looked like normal quiet behaviour. Now: a feed watchdog logs CRITICAL when quotes
go stale, the forecast evaluator refuses to grade against a frozen feed, and the
scanner emits explicit `NO_DATA` rows instead of skipping instruments.

**Make invariants enforced, not advisory.** A misconfiguration warning that logs
and continues will be ignored. If trading on OANDA requires OANDA prices, the code
should *correct* it at startup, not warn.

**Ship a version marker.** `APP_VERSION` logged at startup and served at
`/api/version`. "Is my latest commit actually live?" must be answerable in one
look, not inferred from behaviour. Several days were lost to this.

**Separate discovery from verification, and pre-register the interpretation.**
Any pattern found in data must be sentenced by data it has never seen, and you
must decide what result means what *before* you look. This caught a 77%-accuracy
"edge" that scored 15% out-of-sample.

**Graduated exposure.** shadow (log, never execute) → micro (real but tiny) →
full. Two production bugs were caught for a total of $1.79.

**Measure, don't guess.** Every optimisation in this project was profiled first.
The EMA bottleneck (64% of indicator latency) was not where intuition pointed.

---

## 4. Infrastructure context (reusable)

- **GitHub:** `jtims56/david`. Claude works on a dedicated branch, opens draft PRs.
- **Hosting:** DigitalOcean App Platform, Docker, auto-deploy from branch.
  **Gotcha:** container count must be 1 for stateful apps — scaling to 2 silently
  breaks in-memory state. This cost weeks.
- **Database:** PostgreSQL in production, SQLite for local tests. Schema migrations
  are hand-rolled additive `ALTER TABLE ... IF NOT EXISTS` in `database.py`.
- **Secrets:** environment variables in the DO dashboard.
- **Domain:** dnbfx.io (trading dashboard).
- **Verification habit:** David pastes console output / API JSON from the live
  dashboard into chat. Build "Copy" buttons into dashboards — they make this
  trivial and he uses them constantly.

---

## 5. Project B — MERIDIAN (the IO project, the new work)

**This is what the new session is for: building a full production website.**

What exists today (from David's résumé; **confirm details with him**):
- Full-stack **Flask** application, deployed on **Render**
- Ingests **GDELT** data to monitor global news trends and narrative patterns
  across 100+ countries
- Applies **BERTopic**, transformer-based sentiment analysis, and custom
  **Information Operations frame scoring** to surface emerging narratives
- Themes tracked include AI, fintech, defense tech, logistics, and
  environmental/resource narratives
- Methodology paper targeted at *Parameters* (U.S. Army War College journal)

**Questions the new session should ask David up front:**
1. Is the new site a rebuild of MERIDIAN or a public-facing front end for it?
2. Audience — professional/analyst portfolio, research publication, or a product
   with users?
3. Keep Flask or move to FastAPI + a modern front end (he has React experience,
   and Popper's FastAPI patterns are reusable)?
4. Stay on Render, or move to DigitalOcean where the deploy pipeline is known?
5. Does GDELT ingest run continuously, or on demand?
6. Any OPSEC/classification constraints — he is an active-duty officer working in
   information operations, so **anything touching his professional work needs care**
   about what is published. Ask rather than assume.

**Reusable from Popper:** FastAPI app skeleton, WebSocket push, PostgreSQL async
patterns, Docker/DigitalOcean deploy config, dashboard layout with Copy buttons,
API-key auth middleware, the version-marker and health-check habits.

---

## 6. Career context (why the work matters to him)

He is building toward a civilian career starting August 2027 — targeting
quantitative research and engineering roles, with London quant firms (GSA Capital,
Man AHL, Squarepoint, Qube, G-Research, XTX) on the list. Autumn 2026 is when
their 2027 graduate research intakes open, which fits his timeline.

Popper's value is as a portfolio artifact and demonstrated engineering, not as a
money-maker — this was discussed explicitly and he agreed. **Do not encourage
trading real money on any strategy this project has produced.** He has, at times,
been under financial pressure; the honest position established here is that income
comes from his skills, savings go into index funds, and trading capital stays at
zero until evidence justifies otherwise.

---

## 7. Open threads

- Popper still runs on practice; execution tier `full`, all three asset classes
  live. Fast engine built but disabled by default.
- Ensemble accuracy on genuinely clean data is still accumulating — earlier
  measurements were corrupted by the feed incidents and should be treated as void.
- The DHJ benchmark logs silently as a retired control. Any accuracy number from
  it under ~300 clean samples is noise; watch for duplicate outcome rows inflating it.
- MERIDIAN website: not started.

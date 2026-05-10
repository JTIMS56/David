"""
Rolling 5-year backtest for the DHJ model.

Algorithm:
  For each rebalance date t (spaced by step_days):
    1. Estimate realised vol from a trailing window
    2. Run DHJ model → predicted distribution at horizon h
    3. Look up actual price at t + h
    4. Score: log-likelihood, CRPS, PIT, coverage, call pricing error
    5. Also score Black-Scholes for comparison (Diebold-Mariano)

  Aggregate: mean LL, CRPS, PIT histogram, KS test, coverage rate.

Usage:
  runner = BacktestRunner("EURUSD", horizon_days=22, step_days=5)
  results = runner.run()
  runner.report(results)
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .benchmarks import default_benchmark_suite
from .data import load_prices, estimate_realized_vol, compute_log_returns
from .metrics import (
    BacktestRecord, log_likelihood, crps_from_cdf, pit_score,
    coverage_95, distribution_stats, pit_uniformity_test, diebold_mariano,
)

logger = logging.getLogger(__name__)

# Default Heston + jump parameters (can be overridden)
_DEFAULT_DHJ = dict(
    kappa_H=1.5, xi=0.30, rho=-0.25,
    kappa0=1.0, kappa1=0.002,
    jump_lambda=5.0, jump_p_up=0.55,
    jump_eta_plus=100.0, jump_eta_minus=80.0,
    n_paths=200,
)


@dataclass
class BacktestResults:
    records: list[BacktestRecord] = field(default_factory=list)
    pair: str = ""
    horizon_days: int = 22
    model: str = "DHJ"
    benchmark_scores: dict = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.records])

    @property
    def mean_ll(self) -> float:
        return float(np.mean([r.log_likelihood for r in self.records]))

    @property
    def mean_crps(self) -> float:
        return float(np.mean([r.crps for r in self.records]))

    @property
    def coverage_rate(self) -> float:
        return float(np.mean([r.coverage_95 for r in self.records]))

    @property
    def pit_values(self) -> list[float]:
        return [r.pit for r in self.records]

    @property
    def call_rmse(self) -> float:
        errs = [(r.call_model - r.call_bs)**2 for r in self.records]
        return float(np.sqrt(np.mean(errs)))


class BacktestRunner:
    def __init__(
        self,
        pair: str = "EURUSD",
        start: str = "2019-01-01",
        end: str = "2024-12-31",
        horizon_days: int = 22,    # ≈ 1 calendar month
        step_days: int = 5,        # rebalance weekly
        vol_window: int = 63,      # 3-month rolling realised vol
        csv_path: Optional[str] = None,
        dhj_params: Optional[dict] = None,
        r: float = 0.0525,
    ):
        self.pair = pair
        self.horizon_days = horizon_days
        self.step_days = step_days
        self.vol_window = vol_window
        self.r = r
        self.dhj_params = dhj_params or _DEFAULT_DHJ

        self.prices = load_prices(pair, start, end, csv_path)
        self.rvol   = estimate_realized_vol(self.prices, vol_window)

    def run(self, seed_base: int = 0, include_benchmarks: bool = True) -> BacktestResults:
        """
        Run rolling backtest; returns BacktestResults.
        Set include_benchmarks=True to also score all classical benchmark models
        and print a comparison table at the end.
        """
        from services.dirac_predictor import DiracPredictor
        predictor   = DiracPredictor.instance()
        benchmarks  = default_benchmark_suite() if include_benchmarks else []

        results = BacktestResults(pair=self.pair, horizon_days=self.horizon_days)
        # Per-benchmark accumulators: {name: [ll, crps, coverage, pit]}
        bench_acc: dict[str, dict] = {
            b.name: {"ll": [], "crps": [], "cov": [], "pit": []}
            for b in benchmarks
        }
        dates = self.prices.index
        log_returns = compute_log_returns(self.prices)

        # Identify all forecast dates with a valid horizon outcome
        forecast_dates = []
        for i, d in enumerate(dates):
            future_idx = self._find_horizon_index(dates, i, self.horizon_days)
            if future_idx is None:
                continue
            if i < self.vol_window:
                continue  # not enough history for vol estimate
            if i % self.step_days != 0:
                continue
            forecast_dates.append((i, d, future_idx))

        logger.info("Running %d forecasts for %s (horizon=%dd, step=%dd)",
                    len(forecast_dates), self.pair, self.horizon_days, self.step_days)

        for k, (i, d, fi) in enumerate(forecast_dates):
            S0      = float(self.prices.iloc[i])
            S_real  = float(self.prices.iloc[fi])
            vol_est = float(self.rvol.iloc[i])
            if np.isnan(vol_est) or vol_est <= 0:
                vol_est = 0.085  # fallback

            try:
                pred = predictor.predict_dhj(
                    pair=self.pair,
                    spot=S0,
                    horizon_days=self.horizon_days,
                    seed=seed_base + k,
                    **self.dhj_params,
                )
                bs_pred = predictor.predict(
                    pair=self.pair,
                    spot=S0,
                    horizon_days=self.horizon_days,
                )
            except Exception as e:
                logger.warning("Forecast failed at %s: %s", d.date(), e)
                continue

            prices = pred.prices
            prob   = pred.prob_dhj  # per-unit-S density

            ll      = log_likelihood(S_real, prices, prob)
            crps    = crps_from_cdf(S_real, prices, prob)
            pit     = pit_score(S_real, prices, prob)
            cov95   = coverage_95(S_real, prices, prob)
            dstats  = distribution_stats(prices, prob)

            rec = BacktestRecord(
                date=str(d.date()),
                horizon_days=self.horizon_days,
                S0=S0,
                S_realized=S_real,
                log_ret_realized=np.log(S_real / S0),
                log_likelihood=ll,
                crps=crps,
                pit=pit,
                coverage_95=cov95,
                call_model=pred.call_dhj,
                call_bs=pred.call_bs,
                mean_model=dstats["mean"],
                std_model=dstats["std"],
                pair=self.pair,
                model="DHJ",
            )
            results.records.append(rec)

            # Score benchmark models at the same date
            past_rets = log_returns.iloc[:i].values
            for bm in benchmarks:
                try:
                    from services.dirac_predictor import _PAIR_PARAMS
                    pp = _PAIR_PARAMS.get(self.pair, {"r": 0.0525, "r_f": 0.0})
                    fc = bm.forecast(
                        S0, self.horizon_days, past_rets,
                        r_d=pp["r"], r_f=pp.get("r_f", 0.0),
                    )
                    bm_ll   = log_likelihood(S_real, list(fc.prices), list(fc.density))
                    bm_crps = crps_from_cdf(S_real, list(fc.prices), list(fc.density))
                    bm_pit  = pit_score(S_real, list(fc.prices), list(fc.density))
                    bm_cov  = coverage_95(S_real, list(fc.prices), list(fc.density))
                    bench_acc[bm.name]["ll"].append(bm_ll)
                    bench_acc[bm.name]["crps"].append(bm_crps)
                    bench_acc[bm.name]["cov"].append(bm_cov)
                    bench_acc[bm.name]["pit"].append(bm_pit)
                except Exception as e:
                    logger.debug("Benchmark %s failed at %s: %s", bm.name, d.date(), e)

            if (k + 1) % 50 == 0:
                logger.info("  %d/%d  mean_LL=%.3f  coverage=%.1f%%",
                            k+1, len(forecast_dates),
                            results.mean_ll, 100*results.coverage_rate)

        results.benchmark_scores = bench_acc
        return results

    @staticmethod
    def compare_models(dhj_results: BacktestResults, bs_results: BacktestResults) -> dict:
        """Diebold-Mariano test: DHJ log-likelihood vs BS log-likelihood."""
        dhj_ll = [r.log_likelihood for r in dhj_results.records]
        bs_ll  = [r.log_likelihood for r in bs_results.records]
        n = min(len(dhj_ll), len(bs_ll))
        return diebold_mariano(dhj_ll[:n], bs_ll[:n])

    def report(self, results: BacktestResults) -> str:
        """Print a concise backtest summary."""
        if not results.records:
            return "No records — check data and model setup."

        pit = results.pit_values
        ks  = pit_uniformity_test(pit)
        df  = results.to_dataframe()

        lines = [
            f"\n{'='*60}",
            f"  DHJ Backtest: {results.pair}  horizon={results.horizon_days}d  n={len(results.records)}",
            f"{'='*60}",
            f"  Mean log-likelihood : {results.mean_ll:.4f}",
            f"  Mean CRPS           : {results.mean_crps:.6f}",
            f"  Coverage (95% CI)   : {100*results.coverage_rate:.1f}%  (ideal = 95%)",
            f"  PIT KS p-value      : {ks['p_value']:.4f}  ({'calibrated ✓' if ks.get('calibrated') else 'miscalibrated ✗'})",
            f"  Call price RMSE     : {results.call_rmse:.6f}",
            "",
            f"  Distribution (mean over periods):",
            f"    E[S_T / S_0]      : {df['mean_model'].mean() / df['S0'].mean():.5f}",
            f"    σ forecast        : {df['std_model'].mean():.4f}",
            f"    σ realised        : {df['log_ret_realized'].std() * (252/results.horizon_days)**0.5:.4f}",
            "",
            f"  By year:",
        ]
        df["year"] = pd.to_datetime(df["date"]).dt.year
        for yr, grp in df.groupby("year"):
            cov = 100 * grp["coverage_95"].mean()
            mll = grp["log_likelihood"].mean()
            lines.append(f"    {yr}: n={len(grp):3d}  coverage={cov:.0f}%  mean_LL={mll:.3f}")
        # Benchmark comparison table
        bench = getattr(results, "benchmark_scores", {})
        if bench:
            lines.append("")
            lines.append("  Benchmark comparison (mean log-likelihood / CRPS / coverage):")
            lines.append(f"  {'Model':<20}  {'Mean LL':>9}  {'Mean CRPS':>10}  {'Coverage':>9}")
            lines.append(f"  {'-'*20}  {'-'*9}  {'-'*10}  {'-'*9}")
            # DHJ first
            lines.append(
                f"  {'DHJ (this model)':<20}  {results.mean_ll:>9.4f}  "
                f"{results.mean_crps:>10.6f}  {100*results.coverage_rate:>8.1f}%"
            )
            for name, acc in bench.items():
                if not acc["ll"]:
                    continue
                m_ll   = float(np.mean(acc["ll"]))
                m_crps = float(np.mean(acc["crps"]))
                m_cov  = 100.0 * float(np.mean(acc["cov"]))
                lines.append(
                    f"  {name:<20}  {m_ll:>9.4f}  {m_crps:>10.6f}  {m_cov:>8.1f}%"
                )

        lines.append("=" * 60)
        report_str = "\n".join(lines)
        print(report_str)
        return report_str

    def _find_horizon_index(self, dates, start_idx: int, horizon_days: int):
        """Find the index of the trading day closest to start + horizon_days calendar days."""
        if start_idx >= len(dates):
            return None
        target = dates[start_idx] + pd.Timedelta(days=horizon_days)
        remaining = dates[start_idx + 1:]
        if len(remaining) == 0:
            return None
        diff = abs(remaining - target)
        best = diff.argmin()
        if diff[best].days > 5:  # tolerate up to 5 calendar days mismatch
            return None
        return start_idx + 1 + best

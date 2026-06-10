#!/usr/bin/env python3
"""
CLI entry point for the DHJ backtester.

Usage:
  # Quick smoke test on synthetic data (no real data needed):
  python -m backtester.run_backtest --synthetic

  # Real data via yfinance (pip install yfinance):
  python -m backtester.run_backtest --pair EURUSD --start 2019-01-01 --end 2024-12-31

  # From CSV:
  python -m backtester.run_backtest --csv eurusd_daily.csv

  # Save results:
  python -m backtester.run_backtest --pair EURUSD --output results.csv
"""
import argparse
import logging
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DHJ Model Backtester")
    parser.add_argument("--pair",      default="EURUSD")
    parser.add_argument("--start",     default="2019-01-01")
    parser.add_argument("--end",       default="2024-12-31")
    parser.add_argument("--horizon",   type=int,   default=22,  help="Forecast horizon (trading days)")
    parser.add_argument("--step",      type=int,   default=5,   help="Rebalance frequency (trading days)")
    parser.add_argument("--paths",     type=int,   default=200, help="DHJ MC paths")
    parser.add_argument("--csv",       default=None, help="CSV file with date,price columns")
    parser.add_argument("--output",    default=None, help="Save results to CSV")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic GBM data")
    parser.add_argument("--calibrate", action="store_true",
                        help="Walk-forward calibration: refit sigma, Heston, and jump params each window")
    args = parser.parse_args()

    from backtester.runner import BacktestRunner, _DEFAULT_DHJ
    from backtester.data import synthetic_gbm

    csv_path = args.csv
    if args.synthetic:
        logger.info("Using synthetic GBM data for %s", args.pair)
        prices = synthetic_gbm(args.pair, args.start, args.end)
        prices.to_csv("/tmp/dhj_synthetic.csv", header=True)
        csv_path = "/tmp/dhj_synthetic.csv"
        logger.info("Synthetic prices saved to /tmp/dhj_synthetic.csv")

    dhj_params = {**_DEFAULT_DHJ, "n_paths": args.paths}

    runner = BacktestRunner(
        pair=args.pair,
        start=args.start,
        end=args.end,
        horizon_days=args.horizon,
        step_days=args.step,
        csv_path=csv_path,
        dhj_params=dhj_params,
        use_calibration=args.calibrate,
    )

    results = runner.run()
    runner.report(results)

    if args.output:
        df = results.to_dataframe()
        df.to_csv(args.output, index=False)
        logger.info("Results saved to %s", args.output)
        print(f"\nPIT histogram (should be ~uniform for calibrated model):")
        pit = np.array(results.pit_values)
        bins = np.histogram(pit, bins=10, range=(0, 1))[0]
        expected = len(pit) / 10
        for i, b in enumerate(bins):
            bar = "█" * int(b / max(1, expected) * 20)
            print(f"  [{i/10:.1f}-{(i+1)/10:.1f}]: {bar} {b}")


if __name__ == "__main__":
    main()

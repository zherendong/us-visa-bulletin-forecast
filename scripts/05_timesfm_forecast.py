"""Run TimesFM zero-shot forecasting and compare with trained models.

Usage:
    python scripts/05_timesfm_forecast.py
    python scripts/05_timesfm_forecast.py --cv          # run cross-validation
    python scripts/05_timesfm_forecast.py --predict --country China --eb EB3 --priority-date 2022-01-15
"""

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd

from gc_predict.features.engineer import get_target_columns, get_filing_target_columns
from gc_predict.model.timesfm_model import (
    TIMESFM_AVAILABLE,
    TimesFMConfig,
    forecast_all_targets,
    run_timesfm_cv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_forecast(df: pd.DataFrame, config: TimesFMConfig) -> dict[str, dict]:
    """Run TimesFM forecast on the full dataset."""
    target_cols = get_target_columns(df) + get_filing_target_columns(df)
    logger.info(f"Forecasting {len(target_cols)} targets with TimesFM...")
    return forecast_all_targets(df, target_cols, config)


def display_forecast(forecasts: dict[str, dict], df: pd.DataFrame):
    """Print a summary of forecasts."""
    last_date = df["bulletin_date"].max()
    print(f"\n{'=' * 70}")
    print(f"TIMESFM ZERO-SHOT FORECAST (from {last_date.strftime('%Y-%m-%d')})")
    print(f"{'=' * 70}")

    # Group by action vs filing
    action_targets = {k: v for k, v in forecasts.items() if not k.endswith("_filing_movement")}
    filing_targets = {k: v for k, v in forecasts.items() if k.endswith("_filing_movement")}

    for label, targets in [("FINAL ACTION DATE MOVEMENT", action_targets),
                           ("DATE FOR FILING MOVEMENT", filing_targets)]:
        if not targets:
            continue
        print(f"\n--- {label} (predicted days/month) ---")
        print(f"{'Target':<35} {'Month 1':>8} {'Month 2':>8} {'Month 3':>8} "
              f"{'Month 4':>8} {'Month 5':>8} {'Month 6':>8} {'Total':>8}")
        print("-" * 105)

        for name, fc in sorted(targets.items()):
            movement = fc["point_forecast"]
            total = movement.sum()
            row = f"{name:<35}"
            for m in movement[:6]:
                row += f" {m:>7.1f}"
            row += f" {total:>7.1f}"
            print(row)

    print()


def display_prediction(
    forecasts: dict[str, dict],
    country: str,
    eb: str,
    priority_date: pd.Timestamp,
    df: pd.DataFrame,
):
    """Display timeline prediction using TimesFM forecasts."""
    from gc_predict.data.fetch_visa_bulletin import load_visa_bulletin
    from gc_predict.predict.inference import get_last_known_dates

    country_key = country.replace(" ", "_")

    # Get last known dates
    visa_df = load_visa_bulletin(project_root / "data" / "raw" / "visa_bulletin")
    gov_path = project_root / "data" / "raw" / "visa_bulletin_gov" / "visa_bulletin_gov_full.parquet"
    gov_df = pd.read_parquet(gov_path) if gov_path.exists() else None
    last_known = get_last_known_dates(visa_df, gov_df)

    action_key = f"{eb}_{country_key}_movement"
    filing_key = f"{eb}_{country_key}_filing_movement"
    action_date_key = f"{eb}_{country_key}_action_date"
    filing_date_key = f"{eb}_{country_key}_filing_date"

    last_action = last_known.get(action_date_key)
    last_filing = last_known.get(filing_date_key)

    print(f"\n{'=' * 60}")
    print("TIMESFM GREEN CARD TIMELINE PREDICTION")
    print(f"{'=' * 60}")
    print(f"Country:       {country}")
    print(f"Category:      {eb}")
    print(f"Priority Date: {priority_date.strftime('%Y-%m-%d')}")
    print()

    if last_action:
        print(f"Last Known Final Action Date: {last_action.strftime('%Y-%m-%d')}")
    if last_filing:
        print(f"Last Known Filing Date:       {last_filing.strftime('%Y-%m-%d')}")
    print()

    for label, target_key, last_date in [
        ("FINAL ACTION DATE", action_key, last_action),
        ("DATE FOR FILING", filing_key, last_filing),
    ]:
        print(f"--- {label} ---")
        if target_key not in forecasts:
            print("  No forecast available for this combination.")
            print()
            continue

        if last_date is None:
            print("  No last known date available.")
            print()
            continue

        if priority_date <= last_date:
            print("  Status: CURRENT - Your priority date is already current!")
            print()
            continue

        fc = forecasts[target_key]
        gap_days = (priority_date - last_date).days
        movement = fc["point_forecast"]
        q_movement = fc["quantile_forecast"]

        cum_q10 = float(np.sum(q_movement[:, 0]))
        cum_q50 = float(np.sum(q_movement[:, 1]))
        cum_q90 = float(np.sum(q_movement[:, 2]))

        horizon = len(movement)
        monthly_q10 = cum_q10 / horizon
        monthly_q50 = cum_q50 / horizon
        monthly_q90 = cum_q90 / horizon

        print(f"  Gap to close: {gap_days} days")
        print(f"  Predicted monthly movement: {monthly_q50:.1f} days/month "
              f"(range: {monthly_q10:.1f} to {monthly_q90:.1f})")

        def months_to_period(m):
            if m == float("inf") or m > 300:
                return "25+ years"
            years = int(m) // 12
            remaining = int(m) % 12
            if years == 0:
                return f"{remaining} months"
            if remaining == 0:
                return f"{years} years"
            return f"{years} years {remaining} months"

        if monthly_q50 <= 0:
            print("  The model predicts no forward movement.")
        else:
            months_median = gap_days / monthly_q50
            months_opt = gap_days / monthly_q90 if monthly_q90 > 0 else float("inf")
            months_pes = gap_days / monthly_q10 if monthly_q10 > 0 else float("inf")
            print(f"  Estimated wait:")
            print(f"    Optimistic:  {months_to_period(months_opt)}")
            print(f"    Median:      {months_to_period(months_median)}")
            print(f"    Pessimistic: {months_to_period(months_pes)}")
        print()


def run_cross_validation(df: pd.DataFrame, config: TimesFMConfig):
    """Run TimesFM cross-validation and compare with existing models."""
    target_cols = get_target_columns(df) + get_filing_target_columns(df)

    timesfm_cv = run_timesfm_cv(df, target_cols, config)

    # Load existing model CV results for comparison if available
    print(f"\n{'=' * 70}")
    print("CROSS-VALIDATION COMPARISON")
    print(f"{'=' * 70}")

    models = [("TimesFM (zero-shot)", timesfm_cv["aggregated_metrics"])]

    for name, path in [("LSTM", "lstm_best.pt"), ("TFT", "tft_best.pt")]:
        ckpt_path = project_root / "models" / path
        if ckpt_path.exists():
            import torch
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "aggregated_metrics" in ckpt:
                models.append((name, ckpt["aggregated_metrics"]))

    print(f"\n{'Model':<25} {'MAE (days)':<18} {'Coverage (80%)':<18} {'Dir Accuracy':<18}")
    print("-" * 79)
    for name, agg in models:
        if agg:
            mae = f"{agg['mae_days']['mean']:.1f} +/- {agg['mae_days']['std']:.1f}"
            cov = f"{agg['coverage_80']['mean']:.0%} +/- {agg['coverage_80']['std']:.0%}"
            da = f"{agg['directional_accuracy']['mean']:.0%} +/- {agg['directional_accuracy']['std']:.0%}"
            print(f"{name:<25} {mae:<18} {cov:<18} {da:<18}")
        else:
            print(f"{name:<25} {'No results':<18}")

    print()
    return timesfm_cv


def main():
    parser = argparse.ArgumentParser(description="TimesFM forecasting for visa bulletin")
    parser.add_argument("--cv", action="store_true", help="Run cross-validation")
    parser.add_argument("--predict", action="store_true", help="Run prediction for a specific case")
    parser.add_argument("--country", default="China", help="Country of chargeability")
    parser.add_argument("--eb", default="EB3", help="EB category")
    parser.add_argument("--priority-date", default="2022-01-15", help="Priority date (YYYY-MM-DD)")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu"], help="Compute backend")
    parser.add_argument("--context-len", type=int, default=128, help="Context length (multiple of 32)")
    args = parser.parse_args()

    if not TIMESFM_AVAILABLE:
        logger.error(
            "timesfm is not installed.\n"
            "Install with: pip install timesfm[torch]\n"
            "Note: Requires Python >=3.10, <3.12."
        )
        sys.exit(1)

    # Load data
    merged_path = project_root / "data" / "processed" / "merged.parquet"
    if not merged_path.exists():
        logger.error(f"Merged data not found at {merged_path}. Run 02_process_data.py first.")
        sys.exit(1)

    df = pd.read_parquet(merged_path)
    logger.info(f"Loaded merged data: {df.shape}")
    logger.info(f"Date range: {df['bulletin_date'].min()} to {df['bulletin_date'].max()}")

    config = TimesFMConfig(
        backend=args.backend,
        context_len=args.context_len,
    )

    if args.cv:
        run_cross_validation(df, config)
    elif args.predict:
        priority_date = pd.Timestamp(args.priority_date)
        forecasts = run_forecast(df, config)
        display_prediction(forecasts, args.country, args.eb, priority_date, df)
    else:
        # Default: show forecast summary
        forecasts = run_forecast(df, config)
        display_forecast(forecasts, df)


if __name__ == "__main__":
    main()

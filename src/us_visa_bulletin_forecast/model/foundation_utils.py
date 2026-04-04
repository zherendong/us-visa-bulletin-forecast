"""Shared utilities for foundation model integrations.

Handles the common pattern: extract univariate series from the merged DataFrame,
convert level forecasts to movement, assemble into multi-target format, and
run expanding-window cross-validation.
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_series_for_targets(
    df: pd.DataFrame,
    target_cols: list[str],
    max_context: int | None = None,
    min_points: int = 12,
) -> list[tuple[str, np.ndarray]]:
    """Extract univariate level series for each target column.

    Maps movement columns to their corresponding _days columns
    and returns the historical values as numpy arrays.

    Returns:
        List of (target_name, series_array) tuples.
    """
    results = []
    for target_name in target_cols:
        if target_name.endswith("_filing_movement"):
            days_col = target_name.replace("_filing_movement", "_filing_days")
        elif target_name.endswith("_movement"):
            days_col = target_name.replace("_movement", "_action_days")
        else:
            logger.warning(f"Unexpected target format: {target_name}, skipping")
            continue

        if days_col not in df.columns:
            logger.warning(f"Column {days_col} not found for target {target_name}, skipping")
            continue

        series = df[days_col].ffill().fillna(0).values.astype(np.float64)
        if max_context is not None:
            series = series[-max_context:]
        if len(series) < min_points:
            logger.warning(f"Too few data points for {target_name} ({len(series)}), skipping")
            continue

        results.append((target_name, series))
    return results


def levels_to_movement(
    last_level: float, forecasted_levels: np.ndarray
) -> np.ndarray:
    """Convert level forecasts to month-over-month movement."""
    return np.diff(np.concatenate([[last_level], forecasted_levels]))


def assemble_multitarget(
    forecasts: dict[str, dict],
    target_cols: list[str],
    horizon: int = 6,
    num_quantiles: int = 3,
) -> np.ndarray:
    """Convert per-series forecasts to multi-target array format.

    Matches LSTM/TFT output shape: (1, horizon, num_targets, num_quantiles).
    """
    num_targets = len(target_cols)
    predictions = np.zeros((1, horizon, num_targets, num_quantiles))

    for i, target_name in enumerate(target_cols):
        if target_name in forecasts:
            fc = forecasts[target_name]
            h = min(horizon, len(fc["quantile_forecast"]))
            predictions[0, :h, i, :] = fc["quantile_forecast"][:h]

    return predictions


def run_foundation_model_cv(
    df: pd.DataFrame,
    target_cols: list[str],
    forecast_fn: Callable[[pd.DataFrame, list[str]], dict[str, dict]],
    model_name: str,
    horizon: int = 6,
    num_quantiles: int = 3,
    first_test_year: int = 2020,
    last_test_year: int = 2026,
) -> dict:
    """Run expanding-window cross-validation for any foundation model.

    Args:
        df: merged DataFrame sorted by bulletin_date.
        target_cols: list of target movement columns.
        forecast_fn: callable(train_df, target_cols) -> forecasts dict.
        model_name: display name for logging.
        horizon: forecast horizon in months.
        num_quantiles: number of quantile levels.
        first_test_year: first fiscal year to use as test.
        last_test_year: last fiscal year to use as test.

    Returns:
        dict with per-fold metrics and aggregated results.
    """
    from us_visa_bulletin_forecast.model.evaluate import compute_metrics

    df = df.copy().sort_values("bulletin_date").reset_index(drop=True)
    df[target_cols] = df[target_cols].ffill().fillna(0)

    # Assign fiscal years (FY starts Oct)
    df["fy"] = df["bulletin_date"].dt.year
    df.loc[df["bulletin_date"].dt.month >= 10, "fy"] += 1

    fold_results = []

    for test_fy in range(first_test_year, last_test_year + 1):
        test_mask = df["fy"] == test_fy
        train_mask = df["fy"] < test_fy

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_train < 24 or n_test < horizon:
            logger.info(f"Fold FY{test_fy}: skipping (train={n_train}, test={n_test})")
            continue

        logger.info(f"Fold FY{test_fy}: context={n_train} months, test={n_test} months")

        train_df = df[train_mask]
        forecasts = forecast_fn(train_df, target_cols)

        if not forecasts:
            logger.warning(f"Fold FY{test_fy}: no forecasts produced")
            continue

        predictions = assemble_multitarget(forecasts, target_cols, horizon, num_quantiles)

        test_df = df[test_mask].head(horizon)
        actuals = test_df[target_cols].values
        actual_horizon = min(horizon, len(actuals))
        actuals_shaped = actuals[:actual_horizon][np.newaxis, ...]

        predictions = predictions[:, :actual_horizon, :, :]
        fold_metrics = compute_metrics(predictions, actuals_shaped)

        fold_results.append({
            "test_fy": test_fy,
            "n_context": n_train,
            "n_test": n_test,
            "metrics": fold_metrics,
        })

        logger.info(
            f"  MAE={fold_metrics['mae_days']:.1f}d, "
            f"Coverage={fold_metrics['coverage_80']:.0%}, "
            f"DirAcc={fold_metrics['directional_accuracy']:.0%}"
        )

    # Aggregate
    agg = {}
    if fold_results:
        metric_keys = fold_results[0]["metrics"].keys()
        for key in metric_keys:
            values = [f["metrics"][key] for f in fold_results]
            agg[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

        logger.info("=" * 60)
        logger.info(f"{model_name} CV SUMMARY ({len(fold_results)} folds)")
        for key, stats in agg.items():
            logger.info(f"  {key}: {stats['mean']:.2f} +/- {stats['std']:.2f}")

    return {
        "fold_results": fold_results,
        "aggregated_metrics": agg,
        "model_name": model_name,
    }

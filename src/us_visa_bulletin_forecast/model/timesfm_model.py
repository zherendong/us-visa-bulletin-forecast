"""TimesFM integration for visa bulletin forecasting.

Uses Google's TimesFM foundation model as a zero-shot baseline.
TimesFM forecasts each EB x country time series independently (univariate),
then results are assembled into the same multi-target format used by LSTM/TFT.

Install: pip install timesfm[torch]
Note: TimesFM currently requires Python >=3.10, <3.12.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import timesfm

    TIMESFM_AVAILABLE = True
except ImportError:
    TIMESFM_AVAILABLE = False
    logger.info("timesfm not installed. Install with: pip install timesfm[torch]")


@dataclass
class TimesFMConfig:
    """Configuration for TimesFM forecasting."""

    horizon: int = 6  # months to forecast
    context_len: int = 128  # historical context (multiple of 32, max 2048 for v2.0)
    backend: str = "cpu"  # "cpu" or "gpu"
    per_core_batch_size: int = 32
    freq: int = 1  # 0=high (daily), 1=medium (weekly/monthly), 2=low (quarterly/yearly)
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)


def _load_timesfm_model(config: TimesFMConfig):
    """Load TimesFM model. Downloads checkpoint on first use (~2GB)."""
    if not TIMESFM_AVAILABLE:
        raise ImportError(
            "timesfm is not installed. Install with: pip install timesfm[torch]\n"
            "Note: Requires Python >=3.10, <3.12."
        )

    logger.info("Loading TimesFM model (google/timesfm-2.0-500m-pytorch)...")
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=config.backend,
            per_core_batch_size=config.per_core_batch_size,
            horizon_len=config.horizon,
            num_layers=50,
            context_len=config.context_len,
            use_positional_embedding=False,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        ),
    )
    logger.info("TimesFM model loaded.")
    return tfm


def forecast_all_targets(
    df: pd.DataFrame,
    target_cols: list[str],
    config: TimesFMConfig | None = None,
) -> dict[str, dict]:
    """Forecast all target time series using TimesFM.

    For each target column (e.g. EB3_China_movement), extracts the corresponding
    _days column as a univariate series, forecasts it, and converts back to movement.

    Args:
        df: merged DataFrame with bulletin_date and all feature/target columns, sorted by date.
        target_cols: list of movement target column names.
        config: TimesFM configuration.

    Returns:
        dict mapping target_name -> {
            "point_forecast": np.ndarray of shape (horizon,),
            "quantile_forecast": np.ndarray of shape (horizon, num_quantiles),
            "context_dates": pd.DatetimeIndex,
            "forecast_dates": pd.DatetimeIndex,
        }
    """
    if config is None:
        config = TimesFMConfig()

    tfm = _load_timesfm_model(config)

    df = df.copy().sort_values("bulletin_date").reset_index(drop=True)
    results = {}

    # Collect all series to forecast in a batch
    series_list = []
    series_names = []

    for target_name in target_cols:
        # Map movement column -> days column for the level series
        # e.g. "EB3_China_movement" -> "EB3_China_action_days"
        # e.g. "EB3_China_filing_movement" -> "EB3_China_filing_days"
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

        # Use up to context_len most recent points
        context = series[-config.context_len:]
        if len(context) < 12:
            logger.warning(f"Too few data points for {target_name} ({len(context)}), skipping")
            continue

        series_list.append(context)
        series_names.append(target_name)

    if not series_list:
        logger.error("No valid series to forecast")
        return results

    # Batch forecast all series at once
    freq_input = [config.freq] * len(series_list)
    logger.info(f"Forecasting {len(series_list)} series with TimesFM (horizon={config.horizon})...")

    point_forecasts, quantile_forecasts = tfm.forecast(
        series_list, freq=freq_input
    )
    # point_forecasts: (num_series, horizon)
    # quantile_forecasts: (num_series, horizon, num_quantiles) - fixed quantiles from model

    last_date = pd.Timestamp(df["bulletin_date"].iloc[-1])
    forecast_dates = pd.date_range(last_date, periods=config.horizon + 1, freq="MS")[1:]

    for i, target_name in enumerate(series_names):
        context = series_list[i]
        point_fc = point_forecasts[i, :config.horizon]

        # Convert level forecasts to movement (differences)
        last_level = context[-1]
        forecasted_levels = point_fc
        movement_forecast = np.diff(np.concatenate([[last_level], forecasted_levels]))

        # Extract quantiles and convert to movement
        if quantile_forecasts is not None and len(quantile_forecasts.shape) == 3:
            q_fc = quantile_forecasts[i, :config.horizon, :]
            # Find indices closest to our desired quantiles (0.1, 0.5, 0.9)
            # TimesFM v2.0 outputs quantiles at fixed percentiles
            n_q = q_fc.shape[1]
            q_movements = np.zeros((config.horizon, len(config.quantiles)))
            for qi, target_q in enumerate(config.quantiles):
                # Map to nearest available quantile index
                q_idx = min(int(target_q * n_q), n_q - 1)
                q_levels = q_fc[:, q_idx]
                q_movements[:, qi] = np.diff(np.concatenate([[last_level], q_levels]))
        else:
            # No quantile output - use point forecast with synthetic spread
            q_movements = np.stack([
                movement_forecast * 0.7,  # q10
                movement_forecast,         # q50
                movement_forecast * 1.3,  # q90
            ], axis=-1)

        results[target_name] = {
            "point_forecast": movement_forecast,
            "quantile_forecast": q_movements,
            "forecast_dates": forecast_dates,
            "last_level": last_level,
            "forecasted_levels": forecasted_levels,
        }

    logger.info(f"TimesFM forecasting complete for {len(results)} targets.")
    return results


def timesfm_predict_to_multitarget(
    forecasts: dict[str, dict],
    target_cols: list[str],
    horizon: int = 6,
    num_quantiles: int = 3,
) -> np.ndarray:
    """Convert TimesFM per-series forecasts to multi-target array format.

    Matches the output shape of LSTM/TFT: (1, horizon, num_targets, num_quantiles)
    so it can be evaluated with the same compute_metrics function.

    Args:
        forecasts: output of forecast_all_targets()
        target_cols: ordered list of all target column names
        horizon: forecast horizon
        num_quantiles: number of quantiles (3 for q10/q50/q90)

    Returns:
        np.ndarray of shape (1, horizon, num_targets, num_quantiles)
    """
    num_targets = len(target_cols)
    predictions = np.zeros((1, horizon, num_targets, num_quantiles))

    for i, target_name in enumerate(target_cols):
        if target_name in forecasts:
            fc = forecasts[target_name]
            h = min(horizon, len(fc["quantile_forecast"]))
            predictions[0, :h, i, :] = fc["quantile_forecast"][:h]

    return predictions


def run_timesfm_cv(
    df: pd.DataFrame,
    target_cols: list[str],
    config: TimesFMConfig | None = None,
    first_test_year: int = 2020,
    last_test_year: int = 2026,
) -> dict:
    """Run expanding-window cross-validation with TimesFM.

    Mirrors the CV strategy in cross_validate.py but uses TimesFM zero-shot
    instead of training a model.

    Args:
        df: merged DataFrame sorted by bulletin_date.
        target_cols: list of target movement columns.
        config: TimesFM configuration.
        first_test_year: first fiscal year to use as test.
        last_test_year: last fiscal year to use as test.

    Returns:
        dict with per-fold metrics and aggregated results.
    """
    from us_visa_bulletin_forecast.model.evaluate import compute_metrics

    if config is None:
        config = TimesFMConfig()

    df = df.copy().sort_values("bulletin_date").reset_index(drop=True)

    # Fill targets
    df[target_cols] = df[target_cols].ffill().fillna(0)

    # Assign fiscal years (FY starts Oct)
    df["fy"] = df["bulletin_date"].dt.year
    df.loc[df["bulletin_date"].dt.month >= 10, "fy"] += 1

    fold_results = []

    for test_fy in range(first_test_year, last_test_year + 1):
        # Train = everything before the test FY (TimesFM uses this as context)
        # Test = the test FY
        test_mask = df["fy"] == test_fy
        train_mask = df["fy"] < test_fy

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_train < 24 or n_test < config.horizon:
            logger.info(f"Fold FY{test_fy}: skipping (train={n_train}, test={n_test})")
            continue

        logger.info(f"Fold FY{test_fy}: context={n_train} months, test={n_test} months")

        # Use all data up to test period as context for TimesFM
        train_df = df[train_mask]

        # Forecast
        forecasts = forecast_all_targets(train_df, target_cols, config)

        if not forecasts:
            logger.warning(f"Fold FY{test_fy}: no forecasts produced")
            continue

        # Build predictions array
        predictions = timesfm_predict_to_multitarget(
            forecasts, target_cols, config.horizon, len(config.quantiles)
        )

        # Build actuals array - take first horizon months of test period
        test_df = df[test_mask].head(config.horizon)
        actuals = test_df[target_cols].values  # (test_months, num_targets)
        actual_horizon = min(config.horizon, len(actuals))
        actuals_shaped = actuals[:actual_horizon][np.newaxis, ...]  # (1, horizon, num_targets)

        # Trim predictions to match
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

    # Aggregate across folds
    agg = {}
    if fold_results:
        metric_keys = fold_results[0]["metrics"].keys()
        for key in metric_keys:
            values = [f["metrics"][key] for f in fold_results]
            agg[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

        logger.info("=" * 60)
        logger.info(f"TIMESFM CV SUMMARY ({len(fold_results)} folds)")
        for key, stats in agg.items():
            logger.info(f"  {key}: {stats['mean']:.2f} +/- {stats['std']:.2f}")

    return {
        "fold_results": fold_results,
        "aggregated_metrics": agg,
        "model_name": "TimesFM",
    }

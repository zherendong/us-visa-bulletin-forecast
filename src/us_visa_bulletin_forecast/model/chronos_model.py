"""Chronos-2 integration for visa bulletin forecasting.

Uses Amazon's Chronos-2 foundation model as a zero-shot baseline.
Frequency-agnostic, supports quantile forecasts natively.

Install: uv pip install chronos-forecasting
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from us_visa_bulletin_forecast.model.foundation_utils import (
    extract_series_for_targets,
    levels_to_movement,
    run_foundation_model_cv,
)

logger = logging.getLogger(__name__)

try:
    import torch
    from chronos import BaseChronosPipeline

    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    logger.info("chronos not installed. Install with: uv pip install chronos-forecasting")


@dataclass
class ChronosConfig:
    horizon: int = 6
    model_id: str = "amazon/chronos-2"
    device: str = "cpu"
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)


_pipeline_cache: dict[str, object] = {}


def _load_chronos(config: ChronosConfig):
    """Load Chronos pipeline. Caches across calls."""
    if not CHRONOS_AVAILABLE:
        raise ImportError("chronos not installed. Install with: uv pip install chronos-forecasting")

    cache_key = f"{config.model_id}:{config.device}"
    if cache_key not in _pipeline_cache:
        logger.info(f"Loading Chronos model ({config.model_id})...")
        _pipeline_cache[cache_key] = BaseChronosPipeline.from_pretrained(
            config.model_id, device_map=config.device
        )
        logger.info("Chronos model loaded.")
    return _pipeline_cache[cache_key]


def forecast_all_targets(
    df: pd.DataFrame,
    target_cols: list[str],
    config: ChronosConfig | None = None,
) -> dict[str, dict]:
    """Forecast all target time series using Chronos-2."""
    if config is None:
        config = ChronosConfig()

    pipeline = _load_chronos(config)
    df = df.copy().sort_values("bulletin_date").reset_index(drop=True)

    series_data = extract_series_for_targets(df, target_cols)
    if not series_data:
        logger.error("No valid series to forecast")
        return {}

    # Batch all series as a list of tensors
    names = [name for name, _ in series_data]
    contexts = [torch.tensor(s, dtype=torch.float32) for _, s in series_data]

    logger.info(f"Forecasting {len(contexts)} series with Chronos (horizon={config.horizon})...")

    quantile_results, mean_results = pipeline.predict_quantiles(
        inputs=contexts,
        prediction_length=config.horizon,
        quantile_levels=list(config.quantiles),
    )
    # Returns list of tensors, one per input series
    # Each quantile tensor: (1, horizon, num_quantiles), each mean tensor: (1, horizon)

    last_date = pd.Timestamp(df["bulletin_date"].iloc[-1])
    forecast_dates = pd.date_range(last_date, periods=config.horizon + 1, freq="MS")[1:]

    results = {}
    for i, (target_name, series) in enumerate(series_data):
        last_level = series[-1]
        mean_fc = mean_results[i].squeeze(0).numpy()  # (horizon,)
        q_fc = quantile_results[i].squeeze(0).numpy()  # (horizon, num_quantiles)

        movement = levels_to_movement(last_level, mean_fc)

        q_movements = np.zeros((config.horizon, len(config.quantiles)))
        for qi in range(len(config.quantiles)):
            q_movements[:, qi] = levels_to_movement(last_level, q_fc[:, qi])

        results[target_name] = {
            "point_forecast": movement,
            "quantile_forecast": q_movements,
            "forecast_dates": forecast_dates,
            "last_level": last_level,
            "forecasted_levels": mean_fc,
        }

    logger.info(f"Chronos forecasting complete for {len(results)} targets.")
    return results


def run_chronos_cv(
    df: pd.DataFrame,
    target_cols: list[str],
    config: ChronosConfig | None = None,
    first_test_year: int = 2020,
    last_test_year: int = 2026,
) -> dict:
    """Run expanding-window cross-validation with Chronos-2."""
    if config is None:
        config = ChronosConfig()

    def forecast_fn(train_df, cols):
        return forecast_all_targets(train_df, cols, config)

    return run_foundation_model_cv(
        df, target_cols, forecast_fn, "Chronos-2",
        horizon=config.horizon, num_quantiles=len(config.quantiles),
        first_test_year=first_test_year, last_test_year=last_test_year,
    )

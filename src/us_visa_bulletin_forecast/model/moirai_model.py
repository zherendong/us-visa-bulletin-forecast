"""Moirai 2.0 integration for visa bulletin forecasting.

Uses Salesforce's Moirai 2.0 foundation model as a zero-shot baseline.
Frequency-aware, supports quantile forecasts (9 fixed quantile levels).

Install: uv pip install uni2ts
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
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    MOIRAI_AVAILABLE = True
except ImportError:
    MOIRAI_AVAILABLE = False
    logger.info("uni2ts not installed. Install with: uv pip install uni2ts")

# Moirai 2.0 outputs 9 fixed quantile levels
MOIRAI_QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class MoiraiConfig:
    horizon: int = 6
    context_length: int = 200  # up to full history
    model_id: str = "Salesforce/moirai-2.0-R-small"
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)


def _get_quantile_indices(target_quantiles: tuple[float, ...]) -> list[int]:
    """Map desired quantile levels to Moirai's fixed 9-quantile output indices."""
    indices = []
    for q in target_quantiles:
        idx = min(range(len(MOIRAI_QUANTILE_LEVELS)),
                  key=lambda i: abs(MOIRAI_QUANTILE_LEVELS[i] - q))
        indices.append(idx)
    return indices


def _create_moirai_model(config: MoiraiConfig) -> "Moirai2Forecast":
    """Create Moirai 2.0 forecast model."""
    if not MOIRAI_AVAILABLE:
        raise ImportError("uni2ts not installed. Install with: uv pip install uni2ts")

    logger.info(f"Loading Moirai model ({config.model_id})...")
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(config.model_id),
        prediction_length=config.horizon,
        context_length=config.context_length,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    logger.info("Moirai model loaded.")
    return model


def forecast_all_targets(
    df: pd.DataFrame,
    target_cols: list[str],
    config: MoiraiConfig | None = None,
) -> dict[str, dict]:
    """Forecast all target time series using Moirai 2.0."""
    if config is None:
        config = MoiraiConfig()

    model = _create_moirai_model(config)
    df = df.copy().sort_values("bulletin_date").reset_index(drop=True)

    series_data = extract_series_for_targets(df, target_cols)
    if not series_data:
        logger.error("No valid series to forecast")
        return {}

    # Moirai predict() takes a list of numpy arrays
    names = [name for name, _ in series_data]
    arrays = [s for _, s in series_data]

    logger.info(f"Forecasting {len(arrays)} series with Moirai (horizon={config.horizon})...")

    # predict() returns (batch, num_quantiles, prediction_length)
    predictions = model.predict(past_target=arrays)

    q_indices = _get_quantile_indices(config.quantiles)

    last_date = pd.Timestamp(df["bulletin_date"].iloc[-1])
    forecast_dates = pd.date_range(last_date, periods=config.horizon + 1, freq="MS")[1:]

    results = {}
    for i, (target_name, series) in enumerate(series_data):
        last_level = series[-1]

        # Median (index 4 = 0.5 quantile) as point forecast
        median_idx = 4  # MOIRAI_QUANTILE_LEVELS[4] = 0.5
        point_levels = predictions[i, median_idx, :]
        movement = levels_to_movement(last_level, point_levels)

        q_movements = np.zeros((config.horizon, len(config.quantiles)))
        for qi, q_idx in enumerate(q_indices):
            q_levels = predictions[i, q_idx, :]
            q_movements[:, qi] = levels_to_movement(last_level, q_levels)

        results[target_name] = {
            "point_forecast": movement,
            "quantile_forecast": q_movements,
            "forecast_dates": forecast_dates,
            "last_level": last_level,
            "forecasted_levels": point_levels,
        }

    logger.info(f"Moirai forecasting complete for {len(results)} targets.")
    return results


def run_moirai_cv(
    df: pd.DataFrame,
    target_cols: list[str],
    config: MoiraiConfig | None = None,
    first_test_year: int = 2020,
    last_test_year: int = 2026,
) -> dict:
    """Run expanding-window cross-validation with Moirai 2.0."""
    if config is None:
        config = MoiraiConfig()

    def forecast_fn(train_df, cols):
        return forecast_all_targets(train_df, cols, config)

    return run_foundation_model_cv(
        df, target_cols, forecast_fn, "Moirai-2.0",
        horizon=config.horizon, num_quantiles=len(config.quantiles),
        first_test_year=first_test_year, last_test_year=last_test_year,
    )

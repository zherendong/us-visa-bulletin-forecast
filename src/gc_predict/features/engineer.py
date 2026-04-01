"""Feature engineering pipeline.

Transforms the merged dataset into model-ready features.
Multi-target: predicts movement for ALL EB×country combos simultaneously.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

COUNTRIES = ["China", "India", "Mexico", "Philippines", "Rest_of_World"]
EB_LEVELS = ["EB1", "EB2", "EB3", "EB4"]


@dataclass
class FeatureConfig:
    lookback_window: int = 24
    forecast_horizon: int = 6
    train_end: str = "2025-03-31"  # train through Mar 2025
    val_end: str = "2026-04-30"    # validate on Apr 2025 - Apr 2026 (last ~12 months)


def get_target_columns(df: pd.DataFrame) -> list[str]:
    """Get all movement columns as targets."""
    return [c for c in df.columns if c.endswith("_movement") and not c.endswith("_filing_movement")]


def get_filing_target_columns(df: pd.DataFrame) -> list[str]:
    """Get all filing movement columns as targets."""
    return [c for c in df.columns if c.endswith("_filing_movement")]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get input feature columns (action_days, filing_days, calendar).

    Excludes movement columns (those are targets) and metadata.
    """
    exclude = {"bulletin_date", "month"}
    features = []
    for c in df.columns:
        if c in exclude:
            continue
        if c.endswith("_movement") or c.endswith("_filing_movement"):
            continue
        if df[c].dtype in [np.float64, np.int64, float, int]:
            features.append(c)
    return features


def prepare_features(
    df: pd.DataFrame, config: FeatureConfig | None = None,
) -> dict:
    """Transform merged dataset into train/val/test arrays.

    Multi-target: y contains movement for all EB×country combos.

    Returns dict with X/y arrays, feature/target names, scaler, dates.
    """
    if config is None:
        config = FeatureConfig()

    df = df.copy().sort_values("bulletin_date").reset_index(drop=True)

    feature_cols = get_feature_columns(df)
    target_cols = get_target_columns(df)
    filing_target_cols = get_filing_target_columns(df)

    logger.info(f"Features: {len(feature_cols)}, Targets: {len(target_cols)}, "
                f"Filing targets: {len(filing_target_cols)}")

    # Fill NaN
    df[feature_cols] = df[feature_cols].ffill().fillna(0)
    df[target_cols] = df[target_cols].ffill().fillna(0)
    if filing_target_cols:
        df[filing_target_cols] = df[filing_target_cols].ffill().fillna(0)

    # Combine all targets
    all_target_cols = target_cols + filing_target_cols

    # Split by time
    train_mask = df["bulletin_date"] <= config.train_end
    val_mask = (df["bulletin_date"] > config.train_end) & (df["bulletin_date"] <= config.val_end)
    test_mask = df["bulletin_date"] > config.val_end

    # Fit scaler on training features
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, feature_cols].values)

    all_features = scaler.transform(df[feature_cols].values)
    all_targets = df[all_target_cols].values  # (N, num_targets)
    all_dates = df["bulletin_date"].values

    def create_sequences(features, targets, dates, start_idx, end_idx):
        X, y, d = [], [], []
        for i in range(start_idx + config.lookback_window,
                       min(end_idx, len(features) - config.forecast_horizon + 1)):
            X.append(features[i - config.lookback_window:i])
            # Target: next forecast_horizon months × all targets
            y.append(targets[i:i + config.forecast_horizon])
            d.append(dates[i])
        if not X:
            return np.array([]), np.array([]), np.array([])
        return np.array(X), np.array(y), np.array(d)

    train_indices = df.index[train_mask].tolist()
    val_indices = df.index[val_mask].tolist()
    test_indices = df.index[test_mask].tolist()

    X_train, y_train, d_train = create_sequences(
        all_features, all_targets, all_dates,
        0, train_indices[-1] + 1 if train_indices else 0
    )

    val_start = val_indices[0] if val_indices else len(df)
    val_end_idx = val_indices[-1] + 1 if val_indices else len(df)
    X_val, y_val, d_val = create_sequences(
        all_features, all_targets, all_dates,
        val_start - config.lookback_window, val_end_idx
    )

    test_start = test_indices[0] if test_indices else len(df)
    test_end_idx = test_indices[-1] + 1 if test_indices else len(df)
    X_test, y_test, d_test = create_sequences(
        all_features, all_targets, all_dates,
        test_start - config.lookback_window, test_end_idx
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"X shape: (*, {config.lookback_window}, {len(feature_cols)}), "
                f"y shape: (*, {config.forecast_horizon}, {len(all_target_cols)})")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_names": feature_cols,
        "target_names": all_target_cols,
        "action_target_names": target_cols,
        "filing_target_names": filing_target_cols,
        "scaler": scaler,
        "dates_train": d_train,
        "dates_val": d_val,
        "dates_test": d_test,
        "config": config,
    }

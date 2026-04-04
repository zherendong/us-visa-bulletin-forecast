"""Time series cross-validation with expanding window.

Each fold trains on all data up to year T, validates on T+1, tests on T+2.
The final model is trained on the most recent data for inference.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from us_visa_bulletin_forecast.features.engineer import get_feature_columns, get_target_columns, get_filing_target_columns
from us_visa_bulletin_forecast.model.architecture import LSTMBaseline, TemporalFusionTransformer, QuantileLoss
from us_visa_bulletin_forecast.model.dataset import VisaBulletinDataset, create_dataloaders
from us_visa_bulletin_forecast.model.evaluate import compute_metrics, evaluate_per_target
from us_visa_bulletin_forecast.model.train import train_model

logger = logging.getLogger(__name__)


@dataclass
class CVConfig:
    lookback_window: int = 24
    forecast_horizon: int = 6
    first_test_year: int = 2020  # first FY used as test
    last_test_year: int = 2026   # last FY used as test (partial ok)
    val_months: int = 12         # months for validation in each fold


def _create_sequences(features, targets, dates, start_idx, end_idx, lookback, horizon):
    """Create sliding window sequences between start_idx and end_idx."""
    X, y, d = [], [], []
    for i in range(max(start_idx, lookback), min(end_idx, len(features) - horizon + 1)):
        X.append(features[i - lookback:i])
        y.append(targets[i:i + horizon])
        d.append(dates[i])
    if not X:
        return np.array([]), np.array([]), np.array([])
    return np.array(X), np.array(y), np.array(d)


def run_cv(
    df: pd.DataFrame,
    model_type: str = "tft",
    config: CVConfig | None = None,
    checkpoint_dir: Path | None = None,
    **train_kwargs,
) -> dict:
    """Run time series cross-validation.

    Returns:
        dict with per-fold metrics, aggregated metrics, and final model
    """
    if config is None:
        config = CVConfig()

    df = df.copy().sort_values("bulletin_date").reset_index(drop=True)

    feature_cols = get_feature_columns(df)
    target_cols = get_target_columns(df)
    filing_cols = get_filing_target_columns(df)
    all_target_cols = target_cols + filing_cols

    df[feature_cols] = df[feature_cols].ffill().fillna(0)
    df[all_target_cols] = df[all_target_cols].ffill().fillna(0)

    all_features_raw = df[feature_cols].values
    all_targets = df[all_target_cols].values
    all_dates = df["bulletin_date"].values

    # Determine fiscal year for each row (FY starts Oct)
    df["fy"] = df["bulletin_date"].dt.year
    df.loc[df["bulletin_date"].dt.month >= 10, "fy"] += 1
    fys = df["fy"].values

    fold_results = []
    best_model = None
    best_scaler = None

    for test_fy in range(config.first_test_year, config.last_test_year + 1):
        val_fy = test_fy - 1
        train_end_fy = val_fy - 1

        train_mask = fys <= train_end_fy
        val_mask = fys == val_fy
        test_mask = fys == test_fy

        n_train = train_mask.sum()
        n_val = val_mask.sum()
        n_test = test_mask.sum()

        if n_train < config.lookback_window + config.forecast_horizon:
            logger.info(f"Fold FY{test_fy}: skipping, not enough training data ({n_train} months)")
            continue
        if n_val < config.forecast_horizon:
            logger.info(f"Fold FY{test_fy}: skipping, not enough val data ({n_val} months)")
            continue
        if n_test < config.forecast_horizon:
            logger.info(f"Fold FY{test_fy}: skipping, not enough test data ({n_test} months)")
            continue

        logger.info(f"Fold FY{test_fy}: train={n_train}, val={n_val}, test={n_test}")

        # Fit scaler on training data
        scaler = StandardScaler()
        scaler.fit(all_features_raw[train_mask])
        all_features = scaler.transform(all_features_raw)

        # Get index boundaries
        train_end_idx = np.where(train_mask)[0][-1] + 1
        val_indices = np.where(val_mask)[0]
        val_start_idx = val_indices[0]
        val_end_idx = val_indices[-1] + 1
        test_indices = np.where(test_mask)[0]
        test_start_idx = test_indices[0]
        test_end_idx = test_indices[-1] + 1

        # Create sequences
        X_train, y_train, _ = _create_sequences(
            all_features, all_targets, all_dates,
            0, train_end_idx, config.lookback_window, config.forecast_horizon
        )
        X_val, y_val, _ = _create_sequences(
            all_features, all_targets, all_dates,
            val_start_idx - config.lookback_window, val_end_idx,
            config.lookback_window, config.forecast_horizon
        )
        X_test, y_test, d_test = _create_sequences(
            all_features, all_targets, all_dates,
            test_start_idx - config.lookback_window, test_end_idx,
            config.lookback_window, config.forecast_horizon
        )

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            logger.info(f"Fold FY{test_fy}: skipping, empty sequences")
            continue

        # Create datasets and loaders
        train_ds = VisaBulletinDataset(X_train, y_train, augment=True)
        val_ds = VisaBulletinDataset(X_val, y_val)
        test_ds = VisaBulletinDataset(X_test, y_test)

        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        # Create model
        num_features = len(feature_cols)
        num_targets = len(all_target_cols)

        if model_type == "tft":
            model = TemporalFusionTransformer(
                num_features=num_features, num_targets=num_targets,
                forecast_horizon=config.forecast_horizon,
            )
        else:
            model = LSTMBaseline(
                num_features=num_features, num_targets=num_targets,
                forecast_horizon=config.forecast_horizon,
            )

        # Train
        result = train_model(model, train_loader, val_loader, **train_kwargs)

        # Evaluate on test fold
        model = result["model"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        all_preds, all_actuals = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch)
                all_preds.append(preds.cpu().numpy())
                all_actuals.append(y_batch.numpy())

        predictions = np.concatenate(all_preds, axis=0)
        actuals = np.concatenate(all_actuals, axis=0)

        fold_metrics = compute_metrics(predictions, actuals)
        per_target = evaluate_per_target(predictions, actuals, all_target_cols)

        # Extract key targets
        key_targets = {}
        for name in all_target_cols:
            if "EB3_China" in name or "EB2_India" in name or "EB2_China" in name:
                key_targets[name] = per_target[name]

        fold_results.append({
            "test_fy": test_fy,
            "n_train": n_train,
            "metrics": fold_metrics,
            "key_targets": key_targets,
            "val_loss": result["best_val_loss"],
        })

        logger.info(f"  Test MAE={fold_metrics['mae_days']:.1f}d, "
                    f"Coverage={fold_metrics['coverage_80']:.0%}, "
                    f"DirAcc={fold_metrics['directional_accuracy']:.0%}")
        for name, m in key_targets.items():
            logger.info(f"    {name}: MAE={m['mae_days']:.1f}d, Cov={m['coverage_80']:.0%}")

        # Keep the latest fold's model for inference
        best_model = model
        best_scaler = scaler

    # Aggregate metrics across folds
    if fold_results:
        agg = {}
        metric_keys = fold_results[0]["metrics"].keys()
        for key in metric_keys:
            values = [f["metrics"][key] for f in fold_results]
            agg[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

        logger.info("=" * 60)
        logger.info(f"CROSS-VALIDATION SUMMARY ({len(fold_results)} folds)")
        for key, stats in agg.items():
            logger.info(f"  {key}: {stats['mean']:.2f} ± {stats['std']:.2f}")
    else:
        agg = {}

    # Save final model
    if checkpoint_dir and best_model:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        name = "tft" if model_type == "tft" else "lstm"
        path = checkpoint_dir / f"{name}_best.pt"
        torch.save({
            "model_state_dict": {k: v.cpu().clone() for k, v in best_model.state_dict().items()},
            "cv_results": fold_results,
            "aggregated_metrics": agg,
        }, path)
        logger.info(f"Saved final model to {path}")

    return {
        "fold_results": fold_results,
        "aggregated_metrics": agg,
        "model": best_model,
        "scaler": best_scaler,
        "feature_names": feature_cols,
        "target_names": all_target_cols,
    }

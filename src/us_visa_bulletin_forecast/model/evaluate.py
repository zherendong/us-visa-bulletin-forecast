"""Evaluation metrics and baselines for multi-target prediction."""

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: np.ndarray, actuals: np.ndarray, target_idx: int | None = None
) -> dict:
    """Compute metrics. If target_idx given, evaluate only that target.

    Args:
        predictions: (N, horizon, num_targets, 3) quantile predictions
        actuals: (N, horizon, num_targets)
        target_idx: specific target to evaluate, or None for average across all
    """
    if target_idx is not None:
        predictions = predictions[:, :, target_idx:target_idx+1, :]
        actuals = actuals[:, :, target_idx:target_idx+1]

    median_pred = predictions[..., 1]
    mae = np.nanmean(np.abs(median_pred - actuals))
    rmse = np.sqrt(np.nanmean((median_pred - actuals) ** 2))

    q10, q90 = predictions[..., 0], predictions[..., 2]
    coverage = np.nanmean((actuals >= q10) & (actuals <= q90))

    pred_dir = np.sign(median_pred)
    actual_dir = np.sign(actuals)
    directional_acc = np.nanmean(pred_dir == actual_dir)

    interval_width = np.nanmean(q90 - q10)

    return {
        "mae_days": float(mae),
        "rmse_days": float(rmse),
        "coverage_80": float(coverage),
        "directional_accuracy": float(directional_acc),
        "mean_interval_width": float(interval_width),
    }


def evaluate_model(
    model: nn.Module, test_loader, device: str | None = None
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run model on test set. Returns (metrics, predictions, actuals)."""
    if device is None:
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

    metrics = compute_metrics(predictions, actuals)
    return metrics, predictions, actuals


def evaluate_per_target(
    predictions: np.ndarray, actuals: np.ndarray, target_names: list[str]
) -> dict[str, dict]:
    """Evaluate each target separately."""
    results = {}
    for i, name in enumerate(target_names):
        results[name] = compute_metrics(predictions, actuals, target_idx=i)
    return results


def baseline_naive(y_test: np.ndarray) -> np.ndarray:
    """Predict zero movement for all targets."""
    return np.zeros((*y_test.shape, 3))

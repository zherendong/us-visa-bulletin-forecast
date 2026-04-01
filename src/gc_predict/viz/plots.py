"""Visualization functions for green card timeline predictions."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")
FIGSIZE = (14, 6)


def plot_historical_timeline(
    df: pd.DataFrame,
    output_path: Path | None = None,
):
    """Plot historical EB-3 China Final Action Date timeline."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    eb3_china = df[
        (df["eb_level"].str.contains("3|EB3", case=False))
        & (df["country"] == "China")
        & (~df["is_current"])
        & (df["final_action_date"].notna())
    ].sort_values("bulletin_date")

    ax.plot(
        eb3_china["bulletin_date"],
        eb3_china["final_action_date"],
        "b-", linewidth=1.5, label="EB-3 China Final Action Date",
    )
    ax.set_xlabel("Visa Bulletin Month")
    ax.set_ylabel("Priority Date (Final Action Date)")
    ax.set_title("EB-3 China: Historical Final Action Date Movement")
    ax.legend()
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        logger.info(f"Saved plot to {output_path}")

    return fig


def plot_forecast(
    historical_dates: np.ndarray,
    historical_values: np.ndarray,
    forecast_df: pd.DataFrame,
    output_path: Path | None = None,
):
    """Plot forecast fan chart with historical actuals and future predictions."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Historical
    ax.plot(historical_dates, historical_values, "b-", linewidth=1.5, label="Historical")

    # Forecast
    forecast_x = pd.date_range(
        start=historical_dates[-1], periods=len(forecast_df) + 1, freq="MS"
    )[1:]

    ax.plot(forecast_x, forecast_df["predicted_date_median"], "r-", linewidth=2, label="Forecast (median)")
    ax.fill_between(
        forecast_x,
        forecast_df["predicted_date_low"],
        forecast_df["predicted_date_high"],
        alpha=0.3, color="red", label="80% confidence interval",
    )

    ax.set_xlabel("Bulletin Month")
    ax.set_ylabel("Final Action Date")
    ax.set_title("EB-3 China: Forecast with Confidence Interval")
    ax.legend()
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig


def plot_predictions_vs_actuals(
    predictions: np.ndarray,
    actuals: np.ndarray,
    dates: np.ndarray,
    horizon_idx: int = 0,
    output_path: Path | None = None,
):
    """Plot predicted vs actual values on the test set.

    Args:
        predictions: (N, horizon, 3) quantile predictions
        actuals: (N, horizon) actual values
        dates: (N,) dates for each test sample
        horizon_idx: which forecast horizon to plot (0 = 1-month ahead)
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    median_pred = predictions[:, horizon_idx, 1]
    q10 = predictions[:, horizon_idx, 0]
    q90 = predictions[:, horizon_idx, 2]
    actual = actuals[:, horizon_idx]

    ax.plot(dates, actual, "b-o", markersize=3, label="Actual", linewidth=1.5)
    ax.plot(dates, median_pred, "r-o", markersize=3, label="Predicted (median)", linewidth=1.5)
    ax.fill_between(dates, q10, q90, alpha=0.3, color="red", label="80% CI")

    ax.set_xlabel("Date")
    ax.set_ylabel("Movement (days)")
    ax.set_title(f"EB-3 China: {horizon_idx+1}-Month Ahead Prediction vs Actual")
    ax.legend()
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig


def plot_variable_importance(
    importance: dict[str, float],
    top_n: int = 20,
    output_path: Path | None = None,
):
    """Plot TFT variable importance weights."""
    fig, ax = plt.subplots(figsize=(10, 8))

    items = list(importance.items())[:top_n]
    names = [x[0] for x in items]
    weights = [x[1] for x in items]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, weights, align="center", color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Weight")
    ax.set_title("TFT Variable Importance: Top Features for EB-3 China Prediction")
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig


def plot_training_history(
    history: dict,
    output_path: Path | None = None,
):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Quantile Loss")
    ax.set_title("Training History")
    ax.legend()
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig


def plot_cross_category_correlation(
    df: pd.DataFrame,
    output_path: Path | None = None,
):
    """Plot correlation heatmap of movement across EB categories and countries."""
    movement_cols = [c for c in df.columns if c.endswith("_movement")]
    if not movement_cols:
        logger.warning("No movement columns found for correlation plot")
        return None

    corr = df[movement_cols].corr()

    # Clean up column names for display
    labels = [c.replace("_movement", "").replace("_", " ") for c in movement_cols]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Cross-Category Movement Correlation")
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig

"""Inference pipeline: predict when a priority date becomes current.

Given (country, EB category, priority_date), estimates the time range
for Final Action Date and Date for Filing to reach that priority date.
"""

import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from us_visa_bulletin_forecast.model.architecture import TemporalFusionTransformer, LSTMBaseline

logger = logging.getLogger(__name__)

EPOCH = pd.Timestamp("2000-01-01")


def load_model(
    checkpoint_path: Path, num_features: int, num_targets: int,
    forecast_horizon: int = 6, model_type: str = "tft",
) -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if model_type == "tft":
        model = TemporalFusionTransformer(
            num_features=num_features, num_targets=num_targets,
            forecast_horizon=forecast_horizon,
        )
    else:
        model = LSTMBaseline(
            num_features=num_features, num_targets=num_targets,
            forecast_horizon=forecast_horizon,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _find_target_index(target_names: list[str], country: str, eb: str, table_type: str = "action") -> int | None:
    """Find the index of a specific EB×country target in the target list.

    Args:
        target_names: list like ["EB1_China_movement", "EB2_India_movement", ...]
        country: "China", "India", etc.
        eb: "EB1", "EB2", "EB3", "EB4"
        table_type: "action" for Final Action, "filing" for Dates for Filing
    """
    country_key = country.replace(" ", "_")
    if table_type == "action":
        target = f"{eb}_{country_key}_movement"
    else:
        target = f"{eb}_{country_key}_filing_movement"

    try:
        return target_names.index(target)
    except ValueError:
        logger.warning(f"Target '{target}' not found in {target_names}")
        return None


def predict_timeline(
    model: torch.nn.Module,
    recent_window: np.ndarray,
    scaler,
    target_names: list[str],
    country: str,
    eb_category: str,
    priority_date: pd.Timestamp,
    last_known_dates: dict[str, pd.Timestamp],
    max_years: int = 15,
    device: str | None = None,
) -> dict:
    """Predict when a priority date becomes current.

    Args:
        model: trained multi-target model
        recent_window: (lookback, raw_features) array
        scaler: fitted StandardScaler
        target_names: list of target column names
        country: e.g. "China", "India"
        eb_category: e.g. "EB2", "EB3"
        priority_date: user's priority date
        last_known_dates: dict with keys like "EB3_China_action_date", "EB3_China_filing_date"
            containing the most recent known Final Action Date and Filing Date

    Returns:
        dict with estimated time ranges for both Final Action and Filing dates
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    country_key = country.replace(" ", "_")

    # Find target indices
    action_idx = _find_target_index(target_names, country, eb_category, "action")
    filing_idx = _find_target_index(target_names, country, eb_category, "filing")

    action_key = f"{eb_category}_{country_key}_action_date"
    filing_key = f"{eb_category}_{country_key}_filing_date"

    last_action = last_known_dates.get(action_key)
    last_filing = last_known_dates.get(filing_key)

    result = {
        "country": country,
        "eb_category": eb_category,
        "priority_date": priority_date,
        "last_known_action_date": last_action,
        "last_known_filing_date": last_filing,
    }

    # Run model prediction
    window = np.nan_to_num(recent_window.copy(), nan=0.0)
    scaled = scaler.transform(window)
    X = torch.FloatTensor(scaled).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(X).cpu().numpy().squeeze(0)
        # predictions shape: (horizon, num_targets, 3)

    # Project forward for Final Action Date
    if action_idx is not None and last_action is not None:
        result["final_action"] = _project_to_current(
            predictions, action_idx, priority_date, last_action
        )
    else:
        result["final_action"] = {"status": "unavailable", "reason": "No data for this combo"}

    # Project forward for Filing Date
    if filing_idx is not None and last_filing is not None:
        result["filing"] = _project_to_current(
            predictions, filing_idx, priority_date, last_filing
        )
    else:
        result["filing"] = {"status": "unavailable", "reason": "No filing date data"}

    return result


def _project_to_current(
    predictions: np.ndarray,
    target_idx: int,
    priority_date: pd.Timestamp,
    last_known_date: pd.Timestamp,
) -> dict:
    """Project when the priority date becomes current based on predicted movement.

    Args:
        predictions: (horizon, num_targets, 3) - q10, q50, q90 for each target
        target_idx: which target to look at
        priority_date: when the user's case was filed
        last_known_date: most recent date from the visa bulletin

    Returns:
        dict with status and estimated time range
    """
    if priority_date <= last_known_date:
        return {
            "status": "current",
            "message": "Your priority date is already current!",
        }

    gap_days = (priority_date - last_known_date).days

    # Sum predicted movements over the forecast horizon
    horizon = predictions.shape[0]
    cum_q10 = float(np.sum(predictions[:, target_idx, 0]))
    cum_q50 = float(np.sum(predictions[:, target_idx, 1]))
    cum_q90 = float(np.sum(predictions[:, target_idx, 2]))

    # Average monthly movement rates
    monthly_q10 = cum_q10 / horizon if horizon > 0 else 0
    monthly_q50 = cum_q50 / horizon if horizon > 0 else 0
    monthly_q90 = cum_q90 / horizon if horizon > 0 else 0

    # Estimate months to close the gap
    if monthly_q50 <= 0:
        return {
            "status": "stalled_or_retrogressing",
            "message": f"The model predicts no forward movement (median: {monthly_q50:.1f} days/month). "
                       f"Gap to your priority date: {gap_days} days.",
            "gap_days": gap_days,
            "predicted_monthly_movement_median": monthly_q50,
        }

    months_median = gap_days / monthly_q50
    months_optimistic = gap_days / monthly_q90 if monthly_q90 > 0 else float("inf")
    months_pessimistic = gap_days / monthly_q10 if monthly_q10 > 0 else float("inf")

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

    return {
        "status": "projected",
        "gap_days": gap_days,
        "predicted_monthly_movement": {
            "pessimistic": round(monthly_q10, 1),
            "median": round(monthly_q50, 1),
            "optimistic": round(monthly_q90, 1),
        },
        "estimated_wait": {
            "optimistic": months_to_period(months_optimistic),
            "median": months_to_period(months_median),
            "pessimistic": months_to_period(months_pessimistic),
        },
        "estimated_months": {
            "optimistic": round(months_optimistic, 1),
            "median": round(months_median, 1),
            "pessimistic": round(months_pessimistic, 1),
        },
    }


def get_last_known_dates(visa_df: pd.DataFrame, gov_df: pd.DataFrame | None = None) -> dict[str, pd.Timestamp]:
    """Extract the most recent Final Action and Filing dates for all EB×country combos.

    Returns dict like {"EB3_China_action_date": Timestamp, "EB3_China_filing_date": Timestamp, ...}
    """
    result = {}

    # Final Action Dates from CSV data
    visa_df = visa_df.copy()
    visa_df["eb_level"] = visa_df["eb_level"].apply(
        lambda x: f"EB{x}" if isinstance(x, (int, float)) else str(x).replace("-", "").replace(" ", "").upper()
    )

    for (eb, country), group in visa_df.groupby(["eb_level", "country"]):
        valid = group[group["final_action_date"].notna() & ~group["is_current"]].sort_values("bulletin_date")
        if not valid.empty:
            key = f"{eb}_{country.replace(' ', '_')}_action_date"
            result[key] = valid.iloc[-1]["final_action_date"]

    # Supplement/override with gov-scraped data (may have more recent months)
    if gov_df is not None and not gov_df.empty:
        gov_df = gov_df.copy()
        gov_df["bulletin_date"] = pd.to_datetime(gov_df["bulletin_date"])
        gov_df["eb_level"] = gov_df["eb_level"].str.replace("-", "").str.upper()

        # Final Action Dates from gov (may be newer than CSV)
        action = gov_df[gov_df["table_type"] == "final_action"]
        for (eb, country), group in action.groupby(["eb_level", "country"]):
            valid = group[~group["is_current"] & ~group["is_unavailable"]].sort_values("bulletin_date")
            if not valid.empty:
                last_row = valid.iloc[-1]
                try:
                    date = pd.to_datetime(last_row["date_value"])
                    key = f"{eb}_{country.replace(' ', '_')}_action_date"
                    # Only update if this is more recent (later bulletin_date)
                    if key not in result or last_row["bulletin_date"] > visa_df["bulletin_date"].max():
                        result[key] = date
                except Exception:
                    pass

        # Filing Dates from gov
        filing = gov_df[gov_df["table_type"] == "filing"]
        for (eb, country), group in filing.groupby(["eb_level", "country"]):
            valid = group[~group["is_current"] & ~group["is_unavailable"]].sort_values("bulletin_date")
            if not valid.empty:
                last_row = valid.iloc[-1]
                try:
                    date = pd.to_datetime(last_row["date_value"])
                    key = f"{eb}_{country.replace(' ', '_')}_filing_date"
                    result[key] = date
                except Exception:
                    pass

    return result

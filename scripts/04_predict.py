"""Generate predictions for a given user profile.

Usage:
    python scripts/04_predict.py --country China --eb EB3 --priority-date 2020-06-15
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
from gc_predict.features.engineer import FeatureConfig, prepare_features, get_feature_columns
from gc_predict.model.dataset import create_dataloaders
from gc_predict.predict.inference import load_model, predict_timeline, get_last_known_dates
from gc_predict.data.fetch_visa_bulletin import load_visa_bulletin

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Predict green card timeline")
    parser.add_argument("--country", default="China", help="Country of chargeability")
    parser.add_argument("--eb", default="EB3", help="EB category (EB1, EB2, EB3, EB4)")
    parser.add_argument("--priority-date", default="2020-06-15", help="Priority date (YYYY-MM-DD)")
    args = parser.parse_args()

    priority_date = pd.Timestamp(args.priority_date)

    # Load data
    merged_path = project_root / "data" / "processed" / "merged.parquet"
    df = pd.read_parquet(merged_path)

    config = FeatureConfig()
    data = prepare_features(df, config)
    loaders = create_dataloaders(data, batch_size=32)

    # Load model
    checkpoint_dir = project_root / "models"
    tft_path = checkpoint_dir / "tft_best.pt"
    lstm_path = checkpoint_dir / "lstm_best.pt"

    if tft_path.exists():
        model = load_model(tft_path, loaders["num_features"], loaders["num_targets"], model_type="tft")
        model_name = "TFT"
    elif lstm_path.exists():
        model = load_model(lstm_path, loaders["num_features"], loaders["num_targets"], model_type="lstm")
        model_name = "LSTM"
    else:
        logger.error("No trained model found. Run 03_train.py first.")
        sys.exit(1)

    logger.info(f"Using {model_name} model")

    # Get recent feature window
    feature_cols = get_feature_columns(df)
    recent_window = df[feature_cols].iloc[-config.lookback_window:].values

    # Get last known dates
    visa_df = load_visa_bulletin(project_root / "data" / "raw" / "visa_bulletin")
    gov_path = project_root / "data" / "raw" / "visa_bulletin_gov" / "visa_bulletin_gov_full.parquet"
    gov_df = pd.read_parquet(gov_path) if gov_path.exists() else None
    last_known = get_last_known_dates(visa_df, gov_df)

    # Predict
    result = predict_timeline(
        model=model,
        recent_window=recent_window,
        scaler=data["scaler"],
        target_names=data["target_names"],
        country=args.country,
        eb_category=args.eb,
        priority_date=priority_date,
        last_known_dates=last_known,
    )

    # Display results
    print("\n" + "=" * 60)
    print(f"GREEN CARD TIMELINE PREDICTION")
    print(f"=" * 60)
    print(f"Country:       {result['country']}")
    print(f"Category:      {result['eb_category']}")
    print(f"Priority Date: {result['priority_date'].strftime('%Y-%m-%d')}")
    print()

    if result.get("last_known_action_date"):
        print(f"Last Known Final Action Date: {result['last_known_action_date'].strftime('%Y-%m-%d')}")
    if result.get("last_known_filing_date"):
        print(f"Last Known Filing Date:       {result['last_known_filing_date'].strftime('%Y-%m-%d')}")
    print()

    for label, key in [("FINAL ACTION DATE", "final_action"), ("DATE FOR FILING", "filing")]:
        info = result.get(key, {})
        status = info.get("status", "unavailable")
        print(f"--- {label} ---")

        if status == "current":
            print(f"  Status: CURRENT - {info['message']}")
        elif status == "projected":
            est = info["estimated_wait"]
            movement = info["predicted_monthly_movement"]
            print(f"  Gap to close: {info['gap_days']} days")
            print(f"  Predicted monthly movement: {movement['median']} days/month "
                  f"(range: {movement['pessimistic']} to {movement['optimistic']})")
            print(f"  Estimated wait:")
            print(f"    Optimistic:  {est['optimistic']}")
            print(f"    Median:      {est['median']}")
            print(f"    Pessimistic: {est['pessimistic']}")
        elif status == "stalled_or_retrogressing":
            print(f"  {info['message']}")
        else:
            print(f"  {info.get('reason', 'No data available')}")
        print()

    # Save result
    predictions_dir = project_root / "data" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    # Convert Timestamps for JSON serialization
    serializable = {}
    for k, v in result.items():
        if isinstance(v, pd.Timestamp):
            serializable[k] = v.isoformat()
        elif isinstance(v, dict):
            serializable[k] = v
        else:
            serializable[k] = str(v)

    with open(predictions_dir / "latest_prediction.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)


if __name__ == "__main__":
    main()

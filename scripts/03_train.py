"""Train models using time series cross-validation."""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from us_visa_bulletin_forecast.model.cross_validate import CVConfig, run_cv
from us_visa_bulletin_forecast.model.timesfm_model import TIMESFM_AVAILABLE, run_timesfm_cv, TimesFMConfig
from us_visa_bulletin_forecast.features.engineer import get_target_columns, get_filing_target_columns
from us_visa_bulletin_forecast.viz.plots import plot_training_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    merged_path = project_root / "data" / "processed" / "merged.parquet"
    df = pd.read_parquet(merged_path)
    logger.info(f"Loaded merged data: {df.shape}")
    logger.info(f"Date range: {df['bulletin_date'].min()} to {df['bulletin_date'].max()}")

    checkpoint_dir = project_root / "models"
    config = CVConfig()

    # Run CV for LSTM
    logger.info("=" * 60)
    logger.info("LSTM CROSS-VALIDATION")
    logger.info("=" * 60)
    lstm_cv = run_cv(df, model_type="lstm", config=config, checkpoint_dir=checkpoint_dir)

    # Run CV for TFT
    logger.info("")
    logger.info("=" * 60)
    logger.info("TFT CROSS-VALIDATION")
    logger.info("=" * 60)
    tft_cv = run_cv(df, model_type="tft", config=config, checkpoint_dir=checkpoint_dir)

    # Run TimesFM CV if available
    timesfm_cv = None
    if TIMESFM_AVAILABLE:
        logger.info("")
        logger.info("=" * 60)
        logger.info("TIMESFM CROSS-VALIDATION (zero-shot)")
        logger.info("=" * 60)
        target_cols = get_target_columns(df) + get_filing_target_columns(df)
        timesfm_cv = run_timesfm_cv(df, target_cols, TimesFMConfig(), config.first_test_year, config.last_test_year)
    else:
        logger.info("TimesFM not installed, skipping. Install with: pip install timesfm[torch]")

    # Final comparison
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 60)
    all_models = [("LSTM", lstm_cv), ("TFT", tft_cv)]
    if timesfm_cv:
        all_models.append(("TimesFM", timesfm_cv))
    for name, cv in all_models:
        agg = cv["aggregated_metrics"]
        if agg:
            logger.info(f"{name}:")
            logger.info(f"  MAE:      {agg['mae_days']['mean']:.1f} ± {agg['mae_days']['std']:.1f} days")
            logger.info(f"  Coverage: {agg['coverage_80']['mean']:.0%} ± {agg['coverage_80']['std']:.0%}")
            logger.info(f"  Dir Acc:  {agg['directional_accuracy']['mean']:.0%} ± {agg['directional_accuracy']['std']:.0%}")


if __name__ == "__main__":
    main()

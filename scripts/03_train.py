"""Train models using time series cross-validation."""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from us_visa_bulletin_forecast.model.cross_validate import CVConfig, run_cv
from us_visa_bulletin_forecast.model.timesfm_model import TIMESFM_AVAILABLE, run_timesfm_cv, TimesFMConfig
from us_visa_bulletin_forecast.model.chronos_model import CHRONOS_AVAILABLE, run_chronos_cv, ChronosConfig
from us_visa_bulletin_forecast.model.moirai_model import MOIRAI_AVAILABLE, run_moirai_cv, MoiraiConfig
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

    # Foundation model CV (zero-shot baselines)
    target_cols = get_target_columns(df) + get_filing_target_columns(df)
    foundation_results = []

    for name, available, run_fn, make_config in [
        ("TimesFM", TIMESFM_AVAILABLE, run_timesfm_cv, TimesFMConfig),
        ("Chronos-2", CHRONOS_AVAILABLE, run_chronos_cv, ChronosConfig),
        ("Moirai-2.0", MOIRAI_AVAILABLE, run_moirai_cv, MoiraiConfig),
    ]:
        if available:
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"{name} CROSS-VALIDATION (zero-shot)")
            logger.info("=" * 60)
            cv = run_fn(df, target_cols, make_config(), config.first_test_year, config.last_test_year)
            foundation_results.append((name, cv))
        else:
            logger.info(f"{name} not installed, skipping.")

    # Final comparison
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 60)
    all_models = [("LSTM", lstm_cv), ("TFT", tft_cv)] + foundation_results
    for name, cv in all_models:
        agg = cv["aggregated_metrics"]
        if agg:
            logger.info(f"{name}:")
            logger.info(f"  MAE:      {agg['mae_days']['mean']:.1f} ± {agg['mae_days']['std']:.1f} days")
            logger.info(f"  Coverage: {agg['coverage_80']['mean']:.0%} ± {agg['coverage_80']['std']:.0%}")
            logger.info(f"  Dir Acc:  {agg['directional_accuracy']['mean']:.0%} ± {agg['directional_accuracy']['std']:.0%}")


if __name__ == "__main__":
    main()

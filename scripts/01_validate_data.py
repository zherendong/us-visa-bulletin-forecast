"""Run data cross-validation checks."""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from gc_predict.data.fetch_visa_bulletin import load_visa_bulletin
from gc_predict.data.fetch_i485_inventory import load_i485_inventory
from gc_predict.data.validate_sources import run_validation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    # Load visa bulletin CSV data
    visa_dir = project_root / "data" / "raw" / "visa_bulletin"
    visa_csv_df = load_visa_bulletin(visa_dir) if visa_dir.exists() else None

    # Load gov-scraped sample
    gov_path = project_root / "data" / "raw" / "visa_bulletin_gov" / "visa_bulletin_gov_sample.parquet"
    visa_gov_df = pd.read_parquet(gov_path) if gov_path.exists() else None

    # Load I-485 data
    i485_dir = project_root / "data" / "raw" / "i485_inventory"
    i485_df = None
    if i485_dir.exists() and list(i485_dir.glob("*.xlsx")):
        i485_df = load_i485_inventory(i485_dir)

    report_path = project_root / "data" / "processed" / "validation_report.txt"
    result = run_validation(visa_csv_df, visa_gov_df, i485_df, report_path)

    if not result["passed"]:
        print("\nWARNING: Validation found HIGH severity issues. Review report before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()

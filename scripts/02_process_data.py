"""Process raw data into merged feature dataset."""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from us_visa_bulletin_forecast.data.fetch_visa_bulletin import load_visa_bulletin
from us_visa_bulletin_forecast.data.merge_datasets import merge_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    visa_df = load_visa_bulletin(project_root / "data" / "raw" / "visa_bulletin")

    # Load gov-scraped data (for filing dates)
    gov_path = project_root / "data" / "raw" / "visa_bulletin_gov" / "visa_bulletin_gov_full.parquet"
    gov_df = pd.read_parquet(gov_path) if gov_path.exists() else None

    output_path = project_root / "data" / "processed" / "merged.parquet"
    df = merge_all(visa_df, gov_df, output_path)

    print(f"\nMerged dataset shape: {df.shape}")
    print(f"Date range: {df['bulletin_date'].min()} to {df['bulletin_date'].max()}")
    print(f"\nFeature columns:")
    for c in sorted(df.columns):
        print(f"  {c}")


if __name__ == "__main__":
    main()

"""Merge all data sources into a single unified dataset.

Combines visa bulletin CSV data (Final Action Dates) with gov-scraped
Filing Dates into a monthly time series with all EB×country combinations.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COUNTRIES = ["China", "India", "Mexico", "Philippines", "Rest of World"]
EB_LEVELS = ["EB1", "EB2", "EB3", "EB4"]


def pivot_visa_bulletin(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot visa bulletin data into wide format with one row per month.

    Creates columns: {EB}_{Country}_action_days for each combo.
    """
    df = df.copy()

    # Normalize eb_level: integers -> "EB1", strings -> strip/uppercase
    df["eb_level"] = df["eb_level"].apply(
        lambda x: f"EB{x}" if isinstance(x, (int, float)) else str(x).replace("-", "").replace(" ", "").upper()
    )

    # Reference epoch for converting dates to numeric
    epoch = pd.Timestamp("2000-01-01")
    df["action_days"] = (df["final_action_date"] - epoch).dt.days

    df["year_month"] = df["bulletin_date"].dt.to_period("M")

    pivoted_frames = []
    for _, group in df.groupby("year_month"):
        ym = group["year_month"].iloc[0]
        row = {"year_month": ym, "bulletin_date": group["bulletin_date"].iloc[0]}

        for _, r in group.iterrows():
            prefix = f"{r['eb_level']}_{r['country'].replace(' ', '_')}"
            row[f"{prefix}_action_days"] = r["action_days"]

        pivoted_frames.append(row)

    result = pd.DataFrame(pivoted_frames)
    result = result.sort_values("bulletin_date").reset_index(drop=True)
    return result


def add_filing_dates(main_df: pd.DataFrame, gov_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Dates for Filing from gov-scraped data."""
    if gov_df is None or gov_df.empty:
        logger.warning("No gov-scraped data for Filing Dates")
        return main_df

    filing = gov_df[gov_df["table_type"] == "filing"].copy()
    if filing.empty:
        logger.warning("No Filing Date rows in gov data")
        return main_df

    epoch = pd.Timestamp("2000-01-01")
    filing["bulletin_date"] = pd.to_datetime(filing["bulletin_date"])

    # Parse date_value to datetime
    def parse_date(row):
        val = str(row["date_value"]).strip().upper()
        if val in ("C", "U"):
            return pd.NaT if val == "U" else row["bulletin_date"]
        try:
            return pd.to_datetime(row["date_value"])
        except Exception:
            return pd.NaT

    filing["filing_date"] = filing.apply(parse_date, axis=1)
    filing["filing_days"] = (filing["filing_date"] - epoch).dt.days
    filing["year_month"] = filing["bulletin_date"].dt.to_period("M")

    # Normalize eb_level
    filing["eb_level"] = filing["eb_level"].str.replace("-", "").str.upper()
    # Only keep main EB levels
    filing = filing[filing["eb_level"].isin(EB_LEVELS)]

    # Pivot filing dates
    filing_rows = []
    for _, group in filing.groupby("year_month"):
        ym = group["year_month"].iloc[0]
        row = {"year_month": ym}
        for _, r in group.iterrows():
            prefix = f"{r['eb_level']}_{r['country'].replace(' ', '_')}"
            row[f"{prefix}_filing_days"] = r["filing_days"]
        filing_rows.append(row)

    filing_wide = pd.DataFrame(filing_rows)

    main_df["year_month"] = main_df["bulletin_date"].dt.to_period("M")
    merged = main_df.merge(filing_wide, on="year_month", how="left")

    # Forward-fill filing dates (they don't change every month for some combos)
    filing_cols = [c for c in merged.columns if c.endswith("_filing_days")]
    merged[filing_cols] = merged[filing_cols].ffill()

    return merged


def add_movement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add monthly movement deltas for each action_days column."""
    df = df.copy()
    action_cols = [c for c in df.columns if c.endswith("_action_days")]
    for col in action_cols:
        prefix = col.replace("_action_days", "")
        df[f"{prefix}_movement"] = df[col].diff()
    return df


def add_filing_movement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add monthly movement deltas for filing dates."""
    df = df.copy()
    filing_cols = [c for c in df.columns if c.endswith("_filing_days")]
    for col in filing_cols:
        prefix = col.replace("_filing_days", "")
        df[f"{prefix}_filing_movement"] = df[col].diff()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features."""
    df = df.copy()
    df["month"] = df["bulletin_date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # Fiscal year quarter (Oct=Q1)
    df["fy_quarter"] = ((df["month"] - 10) % 12) // 3 + 1
    return df


def supplement_with_gov_action_dates(visa_df: pd.DataFrame, gov_df: pd.DataFrame) -> pd.DataFrame:
    """Add Final Action Date rows from gov-scraped data for months not in the CSV.

    The CSV (DavidBellamy) may lag behind; the gov scraper covers more recent months.
    """
    if gov_df is None or gov_df.empty:
        return visa_df

    action = gov_df[gov_df["table_type"] == "final_action"].copy()
    if action.empty:
        return visa_df

    action["bulletin_date"] = pd.to_datetime(action["bulletin_date"])
    action["eb_level"] = action["eb_level"].str.replace("-", "").str.upper()
    # Only keep main EB levels
    action = action[action["eb_level"].isin(EB_LEVELS)]

    csv_max_date = visa_df["bulletin_date"].max()
    new_months = action[action["bulletin_date"] > csv_max_date]
    if new_months.empty:
        logger.info("No new months in gov data beyond CSV coverage")
        return visa_df

    logger.info(f"Supplementing with {len(new_months)} gov-scraped rows "
                f"from {new_months['bulletin_date'].min()} to {new_months['bulletin_date'].max()}")

    epoch = pd.Timestamp("2000-01-01")

    def parse_date(row):
        val = str(row["date_value"]).strip().upper()
        if val == "C":
            return row["bulletin_date"]
        if val == "U":
            return pd.NaT
        try:
            return pd.to_datetime(row["date_value"])
        except Exception:
            return pd.NaT

    new_months["final_action_date"] = new_months.apply(parse_date, axis=1)
    new_months["is_current"] = new_months["date_value"].astype(str).str.upper() == "C"
    new_months["is_unavailable"] = new_months["date_value"].astype(str).str.upper() == "U"
    new_months["wait_time_days"] = new_months.apply(
        lambda r: 0.0 if r["is_current"] else (
            float("nan") if r["is_unavailable"] else
            (r["bulletin_date"] - r["final_action_date"]).days if pd.notna(r["final_action_date"]) else float("nan")
        ), axis=1
    )

    # Match CSV column format
    supplement = new_months[["bulletin_date", "eb_level", "country",
                             "final_action_date", "is_current", "is_unavailable",
                             "wait_time_days"]].copy()

    combined = pd.concat([visa_df, supplement], ignore_index=True)
    combined = combined.sort_values(["bulletin_date", "country", "eb_level"]).reset_index(drop=True)
    return combined


def merge_all(
    visa_df: pd.DataFrame,
    gov_df: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Run the full merge pipeline.

    Slimmed down features:
    - action_days per EB×country (~20 features)
    - movement per EB×country (~20 features)
    - filing_days per EB×country (~20 features, where available)
    - filing_movement per EB×country (~20 features, where available)
    - calendar features (3)
    """
    # Supplement CSV with newer gov-scraped Final Action Dates
    if gov_df is not None:
        visa_df = supplement_with_gov_action_dates(visa_df, gov_df)

    logger.info("Pivoting visa bulletin data...")
    df = pivot_visa_bulletin(visa_df)
    logger.info(f"Pivoted shape: {df.shape}")

    if gov_df is not None:
        logger.info("Adding filing dates...")
        df = add_filing_dates(df, gov_df)

    logger.info("Adding movement features...")
    df = add_movement_features(df)

    logger.info("Adding filing movement features...")
    df = add_filing_movement_features(df)

    logger.info("Adding calendar features...")
    df = add_calendar_features(df)

    # Drop helper columns
    if "year_month" in df.columns:
        df = df.drop(columns=["year_month"])

    logger.info(f"Final merged shape: {df.shape}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved merged dataset to {output_path}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).resolve().parents[3]

    from us_visa_bulletin_forecast.data.fetch_visa_bulletin import load_visa_bulletin

    visa_df = load_visa_bulletin(project_root / "data" / "raw" / "visa_bulletin")

    # Load gov-scraped data for filing dates
    gov_path = project_root / "data" / "raw" / "visa_bulletin_gov" / "visa_bulletin_gov_full.parquet"
    gov_df = pd.read_parquet(gov_path) if gov_path.exists() else None

    output_path = project_root / "data" / "processed" / "merged.parquet"
    df = merge_all(visa_df, gov_df, output_path)
    print(f"\nMerged dataset: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

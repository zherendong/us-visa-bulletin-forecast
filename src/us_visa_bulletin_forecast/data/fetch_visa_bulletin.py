"""Fetch visa bulletin data from DavidBellamy/visa_dates GitHub repository.

Downloads CSV files with Final Action Dates and wait times for all EB categories
and countries: India, China, Mexico, Philippines, Rest of World.
"""

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

REPO_BASE_URL = (
    "https://raw.githubusercontent.com/DavidBellamy/visa_dates/master/data"
)

COUNTRIES = {
    "china": "China",
    "india": "India",
    "mexico": "Mexico",
    "philippines": "Philippines",
    "row": "Rest of World",
}

# The repo uses this naming pattern for CSV files
CSV_FILENAME_TEMPLATE = "{country}_visa_backlog_timecourse.csv"


def fetch_country_csv(country_key: str, output_dir: Path) -> Path | None:
    """Download a single country's visa bulletin CSV from GitHub."""
    filename = CSV_FILENAME_TEMPLATE.format(country=country_key)
    url = f"{REPO_BASE_URL}/{filename}"
    output_path = output_dir / filename

    logger.info(f"Fetching {url}")
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        logger.warning(f"Failed to fetch {url}: HTTP {resp.status_code}")
        return None

    output_path.write_text(resp.text)
    logger.info(f"Saved {output_path} ({len(resp.text)} bytes)")
    return output_path


def fetch_all(output_dir: Path) -> dict[str, Path]:
    """Download all country CSVs. Returns mapping of country_key -> file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for key in COUNTRIES:
        path = fetch_country_csv(key, output_dir)
        if path:
            results[key] = path
    return results


def load_visa_bulletin(raw_dir: Path) -> pd.DataFrame:
    """Load and combine all country CSVs into a single DataFrame.

    Handles special values:
    - "C" (current): wait_time = 0, final_action_date = bulletin_date
    - "U" (unavailable): final_action_date = NaT, flagged with is_unavailable column

    Returns a DataFrame with columns:
        country, eb_level, bulletin_date, final_action_date, wait_time_days,
        is_current, is_unavailable
    """
    frames = []
    for country_key, country_name in COUNTRIES.items():
        filename = CSV_FILENAME_TEMPLATE.format(country=country_key)
        path = raw_dir / filename
        if not path.exists():
            logger.warning(f"Missing {path}, skipping {country_name}")
            continue

        df = pd.read_csv(path)
        df["country"] = country_name
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No visa bulletin CSVs found in {raw_dir}")

    combined = pd.concat(frames, ignore_index=True)

    # Standardize column names
    combined = combined.rename(columns={
        "EB_level": "eb_level",
        "visa_bulletin_date": "bulletin_date",
        "final_action_dates": "final_action_date_raw",
        "visa_wait_time": "wait_time_raw",
    })

    # Parse bulletin_date and drop rows where it's missing
    combined["bulletin_date"] = pd.to_datetime(combined["bulletin_date"], errors="coerce")
    combined = combined.dropna(subset=["bulletin_date"])

    # Handle special values in final_action_date
    combined["is_current"] = combined["final_action_date_raw"].astype(str).str.upper() == "C"
    combined["is_unavailable"] = combined["final_action_date_raw"].astype(str).str.upper() == "U"

    # Parse final_action_date: C -> bulletin_date, U -> NaT, else parse
    def parse_action_date(row):
        raw = str(row["final_action_date_raw"]).strip().upper()
        if raw == "C":
            return row["bulletin_date"]
        if raw == "U":
            return pd.NaT
        try:
            return pd.to_datetime(row["final_action_date_raw"])
        except Exception:
            return pd.NaT

    combined["final_action_date"] = combined.apply(parse_action_date, axis=1)

    # Parse wait_time: numeric days or 0 for current, NaN for unavailable
    def parse_wait_time(row):
        if row["is_current"]:
            return 0.0
        if row["is_unavailable"]:
            return float("nan")
        try:
            return float(row["wait_time_raw"])
        except (ValueError, TypeError):
            return float("nan")

    combined["wait_time_days"] = combined.apply(parse_wait_time, axis=1)

    # Drop raw columns
    combined = combined.drop(columns=["final_action_date_raw", "wait_time_raw"])

    # Sort
    combined = combined.sort_values(["bulletin_date", "country", "eb_level"]).reset_index(drop=True)

    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).resolve().parents[3]
    raw_dir = project_root / "data" / "raw" / "visa_bulletin"

    fetch_all(raw_dir)
    df = load_visa_bulletin(raw_dir)
    print(f"Loaded {len(df)} rows, date range: {df['bulletin_date'].min()} to {df['bulletin_date'].max()}")
    print(f"Countries: {df['country'].unique()}")
    print(f"EB levels: {df['eb_level'].unique()}")
    print(f"\nSample:\n{df.head(10)}")

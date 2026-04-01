"""Fetch USCIS processing times data.

Uses the jzebedee/uscis GitHub repository which maintains daily snapshots
of USCIS processing times in a SQLite database.
"""

import io
import logging
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# The jzebedee/uscis repo publishes releases with SQLite databases
USCIS_REPO = "jzebedee/uscis"
RELEASES_API = f"https://api.github.com/repos/{USCIS_REPO}/releases"


def fetch_processing_times_db(output_dir: Path) -> Path | None:
    """Download the latest USCIS processing times SQLite database from GitHub releases."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching releases from {RELEASES_API}")
    try:
        resp = requests.get(RELEASES_API, timeout=30)
        resp.raise_for_status()
        releases = resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch releases: {e}")
        return None

    if not releases:
        logger.warning("No releases found")
        return None

    # Find the latest release with a .db or .sqlite asset
    for release in releases[:5]:  # check last 5 releases
        for asset in release.get("assets", []):
            name = asset["name"].lower()
            if name.endswith((".db", ".sqlite", ".sqlite3", ".zip")):
                download_url = asset["browser_download_url"]
                output_path = output_dir / asset["name"]

                if output_path.exists():
                    logger.info(f"Already exists: {output_path}")
                    return output_path

                logger.info(f"Downloading {download_url}")
                try:
                    resp = requests.get(download_url, timeout=120)
                    resp.raise_for_status()

                    if name.endswith(".zip"):
                        # Extract zip
                        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                            for zi in zf.namelist():
                                if zi.endswith((".db", ".sqlite", ".sqlite3")):
                                    extracted = output_dir / zi
                                    with open(extracted, "wb") as f:
                                        f.write(zf.read(zi))
                                    logger.info(f"Extracted {extracted}")
                                    return extracted
                    else:
                        output_path.write_bytes(resp.content)
                        logger.info(f"Saved {output_path}")
                        return output_path
                except Exception as e:
                    logger.warning(f"Failed to download {download_url}: {e}")

    logger.warning("No suitable database asset found in recent releases")
    return None


def load_processing_times(db_path: Path) -> pd.DataFrame:
    """Load processing times from the SQLite database.

    Focuses on I-485 (adjustment of status) processing times.

    Returns DataFrame with columns:
        snapshot_date, form_type, office, processing_time_days_lower,
        processing_time_days_upper
    """
    conn = sqlite3.connect(str(db_path))

    # First, discover table structure
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    logger.info(f"Tables in database: {tables['name'].tolist()}")

    # Try common table names
    df = None
    for table in tables["name"]:
        try:
            sample = pd.read_sql(f"SELECT * FROM [{table}] LIMIT 5", conn)
            logger.info(f"Table '{table}' columns: {sample.columns.tolist()}")

            # Look for processing time data
            cols_lower = [c.lower() for c in sample.columns]
            if any("form" in c or "type" in c for c in cols_lower):
                df = pd.read_sql(f"SELECT * FROM [{table}]", conn)
                logger.info(f"Using table '{table}' with {len(df)} rows")
                break
        except Exception as e:
            logger.warning(f"Error reading table '{table}': {e}")

    conn.close()

    if df is None:
        raise RuntimeError(f"Could not find processing time data in {db_path}")

    # Filter for I-485 if possible
    form_col = None
    for col in df.columns:
        if "form" in col.lower() or "type" in col.lower():
            form_col = col
            break

    if form_col:
        i485_mask = df[form_col].astype(str).str.contains("I-485|485", case=False, na=False)
        if i485_mask.any():
            df = df[i485_mask].copy()
            logger.info(f"Filtered to {len(df)} I-485 rows")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).resolve().parents[3]
    raw_dir = project_root / "data" / "raw" / "processing_times"

    db_path = fetch_processing_times_db(raw_dir)
    if db_path:
        df = load_processing_times(db_path)
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nSample:\n{df.head()}")
    else:
        print("Could not fetch processing times database")

"""Scrape visa bulletin data directly from travel.state.gov.

Captures both Final Action Dates and Dates for Filing for all EB categories.
Dates for Filing were introduced in October 2015 (FY2016).

The State Department publishes bulletins at:
https://travel.state.gov/content/travel/en/legal/visa-law0/visa-bulletin/{year}/
  visa-bulletin-for-{month}-{year}.html
"""

import logging
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://travel.state.gov/content/travel/en/legal/visa-law0/visa-bulletin"

MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]

EB_LEVEL_MAP = {
    "1st": "EB1",
    "2nd": "EB2",
    "3rd": "EB3",
    "other workers": "EB3OW",
    "4th": "EB4",
    "certain religious workers": "EB4SR",
    "5th": "EB5",
}

COUNTRY_ALIASES = {
    "china": "China",
    "india": "India",
    "mexico": "Mexico",
    "philippines": "Philippines",
    "all chargeability": "Rest of World",
}


def build_bulletin_url(year: int, month: int) -> str:
    month_name = MONTH_NAMES[month - 1]
    return f"{BASE_URL}/{year}/visa-bulletin-for-{month_name}-{year}.html"


def parse_date_cell(text: str) -> tuple[str, bool, bool]:
    """Parse a date cell. Returns (date_str, is_current, is_unavailable)."""
    text = text.strip().upper()
    if text == "C":
        return "C", True, False
    if text == "U":
        return "U", False, True
    for fmt in ["%d%b%y", "%d%b%Y", "%d %b %y", "%d %b %Y", "%B %d, %Y"]:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%Y-%m-%d"), False, False
        except ValueError:
            continue
    logger.debug(f"Could not parse date: '{text}'")
    return text, False, False


def _parse_eb_table(table, bulletin_date: datetime, table_type: str) -> list[dict]:
    """Parse a single employment-based table (Final Action or Filing).

    Args:
        table_type: "final_action" or "filing"
    """
    rows = table.find_all("tr")
    if len(rows) < 2:
        return []

    # Parse header row for country columns
    header_cells = rows[0].find_all(["th", "td"])
    countries = []
    for cell in header_cells[1:]:
        text = " ".join(cell.get_text().split()).strip().lower()
        matched = False
        for alias, name in COUNTRY_ALIASES.items():
            if alias in text:
                countries.append(name)
                matched = True
                break
        if not matched:
            countries.append(text.title())

    rows_data = []
    for row in rows[1:]:
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue
        category_text = " ".join(cells[0].get_text().split()).strip().lower()

        eb_level = None
        for key, level in EB_LEVEL_MAP.items():
            if key in category_text:
                eb_level = level
                break
        if not eb_level:
            continue

        for i, cell in enumerate(cells[1:]):
            if i >= len(countries):
                break
            date_str, is_current, is_unavailable = parse_date_cell(cell.get_text())
            rows_data.append({
                "bulletin_date": bulletin_date,
                "table_type": table_type,
                "eb_level": eb_level,
                "country": countries[i],
                "date_value": date_str,
                "is_current": is_current,
                "is_unavailable": is_unavailable,
            })

    return rows_data


def scrape_bulletin(year: int, month: int) -> pd.DataFrame | None:
    """Scrape a single visa bulletin page for both Final Action and Filing dates."""
    url = build_bulletin_url(year, month)
    logger.info(f"Scraping {url}")

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"HTTP {resp.status_code} for {url}")
            return None
    except requests.RequestException as e:
        logger.warning(f"Request failed for {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    bulletin_date = datetime(year, month, 1)

    all_rows = []

    # Strategy: find text markers for "final action" and "filing", then grab
    # the next employment-based table after each marker.
    tables = soup.find_all("table")

    for table in tables:
        # Look at all preceding text elements to determine table type
        preceding_text = ""
        for prev in table.find_all_previous(["h2", "h3", "h4", "p", "strong", "b"]):
            t = " ".join(prev.get_text().split()).strip().lower()
            if "employ" in t and ("final action" in t or "filing" in t):
                preceding_text = t
                break

        if not preceding_text:
            continue

        # Check first row has employment-based header
        first_row = table.find("tr")
        if not first_row:
            continue
        first_cell = first_row.find(["th", "td"])
        if not first_cell:
            continue
        first_text = " ".join(first_cell.get_text().split()).strip().lower()
        if "employ" not in first_text:
            continue

        if "final action" in preceding_text:
            table_type = "final_action"
        elif "filing" in preceding_text:
            table_type = "filing"
        else:
            continue

        rows_data = _parse_eb_table(table, bulletin_date, table_type)
        all_rows.extend(rows_data)

    if not all_rows:
        logger.warning(f"No data found for {year}-{month:02d}")
        return None

    return pd.DataFrame(all_rows)


def scrape_range(
    start_year: int, start_month: int, end_year: int, end_month: int,
    output_dir: Path, delay: float = 0.5,
) -> pd.DataFrame:
    """Scrape visa bulletins for a range of months."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []

    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        df = scrape_bulletin(year, month)
        if df is not None:
            all_frames.append(df)
        time.sleep(delay)

        month += 1
        if month > 12:
            month = 1
            year += 1

    if not all_frames:
        raise RuntimeError("No visa bulletin data scraped successfully")

    combined = pd.concat(all_frames, ignore_index=True)
    output_path = output_dir / "visa_bulletin_gov_full.parquet"
    combined.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(combined)} rows to {output_path}")
    return combined


def scrape_sample_months(output_dir: Path, sample_interval: int = 12) -> pd.DataFrame:
    """Scrape a sample of months for cross-validation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []
    current = datetime.now()

    year, month = 2013, 1
    while (year, month) <= (current.year, current.month):
        df = scrape_bulletin(year, month)
        if df is not None:
            all_frames.append(df)
        time.sleep(1.0)

        month += sample_interval
        while month > 12:
            month -= 12
            year += 1

    if not all_frames:
        raise RuntimeError("No sample data scraped")

    combined = pd.concat(all_frames, ignore_index=True)
    output_path = output_dir / "visa_bulletin_gov_sample.parquet"
    combined.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(combined)} sample rows to {output_path}")
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).resolve().parents[3]
    raw_dir = project_root / "data" / "raw" / "visa_bulletin_gov"

    # Full scrape from Oct 2015 (when Dates for Filing started) to present
    current = datetime.now()
    scrape_range(2015, 10, current.year, current.month, raw_dir)

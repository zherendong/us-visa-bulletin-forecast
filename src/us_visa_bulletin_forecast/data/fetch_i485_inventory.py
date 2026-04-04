"""Fetch I-485 pending inventory data from USCIS.

USCIS publishes quarterly Excel reports showing employment-based I-485 applications
pending by category and country of chargeability.
"""

import logging
import re
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Known USCIS I-485 inventory file URLs
# These are updated periodically; we list known historical files
INVENTORY_URLS = [
    "https://www.uscis.gov/sites/default/files/document/reports/eb_inventory_02222024.xlsx",
    "https://www.uscis.gov/sites/default/files/document/reports/eb_inventory_07132023.xlsx",
    "https://www.uscis.gov/sites/default/files/document/reports/eb_inventory_01252023.xlsx",
    "https://www.uscis.gov/sites/default/files/document/reports/eb_inventory_07132022.xlsx",
    "https://www.uscis.gov/sites/default/files/document/reports/eb_inventory_01312022.xlsx",
    "https://www.uscis.gov/sites/default/files/document/reports/eb_inventory_07312021.xlsx",
    "https://www.uscis.gov/sites/default/files/document/reports/eb_inventory_01312021.xlsx",
    "https://www.uscis.gov/sites/default/files/document/reports/eb_inventory_07312020.xlsx",
    "https://www.uscis.gov/sites/default/files/document/reports/eb_inventory_01312020.xlsx",
]


def fetch_inventory_files(output_dir: Path) -> list[Path]:
    """Download all known I-485 inventory Excel files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for url in INVENTORY_URLS:
        filename = url.split("/")[-1]
        output_path = output_dir / filename
        if output_path.exists():
            logger.info(f"Already exists: {output_path}")
            downloaded.append(output_path)
            continue

        logger.info(f"Fetching {url}")
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                output_path.write_bytes(resp.content)
                logger.info(f"Saved {output_path} ({len(resp.content)} bytes)")
                downloaded.append(output_path)
            else:
                logger.warning(f"HTTP {resp.status_code} for {url}")
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")

    return downloaded


def parse_inventory_excel(path: Path) -> pd.DataFrame | None:
    """Parse a single I-485 inventory Excel file.

    The Excel files have varying formats, but generally contain:
    - Sheets organized by EB category or a combined sheet
    - Rows for different countries
    - Columns for priority date ranges with pending counts

    Returns DataFrame with columns:
        report_date, eb_level, country, priority_date_range, pending_count
    """
    # Extract date from filename (e.g., eb_inventory_02222024.xlsx -> 2024-02-22)
    match = re.search(r"(\d{2})(\d{2})(\d{4})", path.stem)
    if match:
        month, day, year = match.groups()
        report_date = pd.Timestamp(f"{year}-{month}-{day}")
    else:
        logger.warning(f"Cannot parse date from filename: {path.name}")
        return None

    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        logger.warning(f"Cannot open {path}: {e}")
        return None

    all_rows = []
    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        except Exception:
            continue

        # Try to identify EB level from sheet name
        sheet_lower = sheet_name.lower()
        eb_level = None
        for key, level in [("eb-1", "EB-1"), ("eb-2", "EB-2"), ("eb-3", "EB-3"),
                           ("eb-4", "EB-4"), ("eb-5", "EB-5"), ("eb1", "EB-1"),
                           ("eb2", "EB-2"), ("eb3", "EB-3"), ("eb4", "EB-4"),
                           ("eb5", "EB-5")]:
            if key in sheet_lower:
                eb_level = level
                break

        # Find header row (look for "Country" or similar)
        header_row = None
        for idx, row in df.iterrows():
            row_text = " ".join(str(v).lower() for v in row if pd.notna(v))
            if "country" in row_text or "chargeability" in row_text:
                header_row = idx
                break

        if header_row is None:
            continue

        headers = df.iloc[header_row].tolist()
        data_rows = df.iloc[header_row + 1:]

        # Find country column
        country_col = None
        for i, h in enumerate(headers):
            if pd.notna(h) and "country" in str(h).lower():
                country_col = i
                break
        if country_col is None:
            country_col = 0

        # Parse remaining columns as date ranges with counts
        for _, row in data_rows.iterrows():
            country_raw = str(row.iloc[country_col]).strip()
            if not country_raw or country_raw.lower() in ("nan", "total", "grand total"):
                continue

            # Normalize country name
            country = normalize_country(country_raw)
            if not country:
                continue

            total_pending = 0
            for col_idx in range(len(headers)):
                if col_idx == country_col:
                    continue
                val = row.iloc[col_idx] if col_idx < len(row) else None
                if pd.notna(val):
                    try:
                        count = int(float(val))
                        total_pending += count
                    except (ValueError, TypeError):
                        pass

            if total_pending > 0:
                all_rows.append({
                    "report_date": report_date,
                    "eb_level": eb_level or "Unknown",
                    "country": country,
                    "pending_count": total_pending,
                })

    if not all_rows:
        return None

    return pd.DataFrame(all_rows)


def normalize_country(name: str) -> str | None:
    """Map country name variants to standardized names."""
    lower = name.lower().strip()
    mapping = {
        "china": "China",
        "china-mainland born": "China",
        "china - mainland born": "China",
        "china mainland": "China",
        "india": "India",
        "mexico": "Mexico",
        "philippines": "Philippines",
        "el salvador": "El Salvador",
        "all other": "Rest of World",
        "all chargeability": "Rest of World",
        "rest of world": "Rest of World",
    }
    for key, val in mapping.items():
        if key in lower:
            return val
    return name.title()


def load_i485_inventory(raw_dir: Path) -> pd.DataFrame:
    """Load all I-485 inventory files into a single DataFrame."""
    files = sorted(raw_dir.glob("eb_inventory_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No I-485 inventory files found in {raw_dir}")

    frames = []
    for f in files:
        df = parse_inventory_excel(f)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError("Could not parse any I-485 inventory files")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["report_date", "eb_level", "country"]).reset_index(drop=True)
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).resolve().parents[3]
    raw_dir = project_root / "data" / "raw" / "i485_inventory"

    fetch_inventory_files(raw_dir)
    df = load_i485_inventory(raw_dir)
    print(f"Loaded {len(df)} rows")
    print(f"Report dates: {sorted(df['report_date'].unique())}")
    print(f"\nSample:\n{df.head(10)}")

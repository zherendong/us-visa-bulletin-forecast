"""Fetch all data sources."""

import logging
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from us_visa_bulletin_forecast.data.fetch_visa_bulletin import fetch_all as fetch_visa_bulletin
from us_visa_bulletin_forecast.data.fetch_visa_bulletin_gov import scrape_sample_months
from us_visa_bulletin_forecast.data.fetch_i485_inventory import fetch_inventory_files
from us_visa_bulletin_forecast.data.fetch_processing_times import fetch_processing_times_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    data_dir = project_root / "data" / "raw"

    # 1. Visa Bulletin (primary - DavidBellamy/visa_dates)
    logger.info("=" * 60)
    logger.info("Fetching visa bulletin CSV data...")
    visa_dir = data_dir / "visa_bulletin"
    results = fetch_visa_bulletin(visa_dir)
    logger.info(f"Downloaded {len(results)} visa bulletin CSV files")

    # 2. Visa Bulletin (gov - for cross-validation)
    logger.info("=" * 60)
    logger.info("Scraping sample visa bulletins from travel.state.gov...")
    gov_dir = data_dir / "visa_bulletin_gov"
    try:
        scrape_sample_months(gov_dir, sample_interval=12)
    except Exception as e:
        logger.warning(f"Gov scraping failed (non-critical): {e}")

    # 3. I-485 Inventory
    logger.info("=" * 60)
    logger.info("Fetching I-485 inventory data...")
    i485_dir = data_dir / "i485_inventory"
    files = fetch_inventory_files(i485_dir)
    logger.info(f"Downloaded {len(files)} I-485 inventory files")

    # 4. Processing Times
    logger.info("=" * 60)
    logger.info("Fetching USCIS processing times...")
    pt_dir = data_dir / "processing_times"
    db_path = fetch_processing_times_db(pt_dir)
    if db_path:
        logger.info(f"Processing times database: {db_path}")
    else:
        logger.warning("Could not fetch processing times (non-critical)")

    logger.info("=" * 60)
    logger.info("Data fetching complete!")


if __name__ == "__main__":
    main()

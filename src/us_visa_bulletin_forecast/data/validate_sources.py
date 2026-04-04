"""Cross-validate data sources before merging into final dataset.

Performs:
1. Visa bulletin CSV (DavidBellamy) vs State Dept HTML for sample months
2. Internal consistency checks on all datasets
3. Outputs a validation report
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def validate_visa_bulletin_cross_source(
    csv_df: pd.DataFrame, gov_df: pd.DataFrame, tolerance_days: int = 1
) -> list[dict]:
    """Compare visa bulletin CSV data against government-scraped data.

    Args:
        csv_df: DataFrame from DavidBellamy/visa_dates
        gov_df: DataFrame from travel.state.gov scraper
        tolerance_days: Maximum acceptable discrepancy in days

    Returns:
        List of discrepancy records
    """
    discrepancies = []

    # Normalize columns for comparison
    csv_df = csv_df.copy()
    gov_df = gov_df.copy()

    # Ensure datetime types
    csv_df["bulletin_date"] = pd.to_datetime(csv_df["bulletin_date"])
    csv_df["final_action_date"] = pd.to_datetime(csv_df["final_action_date"])
    gov_df["bulletin_date"] = pd.to_datetime(gov_df["bulletin_date"])

    # Filter gov to final_action rows only and parse date_value
    gov_df = gov_df[gov_df["table_type"] == "final_action"].copy()
    gov_df["final_action_date_parsed"] = pd.to_datetime(
        gov_df["date_value"], errors="coerce"
    )

    # Normalize EB level names for matching
    eb_map = {"EB-1": "EB1", "EB-2": "EB2", "EB-3": "EB3", "EB-4": "EB4", "EB-5": "EB5"}

    def normalize_eb(val):
        val = str(val).strip().upper().replace("-", "").replace(" ", "")
        # Handle bare integers: "1" -> "EB1"
        if val.isdigit():
            val = f"EB{val}"
        return val

    csv_df["eb_norm"] = csv_df["eb_level"].apply(normalize_eb)
    gov_df["eb_norm"] = gov_df["eb_level"].apply(normalize_eb)

    # Find overlapping (bulletin_date, eb_level, country) tuples
    csv_keys = set(zip(csv_df["bulletin_date"], csv_df["eb_norm"], csv_df["country"]))
    gov_keys = set(zip(gov_df["bulletin_date"], gov_df["eb_norm"], gov_df["country"]))
    overlap = csv_keys & gov_keys

    logger.info(f"Cross-validation: {len(overlap)} overlapping data points")

    for bdate, eb, country in overlap:
        csv_row = csv_df[
            (csv_df["bulletin_date"] == bdate)
            & (csv_df["eb_norm"] == eb)
            & (csv_df["country"] == country)
        ]
        gov_row = gov_df[
            (gov_df["bulletin_date"] == bdate)
            & (gov_df["eb_norm"] == eb)
            & (gov_df["country"] == country)
        ]

        if csv_row.empty or gov_row.empty:
            continue

        csv_date = csv_row.iloc[0]["final_action_date"]
        csv_current = csv_row.iloc[0]["is_current"]
        csv_wait = csv_row.iloc[0].get("wait_time_days", None)
        gov_current = gov_row.iloc[0]["is_current"]
        gov_date = gov_row.iloc[0]["final_action_date_parsed"]

        # Treat "action_date == bulletin_date with wait=0" as effectively current
        csv_effectively_current = csv_current or (
            pd.notna(csv_date) and pd.notna(csv_wait) and csv_wait == 0
        )

        # Check current/unavailable status match
        if csv_effectively_current != gov_current:
            discrepancies.append({
                "bulletin_date": bdate,
                "eb_level": eb,
                "country": country,
                "issue": "current_status_mismatch",
                "csv_value": f"current={csv_current}",
                "gov_value": f"current={gov_current}",
                "severity": "HIGH",
            })
            continue

        # Compare dates
        if pd.notna(csv_date) and pd.notna(gov_date):
            diff_days = abs((csv_date - gov_date).days)
            if diff_days > tolerance_days:
                discrepancies.append({
                    "bulletin_date": bdate,
                    "eb_level": eb,
                    "country": country,
                    "issue": "date_mismatch",
                    "csv_value": str(csv_date.date()),
                    "gov_value": str(gov_date.date()),
                    "diff_days": diff_days,
                    "severity": "HIGH" if diff_days > 30 else "MEDIUM",
                })

    return discrepancies


def validate_visa_bulletin_internal(df: pd.DataFrame) -> list[dict]:
    """Internal consistency checks on visa bulletin data."""
    issues = []

    # Check for duplicate entries
    dupes = df.duplicated(subset=["bulletin_date", "eb_level", "country"], keep=False)
    if dupes.any():
        n_dupes = dupes.sum()
        issues.append({
            "check": "duplicate_entries",
            "count": n_dupes,
            "severity": "HIGH",
            "detail": f"{n_dupes} duplicate (date, eb, country) rows found",
        })

    # Check date ordering (final_action_date should generally not exceed bulletin_date)
    valid_dates = df[df["final_action_date"].notna() & ~df["is_current"]]
    future_dates = valid_dates[valid_dates["final_action_date"] > valid_dates["bulletin_date"]]
    if len(future_dates) > 0:
        issues.append({
            "check": "future_action_dates",
            "count": len(future_dates),
            "severity": "MEDIUM",
            "detail": f"{len(future_dates)} rows have final_action_date after bulletin_date",
        })

    # Check for gaps in monthly sequence per (eb, country)
    for (eb, country), group in df.groupby(["eb_level", "country"]):
        dates = group["bulletin_date"].sort_values()
        if len(dates) < 2:
            continue
        diffs = dates.diff().dropna()
        gaps = diffs[diffs > pd.Timedelta(days=45)]  # More than ~1.5 months
        if len(gaps) > 0:
            issues.append({
                "check": "monthly_gap",
                "count": len(gaps),
                "severity": "LOW",
                "detail": f"{eb} {country}: {len(gaps)} gaps > 45 days in bulletin sequence",
            })

    # Check wait_time consistency
    valid_waits = df[df["wait_time_days"].notna() & ~df["is_current"]]
    negative_waits = valid_waits[valid_waits["wait_time_days"] < 0]
    if len(negative_waits) > 0:
        issues.append({
            "check": "negative_wait_times",
            "count": len(negative_waits),
            "severity": "HIGH",
            "detail": f"{len(negative_waits)} rows with negative wait times",
        })

    return issues


def validate_i485_inventory(df: pd.DataFrame) -> list[dict]:
    """Validate I-485 inventory data."""
    issues = []

    # Check for reasonable pending counts
    if df["pending_count"].max() > 10_000_000:
        issues.append({
            "check": "extreme_pending_count",
            "count": 1,
            "severity": "HIGH",
            "detail": f"Max pending count is {df['pending_count'].max():,}, seems unreasonable",
        })

    # Check for zero or negative counts
    bad_counts = df[df["pending_count"] <= 0]
    if len(bad_counts) > 0:
        issues.append({
            "check": "non_positive_pending",
            "count": len(bad_counts),
            "severity": "MEDIUM",
            "detail": f"{len(bad_counts)} rows with pending_count <= 0",
        })

    return issues


def run_validation(
    visa_csv_df: pd.DataFrame | None = None,
    visa_gov_df: pd.DataFrame | None = None,
    i485_df: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> dict:
    """Run all validation checks and produce a report.

    Returns dict with 'passed' (bool) and 'report' (str).
    """
    all_issues = []
    cross_discrepancies = []

    # 1. Internal visa bulletin checks
    if visa_csv_df is not None:
        logger.info("Running visa bulletin internal validation...")
        issues = validate_visa_bulletin_internal(visa_csv_df)
        all_issues.extend(issues)

    # 2. Cross-source validation
    if visa_csv_df is not None and visa_gov_df is not None:
        logger.info("Running cross-source validation...")
        cross_discrepancies = validate_visa_bulletin_cross_source(visa_csv_df, visa_gov_df)

    # 3. I-485 inventory checks
    if i485_df is not None:
        logger.info("Running I-485 inventory validation...")
        issues = validate_i485_inventory(i485_df)
        all_issues.extend(issues)

    # Build report
    lines = [
        "=" * 60,
        "DATA VALIDATION REPORT",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 60,
        "",
    ]

    # Internal issues
    high_count = sum(1 for i in all_issues if i.get("severity") == "HIGH")
    med_count = sum(1 for i in all_issues if i.get("severity") == "MEDIUM")
    low_count = sum(1 for i in all_issues if i.get("severity") == "LOW")

    lines.append(f"Internal checks: {len(all_issues)} issues found "
                 f"(HIGH: {high_count}, MEDIUM: {med_count}, LOW: {low_count})")
    lines.append("")

    for issue in all_issues:
        sev = issue.get("severity", "INFO")
        lines.append(f"  [{sev}] {issue.get('check', 'unknown')}: {issue.get('detail', '')}")

    # Cross-source discrepancies
    lines.append("")
    high_disc = sum(1 for d in cross_discrepancies if d.get("severity") == "HIGH")
    lines.append(f"Cross-source discrepancies: {len(cross_discrepancies)} found "
                 f"(HIGH: {high_disc})")
    lines.append("")

    for disc in cross_discrepancies[:20]:  # Show first 20
        sev = disc.get("severity", "INFO")
        lines.append(
            f"  [{sev}] {disc['bulletin_date'].strftime('%Y-%m') if hasattr(disc['bulletin_date'], 'strftime') else disc['bulletin_date']} "
            f"{disc['eb_level']} {disc['country']}: "
            f"{disc.get('issue', '')} - CSV: {disc.get('csv_value', '')} vs GOV: {disc.get('gov_value', '')}"
        )

    if len(cross_discrepancies) > 20:
        lines.append(f"  ... and {len(cross_discrepancies) - 20} more")

    # Overall verdict
    lines.append("")
    lines.append("=" * 60)
    critical_failures = high_count + high_disc
    if critical_failures > 0:
        lines.append(f"RESULT: {critical_failures} HIGH severity issues found. Review before proceeding.")
        passed = False
    else:
        lines.append("RESULT: PASSED - No critical issues found.")
        passed = True
    lines.append("=" * 60)

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        logger.info(f"Validation report saved to {output_path}")

    print(report)
    return {"passed": passed, "report": report, "issues": all_issues, "discrepancies": cross_discrepancies}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).resolve().parents[3]

    # Load visa bulletin CSV data
    from us_visa_bulletin_forecast.data.fetch_visa_bulletin import load_visa_bulletin

    raw_dir = project_root / "data" / "raw" / "visa_bulletin"
    visa_csv_df = load_visa_bulletin(raw_dir) if raw_dir.exists() else None

    # Load gov-scraped sample data
    gov_dir = project_root / "data" / "raw" / "visa_bulletin_gov"
    gov_path = gov_dir / "visa_bulletin_gov_sample.parquet"
    visa_gov_df = pd.read_parquet(gov_path) if gov_path.exists() else None

    # Load I-485 data
    from us_visa_bulletin_forecast.data.fetch_i485_inventory import load_i485_inventory

    i485_dir = project_root / "data" / "raw" / "i485_inventory"
    i485_df = load_i485_inventory(i485_dir) if i485_dir.exists() else None

    report_path = project_root / "data" / "processed" / "validation_report.txt"
    run_validation(visa_csv_df, visa_gov_df, i485_df, report_path)

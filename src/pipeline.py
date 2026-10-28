from pathlib import Path

import pandas as pd

from .fetch_clinicaltrials import fetch_clinical_trials
from .validator import validate_trials
from .anomaly_detector import detect_anomalies
from .report_generator import generate_report
from .parsing import ages_to_years, parse_dates, read_csv_safe


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

RAW_API = PROCESSED_DIR / "clinicaltrials_api.csv"
MASTER = PROCESSED_DIR / "clinical_trials_master.csv"
POWERBI = PROCESSED_DIR / "powerbi_ready.csv"

VALIDATION_ISSUES = (
    REPORTS_DIR / "pipeline_validation_issues.csv"
)
CLEANING_SUMMARY = (
    REPORTS_DIR / "cleaning_impact_summary.csv"
)
MISSING_COMPARISON = (
    REPORTS_DIR / "missing_values_before_after.csv"
)

MAX_RECORDS = 5000


def count_missing_values(series):
    """Count null and blank-string values in a Series."""
    missing = series.isna()

    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        missing = missing | series.astype("string").str.strip().eq(
            ""
        ).fillna(False)

    return int(missing.sum())


def clean_data(df, return_report=False):
    """
    Clean and normalize trial data.

    By default, return only the cleaned DataFrame to preserve
    compatibility with code that calls clean_data(df).

    With return_report=True, also return a cleaning summary
    and a before/after missing-value comparison.
    """
    df = df.copy()

    expected = [
        "nct_id",
        "brief_title",
        "overall_status",
        "study_type",
        "phase",
        "enrollment",
        "start_date",
        "completion_date",
        "condition",
        "intervention",
        "intervention_type",
        "sponsor",
        "agency_class",
        "country",
        "minimum_age",
        "maximum_age",
        "sex",
        "allocation",
        "intervention_model",
        "masking",
        "primary_outcome",
    ]

    original_rows = len(df)
    original_columns = set(df.columns)

    # Capture missing values before modifying the data.
    before_missing = {
        col: count_missing_values(df[col])
        if col in df.columns
        else original_rows
        for col in expected
    }

    # Record transformations performed during cleaning.
    text_values_standardized = 0
    invalid_enrollment_conversions = 0
    invalid_age_conversions = 0
    invalid_date_conversions = 0
    duplicate_records_removed = 0

    # Add missing expected columns without inventing values.
    for col in expected:
        if col not in df.columns:
            df[col] = pd.NA

    text_columns = [
        "nct_id",
        "brief_title",
        "overall_status",
        "study_type",
        "phase",
        "condition",
        "intervention",
        "intervention_type",
        "sponsor",
        "agency_class",
        "country",
        "sex",
        "allocation",
        "intervention_model",
        "masking",
        "primary_outcome",
    ]

    # Standardize text by removing leading/trailing whitespace.
    for col in text_columns:
        original = df[col].copy()
        standardized = original.astype("string").str.strip()

        changed = (
            original.notna()
            & standardized.ne(original.astype("string")).fillna(False)
        )

        text_values_standardized += int(changed.sum())
        df[col] = standardized

    # Convert enrollment to numeric.
    original_enrollment = df["enrollment"].copy()

    converted_enrollment = pd.to_numeric(
        original_enrollment,
        errors="coerce",
    )

    enrollment_was_present = (
        original_enrollment.notna()
        & original_enrollment.astype("string").str.strip().ne("")
    )

    invalid_enrollment_conversions = int(
        (
            enrollment_was_present
            & converted_enrollment.isna()
        ).sum()
    )

    df["enrollment"] = converted_enrollment

    # Convert age descriptions into years.
    for col in ["minimum_age", "maximum_age"]:
        original_age = df[col].copy()

        age_was_present = (
            original_age.notna()
            & original_age.astype("string").str.strip().ne("")
        )

        converted_age = ages_to_years(original_age)

        invalid_age_conversions += int(
            (
                age_was_present
                & converted_age.isna()
            ).sum()
        )

        df[col] = converted_age

    # Normalize date fields.
    for col in ["start_date", "completion_date"]:
        original_date = df[col].copy()

        date_was_present = (
            original_date.notna()
            & original_date.astype("string").str.strip().ne("")
        )

        converted_date = parse_dates(original_date)

        invalid_date_conversions += int(
            (
                date_was_present
                & converted_date.isna()
            ).sum()
        )

        df[col] = converted_date

    # Remove duplicate valid study IDs, preserving the first.
    valid_id = (
        df["nct_id"].notna()
        & df["nct_id"].astype("string").str.strip().ne("")
    )

    valid_records = df.loc[valid_id]
    missing_id_records = df.loc[~valid_id]

    before_deduplication = len(valid_records)

    valid_records = valid_records.drop_duplicates(
        subset=["nct_id"],
        keep="first",
    )

    duplicate_records_removed = (
        before_deduplication - len(valid_records)
    )

    df = pd.concat(
        [valid_records, missing_id_records],
        axis=0,
    ).reset_index(drop=True)

    # Capture missing values after cleaning.
    after_missing = {
        col: count_missing_values(df[col])
        for col in expected
    }

    missing_rows = []

    for col in expected:
        missing_rows.append({
            "column": col,
            "missing_before_cleaning": before_missing[col],
            "missing_after_cleaning": after_missing[col],
            "change_in_missing_values": (
                after_missing[col] - before_missing[col]
            ),
        })

    missing_comparison = pd.DataFrame(missing_rows)

    # Summarize measurable cleaning operations.
    cleaning_summary = pd.DataFrame([
        {
            "metric": "records_before_cleaning",
            "value": original_rows,
        },
        {
            "metric": "records_after_cleaning",
            "value": len(df),
        },
        {
            "metric": "records_removed_as_duplicate_ids",
            "value": duplicate_records_removed,
        },
        {
            "metric": "text_values_standardized",
            "value": text_values_standardized,
        },
        {
            "metric": "invalid_enrollment_values_converted_to_missing",
            "value": invalid_enrollment_conversions,
        },
        {
            "metric": "invalid_age_values_converted_to_missing",
            "value": invalid_age_conversions,
        },
        {
            "metric": "invalid_date_values_converted_to_missing",
            "value": invalid_date_conversions,
        },
        {
            "metric": "expected_columns_added",
            "value": sum(
                col not in original_columns
                for col in expected
            ),
        },
        {
            "metric": "missing_values_before_cleaning",
            "value": sum(before_missing.values()),
        },
        {
            "metric": "missing_values_after_cleaning",
            "value": sum(after_missing.values()),
        },
    ])

    if return_report:
        return df, cleaning_summary, missing_comparison

    return df

def run_pipeline(
    fetch_live=True,
    page_size=1000,
    max_records=MAX_RECORDS,
):
    """Fetch, clean, validate, detect anomalies, and save outputs."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch fresh API data or load the saved API CSV.
    if fetch_live:
        raw, total = fetch_clinical_trials(
            page_size=page_size,
            max_records=max_records,
        )

        raw.to_csv(RAW_API, index=False)
        print(f"Raw API data saved to: {RAW_API}")

    else:
        if not RAW_API.exists():
            raise FileNotFoundError(
                f"No API data found at {RAW_API}. "
                "Run with fetch_live=True first."
            )

        raw = read_csv_safe(RAW_API)
        total = None

    # 2. Clean records and collect cleaning impact metrics.
    cleaned, cleaning_summary, missing_comparison = clean_data(
        raw,
        return_report=True,
    )

    cleaning_summary.to_csv(CLEANING_SUMMARY, index=False)
    missing_comparison.to_csv(
        MISSING_COMPARISON,
        index=False,
    )

    # 3. Validate the cleaned data.
    validated, issues = validate_trials(
        cleaned,
        context_aware=True,
    )

    # 4. Detect anomalies.
    processed = detect_anomalies(validated)

    # 5. Save processed datasets.
    processed.to_csv(MASTER, index=False)

    processed.drop(
        columns=["brief_title", "primary_outcome"],
        errors="ignore",
    ).to_csv(POWERBI, index=False)

    # Save validation issues separately.
    issues.to_csv(VALIDATION_ISSUES, index=False)

    # Preserve the existing report-generation process.
    generate_report(processed, issues)

    # 6. Print pipeline summary.
    print("\n========== PIPELINE SUMMARY ==========")
    print(f"Records fetched: {len(raw):,}")

    if total is not None:
        print(f"Total studies reported by API: {total:,}")

    print(f"Records after cleaning: {len(cleaned):,}")
    print(f"Records processed: {len(processed):,}")
    print(f"Validation issue entries: {len(issues):,}")

    print("\n=== Cleaning Impact ===")
    print(cleaning_summary.to_string(index=False))

    print("\n=== Missing Values Before and After Cleaning ===")
    print(missing_comparison.to_string(index=False))

    if not issues.empty and "issue" in issues.columns:
        print("\n=== Validation Issues by Type ===")
        print(issues["issue"].value_counts().to_string())

        if "nct_id" in issues.columns:
            print(
                "\nUnique studies with validation issues: "
                f"{issues['nct_id'].nunique():,}"
            )

    if "quality_status" in processed.columns:
        print("\n=== Quality Status Distribution ===")
        print(
            processed["quality_status"]
            .value_counts(dropna=False)
            .to_string()
        )

    print(f"\nMaster dataset: {MASTER}")
    print(f"Power BI dataset: {POWERBI}")
    print(f"Validation issues: {VALIDATION_ISSUES}")
    print(f"Cleaning summary: {CLEANING_SUMMARY}")
    print(f"Missing-value comparison: {MISSING_COMPARISON}")
    print(f"Reports folder: {REPORTS_DIR}")
    print("Pipeline completed successfully.")
    print("======================================")

    return processed, issues, total


def main():
    run_pipeline(
        fetch_live=True,
        page_size=1000,
        max_records=MAX_RECORDS,
    )


if __name__ == "__main__":
    main()

"""Evaluate data quality and unusual records in the processed dataset.

This script does not modify the original dataset or pipeline reports.
Anomaly flags are candidates for review, not confirmed errors.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from parsing import ages_to_years
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports" / "real_data_evaluation"

CANDIDATE_FILES = [
    PROJECT_ROOT / "data" / "processed" / "clinical_trials_master.csv",
    PROJECT_ROOT / "data" / "processed" / "clinicaltrials_api.csv",
    PROJECT_ROOT / "data" / "raw" / "clinical_trials_demo.csv",
]

NUMERIC_CANDIDATES = [
    "enrollment",
    "min_age",
    "max_age",
    "minimum_age",
    "maximum_age",
]

# Screening limits, not universal clinical-trial validity standards.
MIN_ALLOWED_AGE = 0
MAX_ALLOWED_AGE = 120


def find_dataset():
    """Find the first available dataset from known project paths."""
    for path in CANDIDATE_FILES:
        if path.exists():
            return path

    checked = "\n".join(str(path) for path in CANDIDATE_FILES)
    raise FileNotFoundError(
        "Could not find a dataset in the expected locations.\n"
        f"Checked:\n{checked}\n"
        "Update CANDIDATE_FILES to match your actual CSV path."
    )


def load_dataset(path):
    """Read CSV while preserving literal NA text values."""
    df = pd.read_csv(
        path,
        keep_default_na=False,
        na_values=[""],
    )

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    return df


def find_column(df, possible_names):
    """Return the first matching column name."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def is_missing(series):
    """Treat blanks and common null markers as missing."""
    text = series.astype("string").str.strip()

    return (
        series.isna()
        | text.eq("").fillna(False)
        | text.str.upper().isin(
            ["NA", "N/A", "NULL", "NONE"]
        ).fillna(False)
    )


def numeric_series(df, column):
    """Convert a column to numeric without modifying the source."""
    return pd.to_numeric(df[column], errors="coerce")


def evaluate_data_quality(df):
    """Produce record-level issues and summary metrics."""
    n = len(df)

    if n == 0:
        raise ValueError("The selected dataset has no records.")

    issues = pd.DataFrame(index=df.index)
    issues["row_number"] = np.arange(2, n + 2)

    id_col = find_column(df, ["nct_id", "study_id", "trial_id"])

    if id_col:
        issues["study_id"] = df[id_col].astype(str)
    else:
        issues["study_id"] = ""

    # Missing values by column.
    missing_counts = []

    for column in df.columns:
        missing = is_missing(df[column])

        missing_counts.append({
            "column": column,
            "missing_count": int(missing.sum()),
            "missing_percentage": round(
                float(missing.mean() * 100), 2
            ),
        })

    missing_df = pd.DataFrame(missing_counts).sort_values(
        "missing_count",
        ascending=False,
    )

    # 1. Duplicate study identifiers.
    if id_col:
        normalized_id = df[id_col].astype("string").str.strip()
        valid_id = ~is_missing(normalized_id)

        duplicate_mask = normalized_id[valid_id].duplicated(
            keep=False
        )

        duplicate_rows = pd.Series(False, index=df.index)
        duplicate_rows.loc[duplicate_mask.index] = duplicate_mask

        issues["duplicate_study_id"] = duplicate_rows.to_numpy()
    else:
        issues["duplicate_study_id"] = False

    # 2. Validate minimum and maximum ages.
    min_col = find_column(
        df,
        ["min_age", "minimum_age", "minimum_age_years"],
    )
    max_col = find_column(
        df,
        ["max_age", "maximum_age", "maximum_age_years"],
    )

    issues["invalid_minimum_age"] = False
    issues["invalid_maximum_age"] = False
    issues["reversed_age_range"] = False

    min_age = None
    max_age = None

    if min_col:
        min_age = ages_to_years(df[min_col])
        min_present = ~is_missing(df[min_col])

        issues["invalid_minimum_age"] = min_present & (
            min_age.isna()
            | min_age.lt(MIN_ALLOWED_AGE)
            | min_age.gt(MAX_ALLOWED_AGE)
        )

    if max_col:
        max_age = ages_to_years(df[max_col])
        max_present = ~is_missing(df[max_col])

        issues["invalid_maximum_age"] = max_present & (
            max_age.isna()
            | max_age.lt(MIN_ALLOWED_AGE)
            | max_age.gt(MAX_ALLOWED_AGE)
        )

    if min_age is not None and max_age is not None:
        issues["reversed_age_range"] = (
            min_age.notna()
            & max_age.notna()
            & min_age.gt(max_age)
        )

    # Combined age flag. This counts each affected row only once.
    issues["invalid_age_range"] = (
        issues["invalid_minimum_age"]
        | issues["invalid_maximum_age"]
        | issues["reversed_age_range"]
    )

    # 3. Validate enrollment.
    enrollment_col = find_column(
        df,
        ["enrollment", "enrollment_count", "target_enrollment"],
    )

    if enrollment_col:
        enrollment = numeric_series(df, enrollment_col)
        enrollment_missing = is_missing(df[enrollment_col])

        issues["invalid_enrollment"] = (
            ~enrollment_missing
            & enrollment.notna()
            & enrollment.le(0)
        )

        issues["unparseable_enrollment"] = (
            ~enrollment_missing
            & enrollment.isna()
        )
    else:
        issues["invalid_enrollment"] = False
        issues["unparseable_enrollment"] = False

    # 4. Check date consistency.
    start_col = find_column(
        df,
        ["start_date", "study_start_date", "startdate"],
    )

    end_col = find_column(
        df,
        [
            "completion_date",
            "primary_completion_date",
            "study_completion_date",
        ],
    )

    if start_col and end_col:
        start_date = pd.to_datetime(
            df[start_col].replace("", pd.NA),
            errors="coerce",
            format="mixed",
        )

        end_date = pd.to_datetime(
            df[end_col].replace("", pd.NA),
            errors="coerce",
            format="mixed",
        )

        issues["completion_before_start"] = (
            start_date.notna()
            & end_date.notna()
            & end_date.lt(start_date)
        )
    else:
        issues["completion_before_start"] = False

    # 5. Summarize rule-based checks.
    check_columns = [
        "duplicate_study_id",
        "invalid_minimum_age",
        "invalid_maximum_age",
        "reversed_age_range",
        "invalid_age_range",
        "invalid_enrollment",
        "unparseable_enrollment",
        "completion_before_start",
    ]

    for column in check_columns:
        issues[column] = issues[column].fillna(False).astype(bool)

    issues["rule_issue_count"] = issues[check_columns].sum(axis=1)
    issues["has_rule_issue"] = issues["rule_issue_count"] > 0

    # Note: invalid_age_range overlaps with its component checks.
    # The counts in this table are per check, not unique-record totals.
    rule_summary = pd.DataFrame([
        {
            "check": column,
            "records_flagged": int(issues[column].sum()),
        }
        for column in check_columns
    ])

    summary = pd.DataFrame([
        {"metric": "records", "value": n},
        {"metric": "columns", "value": len(df.columns)},
        {
            "metric": "records_with_any_rule_issue",
            "value": int(issues["has_rule_issue"].sum()),
        },
        {
            "metric": "duplicate_study_id_rows",
            "value": int(issues["duplicate_study_id"].sum()),
        },
    ])

    return issues, missing_df, rule_summary, summary


def add_anomaly_flags(df, issues):
    """Add Isolation Forest flags using available numeric features."""
    available = [
        column for column in NUMERIC_CANDIDATES
        if column in df.columns
    ]

    if len(available) < 2 or len(df) < 20:
        print(
            "\nSkipping Isolation Forest: need at least two "
            "supported numeric columns and 20 records."
        )
        issues["ml_anomaly_flag"] = pd.NA
        issues["ml_anomaly_score"] = np.nan
        return issues, pd.DataFrame()

    numeric = df[available].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Drop columns with no usable numeric values.
    numeric = numeric.dropna(axis=1, how="all")

    if numeric.shape[1] < 2:
        print(
            "\nSkipping Isolation Forest: fewer than two "
            "usable numeric features."
        )
        issues["ml_anomaly_flag"] = pd.NA
        issues["ml_anomaly_score"] = np.nan
        return issues, pd.DataFrame()

    # Median imputation allows the model to run.
    # Missingness remains visible in the separate report.
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(numeric)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )

    model.fit(scaled)

    predictions = model.predict(scaled)
    scores = -model.decision_function(scaled)

    issues["ml_anomaly_flag"] = (predictions == -1).astype(int)
    issues["ml_anomaly_score"] = scores

    anomaly_report = df.copy()
    anomaly_report["ml_anomaly_flag"] = (
        predictions == -1
    ).astype(int)
    anomaly_report["ml_anomaly_score"] = scores

    return issues, anomaly_report


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = find_dataset()
    print(f"Dataset selected: {dataset_path}")

    df = load_dataset(dataset_path)

    print(f"Records loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    issues, missing_df, rule_summary, summary = (
        evaluate_data_quality(df)
    )

    issues, anomaly_report = add_anomaly_flags(df, issues)

    # Write only to the dedicated evaluation folder.
    issues.to_csv(
        REPORT_DIR / "record_level_quality_issues.csv",
        index=False,
    )

    missing_df.to_csv(
        REPORT_DIR / "missing_values_summary.csv",
        index=False,
    )

    rule_summary.to_csv(
        REPORT_DIR / "rule_check_summary.csv",
        index=False,
    )

    summary.to_csv(
        REPORT_DIR / "dataset_summary.csv",
        index=False,
    )

    if not anomaly_report.empty:
        anomaly_report.to_csv(
            REPORT_DIR / "isolation_forest_candidates.csv",
            index=False,
        )

    print("\n=== Dataset Summary ===")
    print(summary.to_string(index=False))

    print("\n=== Missing Values by Column ===")
    print(missing_df.to_string(index=False))

    print("\n=== Rule-Based Checks ===")
    print(rule_summary.to_string(index=False))

    if "ml_anomaly_flag" in issues.columns:
        flagged = issues["ml_anomaly_flag"].sum()

        if pd.notna(flagged):
            print(
                f"\nIsolation Forest candidates: {int(flagged):,} "
                f"of {len(df):,} records"
            )

    print(f"\nReports saved to: {REPORT_DIR}")
    print(
        "ML anomaly flags are review candidates, "
        "not confirmed errors."
    )


if __name__ == "__main__":
    main()

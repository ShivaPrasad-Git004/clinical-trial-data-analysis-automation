
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "reports" / "real_data_evaluation"
    / "isolation_forest_candidates.csv"
)

OUTPUT_DIR = ROOT / "reports" / "real_data_evaluation"


def is_missing(series):
    """Identify empty values and common missing-value markers."""
    text = series.astype("string").str.strip()
    return (
        text.isna()
        | text.eq("")
        | text.str.upper().isin(["NA", "N/A", "NULL", "NONE"])
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_FILE}\n"
            "Run python -m src.real_data_evaluation first."
        )

    df = pd.read_csv(INPUT_FILE, keep_default_na=False)

    # Keep only Isolation Forest candidates if the file contains all rows.
    if "ml_anomaly_flag" in df.columns:
        flags = pd.to_numeric(
            df["ml_anomaly_flag"], errors="coerce"
        )
        df = df.loc[flags.eq(1)].copy()

    if df.empty:
        print("No anomaly candidates found.")
        return

    reasons = {index: [] for index in df.index}

    def add_reason(mask, message):
        for index in df.index[mask.fillna(False)]:
            reasons[index].append(message)

    # 1. Check enrollment values.
    if "enrollment" in df.columns:
        raw = df["enrollment"].astype("string").str.strip()
        enrollment = pd.to_numeric(raw, errors="coerce")
        missing = is_missing(df["enrollment"])
        invalid = (~missing) & (
            enrollment.isna()
            | enrollment.le(0)
            | enrollment.gt(10_000_000)
        )
        add_reason(invalid, "Check enrollment: invalid, unparseable, or above 10 million")

        unusual = (~missing) & enrollment.gt(100_000)
        add_reason(unusual, "Unusually large enrollment; verify against study details")

    # 2. Check minimum and maximum age.
    min_age = max_age = None

    if "minimum_age" in df.columns:
        min_age = pd.to_numeric(
            df["minimum_age"].replace("", pd.NA), errors="coerce"
        )
        raw = df["minimum_age"].astype("string").str.strip()
        missing = is_missing(df["minimum_age"])
        invalid = (~missing) & (
            min_age.isna() | min_age.lt(0) | min_age.gt(120)
        )
        add_reason(invalid, "Check minimum age: invalid or outside 0–120")

    if "maximum_age" in df.columns:
        max_age = pd.to_numeric(
            df["maximum_age"].replace("", pd.NA), errors="coerce"
        )
        raw = df["maximum_age"].astype("string").str.strip()
        missing = is_missing(df["maximum_age"])
        invalid = (~missing) & (
            max_age.isna() | max_age.lt(0) | max_age.gt(120)
        )
        add_reason(invalid, "Check maximum age: invalid or outside 0–120")

    if min_age is not None and max_age is not None:
        add_reason(
            min_age.notna() & max_age.notna() & min_age.gt(max_age),
            "Check age range: minimum age exceeds maximum age",
        )

    # 3. Check dates.
    start = end = None

    if "start_date" in df.columns:
        raw = df["start_date"].astype("string").str.strip()
        start = pd.to_datetime(
            raw.replace("", pd.NA), errors="coerce"
        )
        bad = (~is_missing(df["start_date"])) & start.isna()
        add_reason(bad, "Check start date: unparseable date")

    if "completion_date" in df.columns:
        raw = df["completion_date"].astype("string").str.strip()
        end = pd.to_datetime(
            raw.replace("", pd.NA), errors="coerce"
        )
        bad = (~is_missing(df["completion_date"])) & end.isna()
        add_reason(bad, "Check completion date: unparseable date")

    if start is not None and end is not None:
        add_reason(
            start.notna() & end.notna() & end.lt(start),
            "Check dates: completion precedes start",
        )

    # 4. Assign a review category; do not automatically delete records.
    df["review_reasons"] = [
        "; ".join(reasons[index]) for index in df.index
    ]

    df["review_category"] = [
        (
            "POTENTIAL DATA-QUALITY ISSUE"
            if any(
                phrase in reason
                for phrase in [
                    "invalid, unparseable",
                    "invalid or outside",
                    "minimum age exceeds",
                    "unparseable date",
                    "completion precedes",
                ]
            )
            else (
                "UNUSUAL - VERIFY"
                if reason
                else "MANUAL REVIEW - NO RULE VIOLATION FOUND"
            )
        )
        for reason in df["review_reasons"]
    ]

    # Include useful fields when available.
    preferred = [
        "nct_id", "brief_title", "study_type", "overall_status",
        "enrollment", "minimum_age", "maximum_age",
        "start_date", "completion_date", "quality_score",
        "quality_status", "ml_anomaly_score",
        "review_category", "review_reasons",
    ]
    columns = [c for c in preferred if c in df.columns]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_path = OUTPUT_DIR / "categorized_anomaly_candidates.csv"
    summary_path = OUTPUT_DIR / "anomaly_category_summary.csv"

    df[columns].to_csv(detail_path, index=False)

    summary = (
        df["review_category"]
        .value_counts()
        .rename_axis("review_category")
        .reset_index(name="record_count")
    )
    summary.to_csv(summary_path, index=False)

    print(f"Anomaly candidates reviewed: {len(df)}")
    print("\n=== Review Category Summary ===")
    print(summary.to_string(index=False))

    print("\n=== Examples requiring attention ===")
    print(
        df.loc[
            df["review_category"] !=
            "MANUAL REVIEW - NO RULE VIOLATION FOUND",
            columns,
        ].head(15).to_string(index=False)
    )

    print(f"\nDetailed report saved to: {detail_path}")
    print(f"Summary report saved to: {summary_path}")
    print(
        "\nThese categories are screening results, not proof of errors. "
        "Verify records before making changes."
    )


if __name__ == "__main__":
    main()

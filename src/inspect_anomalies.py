
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_data_evaluation"
    / "isolation_forest_candidates.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_data_evaluation"
    / "anomaly_candidate_review.csv"
)


def main():
    df = pd.read_csv(DATA_PATH, keep_default_na=False)

    if "ml_anomaly_flag" not in df.columns:
        raise ValueError("The ML anomaly flag column is missing.")

    flagged = df[
        pd.to_numeric(
            df["ml_anomaly_flag"], errors="coerce"
        ).eq(1)
    ].copy()

    # Display the most unusual records first.
    flagged["ml_anomaly_score"] = pd.to_numeric(
        flagged["ml_anomaly_score"], errors="coerce"
    )

    flagged = flagged.sort_values(
        "ml_anomaly_score",
        ascending=False,
    )

    review_columns = [
        "nct_id",
        "brief_title",
        "overall_status",
        "study_type",
        "phase",
        "enrollment",
        "minimum_age",
        "maximum_age",
        "start_date",
        "completion_date",
        "quality_score",
        "quality_status",
        "anomaly_flag",
        "anomaly_score",
        "ml_anomaly_flag",
        "ml_anomaly_score",
    ]

    review_columns = [
        column for column in review_columns
        if column in flagged.columns
    ]

    review = flagged[review_columns]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(OUTPUT_PATH, index=False)

    print(f"Total records flagged: {len(flagged)}")
    print("\n=== Flagged Records by Study Type ===")

    if "study_type" in flagged.columns:
        print(
            flagged["study_type"]
            .replace("", "Missing")
            .value_counts(dropna=False)
            .to_string()
        )

    print("\n=== Flagged Records by Quality Status ===")

    if "quality_status" in flagged.columns:
        print(
            flagged["quality_status"]
            .replace("", "Missing")
            .value_counts(dropna=False)
            .to_string()
        )

    print("\n=== First 10 Candidate Records ===")
    print(review.head(10).to_string(index=False))

    print(f"\nSaved review file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

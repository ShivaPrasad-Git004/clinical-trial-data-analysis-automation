
import pandas as pd

from .parsing import ages_to_years, parse_dates


def validate_trials(df, context_aware=False):
    """Validate clinical-trial metadata and calculate quality scores."""
    df = df.copy()
    issues = []

    required = [
        "nct_id",
        "brief_title",
        "overall_status",
        "phase",
        "sponsor",
        "start_date",
        "enrollment",
    ]

    optional = [
        "condition",
        "minimum_age",
        "maximum_age",
        "completion_date",
        "study_type",
    ]

    # Ensure expected columns exist.
    for col in required + optional:
        if col not in df.columns:
            df[col] = pd.NA

    def missing_mask(series):
        text = series.astype("string").str.strip()

        return (
            series.isna()
            | text.eq("").fillna(False)
            | text.str.upper().isin(
                ["NA", "N/A", "NULL", "NONE"]
            ).fillna(False)
        )

    status_up = (
        df["overall_status"]
        .astype("string")
        .str.strip()
        .str.upper()
        .fillna("")
    )

    study_up = (
        df["study_type"]
        .astype("string")
        .str.strip()
        .str.upper()
        .fillna("")
    )

    # In context-aware mode, phase is required only
    # for interventional studies.
    phase_required = pd.Series(True, index=df.index)

    if context_aware:
        phase_required = study_up.eq("INTERVENTIONAL")

    def required_missing(col):
        mask = missing_mask(df[col])

        if col == "phase":
            return mask & phase_required

        return mask

    # 1. Check required fields.
    for col in required:
        mask = required_missing(col)

        for idx in df.index[mask]:
            issues.append({
                "nct_id": df.at[idx, "nct_id"],
                "issue": f"Missing {col}",
            })

    # 2. Validate enrollment.
    enrollment = pd.to_numeric(
        df["enrollment"],
        errors="coerce",
    )

    invalid_enrollment = (
        missing_mask(df["enrollment"])
        | enrollment.isna()
        | enrollment.le(0)
    )

    # Allow zero enrollment for withdrawn studies
    # when context-aware validation is enabled.
    if context_aware:
        withdrawn_zero = (
            status_up.eq("WITHDRAWN")
            & enrollment.eq(0)
        )
        invalid_enrollment &= ~withdrawn_zero

    for idx in df.index[invalid_enrollment]:
        issues.append({
            "nct_id": df.at[idx, "nct_id"],
            "issue": "Invalid enrollment",
        })

    # 3. Validate age values.
    min_age = ages_to_years(df["minimum_age"])
    max_age = ages_to_years(df["maximum_age"])

    min_present = ~missing_mask(df["minimum_age"])
    max_present = ~missing_mask(df["maximum_age"])

    invalid_min_age = min_present & (
        min_age.isna()
        | min_age.lt(0)
        | min_age.gt(120)
    )

    invalid_max_age = max_present & (
        max_age.isna()
        | max_age.lt(0)
        | max_age.gt(120)
    )

    reversed_age_range = (
        min_age.notna()
        & max_age.notna()
        & min_age.gt(max_age)
    )

    invalid_age = (
        invalid_min_age
        | invalid_max_age
        | reversed_age_range
    )

    for idx in df.index[invalid_min_age]:
        issues.append({
            "nct_id": df.at[idx, "nct_id"],
            "issue": (
                "Invalid minimum age: "
                "outside 0-120 or unparseable"
            ),
        })

    for idx in df.index[invalid_max_age]:
        issues.append({
            "nct_id": df.at[idx, "nct_id"],
            "issue": (
                "Invalid maximum age: "
                "outside 0-120 or unparseable"
            ),
        })

    for idx in df.index[reversed_age_range]:
        issues.append({
            "nct_id": df.at[idx, "nct_id"],
            "issue": "Minimum age greater than maximum age",
        })

    # 4. Validate dates.
    start = parse_dates(df["start_date"])
    completion = parse_dates(df["completion_date"])

    invalid_dates = (
        start.notna()
        & completion.notna()
        & completion.lt(start)
    )

    for idx in df.index[invalid_dates]:
        issues.append({
            "nct_id": df.at[idx, "nct_id"],
            "issue": "Completion date before start date",
        })

    # 5. Calculate quality scores using the existing penalties.
    score = pd.Series(100.0, index=df.index)

    for col in required:
        score -= required_missing(col).astype(float) * 5

    score -= invalid_enrollment.astype(float) * 10
    score -= invalid_age.astype(float) * 10
    score -= invalid_dates.astype(float) * 15

    df["quality_score"] = score.clip(0, 100).round(2)

    df["quality_status"] = pd.cut(
        df["quality_score"],
        bins=[-1, 39, 59, 79, 89, 100],
        labels=[
            "CRITICAL",
            "POOR",
            "MODERATE",
            "GOOD",
            "EXCELLENT",
        ],
    ).astype("string")

    # 6. Build the validation issues report.
    issue_df = pd.DataFrame(
        issues,
        columns=["nct_id", "issue"],
    )

    # 7. Add validation status.
    # Use study IDs so a record with multiple issues is
    # still marked REVIEW_REQUIRED only once.
    issue_ids = set(
        issue_df["nct_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    study_ids = (
        df["nct_id"]
        .astype("string")
        .str.strip()
    )

    df["validation_status"] = study_ids.map(
        lambda study_id: (
            "REVIEW_REQUIRED"
            if pd.notna(study_id)
            and study_id in issue_ids
            else "PASS"
        )
    )

    return df, issue_df

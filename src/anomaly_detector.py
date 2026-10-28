
"""Detect unusual clinical-trial records using Isolation Forest."""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .parsing import ages_to_years, parse_dates


def detect_anomalies(df):
    """Add anomaly flags and scores to clinical-trial records."""
    df = df.copy()

    # Handle an empty dataset safely.
    if df.empty:
        df["anomaly_flag"] = pd.Series(index=df.index, dtype="int64")
        df["anomaly_score"] = pd.Series(index=df.index, dtype="float64")
        return df

    # Convert enrollment to numeric values.
    enrollment = pd.to_numeric(
        df["enrollment"] if "enrollment" in df.columns
        else pd.Series(pd.NA, index=df.index),
        errors="coerce",
    )

    # Convert all age values to years, including hours, days, and months.
    minimum_age = ages_to_years(
        df["minimum_age"] if "minimum_age" in df.columns
        else pd.Series(pd.NA, index=df.index)
    )
    maximum_age = ages_to_years(
        df["maximum_age"] if "maximum_age" in df.columns
        else pd.Series(pd.NA, index=df.index)
    )

    # Parse mixed-format dates using the shared parsing helper.
    start_dates = parse_dates(
        df["start_date"] if "start_date" in df.columns
        else pd.Series(pd.NaT, index=df.index)
    )
    completion_dates = parse_dates(
        df["completion_date"] if "completion_date" in df.columns
        else pd.Series(pd.NaT, index=df.index)
    )

    # Calculate trial duration in days.
    duration = (completion_dates - start_dates).dt.total_seconds() / 86400

    # Build the numeric feature set for Isolation Forest.
    X = pd.DataFrame(
        {
            "enrollment": enrollment,
            "minimum_age": minimum_age,
            "maximum_age": maximum_age,
            "trial_duration_days": duration,
        },
        index=df.index,
    )

    # Replace infinite values and missing values safely.
    X = X.replace([float("inf"), float("-inf")], float("nan"))
    X = X.fillna(X.median()).fillna(0)

    # Isolation Forest needs at least two records to compare.
    if len(X) < 2:
        df["anomaly_flag"] = 0
        df["anomaly_score"] = 0.0
        return df

    # Scale features so they are on comparable numeric scales.
    X_scaled = StandardScaler().fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=42,
    )

    predictions = model.fit_predict(X_scaled)

    df["anomaly_flag"] = (predictions == -1).astype(int)
    df["anomaly_score"] = model.decision_function(X_scaled).round(6)

    return df

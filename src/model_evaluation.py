
"""
Clinical Trial Data Analysis
Mixed-defect anomaly detector and hybrid model evaluation.

IMPORTANT:
- This is a synthetic benchmark, not real clinical-trial ground truth.
- Detectors are trained only on normal training records.
- Hybrid detection = explicit validation rules OR ML anomaly prediction.
"""

import os

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

RAW_FEATURES = [
    "enrollment",
    "minimum_age",
    "maximum_age",
    "duration_days",
]

MISSING_FEATURES = [
    "enrollment_missing",
    "minimum_age_missing",
    "maximum_age_missing",
]

FEATURES = RAW_FEATURES + MISSING_FEATURES

DEFECT_TYPES = [
    "invalid_age_range",
    "suspicious_enrollment",
    "long_duration",
    "missing_numeric_value",
]

REPORT_DIR = "reports"


# ---------------------------------------------------------
# 2. Create a synthetic clinical-trial benchmark
# ---------------------------------------------------------

def create_benchmark(
    n_samples=5000,
    anomaly_fraction=0.05,
    random_state=42,
):
    """Create synthetic normal records and records with injected defects."""

    if n_samples < 100:
        raise ValueError("n_samples must be at least 100.")

    if not 0 < anomaly_fraction < 1:
        raise ValueError("anomaly_fraction must be between 0 and 1.")

    rng = np.random.default_rng(random_state)

    n_anomalies = int(n_samples * anomaly_fraction)
    n_normal = n_samples - n_anomalies

    # Generate normal records.
    min_age = np.clip(
        rng.normal(25, 10, n_normal),
        0,
        70,
    )

    max_age = np.clip(
        min_age + rng.normal(45, 12, n_normal),
        min_age + 1,
        100,
    )

    normal = pd.DataFrame({
        "enrollment": rng.lognormal(5.0, 0.7, n_normal),
        "minimum_age": min_age,
        "maximum_age": max_age,
        "duration_days": rng.lognormal(
            np.log(365), 0.6, n_normal
        ),
        "defect_type": "normal",
    })

    # Distribute the injected anomalies across four defect types.
    defect_types = np.resize(
        np.array(DEFECT_TYPES),
        n_anomalies,
    )
    rng.shuffle(defect_types)

    anomalies = pd.DataFrame({
        "enrollment": rng.lognormal(
            5.0, 0.7, n_anomalies
        ),
        "minimum_age": np.clip(
            rng.normal(25, 10, n_anomalies),
            0,
            70,
        ),
        "maximum_age": np.clip(
            rng.normal(70, 8, n_anomalies),
            1,
            100,
        ),
        "duration_days": rng.lognormal(
            np.log(365), 0.6, n_anomalies
        ),
        "defect_type": defect_types,
    })

    # Defect 1: minimum age exceeds maximum age.
    mask = anomalies["defect_type"].eq(
        "invalid_age_range"
    )
    count = int(mask.sum())

    anomalies.loc[mask, "minimum_age"] = rng.uniform(
        55, 75, count
    )
    anomalies.loc[mask, "maximum_age"] = rng.uniform(
        25, 50, count
    )

    # Defect 2: suspicious enrollment.
    # These values are elevated, but not necessarily impossible.
    mask = anomalies["defect_type"].eq(
        "suspicious_enrollment"
    )
    count = int(mask.sum())

    anomalies.loc[mask, "enrollment"] = rng.uniform(
        600, 1200, count
    )

    # Defect 3: long trial duration.
    mask = anomalies["defect_type"].eq("long_duration")
    count = int(mask.sum())

    anomalies.loc[mask, "duration_days"] = rng.uniform(
        1000, 1800, count
    )

    # Defect 4: missing numeric information.
    mask = anomalies["defect_type"].eq(
        "missing_numeric_value"
    )
    indices = anomalies.index[mask].to_numpy()

    for i, idx in enumerate(indices):
        if i % 3 == 0:
            anomalies.loc[idx, "enrollment"] = np.nan
        elif i % 3 == 1:
            anomalies.loc[idx, "minimum_age"] = np.nan
        else:
            anomalies.loc[idx, "maximum_age"] = np.nan

    # Combine normal and anomalous records.
    df = pd.concat(
        [normal, anomalies],
        ignore_index=True,
    )

    # Synthetic ground-truth labels:
    # 0 = normal, 1 = injected defect.
    df["true_anomaly"] = (
        df["defect_type"] != "normal"
    ).astype(int)

    df = df.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------
# 3. Prepare features
# ---------------------------------------------------------

def prepare_features(
    df,
    imputer=None,
    scaler=None,
    fit=False,
):
    """
    Impute numeric values, scale them, and retain missingness
    indicators so that missing values remain detectable.
    """

    raw = df[RAW_FEATURES].copy()

    # Record which values were originally missing.
    indicators = pd.DataFrame(index=df.index)

    indicators["enrollment_missing"] = (
        raw["enrollment"].isna().astype(int)
    )
    indicators["minimum_age_missing"] = (
        raw["minimum_age"].isna().astype(int)
    )
    indicators["maximum_age_missing"] = (
        raw["maximum_age"].isna().astype(int)
    )

    if fit:
        imputer = SimpleImputer(strategy="median")
        imputed = imputer.fit_transform(raw)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(imputed)
    else:
        if imputer is None or scaler is None:
            raise ValueError(
                "Pass fitted imputer and scaler when fit=False."
            )

        imputed = imputer.transform(raw)
        scaled = scaler.transform(imputed)

    scaled_df = pd.DataFrame(
        scaled,
        columns=RAW_FEATURES,
        index=df.index,
    )

    features = pd.concat(
        [scaled_df, indicators],
        axis=1,
    )

    # Keep the expected feature order consistent.
    features = features[FEATURES]

    return features, imputer, scaler


# ---------------------------------------------------------
# 4. Evaluate one model
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics and confusion counts."""

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "test_records": len(y_true),
        "flagged_records": int(np.sum(y_pred)),
    }


# ---------------------------------------------------------
# 5. Evaluate all models and hybrid configurations
# ---------------------------------------------------------

def evaluate_one_seed(random_state=42):
    """Evaluate three ML detectors and three hybrid detectors."""

    df = create_benchmark(
        n_samples=5000,
        anomaly_fraction=0.05,
        random_state=random_state,
    )

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.20,
        random_state=random_state,
        stratify=df["defect_type"],
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Fit preprocessing only on normal training records.
    train_normal = train_df.loc[
        train_df["true_anomaly"] == 0
    ].reset_index(drop=True)

    X_train, imputer, scaler = prepare_features(
        train_normal,
        fit=True,
    )

    X_test, _, _ = prepare_features(
        test_df,
        imputer=imputer,
        scaler=scaler,
        fit=False,
    )

    y_test = test_df["true_anomaly"].to_numpy()

    # Convert both train and test data to arrays so sklearn
    # does not produce feature-name warnings.
    X_train_array = X_train.to_numpy()
    X_test_array = X_test.to_numpy()

    models = {
        "Isolation Forest": IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=random_state,
        ),
        "One-Class SVM": OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.05,
        ),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.05,
            novelty=True,
        ),
    }

    overall_results = []
    defect_results = []
    prediction_frames = []

    # -----------------------------------------------------
    # Explicit validation rules
    # -----------------------------------------------------

    # These rules do not inspect true_anomaly or defect_type.
    # They inspect only the observed feature values.
    invalid_age_range = (
        test_df["minimum_age"].notna()
        & test_df["maximum_age"].notna()
        & (
            test_df["minimum_age"]
            > test_df["maximum_age"]
        )
    )

    missing_numeric_value = (
        test_df[RAW_FEATURES].isna().any(axis=1)
    )

    rule_flags = (
        invalid_age_range | missing_numeric_value
    ).astype(int).to_numpy()

    # -----------------------------------------------------
    # Evaluate individual models and hybrid combinations
    # -----------------------------------------------------

    for name, model in models.items():

        # Fit only on normal training records.
        model.fit(X_train_array)

        # ML-only prediction: 1 means anomaly.
        y_pred = (
            model.predict(X_test_array) == -1
        ).astype(int)

        # Larger score means more unusual.
        scores = -model.decision_function(
            X_test_array
        )

        # Save ML-only metrics.
        metrics = calculate_metrics(
            y_test,
            y_pred,
        )

        overall_results.append({
            "seed": random_state,
            "model": name,
            **metrics,
            "average_precision": average_precision_score(
                y_test, scores
            ),
            "roc_auc": roc_auc_score(
                y_test, scores
            ),
        })

        # Record ML-only detection rates for each defect type.
        for defect_type in DEFECT_TYPES:
            mask = (
                test_df["defect_type"]
                .eq(defect_type)
                .to_numpy()
            )

            total = int(mask.sum())
            detected = int(y_pred[mask].sum())

            defect_results.append({
                "seed": random_state,
                "model": name,
                "defect_type": defect_type,
                "test_records_of_type": total,
                "detected": detected,
                "missed": total - detected,
                "detection_rate": (
                    detected / total if total else np.nan
                ),
            })

        prediction_frames.append(pd.DataFrame({
            "seed": random_state,
            "true_anomaly": y_test,
            "defect_type": test_df[
                "defect_type"
            ].to_numpy(),
            "predicted_anomaly": y_pred,
            "anomaly_score": scores,
            "model": name,
        }))

        # -------------------------------------------------
        # Hybrid: explicit rules OR ML prediction
        # -------------------------------------------------

        hybrid_pred = (
            (rule_flags == 1) | (y_pred == 1)
        ).astype(int)

        hybrid_metrics = calculate_metrics(
            y_test,
            hybrid_pred,
        )

        hybrid_name = f"Hybrid + {name}"

        overall_results.append({
            "seed": random_state,
            "model": hybrid_name,
            **hybrid_metrics,

            # Binary union predictions do not provide a
            # calibrated continuous anomaly score.
            "average_precision": np.nan,
            "roc_auc": np.nan,
        })

        for defect_type in DEFECT_TYPES:
            mask = (
                test_df["defect_type"]
                .eq(defect_type)
                .to_numpy()
            )

            total = int(mask.sum())
            detected = int(hybrid_pred[mask].sum())

            defect_results.append({
                "seed": random_state,
                "model": hybrid_name,
                "defect_type": defect_type,
                "test_records_of_type": total,
                "detected": detected,
                "missed": total - detected,
                "detection_rate": (
                    detected / total if total else np.nan
                ),
            })

        prediction_frames.append(pd.DataFrame({
            "seed": random_state,
            "true_anomaly": y_test,
            "defect_type": test_df[
                "defect_type"
            ].to_numpy(),
            "predicted_anomaly": hybrid_pred,
            "anomaly_score": np.nan,
            "model": hybrid_name,
        }))

    return (
        pd.DataFrame(overall_results),
        pd.DataFrame(defect_results),
        pd.concat(
            prediction_frames,
            ignore_index=True,
        ),
    )


# ---------------------------------------------------------
# 6. Run multiple seeds and save reports
# ---------------------------------------------------------

def evaluate_multiple_seeds(
    seeds=(11, 22, 33, 44, 55),
):
    """Repeat evaluation and summarize results across seeds."""

    overall_frames = []
    defect_frames = []
    prediction_frames = []

    for seed in seeds:
        print(f"\nRunning seed {seed}...")

        overall, defects, predictions = evaluate_one_seed(
            random_state=seed
        )

        overall_frames.append(overall)
        defect_frames.append(defects)
        prediction_frames.append(predictions)

    overall_df = pd.concat(
        overall_frames,
        ignore_index=True,
    )

    defect_df = pd.concat(
        defect_frames,
        ignore_index=True,
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    os.makedirs(REPORT_DIR, exist_ok=True)

    # Save detailed results.
    overall_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "mixed_defect_evaluation_results.csv",
        ),
        index=False,
    )

    defect_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "mixed_defect_detection_by_type.csv",
        ),
        index=False,
    )

    predictions_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "mixed_defect_predictions.csv",
        ),
        index=False,
    )

    # Summary for precision, recall, F1, AP, and ROC-AUC.
    metric_columns = [
        "precision",
        "recall",
        "f1_score",
        "average_precision",
        "roc_auc",
    ]

    summary = overall_df.groupby("model")[
        metric_columns
    ].agg(["mean", "std"])

    summary.round(4).to_csv(
        os.path.join(
            REPORT_DIR,
            "mixed_defect_evaluation_summary.csv",
        )
    )

    # Mean and standard deviation of defect-level detection.
    defect_summary = defect_df.groupby(
        ["model", "defect_type"]
    )["detection_rate"].agg(["mean", "std"])

    defect_summary.round(4).to_csv(
        os.path.join(
            REPORT_DIR,
            "mixed_defect_type_summary.csv",
        )
    )

    # Print the main results.
    print("\n=== Overall Results by Seed ===")
    print(
        overall_df.round(4).to_string(index=False)
    )

    print("\n=== Mean and Standard Deviation ===")
    print(
        summary.round(4).to_string()
    )

    print("\n=== Detection Rate by Defect Type ===")
    print(
        defect_summary.round(4).to_string()
    )

    print("\n=== Reports Created ===")
    print(
        "reports/mixed_defect_evaluation_results.csv"
    )
    print(
        "reports/mixed_defect_detection_by_type.csv"
    )
    print(
        "reports/mixed_defect_predictions.csv"
    )
    print(
        "reports/mixed_defect_evaluation_summary.csv"
    )
    print(
        "reports/mixed_defect_type_summary.csv"
    )

    print(
        "\nNote: These metrics apply only to this "
        "synthetic benchmark."
    )


# ---------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    evaluate_multiple_seeds()


from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "claim_model_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "claim_anomaly_results.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "claim_anomaly_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "claim_anomaly_metrics.json"

NUMERIC_FEATURES = ["claim_amount", "annual_premium", "age"]
CATEGORICAL_FEATURES = ["product", "customer_type"]
REQUIRED_COLUMNS = [
    "claim_id",
    "claim_amount",
    "annual_premium",
    "age",
    "product",
    "customer_type",
]


def main():
    print("=" * 60)
    print("InsureData360 AI - Unusual Claim Detection")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Modelling dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("The modelling dataset is empty.")

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError(
            "Required columns contain missing values. "
            "Resolve them before anomaly detection."
        )

    numeric_values = df[NUMERIC_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Numeric features contain non-finite values."
        )

    if (df["claim_amount"] < 0).any():
        raise ValueError(
            "Negative claim amounts were found. "
            "Check ETL validation before proceeding."
        )

    # Only use claim and policy characteristics as model features.
    # Claim ID is an identifier, not a meaningful numeric feature.
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=200,
                    contamination=0.05,
                    random_state=42,
                ),
            ),
        ]
    )

    print(f"\nDataset records: {len(df)}")
    print("Features:", NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    print("Expected anomaly proportion: 5% (configured setting)")
    print("\nFitting Isolation Forest...")

    pipeline.fit(X)

    # Isolation Forest returns 1 for inliers and -1 for outliers.
    model = pipeline.named_steps["isolation_forest"]
    transformed_X = pipeline.named_steps["preprocessor"].transform(X)

    predictions = model.predict(transformed_X)
    anomaly_scores = model.decision_function(transformed_X)

    results = df.copy()
    results["anomaly_score"] = anomaly_scores
    results["anomaly_label"] = np.where(
        predictions == -1,
        "Unusual - Review",
        "Typical",
    )

    # Lower decision-function scores indicate more unusual observations.
    results["review_priority"] = results["anomaly_score"].rank(
        method="first",
        ascending=True,
    ).astype(int)

    results = results.sort_values(
        "anomaly_score",
        ascending=True,
    ).reset_index(drop=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    results.to_csv(OUTPUT_PATH, index=False)

    anomaly_count = int((predictions == -1).sum())
    typical_count = int((predictions == 1).sum())

    metrics = {
        "algorithm": "IsolationForest",
        "input_records": int(len(df)),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "configured_contamination": 0.05,
        "unusual_records": anomaly_count,
        "typical_records": typical_count,
        "observed_unusual_percentage": round(
            anomaly_count / len(df) * 100, 2
        ),
        "interpretation": (
            "Unusual records require investigation. "
            "Anomaly detection does not establish fraud."
        ),
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n" + "-" * 60)
    print("DETECTION SUMMARY")
    print("-" * 60)
    print(f"Total records: {len(df)}")
    print(f"Typical records: {typical_count}")
    print(f"Unusual records: {anomaly_count}")
    print(
        "Observed unusual percentage: "
        f"{metrics['observed_unusual_percentage']}%"
    )

    print("\nFive highest-priority records for review:")
    print(
        results[
            [
                "claim_id",
                "claim_amount",
                "product",
                "anomaly_score",
                "anomaly_label",
            ]
        ].head(5).to_string(index=False)
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Results saved to: {OUTPUT_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print("\nAnomaly detection completed successfully.")


if __name__ == "__main__":
    main()

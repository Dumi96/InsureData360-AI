
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "claim_model_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "claim_severity_model.joblib"
METRICS_FILE = MODEL_DIR / "claim_severity_metrics.json"

TARGET = "claim_amount"

NUMERIC_FEATURES = [
    "annual_premium",
    "age",
]

CATEGORICAL_FEATURES = [
    "product",
    "customer_type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main():
    # --------------------------------------------------
    # 2. Load and validate the dataset
    # --------------------------------------------------

    print("=" * 60)
    print("InsureData360 AI - Claim Severity Model")
    print("=" * 60)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run prepare_claim_data.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Missing values found in model features or target."
        )

    if len(df) < 20:
        raise ValueError(
            "Not enough records for this train/test demonstration."
        )

    if not np.isfinite(
        df[NUMERIC_FEATURES + [TARGET]].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite numeric values found in the dataset."
        )

    if (df[TARGET] < 0).any():
        raise ValueError(
            "Negative claim amounts found."
        )

    # X = input features; y = value we want to predict.
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    print(f"\nDataset rows: {len(df)}")
    print(f"Input features: {FEATURES}")
    print(f"Target: {TARGET}")

    # --------------------------------------------------
    # 3. Split before fitting preprocessing or model
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"\nTraining records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # --------------------------------------------------
    # 4. Prepare numeric and categorical features
    # --------------------------------------------------

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    # Preprocessing is fitted on training data only.
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("regressor", LinearRegression()),
        ]
    )

    # --------------------------------------------------
    # 5. Train the model
    # --------------------------------------------------

    print("\nTraining Linear Regression...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # --------------------------------------------------
    # 6. Compare with a simple baseline
    # --------------------------------------------------

    # Predict the training-set mean for every test record.
    baseline_predictions = np.full(
        shape=len(y_test),
        fill_value=y_train.mean(),
    )

    def calculate_metrics(actual, predicted):
        return {
            "mae": round(
                float(mean_absolute_error(actual, predicted)), 2
            ),
            "rmse": round(
                float(np.sqrt(mean_squared_error(actual, predicted))),
                2,
            ),
            "r2": round(
                float(r2_score(actual, predicted)), 4
            ),
        }

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_predictions,
    )

    model_metrics = calculate_metrics(
        y_test,
        predictions,
    )

    print("\n" + "-" * 60)
    print("TEST SET RESULTS")
    print("-" * 60)

    print("\nBaseline (training-set mean prediction):")
    print(f"MAE:  {baseline_metrics['mae']:,.2f}")
    print(f"RMSE: {baseline_metrics['rmse']:,.2f}")
    print(f"R2:   {baseline_metrics['r2']}")

    print("\nLinear Regression:")
    print(f"MAE:  {model_metrics['mae']:,.2f}")
    print(f"RMSE: {model_metrics['rmse']:,.2f}")
    print(f"R2:   {model_metrics['r2']}")

    if model_metrics["mae"] < baseline_metrics["mae"]:
        print("\nThe model has lower MAE than the baseline.")
    else:
        print("\nThe model did not improve MAE over the baseline.")

    # --------------------------------------------------
    # 7. Save model and evaluation metrics
    # --------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    results = {
        "model_name": "Linear Regression",
        "target": TARGET,
        "features": FEATURES,
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "random_state": 42,
        "baseline": baseline_metrics,
        "linear_regression": model_metrics,
        "note": (
            "Synthetic demonstration data. "
            "Not validated for real insurance decisions."
        ),
    }

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"\nModel saved to: {MODEL_FILE}")
    print(f"Metrics saved to: {METRICS_FILE}")
    print("\nTraining and evaluation completed.")


if __name__ == "__main__":
    main()

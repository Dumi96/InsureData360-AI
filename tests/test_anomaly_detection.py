
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "src" / "ml" / "detect_claim_anomalies.py"

spec = spec_from_file_location("detect_claim_anomalies", SCRIPT_PATH)
anomaly_module = module_from_spec(spec)
spec.loader.exec_module(anomaly_module)


REQUIRED_COLUMNS = [
    "claim_id",
    "claim_amount",
    "annual_premium",
    "age",
    "product",
    "customer_type",
]


def sample_claims(count=40):
    """Create test claims without using real project data."""
    return pd.DataFrame(
        {
            "claim_id": range(1001, 1001 + count),
            "claim_amount": [
                1000.0 + (i * 250.0) for i in range(count)
            ],
            "annual_premium": [
                5000.0 + (i * 100.0) for i in range(count)
            ],
            "age": [25 + (i % 45) for i in range(count)],
            "product": [
                ["Motor", "Life", "Property", "Health"][i % 4]
                for i in range(count)
            ],
            "customer_type": [
                ["Individual", "Corporate"][i % 2]
                for i in range(count)
            ],
        }
    )


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Redirect all input and output files to a temporary directory."""
    input_path = tmp_path / "claim_model_data.csv"
    output_path = tmp_path / "claim_anomaly_results.csv"
    model_path = tmp_path / "claim_anomaly_model.joblib"
    metrics_path = tmp_path / "claim_anomaly_metrics.json"

    monkeypatch.setattr(anomaly_module, "INPUT_PATH", input_path)
    monkeypatch.setattr(anomaly_module, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(anomaly_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(anomaly_module, "METRICS_PATH", metrics_path)

    return {
        "input": input_path,
        "output": output_path,
        "model": model_path,
        "metrics": metrics_path,
    }


def test_missing_required_columns_are_rejected(isolated_paths):
    pd.DataFrame({"claim_id": [1]}).to_csv(
        isolated_paths["input"], index=False
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        anomaly_module.main()


def test_empty_dataset_is_rejected(isolated_paths):
    pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(
        isolated_paths["input"], index=False
    )

    with pytest.raises(ValueError, match="dataset is empty"):
        anomaly_module.main()


def test_missing_required_values_are_rejected(isolated_paths):
    df = sample_claims()
    df.loc[0, "claim_amount"] = np.nan
    df.to_csv(isolated_paths["input"], index=False)

    with pytest.raises(ValueError, match="missing values"):
        anomaly_module.main()


def test_non_finite_numeric_values_are_rejected(isolated_paths):
    df = sample_claims()
    df["age"] = df["age"].astype(float)
    df.loc[0, "age"] = np.inf
    df.to_csv(isolated_paths["input"], index=False)

    with pytest.raises(ValueError, match="non-finite values"):
        anomaly_module.main()


def test_negative_claim_amounts_are_rejected(isolated_paths):
    df = sample_claims()
    df.loc[0, "claim_amount"] = -100.0
    df.to_csv(isolated_paths["input"], index=False)

    with pytest.raises(ValueError, match="Negative claim amounts"):
        anomaly_module.main()


def test_valid_data_creates_model_results_and_metrics(isolated_paths):
    df = sample_claims()
    df.to_csv(isolated_paths["input"], index=False)

    anomaly_module.main()

    assert isolated_paths["model"].is_file()
    assert isolated_paths["output"].is_file()
    assert isolated_paths["metrics"].is_file()

    results = pd.read_csv(isolated_paths["output"])

    assert len(results) == len(df)
    assert results["claim_id"].is_unique
    assert results["anomaly_score"].notna().all()
    assert np.isfinite(results["anomaly_score"]).all()

    assert set(results["anomaly_label"].unique()).issubset(
        {"Typical", "Unusual - Review"}
    )

    assert results["review_priority"].is_unique
    assert sorted(results["review_priority"].tolist()) == list(
        range(1, len(df) + 1)
    )

    metrics = __import__("json").loads(
        isolated_paths["metrics"].read_text(encoding="utf-8")
    )

    assert metrics["input_records"] == len(df)
    assert (
        metrics["unusual_records"] + metrics["typical_records"]
        == len(df)
    )
    assert metrics["algorithm"] == "IsolationForest"

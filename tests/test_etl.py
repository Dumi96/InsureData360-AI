
"""Integration tests for the InsureData360 AI ETL pipeline.

These tests use synthetic CSV data in the project folders.
They rerun the ETL pipeline before checking its output.
"""

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


# Project root: the parent folder of tests/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ETL_SCRIPT = PROJECT_ROOT / "src" / "etl" / "run_etl.py"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REJECTED_DIR = PROJECT_ROOT / "data" / "rejected"


@pytest.fixture(scope="session")
def etl_run():
    """Run the ETL pipeline once before the integration tests."""

    result = subprocess.run(
        [sys.executable, str(ETL_SCRIPT)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    combined_output = result.stdout + "\n" + result.stderr

    assert result.returncode == 0, (
        "ETL pipeline failed.\n"
        f"Exit code: {result.returncode}\n"
        f"Output:\n{combined_output}"
    )

    assert "ETL SUCCESS" in combined_output, (
        "ETL success message was not found.\n"
        f"Output:\n{combined_output}"
    )

    return combined_output


def read_csv(directory, filename):
    """Read an output CSV and give a clear error if it is missing."""

    file_path = directory / filename

    assert file_path.exists(), f"Expected output file is missing: {file_path}"

    return pd.read_csv(file_path)


def test_etl_pipeline_completes_successfully(etl_run):
    """The ETL process should finish without an execution error."""

    assert "ETL SUCCESS" in etl_run


def test_reconciliation_accounts_for_every_input_row(etl_run):
    """Input rows must equal accepted rows plus rejected rows."""

    report = read_csv(
        PROCESSED_DIR,
        "reconciliation_report.csv",
    )

    required_columns = {
        "table_name",
        "input_rows",
        "accepted_rows",
        "rejected_rows",
        "reconciliation_difference",
        "status",
    }

    assert required_columns.issubset(report.columns)
    assert len(report) == 4

    for _, row in report.iterrows():
        assert row["input_rows"] == (
            row["accepted_rows"] + row["rejected_rows"]
        )

    assert (report["reconciliation_difference"] == 0).all()
    assert (report["status"] == "PASS").all()


def test_duplicate_policy_ids_are_rejected(etl_run):
    """Both records sharing the injected duplicate policy ID are rejected."""

    rejected = read_csv(
        REJECTED_DIR,
        "policies_rejected.csv",
    )

    duplicates = rejected[rejected["policy_id"] == 1001]

    assert len(duplicates) == 2
    assert duplicates["rejection_reason"].str.contains(
        "Duplicate policy_id",
        case=False,
        na=False,
    ).all()


def test_invalid_claims_are_rejected(etl_run):
    """Negative claim amounts and unknown statuses must be rejected."""

    rejected = read_csv(
        REJECTED_DIR,
        "claims_rejected.csv",
    )

    negative_claim = rejected[rejected["claim_id"] == 8001]
    unknown_status_claim = rejected[rejected["claim_id"] == 8002]

    assert len(negative_claim) == 1
    assert "claim_amount" in negative_claim.iloc[0]["rejection_reason"]

    assert len(unknown_status_claim) == 1
    assert "claim_status" in unknown_status_claim.iloc[0]["rejection_reason"]


def test_payment_for_nonexistent_policy_is_rejected(etl_run):
    """A payment referencing an unknown policy must not be accepted."""

    rejected = read_csv(
        REJECTED_DIR,
        "premium_payments_rejected.csv",
    )

    invalid_payment = rejected[rejected["policy_id"] == 999999]

    assert len(invalid_payment) >= 1
    assert invalid_payment["rejection_reason"].str.contains(
        "policy_id",
        case=False,
        na=False,
    ).all()


def test_accepted_data_has_no_invalid_amounts(etl_run):
    """Accepted premiums, payment amounts and claim amounts must be nonnegative."""

    policies = read_csv(PROCESSED_DIR, "policies.csv")
    payments = read_csv(PROCESSED_DIR, "premium_payments.csv")
    claims = read_csv(PROCESSED_DIR, "claims.csv")

    assert (policies["annual_premium"] >= 0).all()
    assert (payments["payment_amount"] >= 0).all()
    assert (claims["claim_amount"] >= 0).all()


def test_accepted_policy_ids_are_unique(etl_run):
    """The processed policies must not contain duplicate policy IDs."""

    policies = read_csv(PROCESSED_DIR, "policies.csv")

    assert policies["policy_id"].is_unique

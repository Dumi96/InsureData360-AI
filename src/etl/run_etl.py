
"""
InsureData360 AI — ETL Pipeline

Features:
- CSV extraction and schema validation
- Data quality checks and rejection reasons
- Duplicate policy ID detection
- Referential integrity validation
- Logging and reconciliation
- Staged output generation
"""

from __future__ import annotations

import logging
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# =========================================================
# 1. CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REJECTED_DIR = PROJECT_ROOT / "data" / "rejected"
LOG_DIR = PROJECT_ROOT / "logs"

RUN_ID = uuid.uuid4().hex[:12]
RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()

ALLOWED_CUSTOMER_TYPES = {"Individual", "Corporate"}
ALLOWED_PRODUCTS = {"Motor", "Life", "Health", "Property"}
ALLOWED_POLICY_STATUSES = {"Active", "Expired", "Cancelled"}
ALLOWED_PAYMENT_STATUSES = {"Paid", "Pending", "Failed"}
ALLOWED_CLAIM_STATUSES = {
    "Submitted",
    "Under Review",
    "Approved",
    "Rejected",
    "Paid",
}

SCHEMAS = {
    "customers": [
        "customer_id", "age", "city", "customer_type"
    ],
    "policies": [
        "policy_id", "customer_id", "product",
        "annual_premium", "policy_status"
    ],
    "premium_payments": [
        "payment_id", "policy_id",
        "payment_amount", "payment_status"
    ],
    "claims": [
        "claim_id", "policy_id", "claim_date",
        "claim_amount", "claim_status"
    ],
}


# =========================================================
# 2. LOGGING
# =========================================================

def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("insuredata360.etl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_DIR / "etl.log",
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


# =========================================================
# 3. EXTRACTION AND SCHEMA VALIDATION
# =========================================================

def extract_csv(table_name: str) -> pd.DataFrame:
    """Read a CSV and validate its required columns."""

    if table_name not in SCHEMAS:
        raise ValueError(f"Unknown table: {table_name}")

    file_path = RAW_DIR / f"{table_name}.csv"

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required input file not found: {file_path}"
        )

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if df.columns.duplicated().any():
        duplicate_columns = df.columns[
            df.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"{table_name}.csv contains duplicate column names: "
            f"{duplicate_columns}"
        )

    missing_columns = (
        set(SCHEMAS[table_name]) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{table_name}.csv is missing columns: "
            f"{sorted(missing_columns)}"
        )

    logger.info(
        "EXTRACT | table=%s | input_rows=%d",
        table_name,
        len(df)
    )

    return df


# =========================================================
# 4. COMMON VALIDATION HELPERS
# =========================================================

def valid_positive_integer(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    return (
        numeric.notna()
        & numeric.gt(0)
        & numeric.mod(1).eq(0)
    )


def valid_positive_amount(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna() & numeric.gt(0)


def valid_nonempty(series: pd.Series) -> pd.Series:
    return (
        series.notna()
        & series.astype("string").str.strip().ne("")
    )


def add_rejection_reasons(
    df: pd.DataFrame,
    table_name: str,
    rules: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate valid records from invalid records."""

    working = df.copy()
    reasons_by_position = [[] for _ in range(len(working))]

    for column, rule in rules.items():
        try:
            results = rule(working)
        except Exception as exc:
            raise RuntimeError(
                f"Validation failed for {table_name}.{column}: {exc}"
            ) from exc

        results = pd.Series(
            results, index=working.index
        ).fillna(False).astype(bool)

        for position, is_valid in enumerate(results.tolist()):
            if not is_valid:
                reasons_by_position[position].append(
                    f"Invalid {column}"
                )

    working["rejection_reason"] = [
        "; ".join(reasons)
        for reasons in reasons_by_position
    ]

    rejected_mask = working["rejection_reason"].ne("")

    clean = working.loc[~rejected_mask].copy()
    rejected = working.loc[rejected_mask].copy()

    logger.info(
        "VALIDATE | table=%s | accepted=%d | rejected=%d",
        table_name,
        len(clean),
        len(rejected)
    )

    return clean, rejected


# =========================================================
# 5. CUSTOMER VALIDATION
# =========================================================

def validate_customers(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    working = df.copy()

    working["customer_id"] = pd.to_numeric(
        working["customer_id"], errors="coerce"
    )
    working["age"] = pd.to_numeric(
        working["age"], errors="coerce"
    )

    rules = {
        "customer_id": lambda d: valid_positive_integer(
            d["customer_id"]
        ),
        "age": lambda d: (
            d["age"].notna()
            & d["age"].between(18, 100)
        ),
        "city": lambda d: valid_nonempty(d["city"]),
        "customer_type": lambda d: d["customer_type"].isin(
            ALLOWED_CUSTOMER_TYPES
        ),
    }

    clean, rejected = add_rejection_reasons(
        working, "customers", rules
    )

    # Reject every record associated with a duplicate valid ID.
    duplicate_mask = (
        clean["customer_id"].duplicated(keep=False)
    )

    duplicate_rows = clean.loc[duplicate_mask].copy()
    clean = clean.loc[~duplicate_mask].copy()

    if not duplicate_rows.empty:
        duplicate_rows["rejection_reason"] = "Duplicate customer_id"
        rejected = pd.concat(
            [rejected, duplicate_rows], ignore_index=True
        )

    return clean, rejected


# =========================================================
# 6. POLICY VALIDATION
# =========================================================

def validate_policies(
    df: pd.DataFrame,
    valid_customer_ids: set,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    working = df.copy()

    working["policy_id"] = pd.to_numeric(
        working["policy_id"], errors="coerce"
    )
    working["customer_id"] = pd.to_numeric(
        working["customer_id"], errors="coerce"
    )
    working["annual_premium"] = pd.to_numeric(
        working["annual_premium"], errors="coerce"
    )

    # Identify duplicate valid IDs before other validation rules.
    duplicate_mask = (
        working["policy_id"].notna()
        & working["policy_id"].gt(0)
        & working["policy_id"].duplicated(keep=False)
    )

    duplicate_ids = set(
        working.loc[duplicate_mask, "policy_id"].tolist()
    )

    rules = {
        "policy_id": lambda d: valid_positive_integer(
            d["policy_id"]
        ),
        "customer_id": lambda d: (
            valid_positive_integer(d["customer_id"])
            & d["customer_id"].isin(valid_customer_ids)
        ),
        "product": lambda d: d["product"].isin(
            ALLOWED_PRODUCTS
        ),
        "annual_premium": lambda d: valid_positive_amount(
            d["annual_premium"]
        ),
        "policy_status": lambda d: d["policy_status"].isin(
            ALLOWED_POLICY_STATUSES
        ),
    }

    clean, rejected = add_rejection_reasons(
        working, "policies", rules
    )

    # Reject valid rows if their policy ID occurs more than once.
    duplicate_clean_mask = clean["policy_id"].isin(duplicate_ids)
    duplicate_clean = clean.loc[duplicate_clean_mask].copy()
    clean = clean.loc[~duplicate_clean_mask].copy()

    if not duplicate_clean.empty:
        duplicate_clean["rejection_reason"] = "Duplicate policy_id"

    # Add duplicate reason to records already rejected by another rule.
    duplicate_rejected_mask = rejected["policy_id"].isin(
        duplicate_ids
    )

    for index in rejected.index[duplicate_rejected_mask]:
        existing_reason = rejected.at[index, "rejection_reason"]

        if "Duplicate policy_id" not in existing_reason:
            rejected.at[index, "rejection_reason"] = (
                f"{existing_reason}; Duplicate policy_id"
                if existing_reason
                else "Duplicate policy_id"
            )

    rejected = pd.concat(
        [rejected, duplicate_clean], ignore_index=True
    )

    logger.info(
        "DUPLICATES | table=policies | duplicate_ids=%d "
        "| additional_records_rejected=%d",
        len(duplicate_ids),
        len(duplicate_clean)
    )

    return clean, rejected


# =========================================================
# 7. PREMIUM PAYMENT VALIDATION
# =========================================================

def validate_payments(
    df: pd.DataFrame,
    valid_policy_ids: set,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    working = df.copy()

    working["payment_id"] = pd.to_numeric(
        working["payment_id"], errors="coerce"
    )
    working["policy_id"] = pd.to_numeric(
        working["policy_id"], errors="coerce"
    )
    working["payment_amount"] = pd.to_numeric(
        working["payment_amount"], errors="coerce"
    )

    rules = {
        "payment_id": lambda d: valid_positive_integer(
            d["payment_id"]
        ),
        "policy_id": lambda d: (
            valid_positive_integer(d["policy_id"])
            & d["policy_id"].isin(valid_policy_ids)
        ),
        "payment_amount": lambda d: valid_positive_amount(
            d["payment_amount"]
        ),
        "payment_status": lambda d: d["payment_status"].isin(
            ALLOWED_PAYMENT_STATUSES
        ),
    }

    clean, rejected = add_rejection_reasons(
        working, "premium_payments", rules
    )

    duplicate_mask = clean["payment_id"].duplicated(
        keep=False
    )
    duplicate_rows = clean.loc[duplicate_mask].copy()
    clean = clean.loc[~duplicate_mask].copy()

    if not duplicate_rows.empty:
        duplicate_rows["rejection_reason"] = "Duplicate payment_id"
        rejected = pd.concat(
            [rejected, duplicate_rows], ignore_index=True
        )

    return clean, rejected


# =========================================================
# 8. CLAIM VALIDATION
# =========================================================

def validate_claims(
    df: pd.DataFrame,
    valid_policy_ids: set,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    working = df.copy()

    working["claim_id"] = pd.to_numeric(
        working["claim_id"], errors="coerce"
    )
    working["policy_id"] = pd.to_numeric(
        working["policy_id"], errors="coerce"
    )
    working["claim_amount"] = pd.to_numeric(
        working["claim_amount"], errors="coerce"
    )

    parsed_dates = pd.to_datetime(
        working["claim_date"], errors="coerce"
    )

    working["claim_date"] = parsed_dates.dt.strftime(
        "%Y-%m-%d"
    )
    working.loc[parsed_dates.isna(), "claim_date"] = pd.NA

    rules = {
        "claim_id": lambda d: valid_positive_integer(
            d["claim_id"]
        ),
        "policy_id": lambda d: (
            valid_positive_integer(d["policy_id"])
            & d["policy_id"].isin(valid_policy_ids)
        ),
        "claim_amount": lambda d: valid_positive_amount(
            d["claim_amount"]
        ),
        "claim_status": lambda d: d["claim_status"].isin(
            ALLOWED_CLAIM_STATUSES
        ),
        "claim_date": lambda d: valid_nonempty(d["claim_date"]),
    }

    clean, rejected = add_rejection_reasons(
        working, "claims", rules
    )

    duplicate_mask = clean["claim_id"].duplicated(
        keep=False
    )
    duplicate_rows = clean.loc[duplicate_mask].copy()
    clean = clean.loc[~duplicate_mask].copy()

    if not duplicate_rows.empty:
        duplicate_rows["rejection_reason"] = "Duplicate claim_id"
        rejected = pd.concat(
            [rejected, duplicate_rows], ignore_index=True
        )

    return clean, rejected


# =========================================================
# 9. RECONCILIATION
# =========================================================

def reconcile(
    input_tables: dict,
    clean_tables: dict,
    rejected_tables: dict,
) -> pd.DataFrame:

    report_rows = []

    for table_name, input_df in input_tables.items():
        input_count = len(input_df)
        accepted_count = len(clean_tables[table_name])
        rejected_count = len(rejected_tables[table_name])

        difference = (
            input_count - accepted_count - rejected_count
        )

        status = "PASS" if difference == 0 else "FAIL"

        report_rows.append({
            "table_name": table_name,
            "input_rows": input_count,
            "accepted_rows": accepted_count,
            "rejected_rows": rejected_count,
            "reconciliation_difference": difference,
            "status": status,
            "run_id": RUN_ID,
            "run_timestamp_utc": RUN_TIMESTAMP,
        })

        logger.info(
            "RECONCILIATION | table=%s | input=%d "
            "| accepted=%d | rejected=%d | difference=%d | status=%s",
            table_name,
            input_count,
            accepted_count,
            rejected_count,
            difference,
            status
        )

    report = pd.DataFrame(report_rows)

    if not report["status"].eq("PASS").all():
        raise RuntimeError(
            "Reconciliation failed; outputs were not published."
        )

    return report


# =========================================================
# 10. OUTPUT PUBLISHING
# =========================================================

def write_csv(df: pd.DataFrame, destination: Path) -> None:
    """Write to a temporary file before replacing the destination."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(
        destination.name + ".tmp"
    )

    try:
        df.to_csv(temporary_path, index=False)
        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def publish_outputs(
    clean_tables: dict,
    rejected_tables: dict,
    report: pd.DataFrame,
) -> None:
    """
    Stage all outputs and publish after successful validation.

    Individual file replacements are atomic on the same filesystem,
    but the whole collection of files is not one atomic transaction.
    """

    staging_dir = PROJECT_ROOT / "data" / f".etl_staging_{RUN_ID}"
    staging_processed = staging_dir / "processed"
    staging_rejected = staging_dir / "rejected"

    staging_processed.mkdir(parents=True, exist_ok=False)
    staging_rejected.mkdir(parents=True, exist_ok=False)

    try:
        # Build every output file in staging first.
        for table_name, df in clean_tables.items():
            output_df = df.drop(
                columns=["rejection_reason"], errors="ignore"
            )
            output_df.to_csv(
                staging_processed / f"{table_name}.csv",
                index=False
            )

        for table_name, df in rejected_tables.items():
            df.to_csv(
                staging_rejected / f"{table_name}_rejected.csv",
                index=False
            )

        report.to_csv(
            staging_processed / "reconciliation_report.csv",
            index=False
        )

        # Publish the staged files.
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        REJECTED_DIR.mkdir(parents=True, exist_ok=True)

        for source in staging_processed.glob("*.csv"):
            destination = PROCESSED_DIR / source.name
            write_csv(pd.read_csv(source), destination)

        for source in staging_rejected.glob("*.csv"):
            destination = REJECTED_DIR / source.name
            write_csv(pd.read_csv(source), destination)

        logger.info("OUTPUT | files published successfully")

    except Exception:
        logger.exception(
            "OUTPUT PUBLISHING FAILED | staging=%s",
            staging_dir
        )
        # Keep staging data for investigation.
        raise

    else:
        shutil.rmtree(staging_dir)


# =========================================================
# 11. MAIN ETL ORCHESTRATION
# =========================================================

def main() -> int:
    logger.info(
        "ETL START | run_id=%s | timestamp_utc=%s",
        RUN_ID,
        RUN_TIMESTAMP
    )

    try:
        # Extract all source files.
        input_tables = {
            table_name: extract_csv(table_name)
            for table_name in SCHEMAS
        }

        # Validate customers first.
        customers_clean, customers_bad = validate_customers(
            input_tables["customers"]
        )

        valid_customer_ids = set(
            customers_clean["customer_id"].dropna().tolist()
        )

        # Validate policies against accepted customers.
        policies_clean, policies_bad = validate_policies(
            input_tables["policies"],
            valid_customer_ids
        )

        valid_policy_ids = set(
            policies_clean["policy_id"].dropna().tolist()
        )

        # Validate payments and claims against accepted policies.
        payments_clean, payments_bad = validate_payments(
            input_tables["premium_payments"],
            valid_policy_ids
        )

        claims_clean, claims_bad = validate_claims(
            input_tables["claims"],
            valid_policy_ids
        )

        clean_tables = {
            "customers": customers_clean,
            "policies": policies_clean,
            "premium_payments": payments_clean,
            "claims": claims_clean,
        }

        rejected_tables = {
            "customers": customers_bad,
            "policies": policies_bad,
            "premium_payments": payments_bad,
            "claims": claims_bad,
        }

        # Never publish if record counts do not reconcile.
        report = reconcile(
            input_tables,
            clean_tables,
            rejected_tables
        )

        # Publish the processed and rejected files.
        publish_outputs(
            clean_tables,
            rejected_tables,
            report
        )

        logger.info(
            "ETL SUCCESS | run_id=%s",
            RUN_ID
        )

        return 0

    except Exception:
        logger.exception(
            "ETL FAILED | run_id=%s",
            RUN_ID
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

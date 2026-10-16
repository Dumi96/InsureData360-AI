
"""Load cleaned synthetic insurance CSV files into PostgreSQL."""

import csv
import os
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import psycopg
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Each table is loaded in foreign-key dependency order.
TABLES = [
    {
        "name": "customers",
        "file": "customers.csv",
        "columns": ["customer_id", "age", "city", "customer_type"],
        "primary_key": "customer_id",
        "integers": {"customer_id", "age"},
        "decimals": set(),
        "dates": set(),
    },
    {
        "name": "policies",
        "file": "policies.csv",
        "columns": [
            "policy_id",
            "customer_id",
            "product",
            "annual_premium",
            "policy_status",
        ],
        "primary_key": "policy_id",
        "integers": {"policy_id", "customer_id"},
        "decimals": {"annual_premium"},
        "dates": set(),
    },
    {
        "name": "premium_payments",
        "file": "premium_payments.csv",
        "columns": [
            "payment_id",
            "policy_id",
            "payment_amount",
            "payment_status",
        ],
        "primary_key": "payment_id",
        "integers": {"payment_id", "policy_id"},
        "decimals": {"payment_amount"},
        "dates": set(),
    },
    {
        "name": "claims",
        "file": "claims.csv",
        "columns": [
            "claim_id",
            "policy_id",
            "claim_date",
            "claim_amount",
            "claim_status",
        ],
        "primary_key": "claim_id",
        "integers": {"claim_id", "policy_id"},
        "decimals": {"claim_amount"},
        "dates": {"claim_date"},
    },
]


def convert_value(column, value, config):
    """Convert CSV text to a suitable Python database value."""
    if value is None or value.strip() == "":
        return None

    if column in config["integers"]:
        return int(value)

    if column in config["decimals"]:
        return Decimal(value)

    if column in config["dates"]:
        return date.fromisoformat(value)

    return value.strip()


def read_csv_records(config):
    """Read a CSV and validate its column names."""
    file_path = PROCESSED_DIR / config["file"]

    if not file_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with file_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {file_path}")

        missing = set(config["columns"]) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{config['file']} is missing columns: {sorted(missing)}"
            )

        records = []
        for line_number, row in enumerate(reader, start=2):
            try:
                records.append(
                    tuple(
                        convert_value(column, row.get(column), config)
                        for column in config["columns"]
                    )
                )
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"Invalid value in {config['file']}, "
                    f"line {line_number}: {error}"
                ) from error

    return records


def load_table(connection, config):
    """Insert records, updating matching primary keys on repeat runs."""
    records = read_csv_records(config)
    columns = config["columns"]
    primary_key = config["primary_key"]

    column_sql = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    update_columns = [column for column in columns if column != primary_key]

    if update_columns:
        update_sql = ", ".join(
            f"{column} = EXCLUDED.{column}"
            for column in update_columns
        )
        conflict_sql = (
            f"ON CONFLICT ({primary_key}) DO UPDATE SET {update_sql}"
        )
    else:
        conflict_sql = f"ON CONFLICT ({primary_key}) DO NOTHING"

    sql = (
        f"INSERT INTO {config['name']} ({column_sql}) "
        f"VALUES ({placeholders}) {conflict_sql}"
    )

    with connection.cursor() as cursor:
        if records:
            cursor.executemany(sql, records)

        cursor.execute(f"SELECT COUNT(*) FROM {config['name']}")
        database_count = cursor.fetchone()[0]

    print(
        f"{config['name']}: "
        f"{len(records)} CSV records processed; "
        f"{database_count} total rows in database."
    )


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    required_settings = [
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ]
    missing_settings = [
        key for key in required_settings
        if not os.getenv(key)
    ]

    if missing_settings:
        raise RuntimeError(
            "Missing database settings in .env: "
            + ", ".join(missing_settings)
        )

    print("InsureData360 AI - PostgreSQL data loader")
    print(f"Processed data directory: {PROCESSED_DIR}")
    print("Connecting to PostgreSQL...")

    connection_settings = {
        "host": os.environ["DB_HOST"],
        "port": int(os.environ["DB_PORT"]),
        "dbname": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
    }

    # The connection context commits all table loads together.
    # If any load fails, the transaction is rolled back.
    with psycopg.connect(**connection_settings) as connection:
        current_database = connection.execute(
            "SELECT current_database()"
        ).fetchone()[0]

        if current_database != "insuredata360":
            raise RuntimeError(
                f"Unexpected database: {current_database}. "
                "Expected insuredata360; no data was loaded."
            )

        print(f"Connected to database: {current_database}")

        for config in TABLES:
            load_table(connection, config)

    print("\nSUCCESS: All four CSV files were loaded.")
    print("The database transaction was committed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        print(
            "The database transaction was rolled back if loading had begun.",
            file=sys.stderr,
        )
        sys.exit(1)

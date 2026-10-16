
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv
import os

# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "claim_model_data.csv"

load_dotenv(PROJECT_ROOT / ".env")


def main():
    query = """
        SELECT
            c.claim_id,
            c.claim_date,
            c.claim_amount,
            p.product,
            p.annual_premium,
            cu.age,
            cu.customer_type
        FROM claims c
        INNER JOIN policies p
            ON c.policy_id = p.policy_id
        INNER JOIN customers cu
            ON p.customer_id = cu.customer_id
        WHERE c.claim_amount IS NOT NULL
          AND c.claim_amount >= 0
          AND cu.age IS NOT NULL
          AND p.annual_premium IS NOT NULL
    """

    connection_settings = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }

    print("Connecting to PostgreSQL...")

    with psycopg.connect(**connection_settings) as connection:
        df = pd.read_sql_query(query, connection)

    if df.empty:
        raise ValueError("No valid claims were found for modelling.")

    if df["claim_id"].duplicated().any():
        raise ValueError("Duplicate claim IDs detected in modelling data.")

    if df["claim_amount"].isna().any():
        raise ValueError("Missing target values detected.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nClaim modelling dataset prepared successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nPreview:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()

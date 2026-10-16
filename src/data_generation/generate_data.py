
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

np.random.seed(42)


def generate_customers(n=1000):
    return pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age": np.random.randint(18, 80, n),
        "city": np.random.choice(
            ["Colombo", "Kandy", "Galle", "Jaffna", "Kurunegala"], n
        ),
        "customer_type": np.random.choice(
            ["Individual", "Corporate"], n, p=[0.85, 0.15]
        ),
    })


def generate_policies(customers, n=1500):
    policies = pd.DataFrame({
        "policy_id": range(1001, 1001 + n),
        "customer_id": np.random.choice(customers["customer_id"], n),
        "product": np.random.choice(
            ["Motor", "Health", "Life", "Property"], n
        ),
        "annual_premium": np.round(
            np.random.uniform(15000, 250000, n), 2
        ),
        "policy_status": np.random.choice(
            ["Active", "Expired", "Cancelled"], n,
            p=[0.70, 0.20, 0.10]
        ),
    })

    # Intentional data-quality issues for ETL testing
    policies.loc[0, "annual_premium"] = -5000
    policies.loc[1, "policy_id"] = policies.loc[0, "policy_id"]

    return policies


def generate_payments(policies, n=1500):
    payments = pd.DataFrame({
        "payment_id": range(5001, 5001 + n),
        "policy_id": np.random.choice(
            policies["policy_id"].drop_duplicates(), n
        ),
        "payment_amount": np.round(
            np.random.uniform(1000, 100000, n), 2
        ),
        "payment_status": np.random.choice(
            ["Paid", "Pending", "Failed"], n,
            p=[0.80, 0.15, 0.05]
        ),
    })

    # Intentional reference to a nonexistent policy
    payments.loc[0, "policy_id"] = 999999

    return payments


def generate_claims(policies, customers, n=500):
    # Select policy IDs for the synthetic claim records.
    selected_policy_ids = np.random.choice(
        policies["policy_id"].drop_duplicates(), n
    )

    # Obtain the policy and customer characteristics used to generate
    # the claim amount. These relationships are synthetic assumptions.
    policy_lookup = policies.drop_duplicates(
        subset=["policy_id"], keep="first"
    )

    claim_features = pd.DataFrame({
        "policy_id": selected_policy_ids
    })

    claim_features = claim_features.merge(
        policy_lookup[
            ["policy_id", "customer_id", "product", "annual_premium"]
        ],
        on="policy_id",
        how="left",
        validate="many_to_one",
    )

    claim_features = claim_features.merge(
        customers[["customer_id", "age", "customer_type"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    # Designed product-level differences for this synthetic dataset.
    product_base_amount = {
        "Health": 100000,
        "Life": 115000,
        "Motor": 140000,
        "Property": 155000,
    }

    product_component = (
        claim_features["product"].map(product_base_amount).to_numpy()
    )

    # Deliberately designed effects:
    # - Higher annual premium modestly increases expected claim amount.
    # - Age has a small effect.
    # - Corporate customers have a modest additional component.
    # - Random noise keeps individual claims unpredictable.
    premium_component = (
        claim_features["annual_premium"].to_numpy() * 0.45
    )

    age_component = (
        (claim_features["age"].to_numpy() - 40) * 600
    )

    customer_type_component = (
        claim_features["customer_type"]
        .map({"Individual": 0, "Corporate": 15000})
        .to_numpy()
    )

    random_variation = np.random.normal(
        loc=0,
        scale=45000,
        size=n,
    )

    claim_amounts = (
        product_component
        + premium_component
        + age_component
        + customer_type_component
        + random_variation
    )

    # Keep generated amounts within the defined synthetic range.
    claim_amounts = np.clip(claim_amounts, 5000, 500000)

    claims = pd.DataFrame({
        "claim_id": range(8001, 8001 + n),
        "policy_id": selected_policy_ids,
        "claim_date": pd.to_datetime(np.random.choice(
            pd.date_range("2024-01-01", "2026-09-30"), n
        )),
        "claim_amount": np.round(claim_amounts, 2),
        "claim_status": np.random.choice(
            ["Submitted", "Under Review", "Approved", "Rejected"], n
        ),
    })

    # Intentional data-quality issues for ETL testing
    claims.loc[0, "claim_amount"] = -10000
    claims.loc[1, "claim_status"] = "UnknownStatus"

    return claims


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    customers = generate_customers()
    policies = generate_policies(customers)
    payments = generate_payments(policies)
    claims = generate_claims(policies, customers)

    datasets = {
        "customers.csv": customers,
        "policies.csv": policies,
        "premium_payments.csv": payments,
        "claims.csv": claims,
    }

    for filename, dataframe in datasets.items():
        output_path = RAW_DATA_DIR / filename
        dataframe.to_csv(output_path, index=False)
        print(f"Created {filename} | Rows: {len(dataframe):,}")

    print("Synthetic insurance data generation completed!")


if __name__ == "__main__":
    main()

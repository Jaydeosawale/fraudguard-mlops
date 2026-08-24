from pathlib import Path

import numpy as np
import pandas as pd


def generate_fraud_data(
    n_samples: int = 10_000,
    fraud_rate: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic transaction data for FraudGuard.
    """

    # Create reproducible random number generator
    rng = np.random.default_rng(random_state)

    # Decide which transactions are fraud
    is_fraud = rng.choice(
        [0, 1],
        size=n_samples,
        p=[1 - fraud_rate, fraud_rate],
    )

    # -----------------------------------------
    # Generate baseline transaction behavior
    # -----------------------------------------

    # Transaction amount
    amount = rng.lognormal(
        mean=7.0,
        sigma=1.0,
        size=n_samples,
    )

    # Number of transactions in last 24 hours
    transactions_last_24h = rng.poisson(
        lam=3,
        size=n_samples,
    )

    # Country risk score between 0 and 1
    country_risk = rng.beta(
        a=2,
        b=8,
        size=n_samples,
    )

    # Transaction hour
    hour = rng.integers(
        0,
        24,
        size=n_samples,
    )

    # Account age in days
    account_age_days = rng.integers(
        1,
        3650,
        size=n_samples,
    )

    # -----------------------------------------
    # Add fraud patterns with realistic overlap
    # -----------------------------------------

    fraud_mask = is_fraud == 1
    n_fraud = fraud_mask.sum()

    # Fraud is often higher value,
    # but some fraud looks similar to normal transactions
    amount[fraud_mask] *= rng.uniform(
        1.2,
        4.0,
        size=n_fraud,
    )

    # Fraud often has more activity,
    # but some fraud has normal activity levels
    transactions_last_24h[fraud_mask] += rng.integers(
        0,
        10,
        size=n_fraud,
    )

    # Fraud tends toward higher country risk,
    # but there is overlap with normal transactions
    country_risk[fraud_mask] = rng.beta(
        a=4,
        b=4,
        size=n_fraud,
    )

    # Fraud tends to involve newer accounts,
    # but fraud can also come from older accounts
    account_age_days[fraud_mask] = rng.integers(
        1,
        1000,
        size=n_fraud,
    )

    # -----------------------------------------
    # Create and return DataFrame
    # -----------------------------------------

    return pd.DataFrame(
        {
            "amount": amount,
            "transactions_last_24h": transactions_last_24h,
            "country_risk": country_risk,
            "hour": hour,
            "account_age_days": account_age_days,
            "is_fraud": is_fraud,
        }
    )


if __name__ == "__main__":
    # Generate synthetic data
    df = generate_fraud_data()

    # Define output location
    output_path = Path("data/raw/fraud_data.csv")

    # Create directories if they do not exist
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save data
    df.to_csv(
        output_path,
        index=False,
    )

    # Display results
    print(f"Generated {len(df)} transactions")
    print(f"Saved to {output_path}")

    print("\nFraud distribution:")
    print(df["is_fraud"].value_counts())
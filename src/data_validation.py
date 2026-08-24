import pandas as pd


REQUIRED_COLUMNS = {
    "amount",
    "transactions_last_24h",
    "country_risk",
    "is_fraud",
}


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate FraudGuard training data.

    Raises:
        ValueError: If the data violates the data contract.
    """

    # 1. Check required columns
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # 2. Check for missing values
    if df[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("Required columns contain missing values")

    # 3. Transaction amount cannot be negative
    if (df["amount"] < 0).any():
        raise ValueError("Negative transaction amount detected")

    # 4. Transaction count cannot be negative
    if (df["transactions_last_24h"] < 0).any():
        raise ValueError("Negative transaction count detected")

    # 5. country_risk must be between 0 and 1
    invalid_risk = (
        (df["country_risk"] < 0)
        | (df["country_risk"] > 1)
    )

    if invalid_risk.any():
        raise ValueError(
            "country_risk must be between 0 and 1"
        )

    # 6. is_fraud must contain only 0 or 1
    if not df["is_fraud"].isin([0, 1]).all():
        raise ValueError(
            "is_fraud must contain only 0 or 1"
        )

    print("✅ Data validation passed")
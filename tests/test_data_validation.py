import pandas as pd
import pytest

from src.data_validation import validate_data


def valid_dataframe():
    return pd.DataFrame({
        "amount": [100.0, 500.0],
        "transactions_last_24h": [2, 5],
        "country_risk": [0.2, 0.8],
        "is_fraud": [0, 1],
    })


def test_valid_data_passes():
    df = valid_dataframe()

    validate_data(df)


def test_negative_amount_fails():
    df = valid_dataframe()
    df.loc[0, "amount"] = -100

    with pytest.raises(
        ValueError,
        match="Negative transaction amount detected"
    ):
        validate_data(df)


def test_invalid_risk_fails():
    df = valid_dataframe()
    df.loc[0, "country_risk"] = 2.5

    with pytest.raises(
        ValueError,
        match="country_risk must be between 0 and 1"
    ):
        validate_data(df)


def test_invalid_target_fails():
    df = valid_dataframe()
    df.loc[0, "is_fraud"] = 7

    with pytest.raises(
        ValueError,
        match="is_fraud must contain only 0 or 1"
    ):
        validate_data(df)
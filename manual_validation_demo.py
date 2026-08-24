import pandas as pd

from src.data_validation import validate_data


df = pd.DataFrame({
    "amount": [100, -500, 1200],  # 😈 Invalid
    "transactions_last_24h": [2, 5, 8],
    "country_risk": [0.2, 0.7, 0.9],
    "is_fraud": [0, 0, 1],
})

validate_data(df)
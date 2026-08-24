import mlflow.sklearn
import pandas as pd

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="FraudGuard API",
    description="Fraud detection API using MLflow Model Registry",
    version="1.0.0",
)


MODEL_URI = "models:/FraudGuardModel@champion"


class Transaction(BaseModel):
    amount: float = Field(gt=0)
    transactions_last_24h: int = Field(ge=0)
    country_risk: float = Field(ge=0, le=1)
    hour: int = Field(ge=0, le=23)
    account_age_days: int = Field(gt=0)


@lru_cache()
def get_model():
    """
    Load the champion model once and cache it.
    """

    print("📦 Loading champion model from MLflow...")

    model = mlflow.sklearn.load_model(MODEL_URI)

    print("✅ Model loaded successfully")

    return model


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model": "FraudGuardModel@champion",
    }


@app.post("/predict")
def predict(transaction: Transaction):

    # Get cached model
    model = get_model()

    # Convert API input into DataFrame
    input_data = pd.DataFrame(
        [
            {
                "amount": transaction.amount,
                "transactions_last_24h": transaction.transactions_last_24h,
                "country_risk": transaction.country_risk,
                "hour": transaction.hour,
                "account_age_days": transaction.account_age_days,
            }
        ]
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get fraud probability
    fraud_probability = model.predict_proba(input_data)[0][1]
    return {
        "prediction": "fraud" if int(prediction) == 1 else "not_fraud",
        "fraud_probability": round(float(fraud_probability), 4),
    }
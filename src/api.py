import os
import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="FraudGuard API",
    description="Fraud detection API using MLflow Model Registry",
    version="1.0.0",
)


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


MODEL_URI = "models:/FraudGuardModel@champion"


class Transaction(BaseModel):
    amount: float = Field(gt=0)
    transactions_last_24h: int = Field(ge=0)
    country_risk: float = Field(ge=0, le=1)
    hour: int = Field(ge=0, le=23)
    account_age_days: int = Field(gt=0)


print("📦 Loading champion model from MLflow...")
model = mlflow.sklearn.load_model(MODEL_URI)
print("✅ Model loaded successfully")


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model": "FraudGuardModel@champion",
    }


@app.post("/predict")
def predict(transaction: Transaction):

    # Convert validated API input into a DataFrame
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

    # Convert NumPy values to normal Python values
    prediction = int(prediction)
    fraud_probability = float(fraud_probability)

    return {
        "prediction": "fraud" if prediction == 1 else "not_fraud",
        "fraud_probability": round(fraud_probability, 4),
    }
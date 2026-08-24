import mlflow.sklearn
import pandas as pd


MODEL_URI = "models:/FraudGuardModel@champion"


def main():
    print("📦 Loading champion model from MLflow...")

    model = mlflow.sklearn.load_model(MODEL_URI)

    # Example suspicious transaction
    transaction = pd.DataFrame(
        [
            {
                "amount": 15000.0,
                "transactions_last_24h": 20,
                "country_risk": 0.9,
                "hour": 3,
                "account_age_days": 10,
            }
        ]
    )

    print("\n🔍 Predicting transaction...")

    prediction = model.predict(transaction)[0]
    probability = model.predict_proba(transaction)[0][1]

    print("\n📊 PREDICTION RESULT")
    print("=" * 40)
    print(f"Fraud prediction: {prediction}")
    print(f"Fraud probability: {probability:.4f}")

    if prediction == 1:
        print("🚨 Transaction classified as FRAUD")
    else:
        print("✅ Transaction classified as NOT FRAUD")


if __name__ == "__main__":
    main()
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.data_validation import validate_data


DATA_PATH = Path("data/raw/fraud_data.csv")
MODEL_PATH = Path("models/fraud_model.joblib")

FEATURES = [
    "amount",
    "transactions_last_24h",
    "country_risk",
    "hour",
    "account_age_days",
]

TARGET = "is_fraud"
EXPERIMENT_NAME = "FraudGuard"

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run()
    # 1. Load data
    print("📥 Loading data...")
    df = pd.read_csv(DATA_PATH)

    # 2. Validate data
    print("🛡️ Validating data...")
    validate_data(df)

    # 3. Separate features and target
    X = df[FEATURES]
    y = df[TARGET]

    # 4. Split data
    print("✂️ Splitting train/test data...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(
        f"Training samples: {len(X_train)} | "
        f"Test samples: {len(X_test)}"
    )

    # 5. Train
    print("🧠 Training Random Forest...")
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("random_state", 42)
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # 6. Evaluate ONLY on unseen test data
    print("📊 Evaluating on unseen test data...")

    predictions = model.predict(X_test)

    signature = infer_signature(
    X_test,
    predictions,
)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, predictions)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    print("\n📊 MODEL RESULTS")
    print("=" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # 7. Save model
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, MODEL_PATH)
    input_example = X_test.head(5)
    mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    signature=signature,
    input_example=input_example,
    registered_model_name="FraudGuardModel",
)

    print(f"\n✅ Model saved to: {MODEL_PATH}")
    mlflow.end_run()

if __name__ == "__main__":
    main()
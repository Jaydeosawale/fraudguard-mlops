from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.data_validation import validate_data
from src.train import FEATURES, TARGET


DATA_PATH = Path("data/raw/fraud_data.csv")
MODEL_PATH = Path("models/fraud_model.joblib")


def main():
    print("📥 Loading data...")
    df = pd.read_csv(DATA_PATH)

    print("🛡️ Validating data...")
    validate_data(df)

    X = df[FEATURES]
    y = df[TARGET]

    print("🧠 Loading model...")
    model = joblib.load(MODEL_PATH)

    print("🔮 Making predictions...")
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)

    cm = confusion_matrix(y, predictions)

    print("\n📊 MODEL RESULTS")
    print("=" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
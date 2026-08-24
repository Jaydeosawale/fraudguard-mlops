from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_check():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "FraudGuardModel@champion"


def test_predict_valid_transaction():
    transaction = {
        "amount": 15000,
        "transactions_last_24h": 20,
        "country_risk": 0.9,
        "hour": 3,
        "account_age_days": 10,
    }

    response = client.post(
        "/predict",
        json=transaction,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "fraud_probability" in data

    assert data["prediction"] in [
        "fraud",
        "not_fraud",
    ]

    assert 0 <= data["fraud_probability"] <= 1


def test_predict_invalid_transaction():
    transaction = {
        "amount": -500,
        "transactions_last_24h": 20,
        "country_risk": 1.5,
        "hour": 30,
        "account_age_days": -10,
    }

    response = client.post(
        "/predict",
        json=transaction,
    )

    assert response.status_code == 422
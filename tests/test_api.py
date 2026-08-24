from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_check():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "FraudGuardModel@champion"


@patch("src.api.get_model")
def test_predict_valid_transaction(mock_get_model):

    # Create fake ML model
    mock_model = Mock()

    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.01, 0.99]]

    # Return fake model instead of real MLflow model
    mock_get_model.return_value = mock_model

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

    assert data["prediction"] == "fraud"
    assert data["fraud_probability"] == 0.99

    # Verify fake model was used
    mock_model.predict.assert_called_once()
    mock_model.predict_proba.assert_called_once()


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
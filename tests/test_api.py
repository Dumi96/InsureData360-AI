
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to InsureData360 AI API"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_anomaly_prediction():
    response = client.post(
        "/anomalies",
        json={
            "claim_amount": 291073.84,
            "annual_premium": 247728.09,
            "age": 70,
            "product": "Property",
            "customer_type": "Corporate",
        },
    )

    assert response.status_code == 200

    result = response.json()
    assert result["anomaly_label"] in ["Typical", "Unusual - Review"]
    assert isinstance(result["anomaly_score"], float)
    assert isinstance(result["review_recommended"], bool)
    assert "interpretation" in result


def test_negative_claim_amount_is_rejected():
    response = client.post(
        "/anomalies",
        json={
            "claim_amount": -100,
            "annual_premium": 5000,
            "age": 40,
            "product": "Property",
            "customer_type": "Corporate",
        },
    )

    assert response.status_code == 422


def test_missing_required_field_is_rejected():
    response = client.post(
        "/anomalies",
        json={
            "claim_amount": 10000,
            "annual_premium": 5000,
            "age": 40,
            "product": "Property",
        },
    )

    assert response.status_code == 422

from fastapi.testclient import TestClient

import src.api as api


class FakeModel:
    def predict(self, input_data):
        return [3.5]


api.model = FakeModel()

client = TestClient(api.app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Energy Consumption Prediction API is running"
    )


def test_predict():
    response = client.post(
        "/predict",
        json={
            "hour": 12,
            "day_of_week": 2,
            "month": 9
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_consumption_kwh" in data
    assert isinstance(
        data["predicted_consumption_kwh"],
        float
    )

def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model"] == "EnergyConsumptionModel"
    assert data["alias"] == "champion"
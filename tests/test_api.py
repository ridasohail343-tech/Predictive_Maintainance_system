
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_home():
    response = client.get("/home")

    assert response.status_code == 200
    assert response.json() == {
        "message": "API is running"
    }


def test_prediction():

    sensor_data = {
        "TP2": 5.5,
        "TP3": 9.0,
        "H1": 3.3,
        "Motor_current": 4.4
    }

    response = client.post(
        "/predict",
        json=sensor_data
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in [
        "Normal",
        "Anomaly Detected"
    ]

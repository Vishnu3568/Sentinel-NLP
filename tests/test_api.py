from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_spam_prediction():
    response = client.post(
        "/predict",
        json={
            "message": "Congratulations! You have won a free prize. Call 08001234567 now."
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["label"] == "spam"
    assert 0 <= data["spam_probability"] <= 1
    assert data["threshold"] == 0.3


def test_ham_prediction():
    response = client.post(
        "/predict",
        json={
            "message": "Hey, are we still meeting today at 6?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["label"] == "ham"
    assert 0 <= data["spam_probability"] <= 1
    assert data["threshold"] == 0.3


def test_missing_message_is_rejected():
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_empty_message_is_rejected():
    response = client.post(
        "/predict",
        json={"message": ""},
    )

    assert response.status_code == 422


def test_whitespace_message_is_rejected():
    response = client.post(
        "/predict",
        json={"message": "   "},
    )

    assert response.status_code == 422


def test_non_string_message_is_rejected():
    response = client.post(
        "/predict",
        json={"message": 123},
    )

    assert response.status_code == 422


def test_unhandled_exception_returns_500(monkeypatch):
    def fake_predict_spam(message):
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr("src.api.predict_spam", fake_predict_spam)

    custom_client = TestClient(app, raise_server_exceptions=False)
    response = custom_client.post(
        "/predict",
        json={"message": "Hello, how are you?"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Prediction service temporarily unavailable."
    }




import json

import pytest
from fastapi.testclient import TestClient

SAMPLE = {"gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 3,
          "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
          "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
          "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
          "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5,
          "TotalCharges": 280.0}


@pytest.fixture
def client(trained, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(trained / "models"))
    monkeypatch.setenv("PRED_LOG", str(trained / "logs/predictions.jsonl"))
    import api
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["model_name"]


def test_predict_and_log(client, trained):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_band"] in {"Critical", "High", "Medium", "Low"}
    assert isinstance(body["in_call_list"], bool) and isinstance(body["actionable_reasons"], list)
    assert all("tenure" != r and "TotalCharges" != r for r in body["actionable_reasons"])
    log = (trained / "logs/predictions.jsonl").read_text().strip().splitlines()
    assert json.loads(log[-1])["request_id"] == body["request_id"]


def test_blank_total_charges_ok(client):
    assert client.post("/predict", json={**SAMPLE, "TotalCharges": None}).status_code == 200


def test_unknown_category_rejected(client):
    assert client.post("/predict", json={**SAMPLE, "Contract": "Lifetime"}).status_code == 422


def test_negative_tenure_rejected(client):
    assert client.post("/predict", json={**SAMPLE, "tenure": -1}).status_code == 422


def test_feedback_is_logged(client, trained):
    rid = client.post("/predict", json=SAMPLE).json()["request_id"]
    import os
    os.environ["LABEL_LOG"] = str(trained / "logs/labels.jsonl")
    try:
        assert client.post("/feedback", json={"request_id": rid, "churned": True}).json()["status"] == "recorded"
    finally:
        del os.environ["LABEL_LOG"]
    assert rid in (trained / "logs/labels.jsonl").read_text()

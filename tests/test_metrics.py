from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from src.monitoring import instrument


def sample(name, labels=None):
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def make_app():
    app = FastAPI()

    @app.post("/predict")
    def predict():
        return {"probability": 0.2}

    return instrument(app)


def test_predict_updates_dashboard_series():
    client = TestClient(make_app())
    ok = {"method": "POST", "path": "/predict", "status": "200"}
    before_req = sample("api_requests_total", ok)
    before_prob = sample("model_predicted_probability_count")

    assert client.post("/predict").status_code == 200

    assert sample("api_requests_total", ok) == before_req + 1
    assert sample("model_predicted_probability_count") == before_prob + 1


def test_metrics_endpoint_and_unknown_paths():
    client = TestClient(make_app())
    unmatched = {"method": "GET", "path": "unmatched", "status": "404"}
    before = sample("api_requests_total", unmatched)

    assert client.get("/nope").status_code == 404

    # unknown URLs collapse into one label instead of creating a series each
    assert sample("api_requests_total", unmatched) == before + 1
    body = client.get("/metrics").text
    assert "api_request_duration_seconds_bucket" in body

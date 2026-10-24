import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import batch_score as bs


@pytest.fixture(scope="module")
def client(exported_model):
    from src.serve import app

    with TestClient(app) as c:  # the `with` runs the startup (model loading)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_probability(client, customer):
    r = client.post("/predict", json=customer)
    assert r.status_code == 200
    assert 0.0 <= r.json()["probability"] <= 1.0


@pytest.mark.parametrize(
    "change",
    [
        {"job": "astronaut"},  # value not allowed
        {"age": 17},  # below minimum
        {"extra_field": 1},  # unknown field
    ],
)
def test_predict_rejects_bad_input(client, customer, change):
    r = client.post("/predict", json={**customer, **change})
    assert r.status_code == 422


def test_predict_rejects_missing_field(client, customer):
    del customer["age"]
    assert client.post("/predict", json=customer).status_code == 422


def test_api_and_batch_agree(client, customer):
    """Online and batch scoring must never disagree about a customer."""
    api_p = client.post("/predict", json=customer).json()["probability"]
    batch_p, _ = bs.score(pd.DataFrame([customer]))
    assert api_p == pytest.approx(batch_p.iloc[0])

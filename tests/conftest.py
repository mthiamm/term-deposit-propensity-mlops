import os
from pathlib import Path

import pytest

# Must be set before src.serve is imported: serve.py reads it at import time.
os.environ["MODEL_URI"] = "models/champion"


@pytest.fixture(scope="session")
def exported_model():
    if not Path("models/champion/MLmodel").exists():
        pytest.skip("No exported model: run `make export` first")


@pytest.fixture
def customer() -> dict:
    return {
        "age": 41,
        "job": "admin.",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 1200,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }

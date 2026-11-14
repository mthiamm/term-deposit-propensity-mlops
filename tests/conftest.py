import atexit
import os
import shutil
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features import BINARY_COLUMNS, binarize_yes_no, encode

# Tests use a throwaway model in a temp folder, never the real champion.
# MODEL_URI must be set before src.serve / src.batch_score are imported,
# because both read it once at import time.
_TMP = Path(tempfile.mkdtemp(prefix="test-model-"))
_MODEL_DIR = _TMP / "champion"
os.environ["MODEL_URI"] = str(_MODEL_DIR)
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

JOBS = [
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def _fake_customers(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Random customers with the same 15 input columns as the real data."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 90, n),
            "job": rng.choice(JOBS, n),
            "marital": rng.choice(["married", "single", "divorced"], n),
            "education": rng.choice(["primary", "secondary", "tertiary", "unknown"], n),
            "default": rng.choice(["yes", "no"], n),
            "balance": rng.integers(-500, 8000, n),
            "housing": rng.choice(["yes", "no"], n),
            "loan": rng.choice(["yes", "no"], n),
            "contact": rng.choice(["cellular", "telephone", "unknown"], n),
            "day": rng.integers(1, 29, n),
            "month": rng.choice(MONTHS, n),
            "campaign": rng.integers(1, 10, n),
            "pdays": rng.choice([-1, 5, 90], n),
            "previous": rng.integers(0, 4, n),
            "poutcome": rng.choice(["unknown", "success", "failure", "other"], n),
        }
    )


@pytest.fixture(scope="session")
def exported_model():
    """Train a tiny model and export it like src/export_model.py does."""
    old_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{_TMP / 'mlflow.db'}")
    try:
        raw = _fake_customers()
        X = encode(binarize_yes_no(raw, BINARY_COLUMNS)).astype("float64")
        rng = np.random.default_rng(1)
        y = (rng.random(len(X)) < 0.12 + 0.3 * (raw["poutcome"] == "success")).astype(int)
        model = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(X, y)

        exp_id = mlflow.create_experiment(
            "test-model", artifact_location=(_TMP / "artifacts").as_uri()
        )
        with mlflow.start_run(experiment_id=exp_id) as run:
            mlflow.sklearn.log_model(
                model,
                name="model",
                signature=infer_signature(X, model.predict_proba(X)[:, 1]),
                serialization_format="cloudpickle",
            )
        # mlflow.artifacts.download_artifacts(
        #     artifact_uri=f"runs:/{run.info.run_id}/model", dst_path=str(_MODEL_DIR)
        # )
        downloaded = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model", dst_path=str(_TMP / "dl")
        )
        shutil.copytree(downloaded, _MODEL_DIR)
        assert (_MODEL_DIR / "MLmodel").exists(), "test model export is malformed"
    finally:
        mlflow.set_tracking_uri(old_uri)
    return _MODEL_DIR


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

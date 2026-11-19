"""Build a tiny throwaway model, packaged like src/export_model.py's output.

Used by the test suite (conftest.py) and by the CI Docker smoke test:
    python -m tests.fake_model models/champion
"""

import shutil
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features import BINARY_COLUMNS, binarize_yes_no, encode

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


def fake_customers(n: int = 500, seed: int = 0) -> pd.DataFrame:
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


def build(out_dir: Path) -> Path:
    """Train a tiny model and write it to out_dir in the exported-champion layout."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    old_uri = mlflow.get_tracking_uri()
    with tempfile.TemporaryDirectory(prefix="fake-model-") as work:
        work = Path(work)
        mlflow.set_tracking_uri(f"sqlite:///{work / 'mlflow.db'}")
        try:
            raw = fake_customers()
            X = encode(binarize_yes_no(raw, BINARY_COLUMNS)).astype("float64")
            rng = np.random.default_rng(1)
            y = (rng.random(len(X)) < 0.12 + 0.3 * (raw["poutcome"] == "success")).astype(int)
            model = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(X, y)

            exp_id = mlflow.create_experiment(
                "test-model", artifact_location=(work / "artifacts").as_uri()
            )
            with mlflow.start_run(experiment_id=exp_id) as run:
                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    signature=infer_signature(X, model.predict_proba(X)[:, 1]),
                    serialization_format="cloudpickle",
                )
            downloaded = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id, artifact_path="model", dst_path=str(work / "dl")
            )
            shutil.copytree(downloaded, out_dir)
        finally:
            mlflow.set_tracking_uri(old_uri)

    assert (out_dir / "MLmodel").exists(), "fake model export is malformed"
    return out_dir


if __name__ == "__main__":
    target = build(Path(sys.argv[1] if len(sys.argv) > 1 else "models/champion"))
    print(f"Wrote throwaway model -> {target}")

"""Export the @champion model from the registry to a self-contained folder.

Run from the repo root:
    python -m src.export_model

Why: the registry (mlflow.db) is tied to this machine's paths. A plain folder
can be copied into a Docker image, so the container needs no registry.
"""

import shutil
from pathlib import Path

import mlflow

from src.train import TRACKING_URI

MODEL_URI = "models:/bank-marketing-classifier@champion"
OUT_DIR = Path("models/champion")


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)  # always export fresh, never mix versions
    OUT_DIR.parent.mkdir(exist_ok=True)
    mlflow.artifacts.download_artifacts(artifact_uri=MODEL_URI, dst_path=str(OUT_DIR))
    if not (OUT_DIR / "MLmodel").exists():
        raise SystemExit(f"Export looks wrong: no MLmodel file in {OUT_DIR}")
    print(f"Exported {MODEL_URI} -> {OUT_DIR}")


if __name__ == "__main__":
    main()

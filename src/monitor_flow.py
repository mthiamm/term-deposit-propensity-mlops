"""Prefect flow: check one batch of customers for data and prediction drift.

    python -m src.monitor_flow --input data/monitoring/batch_shifted.csv

The run FAILS when the alert rule fires, so the problem shows up in the
Prefect UI and in the shell exit code instead of sitting in a report.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from prefect import flow, get_run_logger, task

from src import drift
from src.features import BINARY_COLUMNS, binarize_yes_no


@task
def load_batch(path: str, sep: str) -> pd.DataFrame:
    """Read the file and fail early if it can't be monitored."""
    df = pd.read_csv(path, sep=sep)
    missing = [c for c in drift.INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing input columns: {missing}")
    if df.empty:
        raise ValueError(f"{path} has no rows")
    return binarize_yes_no(df[drift.INPUT_COLUMNS].copy(), BINARY_COLUMNS)


@task
def check_drift(batch: pd.DataFrame, name: str) -> dict:
    model, feature_columns = drift.load_model()
    reference = drift.load_reference(model, feature_columns)
    current = drift.add_scores(batch, model, feature_columns)

    snapshot = drift.run_report(reference, current, name)
    summary = drift.summarize(snapshot, name)
    (drift.OUT_DIR / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


@flow(name="drift-check")
def drift_check(input_path: str, sep: str = ";", name: str | None = None) -> dict:
    logger = get_run_logger()
    name = name or f"drift_{Path(input_path).stem}"

    batch = load_batch(input_path, sep)
    summary = check_drift(batch, name)

    logger.info(
        "drifted inputs=%s | prediction drifted=%s | report: reports/drift/%s.html",
        summary["drifted_inputs"],
        summary["prediction_drifted"],
        name,
    )
    if summary["alert"]:
        # Raising marks the run as Failed. That IS the alert.
        raise RuntimeError(
            f"Drift alert for {input_path}: inputs={summary['drifted_inputs']}, "
            f"prediction drifted={summary['prediction_drifted']}"
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--sep", default=";")
    args = parser.parse_args()
    drift_check(args.input, args.sep)

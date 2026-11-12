"""Data and prediction drift reports with Evidently.

    python -m src.drift

Compares a reference set (what the model trained on) against a current
batch (what it sees in production). Two scenarios: a clean batch that
should NOT drift, and a deliberately shifted batch that should.
"""

import json
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.data_prep import RAW_DATA_PATH, clean_data, load_data, split_data
from src.features import TARGET, encode

MODEL_DIR = "models/champion"
FEATURES_JSON = Path("data/processed/feature_columns.json")
OUT_DIR = Path("reports/drift")

MAX_DRIFTED_INPUTS = 2  # alert at this many drifted input columns...


# default/housing/loan are 0/1 flags after cleaning, so treat them as categories.
CATEGORICAL = [
    "job",
    "marital",
    "education",
    "contact",
    "month",
    "poutcome",
    "default",
    "housing",
    "loan",
]
NUMERIC = ["age", "balance", "day", "campaign", "pdays", "previous"]

INPUT_COLUMNS = NUMERIC + CATEGORICAL  # the 15 raw fields the API accepts


def add_scores(raw: pd.DataFrame, model, feature_columns: list[str]) -> pd.DataFrame:
    """Return the raw inputs plus the model's predicted probability."""
    inputs = raw.drop(columns=[TARGET], errors="ignore").reset_index(drop=True)
    X = encode(inputs, feature_columns).astype("float64")
    out = inputs.copy()
    out["prediction"] = model.predict_proba(X)[:, 1]
    return out


def load_model():
    model = mlflow.sklearn.load_model(MODEL_DIR)
    feature_columns = json.loads(FEATURES_JSON.read_text())["feature_columns"]
    return model, feature_columns


def load_reference(model, feature_columns) -> pd.DataFrame:
    """Training rows plus their scores: the 'normal' to compare against."""
    df = clean_data(load_data(RAW_DATA_PATH))
    train, _val, _test = split_data(df)
    return add_scores(train, model, feature_columns)


def inject_drift(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Make a batch that clearly differs from training."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["age"] = (out["age"] + 8).clip(upper=95)  # older customers
    out["balance"] = (out["balance"] * 0.5).round()  # lower balances
    switch = (rng.random(len(out)) < 0.5) & (out["contact"] == "cellular")
    out.loc[switch, "contact"] = "telephone"  # contact mix changes
    return out


def run_report(reference: pd.DataFrame, current: pd.DataFrame, name: str):
    definition = DataDefinition(
        numerical_columns=NUMERIC + ["prediction"],
        categorical_columns=CATEGORICAL,
    )
    ref_ds = Dataset.from_pandas(reference, data_definition=definition)
    cur_ds = Dataset.from_pandas(current, data_definition=definition)

    snapshot = Report([DataDriftPreset()]).run(cur_ds, ref_ds)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(OUT_DIR / f"{name}.html"))
    (OUT_DIR / f"{name}.json").write_text(snapshot.json())
    print(f"wrote {OUT_DIR / name}.html and .json")
    return snapshot


def summarize(snapshot, name: str) -> dict:
    """Turn an Evidently snapshot into a small, decision-ready summary."""
    data = json.loads(snapshot.json())
    columns, evidently_count = {}, None

    for m in data["metrics"]:
        cfg = m["config"]
        kind = cfg["type"].split(":")[-1]
        if kind == "DriftedColumnsCount":
            evidently_count = int(m["value"]["count"])
        elif kind == "ValueDrift":
            columns[cfg["column"]] = {
                "score": round(m["value"], 4),
                "threshold": cfg["threshold"],
                "drifted": m["value"] >= cfg["threshold"],
            }

    drifted = sorted(c for c, v in columns.items() if v["drifted"])
    # Sanity check: our reading of the JSON must match Evidently's own count.
    if evidently_count is not None and len(drifted) != evidently_count:
        raise RuntimeError(
            f"{name}: parsed {len(drifted)} drifted columns, Evidently says {evidently_count}"
        )

    inputs_drifted = [c for c in drifted if c != "prediction"]
    prediction_drifted = "prediction" in drifted
    return {
        "name": name,
        "drifted_inputs": inputs_drifted,
        "prediction_drifted": prediction_drifted,
        "alert": len(inputs_drifted) >= MAX_DRIFTED_INPUTS or prediction_drifted,
        "columns": columns,
    }


def main() -> None:
    df = clean_data(load_data(RAW_DATA_PATH))
    train, _val, test = split_data(df)

    model = mlflow.sklearn.load_model(MODEL_DIR)
    feature_columns = json.loads(FEATURES_JSON.read_text())["feature_columns"]

    reference = add_scores(train, model, feature_columns)
    clean = add_scores(test, model, feature_columns)
    shifted = add_scores(inject_drift(test.drop(columns=[TARGET])), model, feature_columns)

    # run_report(reference, clean, "drift_clean")
    # run_report(reference, shifted, "drift_shifted")

    summaries = [
        summarize(run_report(reference, clean, "drift_clean"), "drift_clean"),
        summarize(run_report(reference, shifted, "drift_shifted"), "drift_shifted"),
    ]
    (OUT_DIR / "summary.json").write_text(json.dumps(summaries, indent=2))
    for s in summaries:
        print(
            f"{s['name']}: alert={s['alert']} | drifted inputs={s['drifted_inputs']} "
            f"| prediction drifted={s['prediction_drifted']}"
        )


if __name__ == "__main__":
    main()

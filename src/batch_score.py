"""Batch scoring: score a file of customers and write a ranked call list.

Run from the repo root:
    python -m src.batch_score --input data/raw/bank-full.csv --sep ";" \
        --output data/scored/call_list.csv

Why batch: scoring a whole file in one vectorized pass is far faster and
simpler than calling the API once per customer.
"""

import argparse
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import Model
from pydantic import TypeAdapter, ValidationError

from src.features import BINARY_COLUMNS, binarize_yes_no, encode
from src.schema import Customer

# Same artifact the API serves, so batch and online scores always agree.
MODEL_URI = os.getenv("MODEL_URI", "models/champion")
CUSTOMER_COLUMNS = list(Customer.model_fields)


def load_customers(path: Path, sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    missing = [c for c in CUSTOMER_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Input is missing required columns: {missing}")
    # Keep only the fields the model uses; drops y, duration, anything extra.
    return df[CUSTOMER_COLUMNS]


def validate(df: pd.DataFrame) -> None:
    """Fail loudly on bad rows instead of scoring garbage."""
    try:
        TypeAdapter(list[Customer]).validate_python(df.to_dict("records"))
    except ValidationError as e:
        errors = e.errors()
        print(f"{len(errors)} validation error(s). First 5:", file=sys.stderr)
        for err in errors[:5]:
            row, field = err["loc"][0], err["loc"][1]
            print(f"  row {row}, field '{field}': {err['msg']}", file=sys.stderr)
        raise SystemExit(1)


def score(df: pd.DataFrame) -> tuple[pd.Series, str]:
    meta = Model.load(MODEL_URI)  # reads MLmodel file only, no database
    feature_columns = meta.signature.inputs.input_names()
    model = mlflow.sklearn.load_model(MODEL_URI)
    X = encode(binarize_yes_no(df, BINARY_COLUMNS), feature_columns).astype("float64")
    return pd.Series(model.predict_proba(X)[:, 1], index=df.index), meta.run_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--sep", default=",", help="CSV separator (bank-full.csv uses ';')")
    ap.add_argument("--top-n", type=int, default=None, help="keep only the N best")
    args = ap.parse_args()

    df = load_customers(args.input, args.sep)
    validate(df)
    proba, run_id = score(df)

    out = (
        pd.DataFrame({"row_id": df.index, "probability": proba.to_numpy()})
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )
    out["rank"] = out.index + 1
    out["decile"] = out.index * 10 // len(out) + 1  # 1 = best 10% of THIS batch
    out["model_run_id"] = run_id
    if args.top_n:
        out = out.head(args.top_n)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Scored {len(df):,} customers with model run {run_id}")
    print(
        f"Mean probability {proba.mean():.3f} | share >= 0.5: {(proba >= 0.5).mean():.1%}"
    )
    print(f"Wrote {len(out):,} rows -> {args.output}")


if __name__ == "__main__":
    main()

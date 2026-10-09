# src/train.py
import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


PROCESSED = Path("data/processed")
TARGET = "y"
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "bank-marketing - baseline"


def load_data():
    schema = json.loads((PROCESSED / "feature_columns.json").read_text())
    target = schema["target"]
    cols = [c for c in schema["feature_columns"] if c != target]  # belt and braces

    train = pd.read_parquet(PROCESSED / "train.parquet")
    val = pd.read_parquet(PROCESSED / "val.parquet")
    return (
        train[cols].astype("float64"),
        train[target],
        val[cols].astype("float64"),
        val[target],
    )


def build_model(name: str, seed: int):
    if name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            ),
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.08, max_depth=6, random_state=seed
        )
    raise ValueError(f"Unknown model: {name}")


def main():

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="logreg", choices=["logreg", "rf", "hgb"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # mlflow.set_tracking_uri(TRACKING_URI)
    # mlflow.set_experiment(EXPERIMENT)

    X_tr, y_tr, X_val, y_val = load_data()

    with mlflow.start_run(run_name=args.model):
        model = build_model(args.model, args.seed)
        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_val)[:, 1]
        metrics = {
            "roc_auc": roc_auc_score(y_val, proba),
            "pr_auc": average_precision_score(y_val, proba),
            "f1_at_0.5": f1_score(y_val, proba >= 0.5),
        }

        mlflow.log_params(
            {
                "model": args.model,
                "seed": args.seed,
                "n_features": X_tr.shape[1],
                "n_train": len(X_tr),
            }
        )
        mlflow.log_params(
            {
                f"hp_{k}": v
                for k, v in model.get_params().items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(PROCESSED / "feature_columns.json"))

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                "sklearn.linear_model._logistic.LogisticRegression",
                "numpy.dtype",
            ],
            signature=infer_signature(X_val, proba),
            input_example=X_val.head(3),
        )
        print(args.model, {k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()

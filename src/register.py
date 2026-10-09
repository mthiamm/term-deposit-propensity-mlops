# src/register.py
import json

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.train import PROCESSED, TRACKING_URI, load_data
from src.tune import EXPERIMENT as TUNING_EXPERIMENT, FIXED

MODEL_NAME = "bank-marketing-classifier"
EXPERIMENT = "bank-marketing - final"
SEED = 42


def load_test():
    schema = json.loads((PROCESSED / "feature_columns.json").read_text())
    target = schema["target"]
    cols = [c for c in schema["feature_columns"] if c != target]
    test = pd.read_parquet(PROCESSED / "test.parquet")
    return test[cols].astype("float64"), test[target]


def latest_best_params() -> tuple[str, dict]:
    runs = mlflow.search_runs(
        experiment_names=[TUNING_EXPERIMENT],
        filter_string="tags.mlflow.runName = 'hgb-optuna-study'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    run_id = runs.iloc[0]["run_id"]
    return run_id, mlflow.artifacts.load_dict(f"runs:/{run_id}/best_params.json")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    tuning_run_id, best = latest_best_params()
    X_tr, y_tr, X_val, y_val = load_data()
    X_te, y_te = load_test()

    with mlflow.start_run(run_name="hgb-champion"):
        model = HistGradientBoostingClassifier(random_state=SEED, **best, **FIXED)
        model.fit(X_tr, y_tr)

        val_p = model.predict_proba(X_val)[:, 1]
        te_p = model.predict_proba(X_te)[:, 1]
        mlflow.log_metrics(
            {
                "val_pr_auc": average_precision_score(y_val, val_p),
                "val_roc_auc": roc_auc_score(y_val, val_p),
                "test_pr_auc": average_precision_score(y_te, te_p),
                "test_roc_auc": roc_auc_score(y_te, te_p),
                "n_iter": model.n_iter_,
            }
        )
        mlflow.log_params({**best, "seed": SEED})
        mlflow.set_tag("tuning_run_id", tuning_run_id)
        mlflow.log_artifact(str(PROCESSED / "feature_columns.json"))

        info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
            # skops_trusted_types=[
            #     "sklearn.tree._tree.Tree",
            #     "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
            #     "numpy.dtype",
            # ],
            serialization_format="cloudpickle",
            signature=infer_signature(X_val, val_p),
            input_example=X_val.head(3),
        )

        MlflowClient().set_registered_model_alias(
            MODEL_NAME, "champion", info.registered_model_version
        )
        print(f"Registered {MODEL_NAME} v{info.registered_model_version} as @champion")


if __name__ == "__main__":
    main()

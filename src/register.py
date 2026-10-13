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

from mlflow.exceptions import MlflowException

MODEL_NAME = "bank-marketing-classifier"
EXPERIMENT = "bank-marketing - final"
SEED = 42
MIN_DELTA = 0.002


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


def current_champion_val_pr_auc() -> float | None:
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        return None  # no champion yet
    return client.get_run(mv.run_id).data.metrics.get("val_pr_auc")


def main(min_delta: float = MIN_DELTA) -> dict:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    tuning_run_id, best = latest_best_params()
    X_tr, y_tr, X_val, y_val = load_data()
    X_te, y_te = load_test()
    champ_val = current_champion_val_pr_auc()

    with mlflow.start_run(run_name="hgb-candidate"):
        model = HistGradientBoostingClassifier(random_state=SEED, **best, **FIXED)
        model.fit(X_tr, y_tr)

        val_p = model.predict_proba(X_val)[:, 1]
        te_p = model.predict_proba(X_te)[:, 1]
        val_pr_auc = average_precision_score(y_val, val_p)
        test_pr_auc = average_precision_score(y_te, te_p)
        promote = champ_val is None or val_pr_auc > champ_val + min_delta

        mlflow.log_metrics(
            {
                "val_pr_auc": val_pr_auc,
                "val_roc_auc": roc_auc_score(y_val, val_p),
                "test_pr_auc": test_pr_auc,
                "test_roc_auc": roc_auc_score(y_te, te_p),
                "n_iter": model.n_iter_,
            }
        )
        mlflow.log_params({**best, "seed": SEED})
        mlflow.set_tags(
            {"tuning_run_id": tuning_run_id, "promoted": str(promote).lower()}
        )
        mlflow.log_artifact(str(PROCESSED / "feature_columns.json"))

        result = {
            "promoted": promote,
            "val_pr_auc": val_pr_auc,
            "champion_val_pr_auc": champ_val,
            "test_pr_auc": test_pr_auc,
        }

        if promote:
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name=MODEL_NAME,
                serialization_format="cloudpickle",
                signature=infer_signature(X_val, val_p),
                input_example=X_val.head(3),
            )
            MlflowClient().set_registered_model_alias(
                MODEL_NAME, "champion", info.registered_model_version
            )
            result["version"] = info.registered_model_version
            print(
                f"Registered {MODEL_NAME} v{info.registered_model_version} as @champion"
            )
        else:
            print(
                f"Not promoted: val PR-AUC {val_pr_auc:.4f} vs champion {champ_val:.4f}"
            )

    return result


if __name__ == "__main__":
    main()

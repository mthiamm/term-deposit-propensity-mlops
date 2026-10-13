# src/tune.py
import argparse

import mlflow
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.train import TRACKING_URI, load_data

EXPERIMENT = "bank-marketing - tuning"


# def make_model(trial: optuna.Trial, seed: int):

#     return HistGradientBoostingClassifier(
#         learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         max_depth=trial.suggest_int("max_depth", 3, 10),
#         max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
#         min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 200),
#         l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10, log=True),
#         max_iter=trial.suggest_int("max_iter", 100, 500),
#         random_state=seed,
#     )

FIXED = {
    "max_iter": 1000,
    "early_stopping": True,
    "scoring": "average_precision",
    "n_iter_no_change": 20,
}


def make_model(trial: optuna.Trial, seed: int):
    return HistGradientBoostingClassifier(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 200),
        **FIXED,
        random_state=seed,
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="hgb", choices=["hgb"])
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    X_tr, y_tr, X_val, y_val = load_data()

    def objective(trial):
        with mlflow.start_run(run_name=f"hgb-trial-{trial.number}", nested=True):
            model = make_model(trial, args.seed)
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_val)[:, 1]
            pr_auc = average_precision_score(y_val, proba)
            mlflow.log_params(trial.params)
            mlflow.log_metrics(
                {
                    "pr_auc": pr_auc,
                    "roc_auc": roc_auc_score(y_val, proba),
                    "n_iter": model.n_iter_,
                }
            )
            return pr_auc

    with mlflow.start_run(run_name=f"hgb-optuna-study") as parent:
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed)
        )
        study.optimize(objective, n_trials=args.trials)
        mlflow.log_params({"model": args.model, "n_trials": args.trials})
        mlflow.log_metric("best_pr_auc", study.best_value)
        mlflow.log_dict(study.best_params, "best_params.json")
        print("Parent run:", parent.info.run_id)
        print("Best pr_auc:", round(study.best_value, 4), study.best_params)


if __name__ == "__main__":
    main()

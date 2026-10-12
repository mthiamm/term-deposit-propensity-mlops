# Term Deposit Propensity: End-to-End MLOps Pipeline

Predicts which bank clients will subscribe to a term deposit, using the UCI Bank Marketing dataset. The project covers the full ML lifecycle: data preparation, experiment tracking, hyperparameter tuning, and a model registry.

## Problem

Bank marketing campaigns are costly, and most calls don't convert. The model ranks clients by likelihood to subscribe so outreach can focus on the most promising ones.

- **Data:** UCI Bank Marketing ('bank-full.cs', ~45k rows)
- **Target:** 'y' (subscribed to a term deposit)
- **Metric:** PR-AUC, because the classes are imbalanced
- **Leakage control:** `duration` is dropped. It's only known after the call, so it can't be used for real prediction.

## Pipeline

| Step | Command | Output |
|------|---------|--------|
| Data prep | 'make prep' | Stratified 60/20/20 train/val/test parquet files + feature schema |
| Baselines | 'make train' | HistGradientBoosting run logged to MLflow |
| Tuning | 'make tune' | 30-trial Optuna search, all trials logged to MLflow |
| Registry | 'make register' | Best model registered with a `champion` alias, evaluated once on the test set |

Run everything: `make pipeline`

## Results

| Model | Validation PR-AUC |
|-------|-------------------|
| Logistic regression | 0.408 |
| Random forest | 0.463 |
| HistGradientBoosting | 0.476 |
| HistGradientBoosting (tuned) | 0.482 |

Test-set PR-AUC for the champion model: **0.456**. This is lower than the tuned validation score (0.482), as expected: the search optimizes against validation, so the held-out test set gives the unbiased estimate. For context, a random classifier scores about the positive rate 11.7%.

## Setup

```bash
pip install -r requirements.txt
make data
make pipeline
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Project structure

```
src/        data_prep.py, train.py, tune.py, register.py
data/       raw and processed data (not tracked)
models/     local model artifacts (not tracked)
tests/      (planned) unit tests for data prep and the registry step
Makefile    pipeline shortcuts
```

## Roadmap

- [x] Data preparation and leakage handling
- [x] Experiment tracking (MLflow)
- [x] Hyperparameter tuning (Optuna)
- [x] Model registry
- [ ] Workflow orchestration
- [ ] Model deployment (containerized service)
- [ ] Monitoring and drift detection
- [ ] Tests, CI, and linting
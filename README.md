# Term Deposit Propensity: End-to-End MLOps Pipeline

Predicts which bank clients will subscribe to a term deposit, using the UCI Bank Marketing dataset. The project covers the full ML lifecycle: data preparation, experiment tracking, hyperparameter tuning, a model registry, workflow orchestration, and serving (online API and batch scoring).

## Problem

Bank marketing campaigns are costly, and most calls don't convert. The model ranks clients by likelihood to subscribe so outreach can focus on the most promising ones.

- **Data:** UCI Bank Marketing (`bank-full.csv`, ~45k rows)
- **Target:** `y` (subscribed to a term deposit)
- **Metric:** PR-AUC, because the classes are imbalanced
- **Leakage control:** `duration` is dropped. It's only known after the call, so it can't be used for real prediction.

## Pipeline

| Step | Command | Output |
|------|---------|--------|
| Data prep | `make prep` | Stratified 60/20/20 train/val/test parquet files + feature schema |
| Baselines | `make train` | HistGradientBoosting run logged to MLflow |
| Tuning | `make tune` | 30-trial Optuna search, all trials logged to MLflow |
| Registry | `make register` | Best model registered with a `champion` alias, evaluated once on the test set |
| Orchestration | `make flow` | Runs all four steps as a Prefect flow, with a promotion gate |
| Export | `make export` | Copies the champion to a self-contained `models/champion/` folder |
| Online API | `make docker-build`, `make docker-run` | FastAPI service in a container on port 8000 |
| Batch scoring | `make batch-flow` | Ranked call list for a customer file (Prefect flow) |

Run the training pipeline end to end: `make pipeline`

## Results

| Model | Validation PR-AUC |
|-------|-------------------|
| Logistic regression | 0.408 |
| Random forest | 0.463 |
| HistGradientBoosting | 0.476 |
| HistGradientBoosting (tuned) | 0.482 |
| HistGradientBoosting (tuned, champion) | 0.483 |

Test-set PR-AUC for the champion model: **0.457**. This is lower than the tuned validation score (0.483), as expected: the search optimizes against validation, so the held-out test set gives the unbiased estimate. For context, a random classifier scores about the positive rate (0.117).

### Business impact (test set)

Customers sorted by predicted probability, split into deciles (decile 1 = highest scores):

- The top 10% of customers contain **44%** of all subscribers (4.4x lift over random targeting).
- The top 20% contain **63%**.
- The probabilities are well calibrated: average predicted probability tracks the actual subscription rate in every decile, so a score of 0.5 means roughly a 50% chance. This is what makes a cost-based call cutoff meaningful.

Reproduce with `python -m src.ranking_report` (writes `reports/ranking_report.md`).

## Setup

```bash
pip install -r requirements.txt
make data
make pipeline
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Serving predictions

The champion model is served two ways from the same exported artifact, so online and batch scores always agree.

| Mode | Use case | Entry point |
|---|---|---|
| Online API | Score one customer on demand (e.g. a call-center screen) | `src/serve.py` (FastAPI) |
| Batch job | Score a whole customer file and produce a ranked call list | `src/batch_score.py`, `src/batch_flow.py` (Prefect) |

### Export the champion

```bash
make export          # copies @champion from the registry to models/champion/
```

The exported folder is self-contained (model, input signature, dependency list), so serving needs neither the registry nor `mlflow.db`.

### Online API

```bash
make docker-build    # exports the champion, then builds the image
make docker-run      # serves on http://localhost:8000 (interactive docs at /docs)
```

```bash
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "age": 41, "job": "admin.", "marital": "married", "education": "secondary",
  "default": "no", "balance": 1200, "housing": "yes", "loan": "no",
  "contact": "cellular", "day": 15, "month": "may", "campaign": 2,
  "pdays": -1, "previous": 0, "poutcome": "unknown"}'
# {"probability": 0.0954..., "model_run_id": "..."}
```

- `GET /health` reports status and which model run is loaded.
- Inputs are validated strictly (allowed category values, ranges, no unknown fields). Invalid requests get a 422 instead of a silent wrong score.
- The API returns a probability, not a yes/no. The cutoff is a business decision (call cost vs. subscription value, or call-center capacity).

### Batch scoring

```bash
make batch-flow      # Prefect flow: validate, score, rank
# or without Prefect:
python -m src.batch_score --input data/raw/bank-full.csv --sep ";" --output data/scored/call_list.csv
```

Output columns: `row_id`, `probability`, `rank`, `decile` (1 = best 10% of that batch), `model_run_id`.

- Every row is validated with the same schema as the API. One bad row fails the whole run with the row number and field, and nothing is written.
- The old output file is deleted at the start of each run, so a failed run can never leave a stale call list that looks current.

### Design notes

- **One feature pipeline** (`src/features.py`) is shared by training, the API and batch scoring, to prevent training/serving skew.
- **One input schema** (`src/schema.py`) is shared by the API and batch validation.
- The container installs only serving dependencies (`requirements-serve.txt`), with versions pinned to those the model was trained with.

## Tests

```bash
make test            # 16 tests
```

Covers feature encoding (column order, unseen categories), input validation, call-list ranking, API behavior, and a check that the API and the batch job return the same score for the same customer. API tests need `make export` first and skip themselves otherwise.

## Project structure

```
src/
  data_prep.py      load, clean, split, encode
  features.py       shared feature transforms (training, API, batch)
  schema.py         input contract for one customer
  train.py          baseline models, MLflow tracking
  tune.py           Optuna hyperparameter search
  register.py       model registry and promotion gate
  flow.py           Prefect training pipeline
  export_model.py   export champion to a standalone folder
  serve.py          FastAPI prediction service
  batch_score.py    batch scoring and call-list ranking
  batch_flow.py     Prefect batch scoring flow
  ranking_report.py decile and lift report on the test set
tests/              unit and API tests
reports/            generated evaluation reports
data/               raw, processed and scored data (not tracked)
models/             exported model artifacts (not tracked)
Dockerfile          serving image
requirements.txt          full development dependencies
requirements-serve.txt    pinned serving dependencies
Makefile            pipeline shortcuts
```

## Roadmap

- [x] Data preparation and leakage handling
- [x] Experiment tracking (MLflow)
- [x] Hyperparameter tuning (Optuna)
- [x] Model registry
- [x] Workflow orchestration
- [x] Model deployment (containerized API and batch scoring)
- [x] Unit and API tests
- [ ] Monitoring and drift detection
- [ ] CI and linting
from prefect import flow, task

from src import data_prep, register, train, tune
from src.register import MIN_DELTA


@task
def prep() -> None:
    data_prep.main()


@task
def baseline(model: str) -> None:
    train.main(["--model", model])


@task
def tune_model(trials: int) -> None:
    tune.main(["--trials", str(trials)])


@task
def register_model(min_delta: float = MIN_DELTA) -> dict:
    return register.main(min_delta=min_delta)


@flow(name="term-deposit-training-pipeline", log_prints=True)
def training_pipeline(
    trials: int = 30, model: str = "hgb", min_delta: float = MIN_DELTA
) -> dict:
    prep()
    baseline(model)
    tune_model(trials)
    result = register_model(min_delta)
    status = "promoted to @champion" if result["promoted"] else "not promoted"
    print(f"Candidate val PR-AUC {result['val_pr_auc']:.4f}: {status}")
    return result


if __name__ == "__main__":
    training_pipeline()

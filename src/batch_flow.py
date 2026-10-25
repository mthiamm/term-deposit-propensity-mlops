"""Prefect flow: validate, score and rank a customer file.

Run from the repo root:
    python -m src.batch_flow
"""

from pathlib import Path

from prefect import flow, get_run_logger, task

from src import batch_score as bs
from src.export_model import main as export_champion


@task
def refresh_model() -> None:
    """Re-export the current @champion so the batch uses the latest model."""
    export_champion()


@task
def load_and_validate(input_path: str, sep: str):
    df = bs.load_customers(Path(input_path), sep)
    bs.validate(df)  # raises ValueError on bad rows -> flow fails visibly
    return df


@task
def score_customers(df):
    return bs.score(df)


@task
def write_call_list(df, proba, run_id: str, output_path: str, top_n: int | None) -> int:
    out = bs.build_call_list(df, proba, run_id, top_n)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return len(out)


@flow(name="batch-scoring", log_prints=True)
def batch_scoring(
    input_path: str = "data/raw/bank-full.csv",
    sep: str = ";",
    output_path: str = "data/scored/call_list.csv",
    top_n: int | None = None,
    refresh: bool = False,
) -> str:
    logger = get_run_logger()
    Path(output_path).unlink(
        missing_ok=True
    )  # a failed run must not leave an old list behind
    if refresh:
        refresh_model()
    df = load_and_validate(input_path, sep)
    proba, run_id = score_customers(df)
    n = write_call_list(df, proba, run_id, output_path, top_n)
    logger.info(
        f"Scored {len(df):,} customers with model run {run_id}; "
        f"mean probability {proba.mean():.3f}; wrote {n:,} rows to {output_path}"
    )
    return output_path


if __name__ == "__main__":
    batch_scoring()

"""Write two demo customer files: one normal, one with injected drift.

python -m src.make_batches
"""

from pathlib import Path

import pandas as pd

from src.data_prep import RAW_DATA_PATH
from src.drift import INPUT_COLUMNS, inject_drift

OUT_DIR = Path("data/monitoring")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_DATA_PATH, sep=";")
    sample = raw.sample(5000, random_state=7)[INPUT_COLUMNS].reset_index(drop=True)

    sample.to_csv(OUT_DIR / "batch_clean.csv", sep=";", index=False)
    inject_drift(sample).to_csv(OUT_DIR / "batch_shifted.csv", sep=";", index=False)
    print(f"wrote 2 files x {len(sample):,} rows to {OUT_DIR}")


if __name__ == "__main__":
    main()

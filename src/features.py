"""Feature transforms shared by training (data_prep) and serving (API, batch).

Single source of truth: if training and serving encode differently, the model
silently receives wrong inputs and nobody gets an error.
"""

import pandas as pd

TARGET = "y"
BINARY_COLUMNS = ["default", "housing", "loan"]  # yes/no -> 1/0


def binarize_yes_no(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert explicit yes/no columns to 1/0."""
    df = df.copy()
    for col in columns:
        df[col] = (df[col] == "yes").astype(int)
    return df


def encode(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode categoricals.

    feature_columns=None  -> training: columns are defined by this data.
    feature_columns=[...] -> val/test/serving: force the exact training columns
    and order (missing -> 0, unseen categories are dropped).
    """
    encoded = pd.get_dummies(df, dtype=int)
    if feature_columns is not None:
        encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded

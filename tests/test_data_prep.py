import numpy as np
import pandas as pd
import pytest

from src.data_prep import clean_data, encode_features, split_data
from src.features import TARGET


@pytest.fixture
def raw():
    """Small fake dataset with the same columns as bank-full.csv."""
    rng = np.random.default_rng(0)
    n = 1000
    return pd.DataFrame(
        {
            "age": rng.integers(18, 90, n),
            "job": rng.choice(["admin.", "technician", "retired"], n),
            "marital": rng.choice(["married", "single"], n),
            "education": rng.choice(["secondary", "tertiary"], n),
            "default": rng.choice(["yes", "no"], n),
            "balance": rng.integers(-500, 5000, n),
            "housing": rng.choice(["yes", "no"], n),
            "loan": rng.choice(["yes", "no"], n),
            "contact": rng.choice(["cellular", "telephone"], n),
            "day": rng.integers(1, 29, n),
            "month": rng.choice(["may", "jun", "jul"], n),
            "duration": rng.integers(0, 1000, n),
            "campaign": rng.integers(1, 10, n),
            "pdays": rng.choice([-1, 5, 90], n),
            "previous": rng.integers(0, 4, n),
            "poutcome": rng.choice(["unknown", "success", "failure"], n),
            "y": rng.choice(["yes", "no"], n, p=[0.12, 0.88]),
        }
    )


def test_clean_drops_leakage_and_binarizes(raw):
    out = clean_data(raw)
    assert "duration" not in out.columns
    assert set(out[TARGET].unique()) <= {0, 1}


def test_split_sizes_and_no_overlap(raw):
    train, val, test = split_data(clean_data(raw))
    assert len(train) + len(val) + len(test) == len(raw)
    assert len(train) == pytest.approx(0.6 * len(raw), abs=2)
    assert len(val) == pytest.approx(0.2 * len(raw), abs=2)
    assert len(test) == pytest.approx(0.2 * len(raw), abs=2)
    # no row may appear in two splits
    assert not set(train.index) & set(val.index)
    assert not set(train.index) & set(test.index)
    assert not set(val.index) & set(test.index)


def test_split_is_stratified(raw):
    df = clean_data(raw)
    for part in split_data(df):
        assert part[TARGET].mean() == pytest.approx(df[TARGET].mean(), abs=0.01)


def test_encoding_aligns_splits_and_ignores_unseen_categories(raw):
    train, val, test = split_data(clean_data(raw))
    val = val.copy()
    val.iloc[0, val.columns.get_loc("job")] = "astronaut"  # never seen in train

    X_train, X_val, X_test, feature_columns = encode_features(train, val, test)

    assert list(X_val.columns) == list(X_train.columns)
    assert list(X_test.columns) == list(X_train.columns)
    assert TARGET not in feature_columns
    assert "duration" not in feature_columns

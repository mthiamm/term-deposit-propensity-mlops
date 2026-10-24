import pandas as pd
import pytest

from src import batch_score as bs


def test_validate_rejects_bad_value_and_names_row_and_field(customer):
    bad = {**customer, "job": "astronaut"}
    df = pd.DataFrame([customer, bad])
    with pytest.raises(ValueError, match=r"row 1, field 'job'"):
        bs.validate(df)


def test_validate_accepts_good_rows(customer):
    bs.validate(pd.DataFrame([customer]))  # no exception


def test_load_customers_rejects_missing_columns(tmp_path, customer):
    path = tmp_path / "in.csv"
    pd.DataFrame([customer]).drop(columns=["age"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="age"):
        bs.load_customers(path, ",")


def test_load_customers_drops_extra_columns(tmp_path, customer):
    path = tmp_path / "in.csv"
    pd.DataFrame([{**customer, "y": "no", "duration": 99}]).to_csv(path, index=False)
    df = bs.load_customers(path, ",")
    assert "duration" not in df.columns and "y" not in df.columns


def test_build_call_list_ranks_and_deciles():
    df = pd.DataFrame({"x": range(20)})
    proba = pd.Series([i / 20 for i in range(20)], index=df.index)
    out = bs.build_call_list(df, proba, "run123")
    assert out["probability"].is_monotonic_decreasing
    assert out["rank"].tolist() == list(range(1, 21))
    assert out["decile"].value_counts().eq(2).all()  # 2 customers per decile
    assert out["decile"].min() == 1 and out["decile"].max() == 10
    assert out.loc[0, "row_id"] == 19  # best customer first


def test_build_call_list_top_n():
    df = pd.DataFrame({"x": range(10)})
    proba = pd.Series([i / 10 for i in range(10)], index=df.index)
    assert len(bs.build_call_list(df, proba, "r", top_n=3)) == 3

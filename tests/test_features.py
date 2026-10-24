import pandas as pd

from src.features import binarize_yes_no, encode


def test_encode_forces_training_columns_and_order():
    cols = ["age", "job_admin.", "job_retired", "housing"]
    df = pd.DataFrame([{"age": 30, "job": "retired", "housing": 1}])
    out = encode(df, cols)
    assert list(out.columns) == cols  # exact names AND order
    assert out.loc[0, "job_retired"] == 1
    assert out.loc[0, "job_admin."] == 0  # absent category filled with 0


def test_encode_ignores_unseen_category():
    cols = ["age", "job_admin."]
    out = encode(pd.DataFrame([{"age": 30, "job": "astronaut"}]), cols)
    assert list(out.columns) == cols
    assert out.loc[0, "job_admin."] == 0


def test_binarize_yes_no():
    df = pd.DataFrame({"default": ["yes", "no"], "loan": ["no", "no"]})
    out = binarize_yes_no(df, ["default", "loan"])
    assert out["default"].tolist() == [1, 0]
    assert out["loan"].tolist() == [0, 0]

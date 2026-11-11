import numpy as np
import pandas as pd
import pytest

from src import drift


def make_df(n, seed, age_mean=40, balance_scale=1000, p_cellular=0.65):
    """Fake 'customers' with the same columns the drift check compares."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.normal(age_mean, 10, n).round().clip(18, 95),
            "balance": rng.exponential(balance_scale, n).round(),
            "day": rng.integers(1, 29, n),
            "campaign": rng.integers(1, 6, n),
            "pdays": rng.choice([-1, 90, 180], n, p=[0.8, 0.1, 0.1]),
            "previous": rng.integers(0, 3, n),
            "job": rng.choice(["admin.", "technician", "services"], n),
            "marital": rng.choice(["married", "single"], n),
            "education": rng.choice(["secondary", "tertiary"], n),
            "contact": rng.choice(
                ["cellular", "telephone"], n, p=[p_cellular, 1 - p_cellular]
            ),
            "month": rng.choice(["may", "jun", "jul"], n),
            "poutcome": "unknown",
            "default": rng.choice([0, 1], n, p=[0.98, 0.02]),
            "housing": rng.integers(0, 2, n),
            "loan": rng.integers(0, 2, n),
            "prediction": rng.beta(1, 8, n),
        }
    )
    return df


@pytest.fixture(autouse=True)
def _tmp_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(drift, "OUT_DIR", tmp_path)  # don't write into reports/


def test_no_alert_on_same_distribution():
    ref, cur = make_df(4000, seed=1), make_df(4000, seed=2)
    summary = drift.summarize(drift.run_report(ref, cur, "same"), "same")
    assert summary["drifted_inputs"] == []
    assert summary["alert"] is False


def test_alert_on_shifted_inputs():
    ref = make_df(4000, seed=1)
    cur = make_df(4000, seed=2, age_mean=55, balance_scale=300, p_cellular=0.3)
    summary = drift.summarize(drift.run_report(ref, cur, "shifted"), "shifted")
    assert {"age", "balance", "contact"} <= set(summary["drifted_inputs"])
    assert summary["alert"] is True


def test_summarize_fails_loudly_when_counts_disagree():
    class FakeSnapshot:
        def json(self):
            import json

            return json.dumps(
                {
                    "metrics": [
                        {
                            "config": {
                                "type": "evidently:metric_v2:DriftedColumnsCount"
                            },
                            "value": {"count": 2.0, "share": 0.5},
                        },
                        {
                            "config": {
                                "type": "evidently:metric_v2:ValueDrift",
                                "column": "age",
                                "threshold": 0.1,
                            },
                            "value": 0.5,
                        },
                    ]
                }
            )

    with pytest.raises(RuntimeError):  # we parsed 1 drifted column, Evidently says 2
        drift.summarize(FakeSnapshot(), "fake")

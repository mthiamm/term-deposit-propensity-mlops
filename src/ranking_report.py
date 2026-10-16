"""Business-facing evaluation of the champion model: how well does it rank?

Run from the repo root:
    python -m src.ranking_report

Scores the held-out test set, then reports per-decile conversion, lift,
cumulative capture, and a simple call-economics comparison.
Writes reports/ranking_report.md.
"""

import json
from pathlib import Path

import mlflow
import pandas as pd

from src.train import PROCESSED, TRACKING_URI

MODEL_URI = "models:/bank-marketing-classifier@champion"
REPORT_PATH = Path("reports/ranking_report.md")

# ILLUSTRATIVE assumptions: replace with the bank's real numbers.
CALL_COST = 5.0  # cost of one outbound call
SUBSCRIPTION_VALUE = 100.0  # value of one term-deposit subscription


def score_test() -> pd.DataFrame:
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)
    schema = json.loads((PROCESSED / "feature_columns.json").read_text())
    target = schema["target"]
    cols = [c for c in schema["feature_columns"] if c != target]
    test = pd.read_parquet(PROCESSED / "test.parquet")
    p = model.predict_proba(test[cols].astype("float64"))[:, 1]
    return pd.DataFrame({"p": p, "y": test[target].to_numpy()})


def decile_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = df.index * 10 // len(df) + 1  # decile 1 = highest scores
    base_rate = df["y"].mean()
    total_subs = df["y"].sum()

    g = df.groupby("decile").agg(
        customers=("y", "size"),
        subscribers=("y", "sum"),
        avg_pred=("p", "mean"),
    )
    g["actual_rate"] = g["subscribers"] / g["customers"]
    g["lift"] = g["actual_rate"] / base_rate
    g["cum_customers"] = g["customers"].cumsum() / len(df)
    g["cum_subscribers"] = g["subscribers"].cumsum() / total_subs
    return g


def economics(df: pd.DataFrame) -> dict:
    """Compare calling everyone vs calling only customers above break-even.

    Break-even probability = call cost / subscription value. Using the raw
    probability as the cutoff is valid because the model is well calibrated.
    """
    break_even = CALL_COST / SUBSCRIPTION_VALUE
    called = df[df["p"] >= break_even]

    def profit(d: pd.DataFrame) -> float:
        return d["y"].sum() * SUBSCRIPTION_VALUE - len(d) * CALL_COST

    return {
        "break_even": break_even,
        "n_called": len(called),
        "share_called": len(called) / len(df),
        "capture": called["y"].sum() / df["y"].sum(),
        "profit_all": profit(df),
        "profit_targeted": profit(called),
    }


def build_report(df: pd.DataFrame) -> str:
    g = decile_table(df)
    e = economics(df)
    n, subs = len(df), int(df["y"].sum())
    top1, top2 = g.loc[1], g.loc[2]

    lines = [
        "## Ranking performance (test set)",
        "",
        f"{n:,} customers, {subs:,} subscribers (base rate {subs / n:.1%}). "
        "Decile 1 = highest-scored 10% of customers.",
        "",
        "| Decile | Customers | Subscribers | Avg predicted | Actual rate | Lift | Cum. % customers | Cum. % subscribers |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d, r in g.iterrows():
        lines.append(
            f"| {d} | {int(r.customers):,} | {int(r.subscribers):,} | {r.avg_pred:.1%} | "
            f"{r.actual_rate:.1%} | {r.lift:.1f}x | {r.cum_customers:.0%} | {r.cum_subscribers:.0%} |"
        )

    calls_all = n / subs
    calls_top2 = (top1.customers + top2.customers) / (
        top1.subscribers + top2.subscribers
    )
    lines += [
        "",
        f"- Top 10% of customers capture **{top1.cum_subscribers:.0%}** of subscribers "
        f"(lift {top1.lift:.1f}x).",
        f"- Top 20% capture **{top2.cum_subscribers:.0%}**.",
        f"- Calls per subscriber: {calls_all:.1f} calling everyone vs "
        f"{calls_top2:.1f} calling the top 20% (**{calls_all / calls_top2:.1f}x cheaper**).",
        "",
        "### Call economics (illustrative: "
        f"${CALL_COST:.0f} per call, ${SUBSCRIPTION_VALUE:.0f} per subscription)",
        "",
        f"- Break-even probability = {CALL_COST:.0f} / {SUBSCRIPTION_VALUE:.0f} = **{e['break_even']:.0%}**.",
        f"- Calling everyone: net **${e['profit_all']:,.0f}**.",
        f"- Calling only customers scored >= {e['break_even']:.0%} "
        f"({e['n_called']:,} calls, {e['share_called']:.0%} of the list, "
        f"capturing {e['capture']:.0%} of subscribers): net **${e['profit_targeted']:,.0f}**.",
    ]
    return "\n".join(lines)


def main() -> None:
    report = build_report(score_test())
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(report + "\n")
    print(report)
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

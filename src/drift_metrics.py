"""Publish drift-check results to Prometheus through the Pushgateway."""

import os
import time

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

GATEWAY = os.getenv("PUSHGATEWAY_URL", "localhost:9091")


def push_drift_summary(summary: dict) -> None:
    registry = CollectorRegistry()

    Gauge(
        "drift_check_drifted_inputs",
        "Input columns that drifted in the last check",
        registry=registry,
    ).set(len(summary["drifted_inputs"]))
    Gauge(
        "drift_check_prediction_drifted",
        "1 if the score distribution drifted",
        registry=registry,
    ).set(int(summary["prediction_drifted"]))
    Gauge(
        "drift_check_alert",
        "1 if the alert rule fired in the last check",
        registry=registry,
    ).set(int(summary["alert"]))
    Gauge(
        "drift_check_last_run_timestamp_seconds",
        "Unix time of the last check",
        registry=registry,
    ).set(time.time())

    scores = Gauge(
        "drift_check_column_score",
        "Drift score per column",
        ["column"],
        registry=registry,
    )
    for column, info in summary["columns"].items():
        scores.labels(column).set(info["score"])

    # push_to_gateway REPLACES everything stored under this job, so the
    # gateway always holds exactly the latest check.
    push_to_gateway(GATEWAY, job="drift_check", registry=registry)

"""Prometheus monitoring for the prediction API.

Kept separate from serve.py so the service logic stays untouched.
`instrument(app)` attaches request metrics, a predicted-probability
histogram, and a /metrics endpoint to any FastAPI app.
"""

import json
import time

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUESTS = Counter(
    "api_requests_total", "HTTP requests handled", ["method", "path", "status"]
)
LATENCY = Histogram(
    "api_request_duration_seconds",
    "Request latency in seconds",
    ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTED_PROBA = Histogram(
    "model_predicted_probability",
    "Distribution of predicted subscription probabilities",
    buckets=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


def instrument(app: FastAPI) -> FastAPI:
    """Attach monitoring to an existing app and return it."""

    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        if request.url.path == "/metrics":  # don't measure the scraper itself
            return await call_next(request)

        start = time.perf_counter()
        status = 500  # if call_next raises, record it as a server error
        try:
            response = await call_next(request)
            status = response.status_code

            # Read the score out of /predict responses for the histogram.
            if request.url.path == "/predict" and status == 200:
                body = b"".join([chunk async for chunk in response.body_iterator])
                try:
                    PREDICTED_PROBA.observe(json.loads(body)["probability"])
                except (ValueError, KeyError):
                    pass  # never let monitoring break a prediction
                response = Response(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
            return response
        finally:
            route = request.scope.get("route")
            path = route.path if route else "unmatched"
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(path).observe(time.perf_counter() - start)

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        """Prometheus scrapes this endpoint every few seconds."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app

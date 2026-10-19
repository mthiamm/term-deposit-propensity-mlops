"""Online prediction service: scores one customer per request.

Run from the repo root:
    uvicorn src.serve:app --port 8000
Docs UI (auto-generated): http://localhost:8000/docs
"""

import os
from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI

# from mlflow.models import get_model_info
from mlflow.models import Model
from pydantic import BaseModel, ConfigDict, Field

from src.features import BINARY_COLUMNS, binarize_yes_no, encode

# Config comes from the environment so Docker can override it later.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.getenv("MODEL_URI", "models:/bank-marketing-classifier@champion")

YesNo = Literal["yes", "no"]


class Customer(BaseModel):
    """One customer's raw attributes, exactly as the bank would hold them."""

    model_config = ConfigDict(extra="forbid")  # reject unknown/typo'd fields

    age: int = Field(ge=18, le=120)
    job: Literal[
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    ]
    marital: Literal["married", "single", "divorced"]
    education: Literal["primary", "secondary", "tertiary", "unknown"]
    default: YesNo
    balance: int
    housing: YesNo
    loan: YesNo
    contact: Literal["cellular", "telephone", "unknown"]
    day: int = Field(ge=1, le=31)
    month: Literal[
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    campaign: int = Field(ge=1)
    pdays: int = Field(ge=-1)
    previous: int = Field(ge=0)
    poutcome: Literal["failure", "other", "success", "unknown"]


class Prediction(BaseModel):
    probability: float  # P(customer subscribes to a term deposit)
    model_run_id: str  # which training run produced the model that answered


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup: load the champion model and its column list."""

    if MODEL_URI.startswith("models:/"):
        mlflow.set_tracking_uri(TRACKING_URI)

    # info = get_model_info(MODEL_URI)
    meta = Model.load(MODEL_URI)  # reads the MLmodel file only, no database
    app.state.feature_columns = meta.signature.inputs.input_names()
    app.state.model = mlflow.sklearn.load_model(MODEL_URI)
    app.state.run_id = meta.run_id
    yield


app = FastAPI(title="Term-deposit propensity API", lifespan=lifespan)


@app.get("/health")
def health():
    """Liveness check. Orchestrators and load balancers poll this."""
    return {"status": "ok", "model_run_id": app.state.run_id}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    raw = pd.DataFrame([customer.model_dump()])
    X = encode(binarize_yes_no(raw, BINARY_COLUMNS), app.state.feature_columns)
    X = X.astype("float64")  # the type the model was trained/logged with
    proba = float(app.state.model.predict_proba(X)[0, 1])
    return Prediction(probability=proba, model_run_id=app.state.run_id)

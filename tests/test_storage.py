import os
import urllib.request
import uuid

import mlflow
import pytest
from mlflow.models import Model

from src import storage
from src.features import BINARY_COLUMNS, binarize_yes_no, encode
from tests.fake_model import build, fake_customers

ENDPOINT = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")


def _localstack_up() -> bool:
    try:
        urllib.request.urlopen(f"{ENDPOINT}/_localstack/health", timeout=2)
        return True
    except OSError:
        return False


# Locally a missing LocalStack skips these; in CI the skip guard turns that into a failure.
pytestmark = pytest.mark.skipif(
    not _localstack_up(), reason=f"LocalStack is not running at {ENDPOINT}"
)


@pytest.fixture(scope="module")
def bucket():
    """A fresh, uniquely named bucket per test module, deleted afterwards."""
    mp = pytest.MonkeyPatch()
    mp.setenv("S3_ENDPOINT_URL", ENDPOINT)
    mp.setenv("AWS_ACCESS_KEY_ID", "test")
    mp.setenv("AWS_SECRET_ACCESS_KEY", "test")
    mp.setenv("AWS_DEFAULT_REGION", "us-east-1")
    name = f"test-{uuid.uuid4().hex[:12]}"
    mp.setenv("S3_BUCKET", name)
    client = storage.get_client()
    client.create_bucket(Bucket=name)
    yield name
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=name):
        for obj in page.get("Contents", []):
            client.delete_object(Bucket=name, Key=obj["Key"])
    client.delete_bucket(Bucket=name)
    mp.undo()


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return build(tmp_path_factory.mktemp("local") / "champion")


def _snapshot(folder):
    return {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_roundtrip_preserves_every_file(bucket, model_dir, tmp_path):
    uploaded = storage.upload_model(model_dir)
    out = tmp_path / "downloaded"
    assert storage.download_model(dest=out) == uploaded
    assert _snapshot(out) == _snapshot(model_dir)


def test_downloaded_model_still_scores(bucket, model_dir, tmp_path):
    storage.upload_model(model_dir)
    out = tmp_path / "downloaded"
    storage.download_model(dest=out)

    columns = Model.load(str(out)).signature.inputs.input_names()
    X = encode(binarize_yes_no(fake_customers(5, seed=3), BINARY_COLUMNS), columns)
    proba = mlflow.sklearn.load_model(str(out)).predict_proba(X.astype("float64"))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_download_from_empty_prefix_fails_clearly(bucket, tmp_path):
    with pytest.raises(FileNotFoundError):
        storage.download_model(dest=tmp_path / "x", prefix="nothing-here")


def test_upload_rejects_a_folder_that_is_not_a_model(bucket, tmp_path):
    with pytest.raises(ValueError):
        storage.upload_model(tmp_path)

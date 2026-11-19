import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from tests.fake_model import build

# Tests use a throwaway model in a temp folder, never the real champion.
# MODEL_URI must be set before src.serve / src.batch_score are imported,
# because both read it once at import time.
_TMP = Path(tempfile.mkdtemp(prefix="test-model-"))
_MODEL_DIR = _TMP / "champion"
os.environ["MODEL_URI"] = str(_MODEL_DIR)
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)


@pytest.fixture(scope="session")
def exported_model():
    return build(_MODEL_DIR)


@pytest.fixture
def customer() -> dict:
    return {
        "age": 41,
        "job": "admin.",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 1200,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }


def pytest_sessionfinish(session, exitstatus):
    """In CI, a skipped test means a hole in the safety net: fail the run."""
    if os.getenv("CI") != "true":
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter else []
    if skipped and session.exitstatus == 0:
        print(f"\nCI guard: {len(skipped)} test(s) skipped. Skips are not allowed in CI.")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED

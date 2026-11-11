"""Production entrypoint: the prediction API plus monitoring.

uvicorn src.serve_monitored:app --port 8000
"""

from src.monitoring import instrument
from src.serve import app

instrument(app)

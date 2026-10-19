FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# Dependencies first: this layer is cached and only rebuilt when the
# requirements file changes, not on every code edit.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Then the code and the model.
COPY src/serve.py src/features.py src/
COPY models/champion models/champion

ENV MODEL_URI=models/champion
EXPOSE 8000

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
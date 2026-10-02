# Default build (CI, docker compose):   docker build -t churn-api .            -> mount ./models at runtime
# Kubernetes / self-contained image:    docker build --target baked -t churn-api:1.0 .   -> model copied into image
FROM python:3.11-slim AS base
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY features.py validate.py monitor.py performance.py policy.py explain.py api.py app.py train.py deep.py ./
COPY scripts ./scripts
RUN useradd -m -u 1000 appuser && mkdir -p logs models reports && chown -R appuser /app
ENV MODEL_DIR=/app/models PRED_LOG=/app/logs/predictions.jsonl LABEL_LOG=/app/logs/labels.jsonl
EXPOSE 8000 8501

FROM base AS baked
COPY --chown=appuser models ./models
COPY --chown=appuser reports ./reports
USER appuser
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# last stage = default target (no model inside)
FROM base AS runtime
USER appuser
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

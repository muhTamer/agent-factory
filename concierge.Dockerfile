# ── Concierge API: document upload, analysis, deployment ───────
FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY factory/ ./factory/
COPY scripts/ ./scripts/
COPY tests/fixtures/ ./tests/fixtures/

# .factory/ may not exist in CI (generated at runtime) — copy if present
RUN mkdir -p .factory
COPY .factor[y]/ ./.factory/

# Create workspace directory
RUN mkdir -p .workspace

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/concierge/runtime/health')" || exit 1

CMD ["uvicorn", "app.concierge.api:app", "--host", "0.0.0.0", "--port", "8001"]

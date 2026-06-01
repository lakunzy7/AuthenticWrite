# Multi-stage build: reduce final image size, cache dependencies
FROM python:3.11-slim-bookworm AS builder

WORKDIR /tmp/build
COPY requirements.txt .

# Install build deps for wheels that need compilation (torch, numpy, pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with CPU-only torch, no cache
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Runtime stage: copy only what's needed
FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code
COPY app.py .

# Bake the RoBERTa model and NLTK data at build time (30-60s, avoids cold start in pod)
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta'); \
    AutoModelForSequenceClassification.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta')" \
    && python -c "import nltk; \
    nltk.download('punkt', quiet=True); \
    nltk.download('punkt_tab', quiet=True); \
    nltk.download('stopwords', quiet=True)"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health', timeout=5)" || exit 1

# Entrypoint: gunicorn with single worker (model is ~1.5 GB, only fit 1 per pod)
# Threads for I/O concurrency, long timeout for model inference
ENTRYPOINT ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:5000", "app:app"]

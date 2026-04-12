# ── Organ Transplant Matching Environment — Dockerfile ──────────────────────
# Compatible with Hugging Face Spaces (port 7860)

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY openenv.yaml .
COPY inference.py .

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run inference + server
CMD sh -c "python inference.py & uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1"
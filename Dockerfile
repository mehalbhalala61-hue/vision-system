# =============================================================================
# Dockerfile — Vision System Capstone v3
# =============================================================================
# python:3.11-slim base — small image for Railway free tier
# CPU torch — GPU version too large for Railway ($5/month credit)
# $PORT env var — Railway sets this dynamically
# =============================================================================

FROM python:3.11-slim

# System deps — libgomp1 required by onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python deps — CPU torch only (Railway free tier)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p logs/checkpoints logs/gradcam_outputs data

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose default port (Railway overrides with $PORT)
EXPOSE 8000

# Health check — Railway uses /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start command — $PORT injected by Railway at runtime
CMD sh -c "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"

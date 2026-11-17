# =============================================================================
# ENTERPRISE AML COMPLIANCE & SAR NARRATIVE GENERATOR - DOCKERFILE
# Multi-Service Container: FastAPI MLOps Engine (8000) + Streamlit Web Portal (8501)
# =============================================================================

FROM python:3.11-slim

# Maintainer / Metadata
LABEL maintainer="AML Compliance & MLOps Engineering Team"
LABEL description="AI/ML Anti-Money Laundering Surveillance & SAR Narrative Generator"
LABEL version="2.0.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    STREAMLIT_PORT=8501 \
    API_PORT=8000

# Install system dependencies & curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency specifications first for optimal layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, datasets, and configurations
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY app_web.py .
COPY app_api.py .
COPY entrypoint.py .
COPY checkpoint_5m_fast.json .
COPY overfitting_underfitting_analysis.png .

# Create directory for runtime logs
RUN mkdir -p logs

# Expose Streamlit Dashboard (8501) and FastAPI MLOps REST API (8000)
EXPOSE 8501 8000

# Container Healthcheck (Monitors FastAPI /health endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start both FastAPI and Streamlit concurrently via entrypoint supervisor
CMD ["python", "entrypoint.py"]

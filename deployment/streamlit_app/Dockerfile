# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /install

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install only the packages required by the Streamlit frontend
RUN pip install --upgrade pip && \
    pip install --prefix=/install/deps --no-cache-dir \
        "streamlit>=1.35.0" \
        "plotly>=5.22.0" \
        "pandas>=2.1.0" \
        "Pillow>=10.0.0" \
        "requests>=2.31.0" \
        "python-dotenv>=1.0.0"

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

COPY --from=builder /install/deps /usr/local

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY deployment/streamlit_app/ ./deployment/streamlit_app/

# Streamlit config directory
RUN mkdir -p .streamlit && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Default: point at the API container name defined in docker-compose
    API_BASE_URL=http://api:8000

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "deployment/streamlit_app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

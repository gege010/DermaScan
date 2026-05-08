# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
# Install deps in isolated layer to keep the final image lean
FROM python:3.10-slim AS builder

WORKDIR /install

# System build deps required by TensorFlow, OpenCV, and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install only the packages needed by the API (no Streamlit/Plotly/Pandas/Seaborn/Matplotlib)
RUN pip install --upgrade pip && \
    pip install --prefix=/install/deps --no-cache-dir \
        "tensorflow>=2.12.0" \
        "numpy>=1.24.0" \
        "scikit-learn>=1.3.0" \
        "Pillow>=10.0.0" \
        "opencv-python-headless>=4.8.0" \
        "fastapi>=0.111.0" \
        "uvicorn[standard]>=0.29.0" \
        "python-multipart>=0.0.9" \
        "pydantic>=2.7.0" \
        "groq>=0.9.0" \
        "tavily-python>=0.3.0" \
        "python-dotenv>=1.0.0"

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

# Runtime system libs (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install/deps /usr/local

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copy only the source code required by the API
COPY src/           ./src/
COPY deployment/api/ ./deployment/api/

# The model files are expected to be mounted at /app/models at runtime
# (they are too large to bake into the image)
RUN mkdir -p models logs

# Ensure the non-root user owns the workdir
RUN chown -R appuser:appuser /app
USER appuser

# ─── Environment defaults (override via docker-compose or --env-file) ─────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

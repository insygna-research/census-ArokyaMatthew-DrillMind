FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (for caching)
COPY pyproject.toml .
# Core deps + production add-ons + ML/RAG
RUN pip install --no-cache-dir \
        "pandas>=2.0" "numpy>=1.24" "openpyxl>=3.1" "pyyaml>=6.0" \
        "pydantic>=2.0" "loguru>=0.7" "websockets>=12.0" \
        "fastapi>=0.110" "uvicorn>=0.29" "httpx>=0.27" \
        "prometheus_client>=0.20" "rank_bm25>=0.2.2" && \
    pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        scikit-learn \
        "chromadb>=0.5" "sentence-transformers>=3.0" "datasets>=3.0"

# Copy source code
COPY . .
RUN pip install --no-cache-dir -e .

# Expose API port
EXPOSE 8000

# Default: load 50K rows for reasonable startup time
ENV DRILLMIND_MAX_ROWS=50000
# Sensible defaults — override at runtime
ENV DRILLMIND_LOG_LEVEL=INFO
ENV DRILLMIND_LOG_JSON=0

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "drillmind.api.server:app", \
     "--host", "0.0.0.0", "--port", "8000"]

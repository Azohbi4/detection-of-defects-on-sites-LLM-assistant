# Image de l'API : CPU uniquement, modèle chargé au démarrage.
# Build depuis la racine du dépôt : docker build -f docker/Dockerfile -t chantier-api .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# PyTorch CPU (roues bien plus légères que les roues CUDA), installé dans sa propre couche pour le cache
RUN pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api,llm]"

COPY models ./models

RUN useradd --create-home --uid 1000 app && mkdir -p /app/.cache && chown -R app /app
USER app

ENV CHANTIER_MODEL_PATH=/app/models/model.pt
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "chantier_defects.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

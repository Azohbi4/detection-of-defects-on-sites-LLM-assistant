# Image de l'API : CPU uniquement, optimisée pour Google Cloud Run.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    PORT=8000

WORKDIR /app

# PyTorch CPU (roues bien plus légères que les roues CUDA)
RUN pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api,llm]"

COPY models ./models
COPY knowledge ./knowledge

ENV CHANTIER_MODEL_PATH=/app/models/model.pt \
    CHANTIER_QDRANT_PATH=/app/qdrant_storage

# Base RAG indexée dans l'image (Qdrant embarqué) : le modèle d'embedding est téléchargé une fois ici,
# puis réutilisé au démarrage sans accès réseau. CHANTIER_QDRANT_URL, s'il est défini, reste prioritaire.
RUN python -m chantier_defects.llm.rag ingest knowledge/

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn chantier_defects.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

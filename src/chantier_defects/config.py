"""Configuration centralisée, lue depuis les variables d'environnement (préfixe CHANTIER_) ou un fichier .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHANTIER_", env_file=".env", extra="ignore")

    # Modèle de vision chargé au démarrage de l'API
    model_path: Path = Path("models/model.pt")
    tile_size: int = 256
    tile_stride: int = 256

    # Limites de l'API
    max_images_per_request: int = 20
    max_image_bytes: int = 15 * 1024 * 1024

    # RAG
    qdrant_url: str | None = None
    qdrant_path: str | None = None  # mode embarqué (sans serveur) si qdrant_url est vide
    qdrant_collection: str = "pathologies_beton"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    rag_top_k: int = 3
    # Similarité cosinus minimale d'un passage. Calibrée sur knowledge/ : passages pertinents 0,45-0,73,
    # requêtes hors sujet 0,10-0,47. À recalibrer si l'on change de documents ou de modèle d'embedding.
    rag_min_score: float = 0.40

    # LLM (la clé est lue par le SDK depuis ANTHROPIC_API_KEY)
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "medium"
    llm_max_tokens: int = 16000
    llm_timeout_s: float = 120.0  # par tentative ; reste sous le délai de 300 s de l'UI, retries compris
    llm_max_retries: int = 1

    # Monitoring
    monitor_window: int = 200
    drift_confidence_drop: float = 0.15


@lru_cache
def get_settings() -> Settings:
    return Settings()

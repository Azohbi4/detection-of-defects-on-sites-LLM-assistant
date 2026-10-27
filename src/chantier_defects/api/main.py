"""API FastAPI.

    POST /analyze  image(s) -> détections JSON (+ gravité par image)
    POST /report   détections -> rapport d'inspection (LLM + RAG, avec garde-fous)
    GET  /health   état du service
    GET  /metrics  latence, distribution des scores, alerte de dérive

Lancement : uvicorn chantier_defects.api.main:app --reload
"""

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image, UnidentifiedImageError

from ..config import Settings, get_settings
from ..llm.report import ReportService
from ..models.predictor import Predictor, analyze_image, load_predictor
from ..monitoring import Monitor
from ..schemas import AnalyzeResponse, Report, ReportRequest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 60_000_000  # protège contre les « bombes de décompression »


def default_report_service(settings: Settings) -> ReportService:
    from ..llm.rag import build_retriever

    load_dotenv()  # hors Docker Compose, la clé du .env n'est pas dans l'environnement ; n'écrase rien
    retriever = build_retriever(settings)
    writer = None
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        from ..llm.report import ClaudeReportWriter

        writer = ClaudeReportWriter(
            settings.llm_model,
            settings.llm_effort,
            settings.llm_max_tokens,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
    else:
        log.warning("Pas de clé ANTHROPIC_API_KEY : les rapports seront générés par gabarit.")
    return ReportService(retriever, writer, settings.rag_top_k, settings.rag_min_score)


def create_app(
    predictor: Predictor | None = None,
    report_service: ReportService | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Le modèle est chargé une seule fois, au démarrage.
        if app.state.predictor is None and Path(settings.model_path).exists():
            app.state.predictor = load_predictor(Path(settings.model_path))
            log.info("Modèle chargé : %s", app.state.predictor.version)
        elif app.state.predictor is None:
            log.warning("Aucun modèle trouvé à %s : /analyze répondra 503.", settings.model_path)
        if app.state.report_service is None:
            app.state.report_service = default_report_service(settings)
        ref = getattr(app.state.predictor, "reference_confidence", None)
        app.state.monitor = Monitor(settings.monitor_window, ref, settings.drift_confidence_drop)
        yield

    app = FastAPI(title="Détection de défauts sur chantier", version="0.1.0", lifespan=lifespan)
    app.state.predictor = predictor
    app.state.report_service = report_service

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")  # l'URL de base mène à la documentation interactive

    @app.get("/health")
    def health(request: Request) -> dict:
        p = request.app.state.predictor
        return {
            "status": "ok",
            "model_loaded": p is not None,
            "model_version": p.version if p else None,
            "classes": p.classes if p else [],
        }

    @app.get("/metrics")
    def metrics(request: Request) -> dict:
        return request.app.state.monitor.snapshot()

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze(request: Request, files: list[UploadFile] = File(...)) -> AnalyzeResponse:
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(503, "Aucun modèle chargé.")
        if len(files) > settings.max_images_per_request:
            raise HTTPException(413, f"Maximum {settings.max_images_per_request} images par requête.")

        t0 = time.perf_counter()
        analyses = []
        for i, upload in enumerate(files):
            data = upload.file.read(settings.max_image_bytes + 1)
            if len(data) > settings.max_image_bytes:
                raise HTTPException(413, f"{upload.filename} dépasse {settings.max_image_bytes // 2**20} Mo.")
            try:
                img = Image.open(io.BytesIO(data))
                img.load()
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise HTTPException(400, f"{upload.filename} n'est pas une image lisible.") from exc
            image_id = upload.filename or f"image_{i + 1}"
            analyses.append(analyze_image(img, predictor, image_id, settings.tile_size, settings.tile_stride))

        request.app.state.monitor.record(
            "/analyze",
            (time.perf_counter() - t0) * 1000,
            confidences=[a.mean_confidence for a in analyses],
            scores=[d.score for a in analyses for d in a.detections],
        )
        return AnalyzeResponse(model_version=predictor.version, images=analyses)

    @app.post("/report", response_model=Report)
    def report(request: Request, body: ReportRequest) -> Report:
        t0 = time.perf_counter()
        result = request.app.state.report_service.build(body)
        request.app.state.monitor.record("/report", (time.perf_counter() - t0) * 1000)
        return result

    return app


app = create_app()

"""Contrôle de qualité de bout en bout contre la vraie API Claude (payant, quelques centimes).

Désactivé par défaut : il faut une clé (variable d'environnement ou .env), la base RAG indexée
(python -m chantier_defects.llm.rag ingest knowledge/) et l'option -m slow.
    ANTHROPIC_API_KEY=... pytest -m slow tests/test_live_claude.py
"""

import os

import pytest
from dotenv import load_dotenv

from chantier_defects.config import Settings
from chantier_defects.llm.rag import build_retriever
from chantier_defects.llm.report import ClaudeReportWriter, ReportService
from chantier_defects.models.predictor import analyze_image
from chantier_defects.schemas import ReportRequest
from tests.conftest import concrete_image

load_dotenv()  # accepte la clé placée dans .env, comme indiqué dans le README

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY absente"),
]


def test_real_report_passes_guardrails_untouched(fake_predictor):
    settings = Settings()
    # Vraie base documentaire : un passage factice sans contenu ne justifie aucune action, et Claude
    # (règle 3 du prompt) ne propose alors aucune recommandation.
    retriever = build_retriever(settings)
    if retriever is None:
        pytest.skip("base RAG indisponible : lancer l'ingestion de knowledge/")
    writer = ClaudeReportWriter(settings.llm_model, settings.llm_effort, settings.llm_max_tokens)
    request = ReportRequest(
        analyses=[analyze_image(concrete_image(), fake_predictor, "pile_P2.jpg")],
        context="Pile P2, face nord, inspection visuelle.",
    )
    report = ReportService(retriever, writer, settings.rag_top_k, settings.rag_min_score).build(request)

    assert report.generated_by == "llm"
    # Un rapport bien formé n'a rien à faire retirer par les garde-fous.
    assert report.guardrail_notes == []
    assert [d.defect_type for d in report.defects] == ["crack"]
    assert report.recommendations and all(r.source_ids for r in report.recommendations)
    assert report.summary.strip() and report.uncertainty.strip()

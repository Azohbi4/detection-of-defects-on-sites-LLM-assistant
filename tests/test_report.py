from chantier_defects.llm.report import (
    LLMUnavailable,
    ReportService,
    RetrievedChunk,
    TemplateReportWriter,
    gather_passages,
    location_label,
    mentioned_defects,
)
from chantier_defects.models.predictor import analyze_image
from chantier_defects.schemas import BBox, DraftDefect, DraftRecommendation, ReportDraft, ReportRequest
from tests.conftest import concrete_image


class ScriptedWriter:
    """Simule un LLM qui hallucine : une reco sans source, une source inventée, un défaut absent."""

    def draft(self, payload, passages):
        return ReportDraft(
            summary="Résumé.",
            defects=[
                DraftDefect(image_id="img1", defect_type="crack", location="zone haut-gauche"),
                DraftDefect(image_id="img1", defect_type="exposed_bars", location="partout"),
            ],
            recommendations=[
                DraftRecommendation(action="Injecter la fissure", rationale="r", source_ids=["S1"]),
                DraftRecommendation(action="Démolir l'ouvrage", rationale="r", source_ids=[]),
                DraftRecommendation(action="Action fantôme", rationale="r", source_ids=["S99"]),
            ],
            uncertainty="Incertain.",
        )


class FailingWriter:
    def draft(self, payload, passages):
        raise LLMUnavailable("timeout")


def request_for(predictor):
    return ReportRequest(analyses=[analyze_image(concrete_image(), predictor, "img1")])


def test_guardrails_drop_unsourced_and_hallucinated(fake_predictor, fake_retriever):
    report = ReportService(fake_retriever, ScriptedWriter()).build(request_for(fake_predictor))
    assert report.generated_by == "llm"
    assert [r.action for r in report.recommendations] == ["Injecter la fissure"]
    assert [d.defect_type for d in report.defects] == ["crack"]
    assert [s.id for s in report.sources] == ["S1"]
    assert len(report.guardrail_notes) == 3
    assert "Validation humaine requise" in report.disclaimer


def test_severity_is_recomputed_server_side(fake_predictor, fake_retriever):
    req = request_for(fake_predictor)
    req.analyses[0].severity.level = "élevé"  # un client qui tenterait d'imposer la gravité
    report = ReportService(fake_retriever, ScriptedWriter()).build(req)
    assert report.severity.level == "faible"


def test_fallback_to_template_when_llm_fails(fake_predictor, fake_retriever):
    report = ReportService(fake_retriever, FailingWriter()).build(request_for(fake_predictor))
    assert report.generated_by == "template"
    assert report.recommendations and all(r.source_ids == ["S1"] for r in report.recommendations)
    assert any("LLM indisponible" in n for n in report.guardrail_notes)


def test_no_sources_means_no_recommendation(fake_predictor):
    report = ReportService(None, TemplateReportWriter()).build(request_for(fake_predictor))
    assert report.recommendations == []
    assert any("Aucune source" in n for n in report.guardrail_notes)


def test_location_label():
    assert location_label(BBox(x=0, y=0, w=100, h=100), 900, 900).startswith("zone haut-gauche")
    assert location_label(BBox(x=400, y=400, w=100, h=100), 900, 900).startswith("centre")


class InventingSummaryWriter:
    def draft(self, payload, passages):
        return ReportDraft(
            summary="Fissure et armatures apparentes en zone haute.",
            defects=[],
            recommendations=[],
            uncertainty="u",
        )


def test_summary_citing_undetected_defect_is_replaced(fake_predictor, fake_retriever):
    report = ReportService(fake_retriever, InventingSummaryWriter()).build(request_for(fake_predictor))
    assert "armatures" not in report.summary and "fissure" in report.summary
    assert any("Résumé remplacé" in n for n in report.guardrail_notes)


def test_mentioned_defects_ignores_negations():
    assert mentioned_defects("Fissure diagonale, aucune efflorescence ni trace de rouille.") == {"crack"}
    assert mentioned_defects("Pas d'éclatement visible ; armatures à nu au droit de la fissure.") == {
        "crack",
        "exposed_bars",
    }


def test_low_score_passages_are_dropped():
    class ScoredRetriever:
        def search(self, query, k):
            return [
                RetrievedChunk(chunk_id="a#0", title="A", text="pertinent", score=0.7),
                RetrievedChunk(chunk_id="b#0", title="B", text="hors sujet", score=0.2),
            ]

    assert [p.chunk_id for p in gather_passages(ScoredRetriever(), ["crack"], 3, min_score=0.4)] == ["a#0"]

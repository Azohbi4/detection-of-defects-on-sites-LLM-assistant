"""Contrats de données (Pydantic) partagés entre le modèle, la règle de gravité, le LLM et l'API."""

from typing import Literal

from pydantic import BaseModel, Field

SeverityLevel = Literal["faible", "moyen", "élevé"]
# Codes des classes (labels.CODEBRIM_CLASSES). Imposés au LLM par le schéma : sans cela il écrit le libellé
# (« fissure »), qui ne correspond plus aux détections et fait retirer le défaut par les garde-fous.
DefectCode = Literal["crack", "spallation", "efflorescence", "exposed_bars", "corrosion_stain"]

DISCLAIMER = (
    "Aide à la décision générée automatiquement. Validation humaine requise : ce rapport ne remplace "
    "pas une inspection par un personnel qualifié ni un diagnostic d'ouvrage."
)


class BBox(BaseModel):
    """Boîte en pixels dans le repère de l'image d'origine."""

    x: int
    y: int
    w: int
    h: int


class Detection(BaseModel):
    defect_type: str
    score: float = Field(ge=0.0, le=1.0, description="Probabilité maximale sur la zone")
    bbox: BBox
    n_tiles: int = Field(ge=1, description="Nombre de tuiles contiguës qui composent la zone")


class SeverityAssessment(BaseModel):
    level: SeverityLevel
    reasons: list[str]


class ImageAnalysis(BaseModel):
    image_id: str
    width: int
    height: int
    detections: list[Detection]
    affected_ratio: dict[str, float] = Field(
        default_factory=dict, description="Part des tuiles de l'image concernées, par type de défaut"
    )
    mean_confidence: float = Field(ge=0.0, le=1.0, description="Confiance moyenne (max des probas par tuile)")
    severity: SeverityAssessment | None = None


class AnalyzeResponse(BaseModel):
    model_version: str
    images: list[ImageAnalysis]


class ReportRequest(BaseModel):
    analyses: list[ImageAnalysis] = Field(min_length=1)
    context: str | None = Field(default=None, max_length=2000, description="Ouvrage, localisation, date…")


# --- Sortie structurée demandée au LLM (il rédige, il ne décide pas de la gravité) ---


class DraftDefect(BaseModel):
    image_id: str
    defect_type: DefectCode = Field(description="Code exact du champ « type » de la détection, pas son libellé")
    location: str = Field(description="Localisation lisible dans l'image, ex. « quart supérieur gauche »")


class DraftRecommendation(BaseModel):
    action: str
    rationale: str
    source_ids: list[str] = Field(description="Identifiants des passages cités, ex. ['S1']")


class ReportDraft(BaseModel):
    summary: str
    defects: list[DraftDefect]
    recommendations: list[DraftRecommendation]
    uncertainty: str


# --- Rapport final renvoyé par l'API ---


class SourceRef(BaseModel):
    id: str
    title: str
    url: str | None = None
    excerpt: str


class Report(BaseModel):
    summary: str
    severity: SeverityAssessment
    severity_by_image: dict[str, SeverityAssessment]
    defects: list[DraftDefect]
    recommendations: list[DraftRecommendation]
    uncertainty: str
    sources: list[SourceRef]
    guardrail_notes: list[str] = Field(default_factory=list)
    generated_by: Literal["llm", "template"]
    disclaimer: str = DISCLAIMER

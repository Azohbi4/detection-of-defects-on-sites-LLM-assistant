"""Règle de gravité déterministe : type de défaut × étendue relative × nombre de zones.

Choix de conception : la gravité est calculée ici, pas par le LLM. Une règle explicite est
reproductible, testable et auditable ; le LLM se contente de rédiger à partir de ce résultat.
Les seuils ci-dessous sont des hypothèses de travail, pas une norme officielle (voir README, Limites).
"""

from collections.abc import Iterable

from .labels import label_fr
from .schemas import ImageAnalysis, SeverityAssessment, SeverityLevel

LEVELS: tuple[SeverityLevel, ...] = ("faible", "moyen", "élevé")

# Niveau de départ par type (0 = faible, 1 = moyen, 2 = élevé)
BASE_LEVEL = {
    "crack": 0,
    "efflorescence": 0,
    "spallation": 1,
    "corrosion_stain": 1,
    "exposed_bars": 2,  # perte d'enrobage : risque structurel et de corrosion accélérée
}
DEFAULT_BASE_LEVEL = 1  # type inconnu : prudence

EXTENT_RATIO = 0.20  # part des tuiles de l'image au-delà de laquelle on aggrave d'un niveau
MANY_ZONES = 3  # nombre de zones distinctes d'un même type au-delà duquel on aggrave d'un niveau
WATER_SIGNS = {"efflorescence", "corrosion_stain"}  # fissure + signe d'eau/corrosion => au moins « moyen »


def assess_image(analysis: ImageAnalysis) -> SeverityAssessment:
    if not analysis.detections:
        return SeverityAssessment(level="faible", reasons=["Aucun défaut détecté au-dessus du seuil."])

    zones: dict[str, int] = {}
    for det in analysis.detections:
        zones[det.defect_type] = zones.get(det.defect_type, 0) + 1

    worst = 0
    reasons = []
    for defect_type, n_zones in sorted(zones.items()):
        level = BASE_LEVEL.get(defect_type, DEFAULT_BASE_LEVEL)
        why = [f"type « {label_fr(defect_type)} » (base {LEVELS[level]})"]
        ratio = analysis.affected_ratio.get(defect_type, 0.0)
        if ratio >= EXTENT_RATIO or n_zones >= MANY_ZONES:
            level += 1
            why.append(f"étendue {ratio:.0%} des tuiles, {n_zones} zone(s)")
        if defect_type == "crack" and WATER_SIGNS & zones.keys() and level < 1:
            level = 1
            why.append("fissure associée à des signes d'eau ou de corrosion")
        level = min(level, len(LEVELS) - 1)
        worst = max(worst, level)
        reasons.append(f"{label_fr(defect_type)} → {LEVELS[level]} : " + ", ".join(why))
    return SeverityAssessment(level=LEVELS[worst], reasons=reasons)


def assess_batch(
    analyses: Iterable[ImageAnalysis],
) -> tuple[SeverityAssessment, dict[str, SeverityAssessment]]:
    """Gravité globale = la pire gravité observée sur le lot."""
    by_image = {a.image_id: assess_image(a) for a in analyses}
    if not by_image:
        return SeverityAssessment(level="faible", reasons=["Aucune image analysée."]), by_image
    worst = max(by_image.values(), key=lambda s: LEVELS.index(s.level))
    reasons = [
        f"{image_id} : {s.level}" for image_id, s in by_image.items() if s.level == worst.level
    ] + worst.reasons
    return SeverityAssessment(level=worst.level, reasons=reasons), by_image

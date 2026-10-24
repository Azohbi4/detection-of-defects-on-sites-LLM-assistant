from chantier_defects.schemas import BBox, Detection, ImageAnalysis
from chantier_defects.severity import assess_batch, assess_image


def analysis(image_id="img", detections=(), ratios=None):
    dets = [
        Detection(defect_type=t, score=0.9, bbox=BBox(x=0, y=0, w=10, h=10), n_tiles=1) for t in detections
    ]
    return ImageAnalysis(
        image_id=image_id,
        width=100,
        height=100,
        detections=dets,
        affected_ratio=ratios or {},
        mean_confidence=0.9,
    )


def test_no_detection_is_low():
    assert assess_image(analysis()).level == "faible"


def test_small_crack_is_low():
    assert assess_image(analysis(detections=["crack"], ratios={"crack": 0.05})).level == "faible"


def test_extended_crack_escalates():
    assert assess_image(analysis(detections=["crack"], ratios={"crack": 0.30})).level == "moyen"


def test_many_zones_escalate():
    assert assess_image(analysis(detections=["crack"] * 3, ratios={"crack": 0.05})).level == "moyen"


def test_crack_with_water_signs_is_at_least_medium():
    result = assess_image(analysis(detections=["crack", "efflorescence"], ratios={"crack": 0.02}))
    assert result.level == "moyen"


def test_exposed_bars_is_high_and_capped():
    result = assess_image(analysis(detections=["exposed_bars"] * 4, ratios={"exposed_bars": 0.5}))
    assert result.level == "élevé"
    assert any("armatures apparentes" in r for r in result.reasons)


def test_batch_takes_worst_image():
    overall, by_image = assess_batch([analysis("a"), analysis("b", ["spallation"], {"spallation": 0.4})])
    assert by_image["a"].level == "faible"
    assert overall.level == "élevé"

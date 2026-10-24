from chantier_defects.models.predictor import _positions, analyze_image
from tests.conftest import concrete_image


def test_positions_cover_edges():
    assert _positions(768, 256, 256) == [0, 256, 512]
    assert _positions(700, 256, 256) == [0, 256, 444]  # dernière tuile alignée sur le bord
    assert _positions(100, 256, 256) == [0]


def test_crack_tiles_merge_into_one_zone(fake_predictor):
    result = analyze_image(concrete_image(768, 512), fake_predictor, "img1")
    assert len(result.detections) == 1
    det = result.detections[0]
    assert det.defect_type == "crack" and det.bbox.x == 0 and det.bbox.y == 0
    assert det.n_tiles == 1
    assert result.affected_ratio == {"crack": round(1 / 6, 4)}
    assert result.severity.level == "faible"


def test_clean_image_has_no_detection(fake_predictor):
    result = analyze_image(concrete_image(512, 512, crack=False), fake_predictor, "clean")
    assert result.detections == [] and result.severity.level == "faible"


def test_small_image_single_tile(fake_predictor):
    result = analyze_image(concrete_image(200, 150), fake_predictor, "small")
    assert result.detections[0].bbox.w == 200 and result.detections[0].bbox.h == 150

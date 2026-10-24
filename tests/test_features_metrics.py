import numpy as np

from chantier_defects.eval.metrics import compute_metrics, sample_confidence, tune_thresholds
from chantier_defects.models.features import extract_features
from tests.conftest import concrete_image


def test_features_fixed_size_and_finite():
    a = extract_features(np.asarray(concrete_image(256, 256, crack=True)))
    b = extract_features(np.asarray(concrete_image(300, 200, crack=False)))
    assert a.shape == b.shape and np.isfinite(a).all()


def test_crack_raises_morphology_signal():
    with_crack = extract_features(np.asarray(concrete_image(256, 256, crack=True)))
    without = extract_features(np.asarray(concrete_image(256, 256, crack=False)))
    # Dernières features : morphologie ; la p95 du black top-hat (index -6) doit monter avec une fissure
    assert with_crack[-6] > without[-6]


def test_thresholds_favor_recall():
    y = np.array([[1], [1], [1], [0], [0], [0], [0], [0]])
    p = np.array([[0.9], [0.6], [0.35], [0.4], [0.3], [0.2], [0.1], [0.05]])
    thr = tune_thresholds(y, p, beta=2.0)
    m = compute_metrics(y, p, thr, ["crack"])
    assert m["per_class"]["crack"]["recall"] == 1.0  # F2 préfère tout rattraper quitte à une fausse alerte


def test_metrics_handle_absent_class():
    y = np.array([[1, 0], [0, 0]])
    p = np.array([[0.8, 0.1], [0.2, 0.3]])
    m = compute_metrics(y, p, np.array([0.5, 0.5]), ["crack", "spallation"])
    assert m["per_class"]["spallation"]["pr_auc"] is None
    assert m["macro_f1"] == 1.0  # moyenne sur les classes présentes seulement


def test_sample_confidence():
    assert np.allclose(sample_confidence(np.array([[0.9, 0.2], [0.5, 0.99]])), [0.8, 0.5])

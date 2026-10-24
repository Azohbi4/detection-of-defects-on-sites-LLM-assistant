from chantier_defects.monitoring import Monitor


def test_drift_alert_triggers_and_recovers():
    m = Monitor(window=8, reference_confidence=0.9, drop=0.15)
    m.record("/analyze", 10, confidences=[0.9, 0.92])
    assert not m.drift_alert
    for _ in range(4):
        m.record("/analyze", 10, confidences=[0.6, 0.6])
    assert m.drift_alert
    for _ in range(4):
        m.record("/analyze", 10, confidences=[0.95, 0.95])
    assert not m.drift_alert


def test_no_alert_without_reference():
    m = Monitor(window=4, reference_confidence=None)
    m.record("/analyze", 10, confidences=[0.1] * 10)
    assert not m.drift_alert and m.snapshot()["rolling_confidence"] < 0.2

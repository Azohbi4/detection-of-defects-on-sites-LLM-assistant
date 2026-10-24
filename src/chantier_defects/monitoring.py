"""Monitoring léger : latence, distribution des scores, alerte de dérive sur la confiance moyenne.

Les événements sont écrits en JSON sur la sortie standard : Azure Container Apps les envoie dans
Log Analytics, où une règle d'alerte peut se déclencher sur `"event": "drift_alert"`.
"""

import json
import logging
import time
from collections import deque

import numpy as np

log = logging.getLogger("chantier_defects.monitoring")


class Monitor:
    def __init__(self, window: int = 200, reference_confidence: float | None = None, drop: float = 0.15):
        self.reference = reference_confidence
        self.drop = drop
        self.min_samples = max(window // 4, 1)
        self.confidences: deque[float] = deque(maxlen=window)
        self.latencies: deque[float] = deque(maxlen=1000)
        self.scores: deque[float] = deque(maxlen=5000)
        self.requests = 0
        self.drift_alert = False

    def record(self, endpoint: str, latency_ms: float, confidences=(), scores=()) -> None:
        self.requests += 1
        self.latencies.append(latency_ms)
        self.confidences.extend(confidences)
        self.scores.extend(scores)
        self._emit(
            "request",
            endpoint=endpoint,
            latency_ms=round(latency_ms, 1),
            n_images=len(confidences),
            mean_confidence=round(float(np.mean(confidences)), 4) if len(confidences) else None,
        )
        self._check_drift()

    def _check_drift(self) -> None:
        if self.reference is None or len(self.confidences) < self.min_samples:
            return
        rolling = float(np.mean(self.confidences))
        drifting = rolling < self.reference - self.drop
        if drifting and not self.drift_alert:
            self._emit(
                "drift_alert",
                level=logging.WARNING,
                rolling_confidence=round(rolling, 4),
                reference_confidence=self.reference,
            )
        elif not drifting and self.drift_alert:
            self._emit("drift_recovered", rolling_confidence=round(rolling, 4))
        self.drift_alert = drifting

    def _emit(self, event: str, level: int = logging.INFO, **fields) -> None:
        log.log(
            level, json.dumps({"event": event, "ts": round(time.time(), 3), **fields}, ensure_ascii=False)
        )

    def snapshot(self) -> dict:
        lat = np.array(self.latencies) if self.latencies else None
        hist, edges = np.histogram(list(self.scores), bins=10, range=(0, 1))
        return {
            "requests": self.requests,
            "latency_ms_p50": float(np.percentile(lat, 50)) if lat is not None else None,
            "latency_ms_p95": float(np.percentile(lat, 95)) if lat is not None else None,
            "rolling_confidence": float(np.mean(self.confidences)) if self.confidences else None,
            "reference_confidence": self.reference,
            "drift_alert": self.drift_alert,
            "detection_score_histogram": {
                f"{edges[i]:.1f}-{edges[i + 1]:.1f}": int(h) for i, h in enumerate(hist)
            },
        }

"""Inférence sur des images entières par découpage en tuiles.

Les modèles sont entraînés sur des patches (SDNET2018 : 256 px, CODEBRIM : recadrages de défauts).
Sur une photo de chantier, on classe donc chaque tuile, puis on fusionne les tuiles positives
voisines (8-connexité, les fissures sont souvent diagonales) en zones : chaque zone devient une
détection avec une boîte englobante, un score et une étendue.
"""

from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image
from scipy import ndimage

from ..eval.metrics import sample_confidence
from ..schemas import BBox, Detection, ImageAnalysis
from ..severity import assess_image


class Predictor(Protocol):
    classes: list[str]
    thresholds: np.ndarray
    version: str
    reference_confidence: float | None

    def predict_proba(self, tiles: list[Image.Image]) -> np.ndarray:
        """Retourne une matrice (n_tuiles, n_classes) de probabilités."""
        ...


class TorchPredictor:
    def __init__(self, path: Path, device: str | None = None, batch_size: int = 32):
        import torch

        from ..data.dataset import eval_transforms
        from .cnn import build_model

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.classes = list(ckpt["classes"])
        self.thresholds = np.asarray(ckpt["thresholds"], dtype=float)
        self.reference_confidence = ckpt.get("reference_confidence")
        self.version = f"{ckpt['run_name']}@{ckpt.get('created', 'unknown')}"
        self.transform = eval_transforms(ckpt["img_size"])
        self.batch_size = batch_size
        self.model = build_model(ckpt["arch"], len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    def predict_proba(self, tiles: list[Image.Image]) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(tiles), self.batch_size):
                batch = self.torch.stack([self.transform(t) for t in tiles[i : i + self.batch_size]])
                out.append(self.torch.sigmoid(self.model(batch.to(self.device))).cpu().numpy())
        return np.concatenate(out)


class BaselinePredictor:
    def __init__(self, path: Path):
        import joblib

        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.classes = list(bundle["classes"])
        self.thresholds = np.asarray(bundle["thresholds"], dtype=float)
        self.reference_confidence = None
        self.version = f"baseline-rf@{path.name}"

    def predict_proba(self, tiles: list[Image.Image]) -> np.ndarray:
        from .baseline import rf_proba
        from .features import extract_features

        X = np.stack([extract_features(np.asarray(t)) for t in tiles])
        return rf_proba(self.model, X)


def load_predictor(path: Path) -> Predictor:
    if path.suffix == ".pt":
        return TorchPredictor(path)
    if path.suffix == ".joblib":
        return BaselinePredictor(path)
    raise ValueError(f"Format de modèle non reconnu : {path}")


def _positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:  # la dernière tuile colle au bord, rien n'est ignoré
        starts.append(length - tile)
    return starts


def analyze_image(
    img: Image.Image, predictor: Predictor, image_id: str, tile: int = 256, stride: int = 256
) -> ImageAnalysis:
    img = img.convert("RGB")
    width, height = img.size
    xs, ys = _positions(width, tile, stride), _positions(height, tile, stride)
    tw, th = min(tile, width), min(tile, height)
    boxes = [(x, y) for y in ys for x in xs]
    probs = predictor.predict_proba([img.crop((x, y, x + tw, y + th)) for x, y in boxes])
    grid_shape = (len(ys), len(xs))

    detections, affected = [], {}
    for c, defect_type in enumerate(predictor.classes):
        positive = (probs[:, c] >= predictor.thresholds[c]).reshape(grid_shape)
        if not positive.any():
            continue
        affected[defect_type] = round(float(positive.mean()), 4)
        labels, n_zones = ndimage.label(positive, structure=np.ones((3, 3)))
        scores = probs[:, c].reshape(grid_shape)
        for zone in range(1, n_zones + 1):
            rows, cols = np.nonzero(labels == zone)
            x0, y0 = xs[cols.min()], ys[rows.min()]
            x1, y1 = xs[cols.max()] + tw, ys[rows.max()] + th
            detections.append(
                Detection(
                    defect_type=defect_type,
                    score=round(float(scores[rows, cols].max()), 4),
                    bbox=BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0),
                    n_tiles=len(rows),
                )
            )

    analysis = ImageAnalysis(
        image_id=image_id,
        width=width,
        height=height,
        detections=sorted(detections, key=lambda d: d.score, reverse=True),
        affected_ratio=affected,
        mean_confidence=round(float(sample_confidence(probs).mean()), 4),
    )
    analysis.severity = assess_image(analysis)
    return analysis

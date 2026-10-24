import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from chantier_defects.llm.report import RetrievedChunk


class FakePredictor:
    """Classe « crack » toute tuile contenant des pixels très sombres : suffisant pour tester le pipeline."""

    classes = ["crack", "spallation"]
    thresholds = np.array([0.5, 0.5])
    version = "fake@test"
    reference_confidence = 0.9

    def predict_proba(self, tiles):
        out = []
        for tile in tiles:
            dark = (np.asarray(tile.convert("L")) < 60).mean()
            out.append([0.95 if dark > 0.002 else 0.05, 0.02])
        return np.array(out)


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, k):
        self.queries.append(query)
        defect = "fissure" if "fissure" in query else "autre"
        return [
            RetrievedChunk(
                chunk_id=f"{defect}-guide#0",
                title=f"Fiche {defect}",
                url=None,
                text=f"Texte de référence sur {defect}.",
                score=0.8,
            )
        ]


def concrete_image(width=768, height=512, crack=True) -> Image.Image:
    """Fond « béton » gris bruité, avec éventuellement une fissure sombre dans le quart haut-gauche."""
    rng = np.random.default_rng(0)
    arr = np.clip(rng.normal(150, 10, (height, width, 3)), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    if crack:
        ImageDraw.Draw(img).line([(20, 20), (230, 200)], fill=(20, 20, 20), width=3)
    return img


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def fake_predictor():
    return FakePredictor()


@pytest.fixture
def fake_retriever():
    return FakeRetriever()

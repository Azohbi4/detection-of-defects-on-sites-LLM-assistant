"""Tests de fumée : les scripts d'entraînement tournent de bout en bout sur un mini jeu synthétique
et produisent un modèle que l'API sait charger."""

import sys

import pandas as pd
import pytest

from chantier_defects.models.predictor import analyze_image, load_predictor
from tests.conftest import concrete_image


@pytest.fixture
def tiny_splits(tmp_path):
    rows = []
    for i in range(36):
        cracked = i % 2 == 0
        path = tmp_path / "img" / f"{i}.png"
        path.parent.mkdir(exist_ok=True)
        concrete_image(96, 96, crack=cracked).save(path)
        split = "train" if i < 24 else "val" if i < 30 else "test"
        rows.append({"path": path.as_posix(), "group": f"g{i}", "crack": int(cracked), "split": split})
    out = tmp_path / "splits.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def run_cli(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    module.main()


def test_baseline_train_and_predict(tiny_splits, tmp_path, monkeypatch):
    from chantier_defects.models import baseline

    model = tmp_path / "baseline.joblib"
    run_cli(
        baseline,
        [
            str(tiny_splits),
            "--out",
            str(model),
            "--reports",
            str(tmp_path / "rep"),
            "--cache",
            str(tmp_path / "cache"),
            "--n-estimators",
            "20",
            "--no-mlflow",
        ],
        monkeypatch,
    )
    assert (tmp_path / "rep" / "metrics_test.json").exists()
    analysis = analyze_image(concrete_image(256, 256), load_predictor(model), "x")
    assert analysis.mean_confidence > 0


def test_cnn_train_and_predict(tiny_splits, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from chantier_defects.models import train

    model = tmp_path / "model.pt"
    run_cli(
        train,
        [
            str(tiny_splits),
            "--no-pretrained",
            "--img-size",
            "64",
            "--batch-size",
            "8",
            "--epochs-head",
            "1",
            "--epochs-ft",
            "1",
            "--num-workers",
            "0",
            "--no-mlflow",
            "--out",
            str(model),
            "--reports",
            str(tmp_path / "rep"),
        ],
        monkeypatch,
    )
    predictor = load_predictor(model)
    assert predictor.classes == ["crack"] and 0 <= predictor.reference_confidence <= 1
    analysis = analyze_image(concrete_image(512, 256), predictor, "x")
    assert analysis.image_id == "x" and analysis.severity is not None

"""Baseline : features classiques (HSV, HOG, LBP, morphologie) + Random Forest multi-label.

Usage :
    python -m chantier_defects.models.baseline data/processed/splits.csv --out models/baseline.joblib
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

from ..data.split import class_columns
from ..eval.metrics import compute_metrics, save_metrics, save_predictions, tune_thresholds
from ..tracking import Tracker, flat_metrics
from .features import extract_features


def _features_of(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return extract_features(np.asarray(img.convert("RGB")))


def features_for(paths: list[str], cache: Path | None = None, n_jobs: int = -1) -> np.ndarray:
    """Calcule (ou relit depuis le cache .npz) les features d'une liste d'images."""
    if cache and cache.exists():
        data = np.load(cache, allow_pickle=True)
        if list(data["paths"]) == list(paths):
            return data["X"]
    X = np.stack(Parallel(n_jobs=n_jobs)(delayed(_features_of)(p) for p in paths))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, paths=np.array(paths, dtype=object))
    return X


def rf_proba(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """Probabilité de la classe positive pour chaque sortie, robuste aux classes absentes du train."""
    raw = model.predict_proba(X)
    if isinstance(raw, list):  # multi-sorties : une matrice et un jeu de classes par défaut
        class_sets = model.classes_
    else:
        raw, class_sets = [raw], [model.classes_]
    cols = []
    for proba, classes in zip(raw, class_sets, strict=True):
        classes = list(classes)
        cols.append(proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X)))
    return np.column_stack(cols)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("splits", type=Path)
    parser.add_argument("--out", type=Path, default=Path("models/baseline.joblib"))
    parser.add_argument("--reports", type=Path, default=Path("reports/baseline"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/features"))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument(
        "--max-train", type=int, default=None, help="Sous-échantillonne le train (essais rapides)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.splits)
    classes = class_columns(df)
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    if args.max_train and len(parts["train"]) > args.max_train:
        parts["train"] = parts["train"].sample(args.max_train, random_state=args.seed)

    t0 = time.time()
    X = {s: features_for(p["path"].tolist(), args.cache / f"{s}.npz") for s, p in parts.items()}
    y = {s: p[classes].to_numpy() for s, p in parts.items()}
    print(f"Features : {X['train'].shape[1]} dimensions, {time.time() - t0:.0f} s")

    tracker = Tracker(enabled=not args.no_mlflow)
    with tracker.run("chantier-defects", "baseline-rf"):
        params = {
            "model": "random_forest",
            "n_estimators": args.n_estimators,
            "classes": ",".join(classes),
            "n_train": len(y["train"]),
            "seed": args.seed,
        }
        tracker.params(params)

        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=2,
            class_weight="balanced_subsample",  # compense le déséquilibre fond / défauts
            n_jobs=-1,
            random_state=args.seed,
        )
        rf.fit(X["train"], y["train"] if len(classes) > 1 else y["train"].ravel())

        prob = {s: rf_proba(rf, X[s]) for s in ("val", "test")}
        thresholds = tune_thresholds(y["val"], prob["val"])
        for split in ("val", "test"):
            metrics = compute_metrics(y[split], prob[split], thresholds, classes)
            save_metrics(metrics, args.reports / f"metrics_{split}.json")
            tracker.metrics(flat_metrics(metrics, split))
        save_predictions(
            parts["test"]["path"].tolist(),
            y["test"],
            prob["test"],
            thresholds,
            classes,
            args.reports / "predictions_test.csv",
        )
        print(f"Test : macro-F1 {metrics['macro_f1']:.3f}, macro-rappel {metrics['macro_recall']:.3f}")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kind": "baseline", "model": rf, "classes": classes, "thresholds": thresholds}, args.out)
        tracker.artifact(args.reports / "metrics_test.json")
        print(f"Modèle -> {args.out}, métriques -> {args.reports}")


if __name__ == "__main__":
    main()

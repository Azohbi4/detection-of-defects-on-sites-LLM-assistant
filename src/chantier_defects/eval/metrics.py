"""Métriques orientées métier pour la classification multi-label.

Un défaut manqué coûte plus cher qu'une fausse alerte : les seuils de décision sont choisis sur la
validation en maximisant le F2 (le rappel compte deux fois plus que la précision), puis appliqués tels
quels au test.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, precision_recall_curve

DEFAULT_THRESHOLD = 0.5


def sample_confidence(probs: np.ndarray) -> np.ndarray:
    """Confiance par échantillon = max(p, 1-p) de la classe la plus incertaine.

    Proche de 1 : le modèle tranche nettement ; proche de 0,5 : il hésite. Sa moyenne sur la validation
    sert de référence pour détecter une dérive en production (images trop différentes de l'entraînement).
    """
    return np.maximum(probs, 1 - probs).min(axis=1)


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 2.0) -> np.ndarray:
    thresholds = np.full(y_true.shape[1], DEFAULT_THRESHOLD)
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            continue
        precision, recall, cuts = precision_recall_curve(y_true[:, c], y_prob[:, c])
        precision, recall = precision[:-1], recall[:-1]  # le dernier point n'a pas de seuil associé
        fbeta = (1 + beta**2) * precision * recall / np.maximum(beta**2 * precision + recall, 1e-12)
        thresholds[c] = float(cuts[np.argmax(fbeta)])
    return thresholds


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    has_pos = y_true.sum() > 0
    return {
        "support": int(y_true.sum()),
        "precision": precision,
        "recall": recall,
        "f1": float(fbeta_score(y_true, y_pred, beta=1.0, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)) if has_pos else None,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray, classes: list[str]
) -> dict:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= thresholds).astype(int)
    per_class = {c: _binary_metrics(y_true[:, i], y_pred[:, i], y_prob[:, i]) for i, c in enumerate(classes)}
    evaluated = [m for m in per_class.values() if m["support"] > 0]
    pr_aucs = [m["pr_auc"] for m in evaluated]
    return {
        "n_samples": int(len(y_true)),
        "thresholds": dict(zip(classes, map(float, thresholds), strict=True)),
        "macro_f1": float(np.mean([m["f1"] for m in evaluated])) if evaluated else 0.0,
        "macro_recall": float(np.mean([m["recall"] for m in evaluated])) if evaluated else 0.0,
        "macro_pr_auc": float(np.mean(pr_aucs)) if pr_aucs else None,
        # Vue « inspecteur » : le patch contient-il au moins un défaut ?
        "any_defect": _binary_metrics(y_true.max(axis=1), y_pred.max(axis=1), y_prob.max(axis=1)),
        "per_class": per_class,
    }


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def save_predictions(
    paths: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    classes: list[str],
    out: Path,
) -> None:
    df = pd.DataFrame({"path": paths})
    for i, c in enumerate(classes):
        df[f"true_{c}"] = y_true[:, i].astype(int)
        df[f"prob_{c}"] = y_prob[:, i]
        df[f"pred_{c}"] = (y_prob[:, i] >= thresholds[i]).astype(int)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

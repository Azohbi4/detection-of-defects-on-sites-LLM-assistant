"""Suivi d'expériences MLflow, désactivable (tests, machine sans MLflow).

Par défaut les runs sont stockés dans ./mlruns ; interface : `mlflow ui` puis http://localhost:5000.
"""

from contextlib import contextmanager
from pathlib import Path


class Tracker:
    def __init__(self, enabled: bool):
        self.mlflow = None
        if enabled:
            try:
                import mlflow
            except ImportError:
                print("MLflow non installé : suivi d'expériences désactivé.")
            else:
                self.mlflow = mlflow

    @contextmanager
    def run(self, experiment: str, run_name: str):
        if self.mlflow is None:
            yield self
            return
        self.mlflow.set_experiment(experiment)
        with self.mlflow.start_run(run_name=run_name):
            yield self

    def params(self, params: dict) -> None:
        if self.mlflow:
            self.mlflow.log_params(params)

    def metrics(self, metrics: dict, step: int | None = None) -> None:
        if self.mlflow:
            self.mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None}, step=step)

    def artifact(self, path: Path) -> None:
        if self.mlflow and path.exists():
            self.mlflow.log_artifact(str(path))


def flat_metrics(metrics: dict, prefix: str) -> dict[str, float]:
    """Aplatit le dict de compute_metrics pour MLflow (ex. test_recall_crack)."""
    out = {f"{prefix}_macro_f1": metrics["macro_f1"], f"{prefix}_macro_recall": metrics["macro_recall"]}
    if metrics["macro_pr_auc"] is not None:
        out[f"{prefix}_macro_pr_auc"] = metrics["macro_pr_auc"]
    for cls, m in metrics["per_class"].items():
        out[f"{prefix}_recall_{cls}"] = m["recall"]
        out[f"{prefix}_f1_{cls}"] = m["f1"]
    return out

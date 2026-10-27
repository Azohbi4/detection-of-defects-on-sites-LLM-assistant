import pytest

from chantier_defects.eval.rag import check_labels, evaluate
from chantier_defects.llm.report import RetrievedChunk

EVAL_SET = {
    "defects": {
        "crack": {"queries": ["fissure"], "relevant": ["doc#0"]},
        "spallation": {"queries": ["éclatement"], "relevant": ["doc#1"]},
    },
    "off_topic": ["plomberie"],
}


class TableRetriever:
    """Renvoie, pour chaque requête, des résultats fixés à l'avance."""

    def __init__(self, table, default):
        self.table, self.default = table, default

    def search(self, query, k):
        rows = self.table.get(query, self.default)
        return [RetrievedChunk(chunk_id=c, title="t", text="x", score=s) for c, s in rows[:k]]


def make_retriever():
    # crack : toujours bien servi ; spallation : passage pertinent seulement en 2e position, score 0.55
    # « plomberie » (hors sujet) : meilleur score 0.45
    table = {"plomberie": [("doc#0", 0.45)], "éclatement": [("doc#0", 0.7), ("doc#1", 0.55)]}
    default_rows = [("doc#0", 0.8), ("doc#1", 0.6)]
    return TableRetriever(table, default_rows)


def test_evaluate_metrics_and_threshold():
    metrics = evaluate(make_retriever(), EVAL_SET, k=2, thresholds=[0.4, 0.5, 0.6])

    assert metrics["n_queries"] == 4  # requête de production + reformulation, pour 2 défauts
    assert metrics["hit@2"] == 1.0
    # spallation : pertinent au rang 2 pour la reformulation et pour la requête de production (défaut)
    assert metrics["mrr@2"] == pytest.approx((1 + 1 + 0.5 + 0.5) / 4)
    sweep = {row["threshold"]: row for row in metrics["sweep"]}
    assert sweep[0.4]["off_topic_leak"] == 1.0 and sweep[0.5]["off_topic_leak"] == 0.0
    assert sweep[0.6]["queries_served"] == 0.75  # seule la reformulation (pertinent à 0.55) perd sa source
    # milieu entre le meilleur hors sujet (0.45) et le plus faible pertinent en production (0.6)
    assert metrics["recommended_threshold"] == pytest.approx(0.525)


def test_check_labels_rejects_stale_ids():
    with pytest.raises(ValueError, match="doc#1"):
        check_labels(EVAL_SET, known_ids={"doc#0"})

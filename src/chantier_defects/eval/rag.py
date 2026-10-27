"""Évaluation de la recherche documentaire (RAG) et calibration du seuil rag_min_score.

Indexe knowledge/ dans un Qdrant en mémoire, interroge avec les requêtes de production (labels.RAG_QUERIES)
et les reformulations de knowledge/eval_queries.yaml, puis compare aux passages étiquetés pertinents.

Usage :
    python -m chantier_defects.eval.rag knowledge/ --out reports/rag
"""

import argparse
import json
from pathlib import Path

import yaml

from ..config import Settings
from ..labels import RAG_QUERIES

THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(8)]  # 0.30 … 0.65


def load_eval_set(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def eval_queries(eval_set: dict) -> list[dict]:
    """Requête de production + reformulations, chacune avec ses passages pertinents."""
    queries = []
    for defect, spec in eval_set["defects"].items():
        relevant = set(spec["relevant"])
        queries.append({"defect": defect, "kind": "production", "text": RAG_QUERIES[defect], "relevant": relevant})
        queries += [{"defect": defect, "kind": "reformulation", "text": q, "relevant": relevant} for q in spec["queries"]]
    return queries


def check_labels(eval_set: dict, known_ids: set[str]) -> None:
    unknown = {cid for spec in eval_set["defects"].values() for cid in spec["relevant"]} - known_ids
    if unknown:
        raise ValueError(f"Passages étiquetés absents de l'index (ré-étiqueter) : {sorted(unknown)}")


def evaluate(retriever, eval_set: dict, k: int = 3, thresholds=THRESHOLDS) -> dict:
    runs = []
    for q in eval_queries(eval_set):
        hits = retriever.search(q["text"], k)
        runs.append({**q, "hits": [(h.chunk_id, h.score, h.chunk_id in q["relevant"]) for h in hits]})
    off_topic = [max((h.score for h in retriever.search(t, k)), default=0.0) for t in eval_set["off_topic"]]

    def first_relevant_rank(hits):
        return next((i + 1 for i, (_, _, rel) in enumerate(hits) if rel), None)

    ranks = [first_relevant_rank(r["hits"]) for r in runs]
    retrieved = [h for r in runs for h in r["hits"]]
    relevant_scores = [s for _, s, rel in retrieved if rel]
    irrelevant_scores = [s for _, s, rel in retrieved if not rel]

    sweep = []
    for t in thresholds:
        sweep.append(
            {
                "threshold": t,
                # requêtes qui gardent au moins un passage pertinent : sinon le rapport perd ses sources
                "queries_served": sum(any(rel and s >= t for _, s, rel in r["hits"]) for r in runs) / len(runs),
                "relevant_kept": sum(s >= t for s in relevant_scores) / max(len(relevant_scores), 1),
                "irrelevant_kept": sum(s >= t for s in irrelevant_scores) / max(len(irrelevant_scores), 1),
                "off_topic_leak": sum(s >= t for s in off_topic) / len(off_topic),
            }
        )
    # Seuil recommandé : milieu de l'écart entre le meilleur passage hors sujet et le plus faible passage
    # pertinent remonté par une requête de production (les seules envoyées par l'API). Sans écart, aucun.
    production_relevant = [s for r in runs if r["kind"] == "production" for _, s, rel in r["hits"] if rel]
    gap_low, gap_high = max(off_topic), min(production_relevant, default=0.0)
    return {
        "k": k,
        "n_queries": len(runs),
        f"hit@{k}": sum(r is not None for r in ranks) / len(runs),
        f"precision@{k}": len(relevant_scores) / max(len(retrieved), 1),
        f"mrr@{k}": sum(1 / r for r in ranks if r) / len(runs),
        "max_off_topic_top1": gap_low,
        "min_production_relevant": gap_high,
        "sweep": sweep,
        "recommended_threshold": (gap_low + gap_high) / 2 if gap_high > gap_low else None,
        "per_query": [
            {"defect": r["defect"], "kind": r["kind"], "query": r["text"],
             "hits": [{"chunk_id": c, "score": round(s, 3), "relevant": rel} for c, s, rel in r["hits"]]}
            for r in runs
        ],
    }


def markdown(metrics: dict) -> str:
    k = metrics["k"]
    t = metrics["recommended_threshold"]
    recommended = "aucun (pas d'écart)" if t is None else f"{t:.2f}"
    lines = [
        f"hit@{k} {metrics[f'hit@{k}']:.2f} · precision@{k} {metrics[f'precision@{k}']:.2f} · "
        f"MRR@{k} {metrics[f'mrr@{k}']:.2f} · hors sujet max {metrics['max_off_topic_top1']:.2f} · "
        f"pertinent min (production) {metrics['min_production_relevant']:.2f} · "
        f"seuil recommandé {recommended}",
        "",
        "| Seuil | Requêtes servies | Pertinents gardés | Non pertinents gardés | Hors sujet qui passent |",
        "|---|---|---|---|---|",
    ]
    for row in metrics["sweep"]:
        lines.append(
            f"| {row['threshold']:.2f} | {row['queries_served']:.0%} | {row['relevant_kept']:.0%} | "
            f"{row['irrelevant_kept']:.0%} | {row['off_topic_leak']:.0%} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("docs_dir", type=Path)
    parser.add_argument("--eval-set", type=Path, default=None, help="défaut : <docs_dir>/eval_queries.yaml")
    parser.add_argument("--out", type=Path, default=Path("reports/rag"))
    args = parser.parse_args()

    from qdrant_client import QdrantClient

    from ..llm.rag import QdrantRetriever, chunk_text, ingest, load_sources, read_document

    settings = Settings()
    retriever = QdrantRetriever(QdrantClient(":memory:"), "eval", settings.embedding_model)
    ingest(args.docs_dir, retriever)
    known = {
        f"{e['file']}#{i}"
        for e in load_sources(args.docs_dir)
        for i, _ in enumerate(chunk_text(read_document(args.docs_dir / e["file"])))
    }
    eval_set = load_eval_set(args.eval_set or args.docs_dir / "eval_queries.yaml")
    check_labels(eval_set, known)

    metrics = evaluate(retriever, eval_set, k=settings.rag_top_k)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(markdown(metrics))


if __name__ == "__main__":
    main()

"""Mini-RAG : documents publics (guides d'inspection, fiches de pathologies du béton) indexés dans Qdrant.

Les embeddings sont calculés localement par fastembed (ONNX, multilingue), sans service externe.

Usage :
    python -m chantier_defects.llm.rag ingest knowledge/            # indexe les documents listés dans knowledge/sources.yaml
    python -m chantier_defects.llm.rag search "fissure traversante"  # vérifie la recherche
"""

import argparse
import logging
import re
import uuid
from pathlib import Path

import yaml

from ..config import Settings, get_settings
from .report import RetrievedChunk

log = logging.getLogger(__name__)

TEXT_EXTS = {".md", ".txt"}


def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:
    """Découpe par paragraphes, en morceaux d'au plus max_chars, avec recouvrement pour garder le contexte."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces = []
    for p in paragraphs:  # un paragraphe trop long est coupé en phrases, puis brutalement si besoin
        if len(p) <= max_chars:
            pieces.append(p)
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", p):
            pieces.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))

    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > max_chars:
            chunks.append(current)
            current = current[-overlap:] if overlap else ""
        current = f"{current}\n{piece}".strip() if current else piece
    if current:
        chunks.append(current)
    return chunks


def read_document(path: Path) -> str:
    if path.suffix.lower() in TEXT_EXTS:
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    raise ValueError(f"Format non pris en charge : {path}")


def load_sources(docs_dir: Path) -> list[dict]:
    """Lit knowledge/sources.yaml : liste de {file, title, url, licence}."""
    manifest = docs_dir / "sources.yaml"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} introuvable : liste les documents et leur licence (voir knowledge/README.md)"
        )
    entries = yaml.safe_load(manifest.read_text(encoding="utf-8")) or []
    for entry in entries:
        if not (docs_dir / entry["file"]).exists():
            raise FileNotFoundError(f"Document listé mais absent : {docs_dir / entry['file']}")
    return entries


class QdrantRetriever:
    def __init__(self, client, collection: str, embedding_model: str):
        self.client, self.collection, self.embedding_model = client, collection, embedding_model

    def _document(self, text: str):
        from qdrant_client import models

        return models.Document(text=text, model=self.embedding_model)

    def recreate_collection(self) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(
                size=self.client.get_embedding_size(self.embedding_model), distance=models.Distance.COSINE
            ),
        )

    def index(self, chunks: list[dict]) -> None:
        from qdrant_client import models

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["chunk_id"])),
                vector=self._document(c["text"]),
                payload=c,
            )
            for c in chunks
        ]
        for i in range(0, len(points), 64):
            self.client.upsert(self.collection, points=points[i : i + 64])

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        hits = self.client.query_points(
            self.collection, query=self._document(query), limit=k, with_payload=True
        )
        return [
            RetrievedChunk(
                chunk_id=h.payload["chunk_id"],
                title=h.payload["title"],
                url=h.payload.get("url"),
                text=h.payload["text"],
                score=h.score,
            )
            for h in hits.points
        ]


def make_client(settings: Settings):
    from qdrant_client import QdrantClient

    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    if settings.qdrant_path:
        return QdrantClient(path=settings.qdrant_path)
    return None


def build_retriever(settings: Settings) -> QdrantRetriever | None:
    try:
        client = make_client(settings)
    except Exception as exc:
        log.warning("Qdrant inaccessible (%s) : rapports sans sources.", exc)
        return None
    if client is None:
        log.warning("Ni CHANTIER_QDRANT_URL ni CHANTIER_QDRANT_PATH : rapports sans sources.")
        return None
    return QdrantRetriever(client, settings.qdrant_collection, settings.embedding_model)


def ingest(docs_dir: Path, retriever: QdrantRetriever) -> int:
    entries = load_sources(docs_dir)
    if not entries:
        raise ValueError(f"{docs_dir / 'sources.yaml'} ne liste aucun document : rien à indexer.")
    chunks = []
    for entry in entries:
        text = read_document(docs_dir / entry["file"])
        for i, chunk in enumerate(chunk_text(text)):
            chunks.append(
                {
                    "chunk_id": f"{entry['file']}#{i}",
                    "title": entry["title"],
                    "url": entry.get("url"),
                    "licence": entry.get("licence"),
                    "text": chunk,
                }
            )
    retriever.recreate_collection()
    retriever.index(chunks)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("docs_dir", type=Path)
    p_search = sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    retriever = build_retriever(get_settings())
    if retriever is None:
        raise SystemExit("Configure CHANTIER_QDRANT_URL (serveur) ou CHANTIER_QDRANT_PATH (local).")
    if args.cmd == "ingest":
        n = ingest(args.docs_dir, retriever)
        print(f"{n} passages indexés dans « {retriever.collection} ».")
    else:
        for hit in retriever.search(args.query, args.k):
            print(f"[{hit.score:.3f}] {hit.title} ({hit.chunk_id})\n{hit.text[:300]}\n")


if __name__ == "__main__":
    main()

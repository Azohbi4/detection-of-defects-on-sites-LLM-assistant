import pytest

from chantier_defects.llm.rag import QdrantRetriever, chunk_text, ingest, load_sources


def test_chunks_respect_size_and_keep_overlap():
    text = "\n\n".join(f"Paragraphe {i}. " + "mot " * 60 for i in range(10))
    chunks = chunk_text(text, max_chars=500, overlap=100)
    assert len(chunks) > 1 and all(len(c) <= 500 + 100 for c in chunks)
    assert chunks[1].startswith(chunks[0][-100:].strip()[:20])  # recouvrement entre morceaux successifs


def test_long_paragraph_is_split():
    assert len(chunk_text("x" * 2500, max_chars=900, overlap=0)) == 3


def test_sources_manifest_required(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sources(tmp_path)
    (tmp_path / "sources.yaml").write_text("- file: absent.md\n  title: T\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_sources(tmp_path)


def test_recreate_collection_drops_old_points_on_disk(tmp_path):
    from qdrant_client import QdrantClient, models

    client = QdrantClient(path=str(tmp_path))
    client.create_collection("c", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE))
    client.upsert("c", points=[models.PointStruct(id=1, vector=[0.1] * 4)])
    retriever = QdrantRetriever(client, "c", "google/embeddinggemma-300m")  # 768 dimensions

    retriever.recreate_collection()
    client.upsert("c", points=[models.PointStruct(id=2, vector=[0.1] * 768)])
    assert client.count("c").count == 1
    client.close()


@pytest.mark.slow
def test_ingest_and_search_in_memory(tmp_path):
    """Intégration réelle Qdrant (mode mémoire) + embeddings fastembed (télécharge ~1,2 Go au premier lancement)."""
    from qdrant_client import QdrantClient

    from chantier_defects.config import Settings

    (tmp_path / "fissures.md").write_text(
        "Les fissures du béton peuvent provenir du retrait, de charges excessives ou de la corrosion des aciers.\n\n"
        "Une fissure active doit être suivie par des témoins avant toute injection.",
        encoding="utf-8",
    )
    (tmp_path / "efflorescence.md").write_text(
        "L'efflorescence est un dépôt blanchâtre de sels lié aux circulations d'eau à travers le béton.",
        encoding="utf-8",
    )
    (tmp_path / "sources.yaml").write_text(
        "- file: fissures.md\n  title: Fiche fissures (test)\n- file: efflorescence.md\n  title: Fiche efflorescence (test)\n",
        encoding="utf-8",
    )

    retriever = QdrantRetriever(QdrantClient(":memory:"), "test", Settings().embedding_model)
    assert ingest(tmp_path, retriever) == 2
    hits = retriever.search("dépôt blanc lié à l'eau sur un mur", k=2)
    assert hits[0].title == "Fiche efflorescence (test)"
    hits = retriever.search("crack in concrete caused by rebar corrosion", k=1)  # requête en anglais
    assert hits[0].title == "Fiche fissures (test)"

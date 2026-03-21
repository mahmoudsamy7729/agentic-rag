import asyncio

import pytest

pytest.importorskip("chromadb")

from src.infrastructure.vector_db.chroma_vectorstore import ChromaVectorStore
from src.rag.models import RAGChunk


def test_chroma_vectorstore_upsert_and_search(tmp_path):
    async def _run():
        store = ChromaVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            collection_name="unit-test-collection",
        )

        chunks = [
            RAGChunk(doc_id="doc-1", chunk_id="chunk-0", source="a.txt", text="alpha"),
            RAGChunk(doc_id="doc-1", chunk_id="chunk-1", source="a.txt", text="beta"),
        ]
        embeddings = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]

        await store.upsert_chunks(chunks=chunks, embeddings=embeddings)
        return await store.similarity_search(query_embedding=[0.9, 0.1], top_k=1)

    results = asyncio.run(_run())

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-0"
    assert results[0].doc_id == "doc-1"
    assert results[0].source == "a.txt"

import asyncio

import pytest

from src.api.v1 import dependencies as deps
from src.infrastructure.llm import huggingface_embeddings as hf_module
from src.infrastructure.llm.openai_embeddings import OpenAIEmbeddingProvider
from src.rag.models import RAGChunk, RetrievedChunk
from src.rag.pipeline.services import RAGIngestionService, RAGRetrievalService
from src.rag.vectorstore.interface import VectorStore
from src.settings.config import settings


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._items: list[tuple[RAGChunk, list[float]]] = []

    async def upsert_chunks(self, *, chunks: list[RAGChunk], embeddings: list[list[float]]) -> None:
        for chunk, emb in zip(chunks, embeddings):
            self._items.append((chunk, emb))

    async def similarity_search(self, *, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        scored: list[tuple[float, RAGChunk]] = []
        for chunk, emb in self._items:
            dot = sum(a * b for a, b in zip(query_embedding, emb))
            scored.append((dot, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            RetrievedChunk(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                text=chunk.text,
                score=float(score),
            )
            for score, chunk in scored[:top_k]
        ]


@pytest.fixture(autouse=True)
def _restore_settings_and_cache():
    fields = [
        "embedding_provider",
        "openai_key",
        "embedding_model",
        "embedding_base_url",
    ]
    snapshot = {field: getattr(settings, field) for field in fields}
    deps.get_embedding_provider.cache_clear()
    try:
        yield
    finally:
        for field, value in snapshot.items():
            setattr(settings, field, value)
        deps.get_embedding_provider.cache_clear()


def test_openai_embedding_mode_ingest_retrieve(monkeypatch):
    async def fake_embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def fake_embed_query(self, text):
        return [1.0, 0.0]

    monkeypatch.setattr(OpenAIEmbeddingProvider, "embed_documents", fake_embed_documents)
    monkeypatch.setattr(OpenAIEmbeddingProvider, "embed_query", fake_embed_query)

    settings.embedding_provider = "openai"
    settings.openai_key = "openai-key"
    settings.embedding_model = "text-embedding-3-small"

    provider = deps.get_embedding_provider()
    store = InMemoryVectorStore()
    ingest_service = RAGIngestionService(
        embedding_provider=provider,
        vector_store=store,
        chunk_size=200,
        chunk_overlap=50,
    )
    retrieve_service = RAGRetrievalService(
        embedding_provider=provider,
        vector_store=store,
        default_top_k=2,
        prefetch_k=50,
    )

    async def _run():
        await ingest_service.ingest_text(text="Paris is the capital of France.", source="wiki", doc_id="doc-1")
        return await retrieve_service.retrieve(query="capital of france", top_k=1)

    results = asyncio.run(_run())

    assert len(results) == 1
    assert results[0].doc_id == "doc-1"


def test_huggingface_embedding_mode_ingest_retrieve(monkeypatch):
    class FakeHF:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    monkeypatch.setattr(hf_module, "_HuggingFaceEmbeddings", FakeHF)

    settings.embedding_provider = "huggingface"
    settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    provider = deps.get_embedding_provider()
    store = InMemoryVectorStore()
    ingest_service = RAGIngestionService(
        embedding_provider=provider,
        vector_store=store,
        chunk_size=200,
        chunk_overlap=50,
    )
    retrieve_service = RAGRetrievalService(
        embedding_provider=provider,
        vector_store=store,
        default_top_k=2,
        prefetch_k=50,
    )

    async def _run():
        await ingest_service.ingest_text(text="Cairo is the capital of Egypt.", source="wiki", doc_id="doc-2")
        return await retrieve_service.retrieve(query="capital of egypt", top_k=1)

    results = asyncio.run(_run())

    assert len(results) == 1
    assert results[0].doc_id == "doc-2"

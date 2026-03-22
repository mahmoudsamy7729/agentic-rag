from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from src.rag.embeddings.interface import EmbeddingProvider
from src.rag.ingestion.chunker import chunk_text
from src.rag.models import RetrievedChunk
from src.rag.reranker import Reranker
from src.rag.vectorstore.interface import VectorStore


@dataclass(slots=True)
class IngestionResult:
    doc_id: str
    chunks_ingested: int


class RAGIngestionService:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def ingest_text(
        self,
        *,
        text: str,
        source: str | None = None,
        doc_id: str | None = None,
    ) -> IngestionResult:
        resolved_doc_id = doc_id or str(uuid4())
        resolved_source = source or "inline-text"

        chunks = chunk_text(
            text=text,
            doc_id=resolved_doc_id,
            source=resolved_source,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        embeddings = await self._embedding_provider.embed_documents(
            [chunk.text for chunk in chunks]
        )
        await self._vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)

        return IngestionResult(
            doc_id=resolved_doc_id,
            chunks_ingested=len(chunks),
        )


class RAGRetrievalService:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        default_top_k: int,
        prefetch_k: int,
        reranker: Reranker | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._default_top_k = default_top_k
        self._prefetch_k = prefetch_k
        self._reranker = reranker

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int | None = None,
        doc_id: str | None = None,
    ) -> list[RetrievedChunk]:
        final_k = top_k or self._default_top_k
        if final_k < 1:
            raise ValueError("top_k must be >= 1")

        candidate_k = max(self._prefetch_k, final_k)
        query_embedding = await self._embedding_provider.embed_query(query)
        candidates = await self._vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=candidate_k,
            doc_id=doc_id,
        )
        if not candidates:
            return []
        if self._reranker is None:
            return candidates[:final_k]

        attempts = 2
        for attempt in range(1, attempts + 1):
            try:
                reranked = await self._reranker.rerank(
                    query=query,
                    chunks=candidates,
                    top_n=final_k,
                )
                return reranked[:final_k]
            except Exception as exc:
                if attempt >= attempts:
                    raise RuntimeError("Reranker failed after one retry.") from exc

        return candidates[:final_k]

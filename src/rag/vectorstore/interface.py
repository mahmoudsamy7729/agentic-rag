from __future__ import annotations

from abc import ABC, abstractmethod

from src.rag.models import RAGChunk, RetrievedChunk


class VectorStore(ABC):
    @abstractmethod
    async def upsert_chunks(
        self,
        *,
        chunks: list[RAGChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunks and their embeddings."""

    @abstractmethod
    async def similarity_search(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        doc_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return top-k most similar chunks."""

    @abstractmethod
    async def list_chunks(self, *, doc_id: str) -> list[RAGChunk]:
        """Return all indexed chunks for a document."""

    @abstractmethod
    async def delete_by_doc_id(self, *, doc_id: str) -> None:
        """Delete all chunks for a given document id."""

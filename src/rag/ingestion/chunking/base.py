from __future__ import annotations

from abc import ABC, abstractmethod

from src.rag.models import RAGChunk


class ChunkingStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Stable strategy identifier."""

    @abstractmethod
    def chunk(
        self,
        *,
        text: str,
        doc_id: str,
        source: str,
        chunk_size: int,
        chunk_overlap: int,
        page_number: int | None = None,
    ) -> list[RAGChunk]:
        """Chunk text into RAG chunks."""

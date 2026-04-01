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


class FixedWindowChunkingStrategy(ChunkingStrategy):
    @property
    def name(self) -> str:
        return "fixed_window"

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
        normalized = text.strip()
        if not normalized:
            return []
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        chunks: list[RAGChunk] = []
        start = 0
        step = chunk_size - chunk_overlap
        index = 0

        while start < len(normalized):
            end = start + chunk_size
            chunk_body = normalized[start:end].strip()
            if chunk_body:
                chunks.append(
                    RAGChunk(
                        doc_id=doc_id,
                        chunk_id=f"chunk-{index}",
                        source=source,
                        text=chunk_body,
                        page_number=page_number,
                        chunking_strategy=self.name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )
                index += 1
            start += step

        return chunks


class ChunkingStrategyRegistry:
    def __init__(self, strategies: list[ChunkingStrategy]) -> None:
        self._strategies = {strategy.name: strategy for strategy in strategies}
        if not self._strategies:
            raise ValueError("At least one chunking strategy must be registered.")

    def resolve(self, name: str) -> ChunkingStrategy:
        strategy = self._strategies.get(name)
        if strategy is None:
            available = ", ".join(sorted(self._strategies))
            raise ValueError(
                f"Unsupported chunking_strategy '{name}'. Available: {available}."
            )
        return strategy

    def names(self) -> list[str]:
        return sorted(self._strategies)


def chunk_text(
    *,
    text: str,
    doc_id: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
    page_number: int | None = None,
) -> list[RAGChunk]:
    return FixedWindowChunkingStrategy().chunk(
        text=text,
        doc_id=doc_id,
        source=source,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        page_number=page_number,
    )

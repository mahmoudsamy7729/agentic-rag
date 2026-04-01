from __future__ import annotations

from src.rag.ingestion.chunking.base import ChunkingStrategy
from src.rag.models import RAGChunk


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

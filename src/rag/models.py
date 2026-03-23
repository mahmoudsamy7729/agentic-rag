from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RAGChunk:
    doc_id: str
    chunk_id: str
    source: str
    text: str
    page_number: int | None = None


@dataclass(slots=True)
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    source: str
    text: str
    score: float
    page_number: int | None = None

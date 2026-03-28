from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EvaluationRunConfig:
    rag_top_k: int
    rag_prefetch_k: int
    embedding_provider: str
    embedding_model: str
    reranker_enabled: bool
    reranker_model: str | None
    answer_model: str
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int


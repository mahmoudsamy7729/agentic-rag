from __future__ import annotations

from abc import ABC, abstractmethod

from src.rag.models import RetrievedChunk
from src.rag.pipeline import RAGRetrievalService


class EvaluationRetriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        *,
        question: str,
        file_id: str,
        k: int,
    ) -> list[RetrievedChunk]:
        """Return top-k retrieved chunks for a single evaluation question."""


class RAGRetrievalEvaluatorAdapter(EvaluationRetriever):
    def __init__(self, *, retrieval_service: RAGRetrievalService) -> None:
        self._retrieval_service = retrieval_service

    async def retrieve(
        self,
        *,
        question: str,
        file_id: str,
        k: int,
    ) -> list[RetrievedChunk]:
        return await self._retrieval_service.retrieve(
            query=question,
            top_k=k,
            doc_id=file_id,
        )

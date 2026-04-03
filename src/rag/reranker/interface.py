from __future__ import annotations

from abc import ABC, abstractmethod

from src.rag.models import RetrievedChunk


class Reranker(ABC):
    @property
    def model_name(self) -> str | None:
        return None

    @abstractmethod
    async def rerank(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        """Return top_n chunks ranked by relevance to query."""


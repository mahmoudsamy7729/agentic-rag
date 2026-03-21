from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents/chunks."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""

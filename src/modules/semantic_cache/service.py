from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from src.modules.semantic_cache.repository import SemanticCacheRepository


@dataclass(slots=True)
class SemanticCacheHit:
    answer: str
    citations: list[dict]


class SemanticCacheService:
    def __init__(
        self,
        *,
        repository: "SemanticCacheRepository",
        enabled: bool,
        similarity_threshold: float,
    ) -> None:
        self._repository = repository
        self._enabled = enabled
        self._similarity_threshold = similarity_threshold

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def normalize_question(question: str) -> str:
        return re.sub(r"\s+", " ", question.strip().lower())

    async def lookup(
        self,
        *,
        owner_user_id: UUID,
        doc_id: str,
        doc_version: datetime,
        model_name: str,
        query_embedding: list[float],
    ) -> SemanticCacheHit | None:
        if not self._enabled:
            return None
        entry = await self._repository.lookup(
            owner_user_id=owner_user_id,
            doc_id=doc_id,
            doc_version=doc_version,
            model_name=model_name,
            query_embedding=query_embedding,
            similarity_threshold=self._similarity_threshold,
        )
        if entry is None:
            return None
        return SemanticCacheHit(
            answer=entry.answer,
            citations=list(entry.citations or []),
        )

    async def store(
        self,
        *,
        owner_user_id: UUID,
        doc_id: str,
        doc_version: datetime,
        model_name: str,
        question_normalized: str,
        question_embedding: list[float],
        answer: str,
        citations: list[dict],
    ) -> None:
        if not self._enabled:
            return
        await self._repository.store(
            owner_user_id=owner_user_id,
            doc_id=doc_id,
            doc_version=doc_version,
            model_name=model_name,
            question_normalized=question_normalized,
            question_embedding=question_embedding,
            answer=answer,
            citations=citations,
        )
        await self._repository.commit()

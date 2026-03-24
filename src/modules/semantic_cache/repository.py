from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.semantic_cache.models import SemanticCacheEntry


class SemanticCacheRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def lookup(
        self,
        *,
        owner_user_id: UUID,
        doc_id: str,
        doc_version: datetime,
        model_name: str,
        query_embedding: list[float],
        similarity_threshold: float,
    ) -> SemanticCacheEntry | None:
        if not query_embedding:
            return None

        distance = SemanticCacheEntry.question_embedding.cosine_distance(query_embedding)
        similarity = (1 - distance).label("similarity")
        stmt = (
            select(SemanticCacheEntry, similarity)
            .where(
                SemanticCacheEntry.owner_user_id == owner_user_id,
                SemanticCacheEntry.doc_id == doc_id,
                SemanticCacheEntry.doc_version == doc_version,
                SemanticCacheEntry.model_name == model_name,
            )
            .order_by(distance.asc())
            .limit(1)
        )
        row = (await self._session.execute(stmt)).first()
        if row is None:
            return None

        entry, score = row
        if score is None or float(score) < similarity_threshold:
            return None
        return entry

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
    ) -> SemanticCacheEntry:
        entry = SemanticCacheEntry(
            owner_user_id=owner_user_id,
            doc_id=doc_id,
            doc_version=doc_version,
            model_name=model_name,
            question_normalized=question_normalized,
            question_embedding=question_embedding,
            answer=answer,
            citations=citations,
        )
        self._session.add(entry)
        await self._session.flush()
        return entry

    async def commit(self) -> None:
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()

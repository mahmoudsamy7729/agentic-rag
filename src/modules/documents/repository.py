from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.documents.models import Document


class DocumentsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_document(
        self,
        *,
        owner_user_id: UUID,
        doc_id: str,
        source: str | None,
        chunking_strategy: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> Document:
        document = Document(
            id=doc_id,
            owner_user_id=owner_user_id,
            source=source,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._session.add(document)
        await self._session.flush()
        return document

    async def get_owned_document(
        self,
        *,
        owner_user_id: UUID,
        doc_id: str,
        include_deleted: bool = False,
    ) -> Document | None:
        stmt = select(Document).where(
            Document.id == doc_id,
            Document.owner_user_id == owner_user_id,
        )
        if not include_deleted:
            stmt = stmt.where(Document.deleted_at.is_(None))

        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_owned_documents(
        self,
        *,
        owner_user_id: UUID,
        limit: int,
        offset: int,
    ) -> list[Document]:
        stmt = (
            select(Document)
            .where(
                Document.owner_user_id == owner_user_id,
                Document.deleted_at.is_(None),
            )
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_owned_documents_by_ids(
        self,
        *,
        owner_user_id: UUID,
        doc_ids: list[str],
        include_deleted: bool = False,
    ) -> list[Document]:
        if not doc_ids:
            return []
        stmt = select(Document).where(
            Document.owner_user_id == owner_user_id,
            Document.id.in_(doc_ids),
        )
        if not include_deleted:
            stmt = stmt.where(Document.deleted_at.is_(None))
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def soft_delete_owned_document(
        self,
        *,
        owner_user_id: UUID,
        doc_id: str,
    ) -> Document | None:
        document = await self.get_owned_document(
            owner_user_id=owner_user_id,
            doc_id=doc_id,
            include_deleted=True,
        )
        if document is None:
            return None
        if document.deleted_at is None:
            now = datetime.now(timezone.utc)
            document.deleted_at = now
            document.updated_at = now
            await self._session.flush()
        return document

    async def doc_id_exists(
        self,
        *,
        doc_id: str,
        include_deleted: bool = True,
    ) -> bool:
        stmt = select(Document.id).where(Document.id == doc_id)
        if not include_deleted:
            stmt = stmt.where(Document.deleted_at.is_(None))

        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def mark_document_indexed(
        self,
        *,
        owner_user_id: UUID,
        doc_id: str,
        indexed_at: datetime | None = None,
        chunking_strategy: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> Document | None:
        document = await self.get_owned_document(
            owner_user_id=owner_user_id,
            doc_id=doc_id,
            include_deleted=False,
        )
        if document is None:
            return None

        now = indexed_at or datetime.now(timezone.utc)
        document.last_indexed_at = now
        document.updated_at = now
        if chunking_strategy is not None:
            document.chunking_strategy = chunking_strategy
        if chunk_size is not None:
            document.chunk_size = chunk_size
        if chunk_overlap is not None:
            document.chunk_overlap = chunk_overlap
        await self._session.flush()
        return document

    async def commit(self) -> None:
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()


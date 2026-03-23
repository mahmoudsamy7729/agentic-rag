from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database import get_db
from src.modules.documents.repository import DocumentsRepository

DbSessionDep = Annotated[AsyncSession, Depends(get_db)]


def get_documents_repository(session: DbSessionDep) -> DocumentsRepository:
    return DocumentsRepository(session)


DocumentsRepositoryDep = Annotated[DocumentsRepository, Depends(get_documents_repository)]


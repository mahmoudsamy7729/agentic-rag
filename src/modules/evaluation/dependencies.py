from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database import get_db
from src.modules.evaluation.repository import EvaluationRepository

DbSessionDep = Annotated[AsyncSession, Depends(get_db)]


def get_evaluation_repository(session: DbSessionDep) -> EvaluationRepository:
    return EvaluationRepository(session)


EvaluationRepositoryDep = Annotated[EvaluationRepository, Depends(get_evaluation_repository)]


from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.modules.evaluation.models import EvaluationCase, EvaluationRun


class EvaluationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_run(self, *, run: EvaluationRun) -> EvaluationRun:
        self._session.add(run)
        await self._session.flush()
        return run

    async def add_cases(self, *, cases: list[EvaluationCase]) -> None:
        self._session.add_all(cases)
        await self._session.flush()

    async def get_owned_run(self, *, owner_user_id: UUID, run_id: UUID) -> EvaluationRun | None:
        stmt = (
            select(EvaluationRun)
            .where(
                EvaluationRun.id == run_id,
                EvaluationRun.owner_user_id == owner_user_id,
            )
            .options(selectinload(EvaluationRun.cases))
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_run(self, *, run_id: UUID) -> EvaluationRun | None:
        stmt = (
            select(EvaluationRun)
            .where(EvaluationRun.id == run_id)
            .options(selectinload(EvaluationRun.cases))
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_owned_runs(
        self,
        *,
        owner_user_id: UUID,
        limit: int,
        offset: int,
    ) -> list[EvaluationRun]:
        stmt = (
            select(EvaluationRun)
            .where(EvaluationRun.owner_user_id == owner_user_id)
            .order_by(EvaluationRun.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_owned_cases(
        self,
        *,
        owner_user_id: UUID,
        run_id: UUID,
        limit: int,
        offset: int,
    ) -> list[EvaluationCase]:
        stmt = (
            select(EvaluationCase)
            .join(EvaluationRun, EvaluationCase.run_id == EvaluationRun.id)
            .where(
                EvaluationCase.run_id == run_id,
                EvaluationRun.owner_user_id == owner_user_id,
            )
            .order_by(EvaluationCase.case_index.asc())
            .offset(offset)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def count_owned_cases(self, *, owner_user_id: UUID, run_id: UUID) -> int:
        stmt = (
            select(func.count(EvaluationCase.id))
            .select_from(EvaluationCase)
            .join(EvaluationRun, EvaluationCase.run_id == EvaluationRun.id)
            .where(
                EvaluationCase.run_id == run_id,
                EvaluationRun.owner_user_id == owner_user_id,
            )
        )
        result = await self._session.execute(stmt)
        return int(result.scalar_one())

    async def list_cases_for_run(self, *, run_id: UUID) -> list[EvaluationCase]:
        stmt: Select[tuple[EvaluationCase]] = (
            select(EvaluationCase)
            .where(EvaluationCase.run_id == run_id)
            .order_by(EvaluationCase.case_index.asc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def mark_run_started(self, *, run: EvaluationRun) -> None:
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        run.error_message = None
        await self._session.flush()

    async def mark_case_completed(
        self,
        *,
        case: EvaluationCase,
        retrieved_chunk_ids: list[str],
        retrieved_chunk_texts: list[str],
        matched_phrases: list[str],
        matched_keywords: list[str],
        hit_at_k: float,
        recall_at_k: float,
        precision_at_k: float,
        mrr: float,
        keyword_coverage: float,
        context_relevance_score: int | None,
        context_relevance_explanation: str | None,
        first_correct_rank: int | None,
        useful_chunk_count: int,
    ) -> None:
        case.status = "completed"
        case.retrieved_chunk_ids = retrieved_chunk_ids
        case.retrieved_chunk_texts = retrieved_chunk_texts
        case.matched_phrases = matched_phrases
        case.matched_keywords = matched_keywords
        case.hit_at_k = hit_at_k
        case.recall_at_k = recall_at_k
        case.precision_at_k = precision_at_k
        case.mrr = mrr
        case.keyword_coverage = keyword_coverage
        case.context_relevance_score = context_relevance_score
        case.context_relevance_explanation = context_relevance_explanation
        case.first_correct_rank = first_correct_rank
        case.useful_chunk_count = useful_chunk_count
        case.error_message = None
        await self._session.flush()

    async def mark_case_failed(self, *, case: EvaluationCase, error_message: str) -> None:
        case.status = "failed"
        case.error_message = error_message
        await self._session.flush()

    async def update_processed_count(self, *, run: EvaluationRun, processed_cases: int) -> None:
        run.processed_cases = processed_cases
        await self._session.flush()

    async def mark_run_completed(
        self,
        *,
        run: EvaluationRun,
        hit_at_k_avg: float,
        recall_at_k_avg: float,
        precision_at_k_avg: float,
        mrr_avg: float,
        keyword_coverage_avg: float,
        context_relevance_score_avg: float | None,
        grouped_summary: dict,
    ) -> None:
        run.status = "completed"
        run.processed_cases = run.total_cases
        run.hit_at_k_avg = hit_at_k_avg
        run.recall_at_k_avg = recall_at_k_avg
        run.precision_at_k_avg = precision_at_k_avg
        run.mrr_avg = mrr_avg
        run.keyword_coverage_avg = keyword_coverage_avg
        run.context_relevance_score_avg = context_relevance_score_avg
        run.grouped_summary = grouped_summary
        run.finished_at = datetime.now(timezone.utc)
        run.error_message = None
        await self._session.flush()

    async def mark_run_failed(self, *, run: EvaluationRun, error_message: str) -> None:
        run.status = "failed"
        run.error_message = error_message
        run.finished_at = datetime.now(timezone.utc)
        await self._session.flush()

    async def commit(self) -> None:
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()

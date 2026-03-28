from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.evaluation.config import EvaluationRunConfig
from src.modules.evaluation.models import EvaluationCase, EvaluationRun


class EvaluationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_run(
        self,
        *,
        owner_user_id: UUID,
        doc_id: str,
        dataset_name: str,
        dataset_sha256: str,
        total_cases: int,
        run_config: EvaluationRunConfig,
    ) -> EvaluationRun:
        run = EvaluationRun(
            owner_user_id=owner_user_id,
            doc_id=doc_id,
            status="queued",
            dataset_name=dataset_name,
            dataset_sha256=dataset_sha256,
            total_cases=total_cases,
            processed_cases=0,
            cfg_rag_top_k=run_config.rag_top_k,
            cfg_rag_prefetch_k=run_config.rag_prefetch_k,
            cfg_embedding_provider=run_config.embedding_provider,
            cfg_embedding_model=run_config.embedding_model,
            cfg_reranker_enabled=run_config.reranker_enabled,
            cfg_reranker_model=run_config.reranker_model,
            cfg_answer_model=run_config.answer_model,
            cfg_chunk_strategy=run_config.chunk_strategy,
            cfg_chunk_size=run_config.chunk_size,
            cfg_chunk_overlap=run_config.chunk_overlap,
        )
        self._session.add(run)
        await self._session.flush()
        return run

    async def create_cases(
        self,
        *,
        run_id: UUID,
        rows: list[dict],
    ) -> list[EvaluationCase]:
        cases: list[EvaluationCase] = []
        for index, row in enumerate(rows):
            case = EvaluationCase(
                run_id=run_id,
                case_index=index,
                question=row["question"],
                reference_answer=row["answer"],
                expected_chunk_ids=row["expected_chunk_ids"],
                difficulty=row.get("difficulty"),
                category=row.get("category"),
                status="queued",
            )
            self._session.add(case)
            cases.append(case)
        await self._session.flush()
        return cases

    async def get_owned_run(self, *, owner_user_id: UUID, run_id: UUID) -> EvaluationRun | None:
        stmt = select(EvaluationRun).where(
            EvaluationRun.id == run_id,
            EvaluationRun.owner_user_id == owner_user_id,
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_run(self, *, run_id: UUID) -> EvaluationRun | None:
        result = await self._session.execute(
            select(EvaluationRun).where(EvaluationRun.id == run_id)
        )
        return result.scalar_one_or_none()

    async def list_owned_runs(
        self,
        *,
        owner_user_id: UUID,
        doc_id: str | None,
        limit: int,
        offset: int,
    ) -> list[EvaluationRun]:
        stmt = select(EvaluationRun).where(EvaluationRun.owner_user_id == owner_user_id)
        if doc_id:
            stmt = stmt.where(EvaluationRun.doc_id == doc_id)
        stmt = stmt.order_by(EvaluationRun.created_at.desc()).offset(offset).limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def count_owned_runs(self, *, owner_user_id: UUID, doc_id: str | None) -> int:
        stmt = select(func.count(EvaluationRun.id)).where(EvaluationRun.owner_user_id == owner_user_id)
        if doc_id:
            stmt = stmt.where(EvaluationRun.doc_id == doc_id)
        result = await self._session.execute(stmt)
        return int(result.scalar() or 0)

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
            .join(EvaluationRun, EvaluationRun.id == EvaluationCase.run_id)
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
            .join(EvaluationRun, EvaluationRun.id == EvaluationCase.run_id)
            .where(
                EvaluationCase.run_id == run_id,
                EvaluationRun.owner_user_id == owner_user_id,
            )
        )
        result = await self._session.execute(stmt)
        return int(result.scalar() or 0)

    async def list_run_cases(self, *, run_id: UUID) -> list[EvaluationCase]:
        stmt = (
            select(EvaluationCase)
            .where(EvaluationCase.run_id == run_id)
            .order_by(EvaluationCase.case_index.asc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_failed_cases_for_run(self, *, run_id: UUID) -> list[EvaluationCase]:
        stmt = (
            select(EvaluationCase)
            .where(
                EvaluationCase.run_id == run_id,
                EvaluationCase.status == "failed",
            )
            .order_by(EvaluationCase.case_index.asc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def mark_run_running(self, *, run: EvaluationRun) -> None:
        now = datetime.now(timezone.utc)
        run.status = "running"
        run.started_at = now
        run.error_message = None
        await self._session.flush()

    async def mark_run_rerun_started(self, *, run: EvaluationRun) -> None:
        run.status = "running"
        run.error_message = None
        await self._session.flush()

    async def increment_progress(self, *, run: EvaluationRun) -> None:
        run.processed_cases += 1
        await self._session.flush()

    async def complete_run(
        self,
        *,
        run: EvaluationRun,
        hit_at_k: float | None,
        recall_at_k: float | None,
        mrr: float | None,
        accuracy_avg: float | None,
        completeness_avg: float | None,
        relevance_avg: float | None,
        groundedness_avg: float | None,
    ) -> None:
        run.status = "completed"
        run.hit_at_k = hit_at_k
        run.recall_at_k = recall_at_k
        run.mrr = mrr
        run.accuracy_avg = accuracy_avg
        run.completeness_avg = completeness_avg
        run.relevance_avg = relevance_avg
        run.groundedness_avg = groundedness_avg
        run.error_message = None
        run.finished_at = datetime.now(timezone.utc)
        await self._session.flush()

    async def fail_run(self, *, run: EvaluationRun, error_message: str) -> None:
        run.status = "failed"
        run.error_message = error_message
        run.finished_at = datetime.now(timezone.utc)
        await self._session.flush()

    async def set_run_with_aggregates(
        self,
        *,
        run: EvaluationRun,
        status: str,
        error_message: str | None,
        hit_at_k: float | None,
        recall_at_k: float | None,
        mrr: float | None,
        accuracy_avg: float | None,
        completeness_avg: float | None,
        relevance_avg: float | None,
        groundedness_avg: float | None,
    ) -> None:
        run.status = status
        run.error_message = error_message
        run.hit_at_k = hit_at_k
        run.recall_at_k = recall_at_k
        run.mrr = mrr
        run.accuracy_avg = accuracy_avg
        run.completeness_avg = completeness_avg
        run.relevance_avg = relevance_avg
        run.groundedness_avg = groundedness_avg
        run.processed_cases = run.total_cases
        run.finished_at = datetime.now(timezone.utc)
        await self._session.flush()

    async def commit(self) -> None:
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()


from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.modules.evaluation.dataset import parse_retrieval_dataset_jsonl_bytes
from src.modules.evaluation.judge import ContextRelevanceJudge
from src.modules.evaluation.matching import TextNormalizationConfig, UsefulChunkConfig
from src.modules.evaluation.metrics import (
    aggregate_metric_average,
    compute_retrieval_metrics,
    grouped_metric_summary,
)
from src.modules.evaluation.models import EvaluationCase, EvaluationRun
from src.modules.evaluation.repository import EvaluationRepository
from src.modules.evaluation.retriever import EvaluationRetriever


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationRunConfig:
    k: int
    strip_punctuation: bool
    min_keyword_hits: int
    min_keyword_ratio: float
    store_retrieved_chunk_texts: bool
    judge_enabled: bool
    rag_top_k: int
    rag_prefetch_k: int
    embedding_provider: str
    embedding_model: str
    reranker_enabled: bool
    reranker_model: str | None
    judge_model: str | None


@dataclass(frozen=True, slots=True)
class CreateRetrievalEvaluationRunResult:
    run_id: UUID
    status: str


class RetrievalEvaluationService:
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        dataset_storage_dir: str,
        retriever_factory: Callable[[], EvaluationRetriever],
        judge_factory: Callable[[], ContextRelevanceJudge | None],
    ) -> None:
        self._session_factory = session_factory
        self._dataset_storage_dir = Path(dataset_storage_dir)
        self._retriever_factory = retriever_factory
        self._judge_factory = judge_factory

    async def create_run_from_upload(
        self,
        *,
        owner_user_id: UUID,
        file_id: str,
        dataset_name: str,
        dataset_bytes: bytes,
        config: RetrievalEvaluationRunConfig,
    ) -> CreateRetrievalEvaluationRunResult:
        dataset = parse_retrieval_dataset_jsonl_bytes(dataset_bytes)
        self._dataset_storage_dir.mkdir(parents=True, exist_ok=True)

        async with self._session_factory() as session:
            repository = EvaluationRepository(session)
            run = EvaluationRun(
                owner_user_id=owner_user_id,
                doc_id=file_id,
                status="queued",
                dataset_name=dataset_name,
                dataset_path="",
                dataset_sha256=dataset.sha256,
                total_cases=len(dataset.items),
                processed_cases=0,
                k=config.k,
                config_snapshot=asdict(config),
                grouped_summary={},
            )
            await repository.create_run(run=run)

            dataset_dir = self._dataset_storage_dir / str(run.id)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = dataset_dir / "dataset.jsonl"
            dataset_path.write_bytes(dataset_bytes)
            run.dataset_path = str(dataset_path)

            cases = [
                EvaluationCase(
                    run_id=run.id,
                    case_index=index,
                    status="queued",
                    question=item.question,
                    reference_answer=item.answer,
                    must_include_keywords=item.must_include_keywords,
                    must_include_phrases=item.must_include_phrases,
                    difficulty=item.difficulty,
                    category=item.category,
                )
                for index, item in enumerate(dataset.items)
            ]
            await repository.add_cases(cases=cases)
            await repository.commit()
            return CreateRetrievalEvaluationRunResult(run_id=run.id, status=run.status)

    async def process_run(self, *, run_id: UUID) -> None:
        retriever = self._retriever_factory()

        async with self._session_factory() as session:
            repository = EvaluationRepository(session)
            run = await repository.get_run(run_id=run_id)
            if run is None:
                return
            try:
                await repository.mark_run_started(run=run)
                await repository.commit()

                normalization_config = TextNormalizationConfig(
                    strip_punctuation=bool(run.config_snapshot.get("strip_punctuation", True))
                )
                useful_chunk_config = UsefulChunkConfig(
                    min_keyword_hits=int(run.config_snapshot.get("min_keyword_hits", 2)),
                    min_keyword_ratio=float(run.config_snapshot.get("min_keyword_ratio", 0.4)),
                )
                store_chunk_texts = bool(
                    run.config_snapshot.get("store_retrieved_chunk_texts", False)
                )
                judge = (
                    self._judge_factory()
                    if bool(run.config_snapshot.get("judge_enabled", True))
                    else None
                )

                cases = await repository.list_cases_for_run(run_id=run.id)
                processed = 0
                for case in cases:
                    try:
                        retrieved_chunks = await retriever.retrieve(
                            question=case.question,
                            file_id=run.doc_id,
                            k=run.k,
                        )
                        metrics = compute_retrieval_metrics(
                            retrieved_chunks=retrieved_chunks,
                            must_include_phrases=case.must_include_phrases,
                            must_include_keywords=case.must_include_keywords,
                            k=run.k,
                            normalization_config=normalization_config,
                            useful_chunk_config=useful_chunk_config,
                        )
                        judge_result = None
                        if judge is not None:
                            judge_result = await judge.judge(
                                question=case.question,
                                retrieved_chunks=[
                                    {
                                        "chunk_id": chunk.chunk_id,
                                        "text": chunk.text,
                                    }
                                    for chunk in retrieved_chunks
                                ],
                            )
                        await repository.mark_case_completed(
                            case=case,
                            retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved_chunks],
                            retrieved_chunk_texts=(
                                [chunk.text for chunk in retrieved_chunks]
                                if store_chunk_texts
                                else []
                            ),
                            matched_phrases=metrics.matched_phrases,
                            matched_keywords=metrics.matched_keywords,
                            hit_at_k=metrics.hit_at_k,
                            recall_at_k=metrics.recall_at_k,
                            precision_at_k=metrics.precision_at_k,
                            mrr=metrics.mrr,
                            keyword_coverage=metrics.keyword_coverage,
                            context_relevance_score=(
                                judge_result.score if judge_result is not None else None
                            ),
                            context_relevance_explanation=(
                                judge_result.explanation if judge_result is not None else None
                            ),
                            first_correct_rank=metrics.first_correct_rank,
                            useful_chunk_count=metrics.useful_chunk_count,
                        )
                    except Exception as exc:
                        await repository.mark_case_failed(case=case, error_message=str(exc))
                    processed += 1
                    await repository.update_processed_count(
                        run=run,
                        processed_cases=processed,
                    )
                    await repository.commit()

                refreshed_cases = await repository.list_cases_for_run(run_id=run.id)
                aggregate_cases = [
                    {
                        "category": case.category,
                        "difficulty": case.difficulty,
                        "hit_at_k": case.hit_at_k,
                        "recall_at_k": case.recall_at_k,
                        "precision_at_k": case.precision_at_k,
                        "mrr": case.mrr,
                        "keyword_coverage": case.keyword_coverage,
                        "context_relevance_score": case.context_relevance_score,
                    }
                    for case in refreshed_cases
                    if case.status == "completed"
                ]
                await repository.mark_run_completed(
                    run=run,
                    hit_at_k_avg=aggregate_metric_average(
                        case.get("hit_at_k") for case in aggregate_cases
                    )
                    or 0.0,
                    recall_at_k_avg=aggregate_metric_average(
                        case.get("recall_at_k") for case in aggregate_cases
                    )
                    or 0.0,
                    precision_at_k_avg=aggregate_metric_average(
                        case.get("precision_at_k") for case in aggregate_cases
                    )
                    or 0.0,
                    mrr_avg=aggregate_metric_average(case.get("mrr") for case in aggregate_cases)
                    or 0.0,
                    keyword_coverage_avg=aggregate_metric_average(
                        case.get("keyword_coverage") for case in aggregate_cases
                    )
                    or 0.0,
                    context_relevance_score_avg=aggregate_metric_average(
                        case.get("context_relevance_score") for case in aggregate_cases
                    ),
                    grouped_summary={
                        "category": grouped_metric_summary(
                            cases=aggregate_cases,
                            field_name="category",
                        ),
                        "difficulty": grouped_metric_summary(
                            cases=aggregate_cases,
                            field_name="difficulty",
                        ),
                    },
                )
                await repository.commit()
            except Exception as exc:
                await repository.rollback()
                failed_run = await repository.get_run(run_id=run_id)
                if failed_run is not None:
                    await repository.mark_run_failed(run=failed_run, error_message=str(exc))
                    await repository.commit()

    async def list_runs(
        self,
        *,
        owner_user_id: UUID,
        limit: int,
        offset: int,
    ) -> list[EvaluationRun]:
        async with self._session_factory() as session:
            repository = EvaluationRepository(session)
            return await repository.list_owned_runs(
                owner_user_id=owner_user_id,
                limit=limit,
                offset=offset,
            )

    async def get_run(self, *, owner_user_id: UUID, run_id: UUID) -> EvaluationRun | None:
        async with self._session_factory() as session:
            repository = EvaluationRepository(session)
            return await repository.get_owned_run(owner_user_id=owner_user_id, run_id=run_id)

    async def list_cases(
        self,
        *,
        owner_user_id: UUID,
        run_id: UUID,
        limit: int,
        offset: int,
    ) -> tuple[list[EvaluationCase], int]:
        async with self._session_factory() as session:
            repository = EvaluationRepository(session)
            cases = await repository.list_owned_cases(
                owner_user_id=owner_user_id,
                run_id=run_id,
                limit=limit,
                offset=offset,
            )
            total = await repository.count_owned_cases(owner_user_id=owner_user_id, run_id=run_id)
            return cases, total

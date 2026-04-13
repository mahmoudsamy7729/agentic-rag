from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.modules.evaluation.dataset import (
    LoadedRetrievalDataset,
    RetrievalDatasetItem,
    parse_retrieval_dataset_jsonl_bytes,
)
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


@dataclass(frozen=True, slots=True)
class RerunFailedCasesResult:
    run_id: UUID
    status: str
    rerun_case_count: int
    case_ids: list[UUID]


@dataclass(frozen=True, slots=True)
class StoredRetrievalDataset:
    dataset_sha256: str
    dataset_name: str
    dataset_path: str
    total_cases: int
    categories: list[str]
    difficulties: list[str]
    created_at: datetime
    last_used_at: datetime
    run_count: int


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
        dataset_path = self._persist_dataset_bytes(
            dataset_sha256=dataset.sha256,
            dataset_bytes=dataset_bytes,
        )

        async with self._session_factory() as session:
            repository = EvaluationRepository(session)
            run = EvaluationRun(
                owner_user_id=owner_user_id,
                doc_id=file_id,
                status="queued",
                dataset_name=dataset_name,
                dataset_path=str(dataset_path),
                dataset_sha256=dataset.sha256,
                total_cases=len(dataset.items),
                processed_cases=0,
                k=config.k,
                config_snapshot=asdict(config),
                grouped_summary={},
            )
            await repository.create_run(run=run)

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

    async def create_run_from_existing_dataset(
        self,
        *,
        owner_user_id: UUID,
        file_id: str,
        dataset_sha256: str,
        config: RetrievalEvaluationRunConfig,
    ) -> CreateRetrievalEvaluationRunResult:
        dataset_entry = await self.get_dataset(owner_user_id=owner_user_id, dataset_sha256=dataset_sha256)
        if dataset_entry is None:
            raise FileNotFoundError("Dataset not found.")
        dataset_bytes = Path(dataset_entry.dataset_path).read_bytes()
        return await self.create_run_from_upload(
            owner_user_id=owner_user_id,
            file_id=file_id,
            dataset_name=dataset_entry.dataset_name,
            dataset_bytes=dataset_bytes,
            config=config,
        )

    async def process_run(self, *, run_id: UUID) -> None:
        async with self._session_factory() as session:
            repository = EvaluationRepository(session)
            run = await repository.get_run(run_id=run_id)
            if run is None:
                return
            try:
                await self._process_run_cases(repository=repository, run=run, cases=None)
                await repository.commit()
            except Exception as exc:
                await repository.rollback()
                failed_run = await repository.get_run(run_id=run_id)
                if failed_run is not None:
                    await repository.mark_run_failed(run=failed_run, error_message=str(exc))
                    await repository.commit()

    async def process_selected_cases(self, *, run_id: UUID, case_ids: list[UUID]) -> None:
        async with self._session_factory() as session:
            repository = EvaluationRepository(session)
            run = await repository.get_run(run_id=run_id)
            if run is None:
                return
            try:
                target_case_ids = set(case_ids)
                cases = [
                    case
                    for case in await repository.list_cases_for_run(run_id=run.id)
                    if case.id in target_case_ids
                ]
                if not cases:
                    await self._finalize_run(repository=repository, run=run)
                    await repository.commit()
                    return
                await self._process_run_cases(repository=repository, run=run, cases=cases)
                await repository.commit()
            except Exception as exc:
                await repository.rollback()
                failed_run = await repository.get_run(run_id=run_id)
                if failed_run is not None:
                    await repository.mark_run_failed(run=failed_run, error_message=str(exc))
                    await repository.commit()

    async def rerun_failed_cases(
        self,
        *,
        owner_user_id: UUID,
        run_id: UUID,
    ) -> RerunFailedCasesResult:
        async with self._session_factory() as session:
            repository = EvaluationRepository(session)
            run = await repository.get_owned_run(owner_user_id=owner_user_id, run_id=run_id)
            if run is None:
                raise LookupError("Evaluation run not found.")
            if run.status in {"queued", "running"}:
                raise ValueError("Evaluation run is already in progress.")

            failed_cases = await repository.list_failed_cases_for_run(run_id=run.id)
            if not failed_cases:
                raise RuntimeError("This evaluation run has no failed cases to rerun.")

            for case in failed_cases:
                await repository.reset_case_for_rerun(case=case)

            completed_case_count = sum(
                1 for case in await repository.list_cases_for_run(run_id=run.id) if case.status == "completed"
            )
            await repository.reset_run_for_rerun(
                run=run,
                processed_cases=completed_case_count,
            )
            await repository.commit()
            return RerunFailedCasesResult(
                run_id=run.id,
                status=run.status,
                rerun_case_count=len(failed_cases),
                case_ids=[case.id for case in failed_cases],
            )

    async def list_runs(
        self,
        *,
        owner_user_id: UUID,
        limit: int,
        offset: int,
    ) -> tuple[list[EvaluationRun], int]:
        async with self._session_factory() as session:
            repository = EvaluationRepository(session)
            runs = await repository.list_owned_runs(
                owner_user_id=owner_user_id,
                limit=limit,
                offset=offset,
            )
            total = await repository.count_owned_runs(owner_user_id=owner_user_id)
            return runs, total

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

    async def delete_run(self, *, owner_user_id: UUID, run_id: UUID) -> bool:
        async with self._session_factory() as session:
            repository = EvaluationRepository(session)
            deleted = await repository.delete_owned_run(owner_user_id=owner_user_id, run_id=run_id)
            if deleted:
                await repository.commit()
            else:
                await repository.rollback()
            return deleted

    async def list_datasets(self, *, owner_user_id: UUID) -> list[StoredRetrievalDataset]:
        async with self._session_factory() as session:
            repository = EvaluationRepository(session)
            runs = await repository.list_all_owned_runs(owner_user_id=owner_user_id)
        grouped_runs: dict[str, list[EvaluationRun]] = {}
        for run in runs:
            grouped_runs.setdefault(run.dataset_sha256, []).append(run)

        datasets: list[StoredRetrievalDataset] = []
        for dataset_sha256, items in grouped_runs.items():
            representative = next(
                (
                    run
                    for run in items
                    if run.dataset_path and Path(run.dataset_path).exists()
                ),
                None,
            )
            if representative is None:
                continue
            loaded_dataset = parse_retrieval_dataset_jsonl_bytes(Path(representative.dataset_path).read_bytes())
            datasets.append(
                self._build_dataset_summary(
                    dataset=loaded_dataset,
                    runs=items,
                    representative=representative,
                )
            )
        datasets.sort(key=lambda item: item.created_at, reverse=True)
        return datasets

    async def get_dataset(
        self,
        *,
        owner_user_id: UUID,
        dataset_sha256: str,
    ) -> StoredRetrievalDataset | None:
        datasets = await self.list_datasets(owner_user_id=owner_user_id)
        for dataset in datasets:
            if dataset.dataset_sha256 == dataset_sha256:
                return dataset
        return None

    async def preview_dataset(
        self,
        *,
        owner_user_id: UUID,
        dataset_sha256: str,
        sample_limit: int = 10,
    ) -> tuple[StoredRetrievalDataset, list[RetrievalDatasetItem]] | None:
        dataset = await self.get_dataset(owner_user_id=owner_user_id, dataset_sha256=dataset_sha256)
        if dataset is None:
            return None
        loaded_dataset = parse_retrieval_dataset_jsonl_bytes(Path(dataset.dataset_path).read_bytes())
        return dataset, loaded_dataset.items[:sample_limit]

    async def get_dataset_bytes(
        self,
        *,
        owner_user_id: UUID,
        dataset_sha256: str,
    ) -> tuple[StoredRetrievalDataset, bytes] | None:
        dataset = await self.get_dataset(owner_user_id=owner_user_id, dataset_sha256=dataset_sha256)
        if dataset is None:
            return None
        return dataset, Path(dataset.dataset_path).read_bytes()

    async def delete_dataset(self, *, owner_user_id: UUID, dataset_sha256: str) -> bool:
        dataset = await self.get_dataset(owner_user_id=owner_user_id, dataset_sha256=dataset_sha256)
        if dataset is None:
            return False
        dataset_path = Path(dataset.dataset_path)
        if not dataset_path.exists():
            return False
        dataset_path.unlink()
        return True

    def _persist_dataset_bytes(self, *, dataset_sha256: str, dataset_bytes: bytes) -> Path:
        datasets_dir = self._dataset_storage_dir / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = datasets_dir / f"{dataset_sha256}.jsonl"
        if not dataset_path.exists():
            dataset_path.write_bytes(dataset_bytes)
        return dataset_path

    def _build_dataset_summary(
        self,
        *,
        dataset: LoadedRetrievalDataset,
        runs: list[EvaluationRun],
        representative: EvaluationRun,
    ) -> StoredRetrievalDataset:
        categories = sorted(
            {
                item.category.strip()
                for item in dataset.items
                if item.category and item.category.strip()
            }
        )
        difficulties = sorted(
            {
                item.difficulty.strip()
                for item in dataset.items
                if item.difficulty and item.difficulty.strip()
            }
        )
        created_at = min(run.created_at for run in runs)
        last_used_at = max(run.created_at for run in runs)
        return StoredRetrievalDataset(
            dataset_sha256=dataset.sha256,
            dataset_name=representative.dataset_name,
            dataset_path=representative.dataset_path,
            total_cases=len(dataset.items),
            categories=categories,
            difficulties=difficulties,
            created_at=created_at,
            last_used_at=last_used_at,
            run_count=len(runs),
        )

    async def _process_run_cases(
        self,
        *,
        repository: EvaluationRepository,
        run: EvaluationRun,
        cases: list[EvaluationCase] | None,
    ) -> None:
        retriever = self._retriever_factory()
        normalization_config = TextNormalizationConfig(
            strip_punctuation=bool(run.config_snapshot.get("strip_punctuation", True))
        )
        useful_chunk_config = UsefulChunkConfig(
            min_keyword_hits=int(run.config_snapshot.get("min_keyword_hits", 2)),
            min_keyword_ratio=float(run.config_snapshot.get("min_keyword_ratio", 0.4)),
        )
        store_chunk_texts = bool(run.config_snapshot.get("store_retrieved_chunk_texts", False))
        judge = (
            self._judge_factory()
            if bool(run.config_snapshot.get("judge_enabled", True))
            else None
        )

        await repository.mark_run_started(run=run)
        await repository.commit()

        target_cases = cases if cases is not None else await repository.list_cases_for_run(run_id=run.id)
        processed = int(run.processed_cases)
        for case in target_cases:
            await self._execute_case(
                repository=repository,
                retriever=retriever,
                judge=judge,
                run=run,
                case=case,
                normalization_config=normalization_config,
                useful_chunk_config=useful_chunk_config,
                store_chunk_texts=store_chunk_texts,
            )
            processed += 1
            await repository.update_processed_count(run=run, processed_cases=processed)
            await repository.commit()

        await self._finalize_run(repository=repository, run=run)

    async def _execute_case(
        self,
        *,
        repository: EvaluationRepository,
        retriever: EvaluationRetriever,
        judge: ContextRelevanceJudge | None,
        run: EvaluationRun,
        case: EvaluationCase,
        normalization_config: TextNormalizationConfig,
        useful_chunk_config: UsefulChunkConfig,
        store_chunk_texts: bool,
    ) -> None:
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
                    [chunk.text for chunk in retrieved_chunks] if store_chunk_texts else []
                ),
                matched_phrases=metrics.matched_phrases,
                matched_keywords=metrics.matched_keywords,
                hit_at_k=metrics.hit_at_k,
                recall_at_k=metrics.recall_at_k,
                precision_at_k=metrics.precision_at_k,
                mrr=metrics.mrr,
                keyword_coverage=metrics.keyword_coverage,
                context_relevance_score=judge_result.score if judge_result is not None else None,
                context_relevance_explanation=(
                    judge_result.explanation if judge_result is not None else None
                ),
                first_correct_rank=metrics.first_correct_rank,
                useful_chunk_count=metrics.useful_chunk_count,
            )
        except Exception as exc:
            await repository.mark_case_failed(case=case, error_message=str(exc))

    async def _finalize_run(
        self,
        *,
        repository: EvaluationRepository,
        run: EvaluationRun,
    ) -> None:
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
            hit_at_k_avg=aggregate_metric_average(case.get("hit_at_k") for case in aggregate_cases)
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

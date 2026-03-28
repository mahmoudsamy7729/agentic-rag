from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from src.agents import AgentAskPipeline, AgentService
from src.modules.evaluation.config import EvaluationRunConfig
from src.modules.evaluation.judge import EvaluationJudgeService, JudgeScore
from src.modules.evaluation.models import EvaluationCase, EvaluationRun
from src.modules.evaluation.repository import EvaluationRepository
from src.rag.models import RetrievedChunk
from src.rag.pipeline import RAGRetrievalService


@dataclass(slots=True)
class RetrievalMetrics:
    hit: bool
    recall: float
    first_relevant_rank: int | None
    reciprocal_rank: float


class EvaluationService:
    def __init__(
        self,
        *,
        repository: EvaluationRepository,
        retrieval_service: RAGRetrievalService,
        agent_service: AgentService,
        ask_pipeline: AgentAskPipeline,
        judge_service: EvaluationJudgeService,
        max_cases: int,
        run_config: EvaluationRunConfig,
    ) -> None:
        self._repository = repository
        self._retrieval_service = retrieval_service
        self._agent_service = agent_service
        self._ask_pipeline = ask_pipeline
        self._judge_service = judge_service
        self._max_cases = max_cases
        self._run_config = run_config

    async def create_run_from_dataset(
        self,
        *,
        owner_user_id: UUID,
        doc_id: str,
        dataset_name: str,
        dataset_bytes: bytes,
    ) -> EvaluationRun:
        rows = self._parse_dataset(dataset_bytes)
        dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
        run = await self._repository.create_run(
            owner_user_id=owner_user_id,
            doc_id=doc_id,
            dataset_name=dataset_name,
            dataset_sha256=dataset_sha256,
            total_cases=len(rows),
            run_config=self._run_config,
        )
        await self._repository.create_cases(run_id=run.id, rows=rows)
        await self._repository.commit()
        return run

    async def list_runs(
        self,
        *,
        owner_user_id: UUID,
        doc_id: str | None,
        limit: int,
        offset: int,
    ) -> tuple[list[EvaluationRun], int]:
        runs = await self._repository.list_owned_runs(
            owner_user_id=owner_user_id,
            doc_id=doc_id,
            limit=limit,
            offset=offset,
        )
        total = await self._repository.count_owned_runs(owner_user_id=owner_user_id, doc_id=doc_id)
        return runs, total

    async def get_run_status(self, *, owner_user_id: UUID, run_id: UUID) -> EvaluationRun | None:
        return await self._repository.get_owned_run(owner_user_id=owner_user_id, run_id=run_id)

    async def get_owned_run_failed_count(
        self,
        *,
        owner_user_id: UUID,
        run_id: UUID,
    ) -> tuple[EvaluationRun, int] | None:
        run = await self._repository.get_owned_run(owner_user_id=owner_user_id, run_id=run_id)
        if run is None:
            return None
        failed_cases = await self._repository.list_failed_cases_for_run(run_id=run_id)
        return run, len(failed_cases)

    async def list_run_cases(
        self,
        *,
        owner_user_id: UUID,
        run_id: UUID,
        limit: int,
        offset: int,
    ) -> tuple[EvaluationRun, list[EvaluationCase], int] | None:
        run = await self._repository.get_owned_run(owner_user_id=owner_user_id, run_id=run_id)
        if run is None:
            return None
        cases = await self._repository.list_owned_cases(
            owner_user_id=owner_user_id,
            run_id=run_id,
            limit=limit,
            offset=offset,
        )
        total = await self._repository.count_owned_cases(owner_user_id=owner_user_id, run_id=run_id)
        return run, cases, total

    async def get_run_report(
        self,
        *,
        owner_user_id: UUID,
        run_id: UUID,
    ) -> tuple[EvaluationRun, list[EvaluationCase]] | None:
        run = await self._repository.get_owned_run(owner_user_id=owner_user_id, run_id=run_id)
        if run is None:
            return None
        cases = await self._repository.list_owned_cases(
            owner_user_id=owner_user_id,
            run_id=run_id,
            limit=run.total_cases,
            offset=0,
        )
        return run, cases

    async def execute_run(self, *, run_id: UUID) -> None:
        run = await self._repository.get_run(run_id=run_id)
        if run is None:
            return

        try:
            await self._repository.mark_run_running(run=run)
            await self._repository.commit()
            cases = await self._repository.list_run_cases(run_id=run_id)

            retrieval_metrics: list[RetrievalMetrics] = []
            judge_scores: list[JudgeScore] = []

            for case in cases:
                case.status = "running"
                case.error_message = None
                await self._repository.commit()
                try:
                    chunks = await self._retrieval_service.retrieve(
                        query=case.question,
                        doc_id=run.doc_id,
                    )
                    retrieved_chunk_ids = [chunk.chunk_id for chunk in chunks]
                    metric = self.compute_retrieval_metrics(
                        expected_chunk_ids=case.expected_chunk_ids,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                    )

                    case.retrieved_chunk_ids = retrieved_chunk_ids
                    case.hit = metric.hit
                    case.recall = metric.recall
                    case.first_relevant_rank = metric.first_relevant_rank
                    case.reciprocal_rank = metric.reciprocal_rank

                    agent_result = await self._agent_service.run(
                        question=case.question,
                        doc_id=run.doc_id,
                        session_id=f"eval-{run.id}-{case.case_index}",
                        user_id=str(run.owner_user_id),
                    )
                    citations = [
                        {
                            "source": citation.source,
                            "doc_id": citation.doc_id,
                            "chunk_id": citation.chunk_id,
                            "snippet": citation.snippet,
                            "page_number": citation.page_number,
                        }
                        for citation in agent_result.citations
                    ]

                    case.generated_answer = agent_result.answer
                    case.citations = citations
                    score = await self._judge_service.evaluate(
                        question=case.question,
                        generated_answer=agent_result.answer,
                        reference_answer=case.reference_answer,
                        citations=citations,
                    )
                    case.accuracy = score.accuracy
                    case.completeness = score.completeness
                    case.relevance = score.relevance
                    case.groundedness = score.groundedness
                    case.judge_feedback = score.feedback
                    case.status = "completed"
                    case.error_message = None

                    retrieval_metrics.append(metric)
                    judge_scores.append(score)
                except Exception as exc:
                    case.status = "failed"
                    case.error_message = str(exc)[:2000]
                finally:
                    await self._repository.increment_progress(run=run)
                    await self._repository.commit()

            await self._repository.complete_run(
                run=run,
                hit_at_k=self._avg([1.0 if m.hit else 0.0 for m in retrieval_metrics]),
                recall_at_k=self._avg([m.recall for m in retrieval_metrics]),
                mrr=self._avg([m.reciprocal_rank for m in retrieval_metrics]),
                accuracy_avg=self._avg([float(score.accuracy) for score in judge_scores]),
                completeness_avg=self._avg([float(score.completeness) for score in judge_scores]),
                relevance_avg=self._avg([float(score.relevance) for score in judge_scores]),
                groundedness_avg=self._avg([float(score.groundedness) for score in judge_scores]),
            )
            await self._repository.commit()
        except Exception as exc:
            await self._repository.rollback()
            run = await self._repository.get_run(run_id=run_id)
            if run is None:
                return
            await self._repository.fail_run(run=run, error_message=str(exc)[:2000])
            await self._repository.commit()

    async def execute_rerun_failed(self, *, run_id: UUID) -> None:
        run = await self._repository.get_run(run_id=run_id)
        if run is None:
            return
        if run.status == "running":
            return

        try:
            await self._repository.mark_run_rerun_started(run=run)
            await self._repository.commit()
            failed_cases = await self._repository.list_failed_cases_for_run(run_id=run_id)

            for case in failed_cases:
                case.status = "running"
                case.error_message = None
                await self._repository.commit()
                try:
                    chunks = await self._retrieval_service.retrieve(
                        query=case.question,
                        doc_id=run.doc_id,
                    )
                    retrieved_chunk_ids = [chunk.chunk_id for chunk in chunks]
                    metric = self.compute_retrieval_metrics(
                        expected_chunk_ids=case.expected_chunk_ids,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                    )

                    case.retrieved_chunk_ids = retrieved_chunk_ids
                    case.hit = metric.hit
                    case.recall = metric.recall
                    case.first_relevant_rank = metric.first_relevant_rank
                    case.reciprocal_rank = metric.reciprocal_rank

                    ask_result = await self._ask_pipeline.ask(
                        owner_user_id=run.owner_user_id,
                        question=case.question,
                        doc_id=run.doc_id,
                        session_id=f"rerun-{run.id}-{case.case_index}",
                        use_cache=False,
                    )

                    case.generated_answer = ask_result.answer
                    case.citations = list(ask_result.citations or [])
                    score = await self._judge_service.evaluate(
                        question=case.question,
                        generated_answer=ask_result.answer,
                        reference_answer=case.reference_answer,
                        citations=case.citations,
                    )
                    case.accuracy = score.accuracy
                    case.completeness = score.completeness
                    case.relevance = score.relevance
                    case.groundedness = score.groundedness
                    case.judge_feedback = score.feedback
                    case.status = "completed"
                    case.error_message = None
                except Exception as exc:
                    case.status = "failed"
                    case.error_message = str(exc)[:2000]
                finally:
                    await self._repository.commit()

            all_cases = await self._repository.list_run_cases(run_id=run_id)
            failed_count = sum(1 for case in all_cases if case.status == "failed")

            await self._repository.set_run_with_aggregates(
                run=run,
                status="completed" if failed_count == 0 else "failed",
                error_message=(
                    None
                    if failed_count == 0
                    else f"{failed_count} case(s) still failed after rerun."
                ),
                hit_at_k=self._avg([1.0 if case.hit else 0.0 for case in all_cases if case.hit is not None]),
                recall_at_k=self._avg([float(case.recall) for case in all_cases if case.recall is not None]),
                mrr=self._avg([float(case.reciprocal_rank) for case in all_cases if case.reciprocal_rank is not None]),
                accuracy_avg=self._avg([float(case.accuracy) for case in all_cases if case.accuracy is not None]),
                completeness_avg=self._avg([float(case.completeness) for case in all_cases if case.completeness is not None]),
                relevance_avg=self._avg([float(case.relevance) for case in all_cases if case.relevance is not None]),
                groundedness_avg=self._avg([float(case.groundedness) for case in all_cases if case.groundedness is not None]),
            )
            await self._repository.commit()
        except Exception as exc:
            await self._repository.rollback()
            run = await self._repository.get_run(run_id=run_id)
            if run is None:
                return
            await self._repository.fail_run(run=run, error_message=str(exc)[:2000])
            await self._repository.commit()

    def _parse_dataset(self, dataset_bytes: bytes) -> list[dict[str, Any]]:
        if not dataset_bytes:
            raise ValueError("Dataset file is empty.")
        try:
            content = dataset_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Dataset must be UTF-8 encoded JSONL.") from exc

        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Dataset does not contain any JSONL rows.")
        if len(lines) > self._max_cases:
            raise ValueError(f"Dataset exceeds max cases limit ({self._max_cases}).")

        rows: list[dict[str, Any]] = []
        for idx, line in enumerate(lines, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx}.") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Line {idx} must be a JSON object.")

            question = str(payload.get("question", "")).strip()
            answer = str(payload.get("answer", "")).strip()
            expected_chunk_ids = payload.get("expected_chunk_ids")

            if not question:
                raise ValueError(f"Line {idx} is missing non-empty 'question'.")
            if not answer:
                raise ValueError(f"Line {idx} is missing non-empty 'answer'.")
            if (
                not isinstance(expected_chunk_ids, list)
                or not expected_chunk_ids
                or not all(isinstance(item, str) and item.strip() for item in expected_chunk_ids)
            ):
                raise ValueError(
                    f"Line {idx} must include non-empty 'expected_chunk_ids' list[str]."
                )

            row = {
                "question": question,
                "answer": answer,
                "expected_chunk_ids": [item.strip() for item in expected_chunk_ids],
                "difficulty": (
                    str(payload["difficulty"]).strip()
                    if payload.get("difficulty") is not None
                    else None
                ),
                "category": (
                    str(payload["category"]).strip()
                    if payload.get("category") is not None
                    else None
                ),
            }
            rows.append(row)
        return rows

    @staticmethod
    def compute_retrieval_metrics(
        *,
        expected_chunk_ids: list[str],
        retrieved_chunk_ids: list[str],
    ) -> RetrievalMetrics:
        expected = {item for item in expected_chunk_ids if item}
        retrieved = [item for item in retrieved_chunk_ids if item]
        if not expected:
            return RetrievalMetrics(
                hit=False,
                recall=0.0,
                first_relevant_rank=None,
                reciprocal_rank=0.0,
            )

        first_relevant_rank: int | None = None
        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in expected:
                first_relevant_rank = rank
                break

        retrieved_set = set(retrieved)
        recall = len(expected & retrieved_set) / len(expected)
        reciprocal_rank = 0.0
        if first_relevant_rank is not None:
            reciprocal_rank = 1.0 / float(first_relevant_rank)

        return RetrievalMetrics(
            hit=first_relevant_rank is not None,
            recall=recall,
            first_relevant_rank=first_relevant_rank,
            reciprocal_rank=reciprocal_rank,
        )

    @staticmethod
    def chunks_to_ids(chunks: list[RetrievedChunk]) -> list[str]:
        return [chunk.chunk_id for chunk in chunks]

    @staticmethod
    def _avg(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)


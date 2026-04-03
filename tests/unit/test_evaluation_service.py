from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

from src.modules.evaluation.config import EvaluationRunConfig
from src.modules.evaluation.judge import JudgeScore
from src.modules.evaluation.service import EvaluationService
from src.rag.models import RetrievedChunk


class _NoopRepository:
    pass


class _NoopRetrieval:
    pass


class _NoopAgent:
    pass


class _NoopJudge:
    pass


class _NoopAskPipeline:
    pass


def _build_service(max_cases: int = 500) -> EvaluationService:
    return EvaluationService(
        repository=_NoopRepository(),  # type: ignore[arg-type]
        retrieval_service=_NoopRetrieval(),  # type: ignore[arg-type]
        agent_service=_NoopAgent(),  # type: ignore[arg-type]
        ask_pipeline=_NoopAskPipeline(),  # type: ignore[arg-type]
        judge_service=_NoopJudge(),  # type: ignore[arg-type]
        max_cases=max_cases,
        run_config=EvaluationRunConfig(
            rag_top_k=4,
            rag_prefetch_k=50,
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            reranker_enabled=False,
            reranker_model=None,
            answer_model="gpt-oss:120b-cloud",
            chunk_strategy="fixed_window",
            chunk_size=800,
            chunk_overlap=120,
        ),
    )


def test_parse_dataset_accepts_valid_jsonl():
    service = _build_service()
    payload = (
        '{"question":"Q1","answer":"A1","expected_chunk_ids":["chunk-1"],"difficulty":"easy"}\n'
        '{"question":"Q2","answer":"A2","expected_chunk_ids":["chunk-2","chunk-3"],"category":"x"}'
    ).encode("utf-8")
    rows = service._parse_dataset(payload)
    assert len(rows) == 2
    assert rows[0]["question"] == "Q1"
    assert rows[0]["expected_chunk_ids"] == ["chunk-1"]
    assert rows[1]["category"] == "x"


def test_parse_dataset_rejects_missing_required_fields():
    service = _build_service()
    payload = b'{"question":"Q1","answer":"","expected_chunk_ids":["chunk-1"]}'
    try:
        service._parse_dataset(payload)
        assert False, "Expected ValueError for empty answer."
    except ValueError as exc:
        assert "missing non-empty 'answer'" in str(exc)


def test_parse_dataset_enforces_max_cases():
    service = _build_service(max_cases=1)
    payload = (
        '{"question":"Q1","answer":"A1","expected_chunk_ids":["chunk-1"]}\n'
        '{"question":"Q2","answer":"A2","expected_chunk_ids":["chunk-2"]}'
    ).encode("utf-8")
    try:
        service._parse_dataset(payload)
        assert False, "Expected ValueError for max cases overflow."
    except ValueError as exc:
        assert "exceeds max cases limit" in str(exc)


def test_retrieval_metrics_hit_recall_mrr():
    metrics = EvaluationService.compute_retrieval_metrics(
        expected_chunk_ids=["chunk-9", "chunk-10"],
        retrieved_chunk_ids=["chunk-3", "chunk-9", "chunk-1"],
    )
    assert metrics.hit is True
    assert metrics.recall == 0.5
    assert metrics.first_relevant_rank == 2
    assert metrics.reciprocal_rank == 0.5


def test_retrieval_metrics_miss():
    metrics = EvaluationService.compute_retrieval_metrics(
        expected_chunk_ids=["chunk-9"],
        retrieved_chunk_ids=["chunk-3", "chunk-1"],
    )
    assert metrics.hit is False
    assert metrics.recall == 0.0
    assert metrics.first_relevant_rank is None
    assert metrics.reciprocal_rank == 0.0


@dataclass
class _Run:
    id: object
    owner_user_id: object
    doc_id: str
    status: str = "queued"
    processed_cases: int = 0
    error_message: str | None = None
    hit_at_k: float | None = None
    recall_at_k: float | None = None
    mrr: float | None = None
    accuracy_avg: float | None = None
    completeness_avg: float | None = None
    relevance_avg: float | None = None
    groundedness_avg: float | None = None


@dataclass
class _Case:
    case_index: int
    question: str
    reference_answer: str
    expected_chunk_ids: list[str]
    status: str = "queued"
    error_message: str | None = None
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    hit: bool | None = None
    recall: float | None = None
    first_relevant_rank: int | None = None
    reciprocal_rank: float | None = None
    generated_answer: str | None = None
    citations: list[dict] = field(default_factory=list)
    accuracy: int | None = None
    completeness: int | None = None
    relevance: int | None = None
    groundedness: int | None = None
    judge_feedback: str | None = None


class _Repository:
    def __init__(self, run: _Run, cases: list[_Case]) -> None:
        self._run = run
        self._cases = cases

    async def get_run(self, *, run_id):
        return self._run if run_id == self._run.id else None

    async def mark_run_running(self, *, run):
        run.status = "running"

    async def list_run_cases(self, *, run_id):
        return self._cases if run_id == self._run.id else []

    async def increment_progress(self, *, run):
        run.processed_cases += 1

    async def complete_run(self, *, run, **aggregate_kwargs):
        run.status = "completed"
        run.error_message = None
        for key, value in aggregate_kwargs.items():
            setattr(run, key, value)

    async def set_run_with_aggregates(self, *, run, status, error_message, **aggregate_kwargs):
        run.status = status
        run.error_message = error_message
        for key, value in aggregate_kwargs.items():
            setattr(run, key, value)

    async def rollback(self):
        return None

    async def fail_run(self, *, run, error_message):
        run.status = "failed"
        run.error_message = error_message

    async def commit(self):
        return None


class _RetrievalService:
    async def retrieve(self, *, query: str, doc_id: str):
        if query == "fail retrieval":
            raise RuntimeError("Reranker failed after one retry.")
        return [
            RetrievedChunk(
                doc_id=doc_id,
                chunk_id="chunk-1",
                source="policy.pdf",
                text="Relevant policy chunk",
                score=0.9,
                page_number=1,
            )
        ]


class _AgentResult:
    def __init__(self, answer: str, citations: list[object]) -> None:
        self.answer = answer
        self.citations = citations


class _Citation:
    def __init__(self, *, doc_id: str, chunk_id: str) -> None:
        self.source = "policy.pdf"
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.snippet = "Relevant policy chunk"
        self.page_number = 1


class _AgentService:
    async def run(
        self,
        *,
        question: str,
        doc_id: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        return _AgentResult(
            answer=f"answer for {question}",
            citations=[_Citation(doc_id=doc_id, chunk_id="chunk-1")],
        )


class _JudgeService:
    async def evaluate(
        self,
        *,
        question: str,
        generated_answer: str,
        reference_answer: str,
        citations: list[dict],
    ) -> JudgeScore:
        return JudgeScore(
            accuracy=4,
            completeness=4,
            relevance=4,
            groundedness=4,
            feedback="ok",
        )


def test_execute_run_marks_only_failed_case_and_continues():
    run = _Run(id=uuid4(), owner_user_id=uuid4(), doc_id="doc-1")
    cases = [
        _Case(
            case_index=0,
            question="fail retrieval",
            reference_answer="first",
            expected_chunk_ids=["chunk-1"],
        ),
        _Case(
            case_index=1,
            question="second question",
            reference_answer="second",
            expected_chunk_ids=["chunk-1"],
        ),
    ]
    repository = _Repository(run, cases)
    service = EvaluationService(
        repository=repository,  # type: ignore[arg-type]
        retrieval_service=_RetrievalService(),  # type: ignore[arg-type]
        agent_service=_AgentService(),  # type: ignore[arg-type]
        ask_pipeline=_NoopAskPipeline(),  # type: ignore[arg-type]
        judge_service=_JudgeService(),  # type: ignore[arg-type]
        max_cases=500,
        run_config=EvaluationRunConfig(
            rag_top_k=4,
            rag_prefetch_k=50,
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            reranker_enabled=True,
            reranker_model="rerank-english-v3.0",
            answer_model="gpt-oss:120b-cloud",
            chunk_strategy="fixed_window",
            chunk_size=800,
            chunk_overlap=120,
        ),
    )

    asyncio.run(service.execute_run(run_id=run.id))

    assert cases[0].status == "failed"
    assert cases[0].error_message == "Reranker failed after one retry."
    assert cases[1].status == "completed"
    assert cases[1].generated_answer == "answer for second question"
    assert run.processed_cases == 2
    assert run.status == "failed"
    assert run.error_message == "1 case(s) failed during evaluation."

from __future__ import annotations

from src.modules.evaluation.service import EvaluationService


class _NoopRepository:
    pass


class _NoopRetrieval:
    pass


class _NoopAgent:
    pass


class _NoopJudge:
    pass


def _build_service(max_cases: int = 500) -> EvaluationService:
    return EvaluationService(
        repository=_NoopRepository(),  # type: ignore[arg-type]
        retrieval_service=_NoopRetrieval(),  # type: ignore[arg-type]
        agent_service=_NoopAgent(),  # type: ignore[arg-type]
        judge_service=_NoopJudge(),  # type: ignore[arg-type]
        max_cases=max_cases,
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

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


EvaluationRunStatus = Literal["queued", "running", "completed", "failed"]


class EvaluationRunCreateResponse(BaseModel):
    status: str = Field(description="API operation status.")
    run_id: UUID = Field(description="Created evaluation run id.")
    run_status: EvaluationRunStatus = Field(description="Initial run status.")
    total_cases: int = Field(description="Number of accepted test cases.")


class EvaluationRerunFailedResponse(BaseModel):
    status: str = Field(description="API operation status.")
    run_id: UUID = Field(description="Evaluation run id.")
    queued_failed_cases: int = Field(description="Number of failed cases queued for rerun.")


class EvaluationRunConfig(BaseModel):
    rag_top_k: int | None = Field(default=None)
    rag_prefetch_k: int | None = Field(default=None)
    embedding_provider: str | None = Field(default=None)
    embedding_model: str | None = Field(default=None)
    reranker_enabled: bool | None = Field(default=None)
    reranker_model: str | None = Field(default=None)
    answer_model: str | None = Field(default=None)
    chunk_strategy: str | None = Field(default=None)
    chunk_size: int | None = Field(default=None)
    chunk_overlap: int | None = Field(default=None)


class EvaluationRunStatusResponse(BaseModel):
    status: str = Field(description="API operation status.")
    run_id: UUID = Field(description="Evaluation run id.")
    doc_id: str = Field(description="Document id used for evaluation.")
    run_status: EvaluationRunStatus = Field(description="Current run status.")
    dataset_name: str = Field(description="Uploaded dataset file name.")
    dataset_sha256: str = Field(description="Dataset checksum.")
    total_cases: int = Field(description="Total cases in this run.")
    processed_cases: int = Field(description="Number of processed cases.")
    hit_at_k: float | None = Field(default=None, description="Run-level Hit@k average.")
    recall_at_k: float | None = Field(default=None, description="Run-level Recall@k average.")
    mrr: float | None = Field(default=None, description="Run-level MRR average.")
    accuracy_avg: float | None = Field(default=None, description="Average judge accuracy score.")
    completeness_avg: float | None = Field(
        default=None,
        description="Average judge completeness score.",
    )
    relevance_avg: float | None = Field(default=None, description="Average judge relevance score.")
    groundedness_avg: float | None = Field(
        default=None,
        description="Average judge groundedness score.",
    )
    error_message: str | None = Field(default=None, description="Top-level run error, if any.")
    config: EvaluationRunConfig = Field(description="Snapshot of config used for this run.")
    created_at: datetime = Field(description="Run creation timestamp.")
    started_at: datetime | None = Field(default=None, description="Run start timestamp.")
    finished_at: datetime | None = Field(default=None, description="Run finish timestamp.")


class EvaluationCaseItem(BaseModel):
    case_id: UUID = Field(description="Evaluation case id.")
    case_index: int = Field(description="Zero-based case index.")
    question: str = Field(description="Case question.")
    reference_answer: str = Field(description="Reference answer.")
    expected_chunk_ids: list[str] = Field(default_factory=list, description="Expected chunk ids.")
    difficulty: str | None = Field(default=None, description="Optional dataset difficulty label.")
    category: str | None = Field(default=None, description="Optional dataset category label.")
    retrieved_chunk_ids: list[str] = Field(default_factory=list, description="Retrieved chunk ids.")
    hit: bool | None = Field(default=None, description="Whether any relevant chunk was retrieved.")
    recall: float | None = Field(default=None, description="Per-case recall value.")
    first_relevant_rank: int | None = Field(default=None, description="Rank of first relevant chunk.")
    reciprocal_rank: float | None = Field(default=None, description="Per-case reciprocal rank.")
    generated_answer: str | None = Field(default=None, description="Generated answer text.")
    citations: list[dict] = Field(default_factory=list, description="Returned citations payload.")
    accuracy: int | None = Field(default=None, description="Judge accuracy score 1..5.")
    completeness: int | None = Field(default=None, description="Judge completeness score 1..5.")
    relevance: int | None = Field(default=None, description="Judge relevance score 1..5.")
    groundedness: int | None = Field(default=None, description="Judge groundedness score 1..5.")
    judge_feedback: str | None = Field(default=None, description="Judge textual feedback.")
    case_status: EvaluationRunStatus = Field(description="Case execution status.")
    error_message: str | None = Field(default=None, description="Per-case error message.")


class EvaluationCaseListResponse(BaseModel):
    status: str = Field(description="API operation status.")
    run_id: UUID = Field(description="Evaluation run id.")
    run_status: EvaluationRunStatus = Field(description="Current run status.")
    total: int = Field(description="Total number of cases in this run.")
    limit: int = Field(description="Pagination limit used.")
    offset: int = Field(description="Pagination offset used.")
    items: list[EvaluationCaseItem] = Field(default_factory=list, description="Paginated case list.")


class EvaluationRunListResponse(BaseModel):
    status: str = Field(description="API operation status.")
    total: int = Field(description="Total number of runs in query scope.")
    limit: int = Field(description="Pagination limit used.")
    offset: int = Field(description="Pagination offset used.")
    items: list[EvaluationRunStatusResponse] = Field(
        default_factory=list,
        description="Paginated evaluation run summaries.",
    )


class EvaluationReportResponse(BaseModel):
    status: str = Field(description="API operation status.")
    run: EvaluationRunStatusResponse = Field(description="Run-level summary.")
    cases: list[EvaluationCaseItem] = Field(default_factory=list, description="All case-level results.")


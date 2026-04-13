from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class EvaluationMetricSummary(BaseModel):
    hit_at_k_avg: float | None = Field(default=None)
    recall_at_k_avg: float | None = Field(default=None)
    precision_at_k_avg: float | None = Field(default=None)
    mrr_avg: float | None = Field(default=None)
    keyword_coverage_avg: float | None = Field(default=None)
    context_relevance_score_avg: float | None = Field(default=None)


class EvaluationGroupedBucket(BaseModel):
    count: int = Field(description="Number of cases in the group.")
    hit_at_k_avg: float | None = Field(default=None)
    recall_at_k_avg: float | None = Field(default=None)
    precision_at_k_avg: float | None = Field(default=None)
    mrr_avg: float | None = Field(default=None)
    keyword_coverage_avg: float | None = Field(default=None)
    context_relevance_score_avg: float | None = Field(default=None)


class EvaluationRunItem(BaseModel):
    run_id: UUID
    file_id: str
    document_name: str | None = None
    chunking_strategy: str | None = None
    status: str
    evaluation_type: str
    dataset_name: str
    dataset_sha256: str
    total_cases: int
    processed_cases: int
    k: int
    config_snapshot: dict
    grouped_summary: dict[str, dict[str, EvaluationGroupedBucket]] = Field(default_factory=dict)
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metrics: EvaluationMetricSummary


class EvaluationRunListResponse(BaseModel):
    status: str
    items: list[EvaluationRunItem] = Field(default_factory=list)
    total: int
    limit: int
    offset: int


class EvaluationRunDetailResponse(BaseModel):
    status: str
    item: EvaluationRunItem


class EvaluationCaseItem(BaseModel):
    case_id: UUID
    case_index: int
    status: str
    question: str
    reference_answer: str
    must_include_keywords: list[str]
    must_include_phrases: list[str]
    difficulty: str | None = None
    category: str | None = None
    hit_at_k: float | None = None
    recall_at_k: float | None = None
    precision_at_k: float | None = None
    mrr: float | None = None
    keyword_coverage: float | None = None
    context_relevance_score: int | None = None
    context_relevance_explanation: str | None = None
    matched_phrases: list[str] = Field(default_factory=list)
    matched_keywords: list[str] = Field(default_factory=list)
    first_correct_rank: int | None = None
    useful_chunk_count: int | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_texts: list[str] = Field(default_factory=list)
    file_id: str
    error_message: str | None = None


class EvaluationCaseListResponse(BaseModel):
    status: str
    run_id: UUID
    total: int
    limit: int
    offset: int
    items: list[EvaluationCaseItem] = Field(default_factory=list)


class EvaluationRunDeleteResponse(BaseModel):
    status: str
    run_id: UUID
    deleted: bool


class EvaluationDatasetPreviewItem(BaseModel):
    question: str
    answer: str
    must_include_keywords: list[str] = Field(default_factory=list)
    must_include_phrases: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    category: str | None = None


class EvaluationDatasetItem(BaseModel):
    dataset_sha256: str
    dataset_name: str
    file_name: str
    total_cases: int
    categories: list[str] = Field(default_factory=list)
    difficulties: list[str] = Field(default_factory=list)
    created_at: datetime
    last_used_at: datetime
    run_count: int


class EvaluationDatasetListResponse(BaseModel):
    status: str
    items: list[EvaluationDatasetItem] = Field(default_factory=list)


class EvaluationDatasetPreviewResponse(BaseModel):
    status: str
    item: EvaluationDatasetItem
    sample_items: list[EvaluationDatasetPreviewItem] = Field(default_factory=list)


class EvaluationDatasetDeleteResponse(BaseModel):
    status: str
    dataset_sha256: str
    deleted: bool

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.infrastructure.database import Base


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    owner_user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    doc_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    dataset_name: Mapped[str] = mapped_column(String(512), nullable=False)
    dataset_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    total_cases: Mapped[int] = mapped_column(Integer, nullable=False)
    processed_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cfg_rag_top_k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cfg_rag_prefetch_k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cfg_embedding_provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    cfg_embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cfg_reranker_enabled: Mapped[bool | None] = mapped_column(nullable=True)
    cfg_reranker_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cfg_answer_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cfg_chunk_strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    cfg_chunk_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cfg_chunk_overlap: Mapped[int | None] = mapped_column(Integer, nullable=True)

    hit_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    mrr: Mapped[float | None] = mapped_column(Float, nullable=True)
    accuracy_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    completeness_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    relevance_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    groundedness_avg: Mapped[float | None] = mapped_column(Float, nullable=True)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EvaluationCase(Base):
    __tablename__ = "evaluation_cases"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    case_index: Mapped[int] = mapped_column(Integer, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    reference_answer: Mapped[str] = mapped_column(Text, nullable=False)
    expected_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    difficulty: Mapped[str | None] = mapped_column(String(64), nullable=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)

    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    hit: Mapped[bool | None] = mapped_column(nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    first_relevant_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reciprocal_rank: Mapped[float | None] = mapped_column(Float, nullable=True)

    generated_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    citations: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, default=list)
    accuracy: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completeness: Mapped[int | None] = mapped_column(Integer, nullable=True)
    relevance: Mapped[int | None] = mapped_column(Integer, nullable=True)
    groundedness: Mapped[int | None] = mapped_column(Integer, nullable=True)
    judge_feedback: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="queued")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructure.database import Base

JSON_PAYLOAD_TYPE = JSON().with_variant(JSONB, "postgresql")


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
    evaluation_type: Mapped[str] = mapped_column(String(32), nullable=False, default="retrieval")
    dataset_name: Mapped[str] = mapped_column(String(512), nullable=False)
    dataset_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    dataset_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    total_cases: Mapped[int] = mapped_column(Integer, nullable=False)
    processed_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    k: Mapped[int] = mapped_column(Integer, nullable=False)
    config_snapshot: Mapped[dict] = mapped_column(JSON_PAYLOAD_TYPE, nullable=False, default=dict)
    hit_at_k_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_at_k_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_at_k_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    mrr_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    keyword_coverage_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_relevance_score_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    grouped_summary: Mapped[dict] = mapped_column(JSON_PAYLOAD_TYPE, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    cases: Mapped[list["EvaluationCase"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="EvaluationCase.case_index",
    )


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
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued", index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    reference_answer: Mapped[str] = mapped_column(Text, nullable=False)
    must_include_keywords: Mapped[list[str]] = mapped_column(
        JSON_PAYLOAD_TYPE,
        nullable=False,
        default=list,
    )
    must_include_phrases: Mapped[list[str]] = mapped_column(
        JSON_PAYLOAD_TYPE,
        nullable=False,
        default=list,
    )
    difficulty: Mapped[str | None] = mapped_column(String(64), nullable=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(
        JSON_PAYLOAD_TYPE,
        nullable=False,
        default=list,
    )
    retrieved_chunk_texts: Mapped[list[str]] = mapped_column(
        JSON_PAYLOAD_TYPE,
        nullable=False,
        default=list,
    )
    matched_phrases: Mapped[list[str]] = mapped_column(
        JSON_PAYLOAD_TYPE,
        nullable=False,
        default=list,
    )
    matched_keywords: Mapped[list[str]] = mapped_column(
        JSON_PAYLOAD_TYPE,
        nullable=False,
        default=list,
    )
    hit_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    mrr: Mapped[float | None] = mapped_column(Float, nullable=True)
    keyword_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_relevance_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    context_relevance_explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    first_correct_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    useful_chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    run: Mapped[EvaluationRun] = relationship(back_populates="cases")

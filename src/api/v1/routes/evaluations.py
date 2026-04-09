from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile

from src.api.v1.dependencies import RetrievalEvaluationServiceDep
from src.api.v1.schemas.evaluation import (
    EvaluationCaseItem,
    EvaluationCaseListResponse,
    EvaluationMetricSummary,
    EvaluationRunDetailResponse,
    EvaluationRunItem,
    EvaluationRunListResponse,
)
from src.modules.documents import DocumentsRepositoryDep
from src.modules.evaluation.dataset import DatasetValidationError
from src.modules.evaluation.service import RetrievalEvaluationRunConfig
from src.modules.users.dependencies import ActiveUserDep
from src.settings.config import settings

router = APIRouter(tags=["evaluations"])


@router.post("/evaluations/retrieval", response_model=EvaluationRunDetailResponse, status_code=202)
async def create_retrieval_evaluation(
    background_tasks: BackgroundTasks,
    current_user: ActiveUserDep,
    documents_repository: DocumentsRepositoryDep,
    evaluation_service: RetrievalEvaluationServiceDep,
    file_id: Annotated[str, Form()],
    k: Annotated[int, Form(ge=1, le=100)],
    dataset_file: Annotated[UploadFile, File()],
    strip_punctuation: Annotated[bool | None, Form()] = None,
    min_keyword_hits: Annotated[int | None, Form(ge=0, le=100)] = None,
    min_keyword_ratio: Annotated[float | None, Form(ge=0.0, le=1.0)] = None,
    store_retrieved_chunk_texts: Annotated[bool | None, Form()] = None,
    judge_enabled: Annotated[bool | None, Form()] = None,
):
    document = await documents_repository.get_owned_document(
        owner_user_id=current_user.id,
        doc_id=file_id,
        include_deleted=False,
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    dataset_bytes = await dataset_file.read()
    if not dataset_bytes:
        raise HTTPException(status_code=422, detail="Dataset file is empty.")

    config = RetrievalEvaluationRunConfig(
        k=k,
        strip_punctuation=(
            settings.evaluation_text_strip_punctuation
            if strip_punctuation is None
            else strip_punctuation
        ),
        min_keyword_hits=(
            settings.evaluation_useful_chunk_min_keyword_hits
            if min_keyword_hits is None
            else min_keyword_hits
        ),
        min_keyword_ratio=(
            settings.evaluation_useful_chunk_min_keyword_ratio
            if min_keyword_ratio is None
            else min_keyword_ratio
        ),
        store_retrieved_chunk_texts=(
            settings.evaluation_store_retrieved_chunk_texts
            if store_retrieved_chunk_texts is None
            else store_retrieved_chunk_texts
        ),
        judge_enabled=(
            settings.evaluation_judge_enabled if judge_enabled is None else judge_enabled
        ),
        rag_top_k=settings.rag_top_k,
        rag_prefetch_k=settings.rag_prefetch_k,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        reranker_enabled=settings.reranker_enabled,
        reranker_model=settings.reranker_model,
        judge_model=(
            settings.evaluation_judge_model
            if (settings.evaluation_judge_enabled if judge_enabled is None else judge_enabled)
            else None
        ),
    )
    try:
        result = await evaluation_service.create_run_from_upload(
            owner_user_id=current_user.id,
            file_id=file_id,
            dataset_name=dataset_file.filename or "dataset.jsonl",
            dataset_bytes=dataset_bytes,
            config=config,
        )
    except DatasetValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    background_tasks.add_task(evaluation_service.process_run, run_id=result.run_id)
    run = await evaluation_service.get_run(owner_user_id=current_user.id, run_id=result.run_id)
    if run is None:
        raise HTTPException(status_code=500, detail="Evaluation run could not be loaded.")
    return EvaluationRunDetailResponse(status="accepted", item=_to_run_item(run=run))


@router.get("/evaluations", response_model=EvaluationRunListResponse)
async def list_evaluations(
    current_user: ActiveUserDep,
    evaluation_service: RetrievalEvaluationServiceDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    runs = await evaluation_service.list_runs(
        owner_user_id=current_user.id,
        limit=limit,
        offset=offset,
    )
    return EvaluationRunListResponse(
        status="ok",
        items=[_to_run_item(run=run) for run in runs],
        limit=limit,
        offset=offset,
    )


@router.get("/evaluations/{run_id}", response_model=EvaluationRunDetailResponse)
async def get_evaluation(
    run_id: UUID,
    current_user: ActiveUserDep,
    evaluation_service: RetrievalEvaluationServiceDep,
):
    run = await evaluation_service.get_run(owner_user_id=current_user.id, run_id=run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found.")
    return EvaluationRunDetailResponse(status="ok", item=_to_run_item(run=run))


@router.get("/evaluations/{run_id}/cases", response_model=EvaluationCaseListResponse)
async def list_evaluation_cases(
    run_id: UUID,
    current_user: ActiveUserDep,
    evaluation_service: RetrievalEvaluationServiceDep,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    run = await evaluation_service.get_run(owner_user_id=current_user.id, run_id=run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found.")
    cases, total = await evaluation_service.list_cases(
        owner_user_id=current_user.id,
        run_id=run_id,
        limit=limit,
        offset=offset,
    )
    return EvaluationCaseListResponse(
        status="ok",
        run_id=run_id,
        total=total,
        limit=limit,
        offset=offset,
        items=[
            EvaluationCaseItem(
                case_id=case.id,
                case_index=case.case_index,
                status=case.status,
                question=case.question,
                reference_answer=case.reference_answer,
                must_include_keywords=case.must_include_keywords,
                must_include_phrases=case.must_include_phrases,
                difficulty=case.difficulty,
                category=case.category,
                hit_at_k=case.hit_at_k,
                recall_at_k=case.recall_at_k,
                precision_at_k=case.precision_at_k,
                mrr=case.mrr,
                keyword_coverage=case.keyword_coverage,
                context_relevance_score=case.context_relevance_score,
                context_relevance_explanation=case.context_relevance_explanation,
                matched_phrases=case.matched_phrases,
                matched_keywords=case.matched_keywords,
                first_correct_rank=case.first_correct_rank,
                useful_chunk_count=case.useful_chunk_count,
                retrieved_chunk_ids=case.retrieved_chunk_ids,
                retrieved_chunk_texts=case.retrieved_chunk_texts,
                file_id=run.doc_id,
                error_message=case.error_message,
            )
            for case in cases
        ],
    )


def _to_run_item(*, run) -> EvaluationRunItem:
    grouped_summary = {
        bucket_name: {
            key: data
            for key, data in bucket_data.items()
        }
        for bucket_name, bucket_data in (run.grouped_summary or {}).items()
    }
    return EvaluationRunItem(
        run_id=run.id,
        file_id=run.doc_id,
        status=run.status,
        evaluation_type=run.evaluation_type,
        dataset_name=run.dataset_name,
        dataset_sha256=run.dataset_sha256,
        total_cases=run.total_cases,
        processed_cases=run.processed_cases,
        k=run.k,
        config_snapshot=run.config_snapshot,
        grouped_summary=grouped_summary,
        error_message=run.error_message,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        metrics=EvaluationMetricSummary(
            hit_at_k_avg=run.hit_at_k_avg,
            recall_at_k_avg=run.recall_at_k_avg,
            precision_at_k_avg=run.precision_at_k_avg,
            mrr_avg=run.mrr_avg,
            keyword_coverage_avg=run.keyword_coverage_avg,
            context_relevance_score_avg=run.context_relevance_score_avg,
        ),
    )

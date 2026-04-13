from __future__ import annotations

from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from src.api.v1.dependencies import RetrievalEvaluationServiceDep
from src.api.v1.schemas.evaluation import (
    EvaluationCaseItem,
    EvaluationCaseListResponse,
    EvaluationDatasetDeleteResponse,
    EvaluationDatasetItem,
    EvaluationDatasetListResponse,
    EvaluationDatasetPreviewItem,
    EvaluationDatasetPreviewResponse,
    EvaluationMetricSummary,
    EvaluationRunDeleteResponse,
    EvaluationRunDetailResponse,
    EvaluationRunItem,
    EvaluationRunListResponse,
    EvaluationRunRerunResponse,
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
    dataset_file: Annotated[UploadFile | None, File()] = None,
    dataset_sha256: Annotated[str | None, Form()] = None,
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

    if dataset_file is None and not dataset_sha256:
        raise HTTPException(status_code=422, detail="Either dataset_file or dataset_sha256 is required.")
    if dataset_file is not None and dataset_sha256:
        raise HTTPException(status_code=422, detail="Provide either dataset_file or dataset_sha256, not both.")

    config = _build_config(
        k=k,
        strip_punctuation=strip_punctuation,
        min_keyword_hits=min_keyword_hits,
        min_keyword_ratio=min_keyword_ratio,
        store_retrieved_chunk_texts=store_retrieved_chunk_texts,
        judge_enabled=judge_enabled,
    )
    try:
        if dataset_file is not None:
            dataset_bytes = await dataset_file.read()
            if not dataset_bytes:
                raise HTTPException(status_code=422, detail="Dataset file is empty.")
            result = await evaluation_service.create_run_from_upload(
                owner_user_id=current_user.id,
                file_id=file_id,
                dataset_name=dataset_file.filename or "dataset.jsonl",
                dataset_bytes=dataset_bytes,
                config=config,
            )
        else:
            try:
                result = await evaluation_service.create_run_from_existing_dataset(
                    owner_user_id=current_user.id,
                    file_id=file_id,
                    dataset_sha256=str(dataset_sha256),
                    config=config,
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Dataset not found.") from exc
    except DatasetValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    background_tasks.add_task(evaluation_service.process_run, run_id=result.run_id)
    run = await evaluation_service.get_run(owner_user_id=current_user.id, run_id=result.run_id)
    if run is None:
        raise HTTPException(status_code=500, detail="Evaluation run could not be loaded.")
    return EvaluationRunDetailResponse(
        status="accepted",
        item=_to_run_item(run=run, document=document),
    )


@router.get("/evaluations", response_model=EvaluationRunListResponse)
async def list_evaluations(
    current_user: ActiveUserDep,
    documents_repository: DocumentsRepositoryDep,
    evaluation_service: RetrievalEvaluationServiceDep,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    runs, total = await evaluation_service.list_runs(
        owner_user_id=current_user.id,
        limit=limit,
        offset=offset,
    )
    document_lookup = await _load_document_lookup(
        documents_repository=documents_repository,
        owner_user_id=current_user.id,
        doc_ids=[run.doc_id for run in runs],
    )
    return EvaluationRunListResponse(
        status="ok",
        items=[
            _to_run_item(
                run=run,
                document=document_lookup.get(run.doc_id),
            )
            for run in runs
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/evaluations/{run_id}", response_model=EvaluationRunDetailResponse)
async def get_evaluation(
    run_id: UUID,
    current_user: ActiveUserDep,
    documents_repository: DocumentsRepositoryDep,
    evaluation_service: RetrievalEvaluationServiceDep,
):
    run = await evaluation_service.get_run(owner_user_id=current_user.id, run_id=run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found.")
    document = await documents_repository.get_owned_document(
        owner_user_id=current_user.id,
        doc_id=run.doc_id,
        include_deleted=True,
    )
    return EvaluationRunDetailResponse(
        status="ok",
        item=_to_run_item(run=run, document=document),
    )


@router.delete("/evaluations/{run_id}", response_model=EvaluationRunDeleteResponse)
async def delete_evaluation(
    run_id: UUID,
    current_user: ActiveUserDep,
    evaluation_service: RetrievalEvaluationServiceDep,
):
    deleted = await evaluation_service.delete_run(
        owner_user_id=current_user.id,
        run_id=run_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Evaluation run not found.")
    return EvaluationRunDeleteResponse(status="ok", run_id=run_id, deleted=True)


@router.post(
    "/evaluations/{run_id}/rerun-failed",
    response_model=EvaluationRunRerunResponse,
    status_code=202,
)
async def rerun_failed_evaluation_cases(
    run_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: ActiveUserDep,
    documents_repository: DocumentsRepositoryDep,
    evaluation_service: RetrievalEvaluationServiceDep,
):
    try:
        result = await evaluation_service.rerun_failed_cases(
            owner_user_id=current_user.id,
            run_id=run_id,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    background_tasks.add_task(
        evaluation_service.process_selected_cases,
        run_id=result.run_id,
        case_ids=result.case_ids,
    )
    run = await evaluation_service.get_run(owner_user_id=current_user.id, run_id=result.run_id)
    if run is None:
        raise HTTPException(status_code=500, detail="Evaluation run could not be loaded.")
    document = await documents_repository.get_owned_document(
        owner_user_id=current_user.id,
        doc_id=run.doc_id,
        include_deleted=True,
    )
    return EvaluationRunRerunResponse(
        status="accepted",
        item=_to_run_item(run=run, document=document),
        rerun_case_count=result.rerun_case_count,
    )


@router.get("/evaluations/{run_id}/cases", response_model=EvaluationCaseListResponse)
async def list_evaluation_cases(
    run_id: UUID,
    current_user: ActiveUserDep,
    evaluation_service: RetrievalEvaluationServiceDep,
    limit: int = Query(default=100, ge=1, le=5000),
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


@router.get("/evaluation-datasets", response_model=EvaluationDatasetListResponse)
async def list_evaluation_datasets(
    current_user: ActiveUserDep,
    evaluation_service: RetrievalEvaluationServiceDep,
):
    datasets = await evaluation_service.list_datasets(owner_user_id=current_user.id)
    return EvaluationDatasetListResponse(
        status="ok",
        items=[_to_dataset_item(item) for item in datasets],
    )


@router.get("/evaluation-datasets/{dataset_sha256}", response_model=EvaluationDatasetPreviewResponse)
async def preview_evaluation_dataset(
    dataset_sha256: str,
    current_user: ActiveUserDep,
    evaluation_service: RetrievalEvaluationServiceDep,
    sample_limit: int = Query(default=10, ge=1, le=50),
):
    payload = await evaluation_service.preview_dataset(
        owner_user_id=current_user.id,
        dataset_sha256=dataset_sha256,
        sample_limit=sample_limit,
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    dataset, sample_items = payload
    return EvaluationDatasetPreviewResponse(
        status="ok",
        item=_to_dataset_item(dataset),
        sample_items=[
            EvaluationDatasetPreviewItem(
                question=item.question,
                answer=item.answer,
                must_include_keywords=item.must_include_keywords,
                must_include_phrases=item.must_include_phrases,
                difficulty=item.difficulty,
                category=item.category,
            )
            for item in sample_items
        ],
    )


@router.get("/evaluation-datasets/{dataset_sha256}/download")
async def download_evaluation_dataset(
    dataset_sha256: str,
    current_user: ActiveUserDep,
    evaluation_service: RetrievalEvaluationServiceDep,
):
    payload = await evaluation_service.get_dataset_bytes(
        owner_user_id=current_user.id,
        dataset_sha256=dataset_sha256,
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    dataset, _dataset_bytes = payload
    return FileResponse(
        path=Path(dataset.dataset_path),
        media_type="application/x-ndjson",
        filename=dataset.dataset_name,
    )


@router.delete("/evaluation-datasets/{dataset_sha256}", response_model=EvaluationDatasetDeleteResponse)
async def delete_evaluation_dataset(
    dataset_sha256: str,
    current_user: ActiveUserDep,
    evaluation_service: RetrievalEvaluationServiceDep,
):
    deleted = await evaluation_service.delete_dataset(
        owner_user_id=current_user.id,
        dataset_sha256=dataset_sha256,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return EvaluationDatasetDeleteResponse(
        status="ok",
        dataset_sha256=dataset_sha256,
        deleted=True,
    )


def _build_config(
    *,
    k: int,
    strip_punctuation: bool | None,
    min_keyword_hits: int | None,
    min_keyword_ratio: float | None,
    store_retrieved_chunk_texts: bool | None,
    judge_enabled: bool | None,
) -> RetrievalEvaluationRunConfig:
    resolved_judge_enabled = (
        settings.evaluation_judge_enabled if judge_enabled is None else judge_enabled
    )
    return RetrievalEvaluationRunConfig(
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
        judge_enabled=resolved_judge_enabled,
        rag_top_k=settings.rag_top_k,
        rag_prefetch_k=settings.rag_prefetch_k,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        reranker_enabled=settings.reranker_enabled,
        reranker_model=settings.reranker_model,
        judge_model=(
            settings.evaluation_judge_model if resolved_judge_enabled else None
        ),
    )


async def _load_document_lookup(
    *,
    documents_repository,
    owner_user_id,
    doc_ids: list[str],
) -> dict[str, object]:
    unique_doc_ids = list(dict.fromkeys(doc_ids))
    if not unique_doc_ids:
        return {}
    if hasattr(documents_repository, "list_owned_documents_by_ids"):
        items = await documents_repository.list_owned_documents_by_ids(
            owner_user_id=owner_user_id,
            doc_ids=unique_doc_ids,
            include_deleted=True,
        )
        return {item.id: item for item in items}
    lookup = {}
    for doc_id in unique_doc_ids:
        item = await documents_repository.get_owned_document(
            owner_user_id=owner_user_id,
            doc_id=doc_id,
            include_deleted=True,
        )
        if item is not None:
            lookup[item.id] = item
    return lookup


def _to_run_item(*, run, document=None) -> EvaluationRunItem:
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
        document_name=getattr(document, "source", None),
        chunking_strategy=getattr(document, "chunking_strategy", None),
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


def _to_dataset_item(dataset) -> EvaluationDatasetItem:
    return EvaluationDatasetItem(
        dataset_sha256=dataset.dataset_sha256,
        dataset_name=dataset.dataset_name,
        file_name=dataset.dataset_name,
        total_cases=dataset.total_cases,
        categories=dataset.categories,
        difficulties=dataset.difficulties,
        created_at=dataset.created_at,
        last_used_at=dataset.last_used_at,
        run_count=dataset.run_count,
    )

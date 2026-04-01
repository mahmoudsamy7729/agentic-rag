from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.api.v1.dependencies import (
    EvaluationJobRunnerDep,
    EvaluationRerunFailedJobRunnerDep,
    EvaluationServiceDep,
    build_evaluation_run_config,
)
from src.api.v1.schemas import (
    EvaluationCaseItem,
    EvaluationCaseListResponse,
    EvaluationRerunFailedResponse,
    EvaluationRunConfig,
    EvaluationRunListResponse,
    EvaluationReportResponse,
    EvaluationRunCreateResponse,
    EvaluationRunStatusResponse,
)
from src.modules.documents import DocumentsRepositoryDep
from src.modules.evaluation.models import EvaluationCase, EvaluationRun
from src.modules.users.dependencies import ActiveUserDep
from src.settings.config import settings

router = APIRouter(tags=["evaluations"])
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[4] / "templates"))


@router.get("/evaluation-ui", response_class=HTMLResponse, include_in_schema=False)
async def evaluation_ui(request: Request):
    return templates.TemplateResponse(
        "evaluation_ui.html",
        {
            "request": request,
            "poll_interval_ms": settings.eval_poll_interval_ms,
        },
    )


@router.get("/evaluation-history-ui", response_class=HTMLResponse, include_in_schema=False)
async def evaluation_history_ui(request: Request):
    return templates.TemplateResponse(
        "evaluation_history_ui.html",
        {
            "request": request,
            "poll_interval_ms": settings.eval_poll_interval_ms,
        },
    )


@router.get("/evaluation-runs/{run_id}/ui", response_class=HTMLResponse, include_in_schema=False)
async def evaluation_run_ui(request: Request, run_id: UUID):
    return templates.TemplateResponse(
        "evaluation_run_ui.html",
        {
            "request": request,
            "run_id": str(run_id),
            "poll_interval_ms": settings.eval_poll_interval_ms,
        },
    )


@router.post("/evaluations/rag", response_model=EvaluationRunCreateResponse, status_code=202)
async def create_rag_evaluation_run(
    background_tasks: BackgroundTasks,
    current_user: ActiveUserDep,
    documents_repository: DocumentsRepositoryDep,
    evaluation_service: EvaluationServiceDep,
    evaluation_job_runner: EvaluationJobRunnerDep,
    doc_id: str = Form(...),
    file: UploadFile = File(...),
):
    document = await documents_repository.get_owned_document(
        owner_user_id=current_user.id,
        doc_id=doc_id,
        include_deleted=False,
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if not file.filename or not file.filename.lower().endswith(".jsonl"):
        raise HTTPException(status_code=422, detail="Dataset file must be a .jsonl file.")

    payload = await file.read()
    max_bytes = settings.eval_upload_max_mb * 1024 * 1024
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Dataset exceeds max size of {settings.eval_upload_max_mb} MB.",
        )
    try:
        run = await evaluation_service.create_run_from_dataset(
            owner_user_id=current_user.id,
            doc_id=doc_id,
            dataset_name=file.filename,
            dataset_bytes=payload,
            run_config=build_evaluation_run_config(document=document),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    background_tasks.add_task(evaluation_job_runner, run.id)
    return EvaluationRunCreateResponse(
        status="accepted",
        run_id=run.id,
        run_status="queued",
        total_cases=run.total_cases,
    )


@router.post(
    "/evaluations/{run_id}/rerun-failed",
    response_model=EvaluationRerunFailedResponse,
    status_code=202,
)
async def rerun_failed_evaluation_cases(
    run_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: ActiveUserDep,
    evaluation_service: EvaluationServiceDep,
    evaluation_rerun_failed_job_runner: EvaluationRerunFailedJobRunnerDep,
):
    payload = await evaluation_service.get_owned_run_failed_count(
        owner_user_id=current_user.id,
        run_id=run_id,
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found.")

    run, failed_count = payload
    if run.status == "running":
        raise HTTPException(status_code=409, detail="Cannot rerun while evaluation is running.")

    if failed_count > 0:
        background_tasks.add_task(evaluation_rerun_failed_job_runner, run.id)

    return EvaluationRerunFailedResponse(
        status="accepted",
        run_id=run.id,
        queued_failed_cases=failed_count,
    )


@router.get("/evaluations", response_model=EvaluationRunListResponse)
async def list_rag_evaluation_runs(
    current_user: ActiveUserDep,
    evaluation_service: EvaluationServiceDep,
    doc_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    runs, total = await evaluation_service.list_runs(
        owner_user_id=current_user.id,
        doc_id=doc_id,
        limit=limit,
        offset=offset,
    )
    return EvaluationRunListResponse(
        status="ok",
        total=total,
        limit=limit,
        offset=offset,
        items=[_to_run_status_response(run) for run in runs],
    )


@router.get("/evaluations/{run_id}", response_model=EvaluationRunStatusResponse)
async def get_rag_evaluation_run(run_id: UUID, current_user: ActiveUserDep, evaluation_service: EvaluationServiceDep):
    run = await evaluation_service.get_run_status(owner_user_id=current_user.id, run_id=run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found.")
    return _to_run_status_response(run)


@router.get("/evaluations/{run_id}/cases", response_model=EvaluationCaseListResponse)
async def get_rag_evaluation_cases(
    run_id: UUID,
    current_user: ActiveUserDep,
    evaluation_service: EvaluationServiceDep,
    limit: int = Query(default=20, ge=1, le=settings.eval_max_cases),
    offset: int = Query(default=0, ge=0),
):
    payload = await evaluation_service.list_run_cases(
        owner_user_id=current_user.id,
        run_id=run_id,
        limit=limit,
        offset=offset,
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found.")
    run, cases, total = payload
    return EvaluationCaseListResponse(
        status="ok",
        run_id=run.id,
        run_status=run.status,  # type: ignore[arg-type]
        total=total,
        limit=limit,
        offset=offset,
        items=[_to_case_item(case) for case in cases],
    )


@router.get("/evaluations/{run_id}/report", response_model=EvaluationReportResponse)
async def get_rag_evaluation_report(
    run_id: UUID,
    current_user: ActiveUserDep,
    evaluation_service: EvaluationServiceDep,
):
    payload = await evaluation_service.get_run_report(owner_user_id=current_user.id, run_id=run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found.")
    run, cases = payload
    return EvaluationReportResponse(
        status="ok",
        run=_to_run_status_response(run),
        cases=[_to_case_item(case) for case in cases],
    )


def _to_run_status_response(run: EvaluationRun) -> EvaluationRunStatusResponse:
    return EvaluationRunStatusResponse(
        status="ok",
        run_id=run.id,
        doc_id=run.doc_id,
        run_status=run.status,  # type: ignore[arg-type]
        dataset_name=run.dataset_name,
        dataset_sha256=run.dataset_sha256,
        total_cases=run.total_cases,
        processed_cases=run.processed_cases,
        hit_at_k=run.hit_at_k,
        recall_at_k=run.recall_at_k,
        mrr=run.mrr,
        accuracy_avg=run.accuracy_avg,
        completeness_avg=run.completeness_avg,
        relevance_avg=run.relevance_avg,
        groundedness_avg=run.groundedness_avg,
        error_message=run.error_message,
        config=EvaluationRunConfig(
            rag_top_k=run.cfg_rag_top_k,
            rag_prefetch_k=run.cfg_rag_prefetch_k,
            embedding_provider=run.cfg_embedding_provider,
            embedding_model=run.cfg_embedding_model,
            reranker_enabled=run.cfg_reranker_enabled,
            reranker_model=run.cfg_reranker_model,
            answer_model=run.cfg_answer_model,
            chunk_strategy=run.cfg_chunk_strategy,
            chunk_size=run.cfg_chunk_size,
            chunk_overlap=run.cfg_chunk_overlap,
        ),
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


def _to_case_item(case: EvaluationCase) -> EvaluationCaseItem:
    return EvaluationCaseItem(
        case_id=case.id,
        case_index=case.case_index,
        question=case.question,
        reference_answer=case.reference_answer,
        expected_chunk_ids=list(case.expected_chunk_ids or []),
        difficulty=case.difficulty,
        category=case.category,
        retrieved_chunk_ids=list(case.retrieved_chunk_ids or []),
        hit=case.hit,
        recall=case.recall,
        first_relevant_rank=case.first_relevant_rank,
        reciprocal_rank=case.reciprocal_rank,
        generated_answer=case.generated_answer,
        citations=list(case.citations or []),
        accuracy=case.accuracy,
        completeness=case.completeness,
        relevance=case.relevance,
        groundedness=case.groundedness,
        judge_feedback=case.judge_feedback,
        case_status=case.status,  # type: ignore[arg-type]
        error_message=case.error_message,
    )


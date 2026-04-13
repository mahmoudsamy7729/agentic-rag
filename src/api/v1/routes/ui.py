from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.api.v1.dependencies import get_chunking_registry
from src.settings.config import settings

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[4] / "templates"))


@router.get("/login-ui", response_class=HTMLResponse, include_in_schema=False)
async def login_ui(request: Request):
    return templates.TemplateResponse(
        "login_ui.html",
        {
            "request": request,
        },
    )


@router.get("/ask-ui", response_class=HTMLResponse, include_in_schema=False)
async def ask_ui(request: Request):
    return templates.TemplateResponse(
        "ask_ui.html",
        {
            "request": request,
        },
    )


@router.get("/documents-ui", response_class=HTMLResponse, include_in_schema=False)
async def documents_ui(request: Request):
    return templates.TemplateResponse(
        "documents_ui.html",
        {
            "request": request,
            "chunking_strategies": get_chunking_registry().names(),
            "default_chunking_strategy": settings.default_chunking_strategy,
        },
    )


@router.get("/documents/{doc_id}/chunks-ui", response_class=HTMLResponse, include_in_schema=False)
async def document_chunks_ui(request: Request, doc_id: str):
    return templates.TemplateResponse(
        "document_chunks_ui.html",
        {
            "request": request,
            "doc_id": doc_id,
        },
    )


@router.get("/evaluations-ui", response_class=HTMLResponse, include_in_schema=False)
async def evaluations_ui(request: Request):
    return templates.TemplateResponse(
        "evaluation_runs_ui.html",
        {
            "request": request,
            "active_eval_nav": "runs",
        },
    )


@router.get("/evaluations-create-ui", response_class=HTMLResponse, include_in_schema=False)
async def evaluation_new_ui(request: Request):
    return templates.TemplateResponse(
        "evaluation_new_ui.html",
        {
            "request": request,
            "active_eval_nav": "new",
            "chunking_strategies": get_chunking_registry().names(),
            "evaluation_defaults": {
                "min_keyword_hits": settings.evaluation_useful_chunk_min_keyword_hits,
                "min_keyword_ratio": settings.evaluation_useful_chunk_min_keyword_ratio,
                "judge_enabled": settings.evaluation_judge_enabled,
                "store_retrieved_chunk_texts": settings.evaluation_store_retrieved_chunk_texts,
            },
        },
    )


@router.get("/evaluations/{run_id}/ui", response_class=HTMLResponse, include_in_schema=False)
async def evaluation_run_detail_ui(request: Request, run_id: str):
    return templates.TemplateResponse(
        "evaluation_run_detail_ui.html",
        {
            "request": request,
            "active_eval_nav": "runs",
            "run_id": run_id,
        },
    )


@router.get("/evaluations-compare-ui", response_class=HTMLResponse, include_in_schema=False)
async def evaluation_compare_ui(request: Request):
    return templates.TemplateResponse(
        "evaluation_compare_ui.html",
        {
            "request": request,
            "active_eval_nav": "compare",
        },
    )


@router.get("/evaluation-datasets-ui", response_class=HTMLResponse, include_in_schema=False)
async def evaluation_datasets_ui(request: Request):
    return templates.TemplateResponse(
        "evaluation_datasets_ui.html",
        {
            "request": request,
            "active_eval_nav": "datasets",
        },
    )

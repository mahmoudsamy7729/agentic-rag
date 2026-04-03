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

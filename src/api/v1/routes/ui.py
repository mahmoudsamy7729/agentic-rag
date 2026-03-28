from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

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

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.api.v1.routes.agent import router as agent_router
from src.api.v1.routes.auth import router as auth_router
from src.api.v1.routes.documents import router as documents_router
from src.api.v1.routes.evaluations import router as evaluations_router
from src.api.v1.routes.health import router as health_router
from src.api.v1.routes.rag import router as rag_router
from src.api.v1.routes.ui import router as ui_router
from src.api.v1.routes.users import router as users_router
from src.infrastructure.database import engine
from src.shared.tracing import configure_trace_logging


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        await engine.dispose()


configure_trace_logging()
app = FastAPI(lifespan=lifespan)
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


app.include_router(health_router)
app.include_router(agent_router)
app.include_router(rag_router)
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(documents_router)
app.include_router(evaluations_router)
app.include_router(ui_router)

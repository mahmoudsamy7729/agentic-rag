from fastapi import FastAPI
from src.api.v1.routes.agent import router as agent_router
from src.api.v1.routes.health import router as health_router


app = FastAPI()


app.include_router(health_router)
app.include_router(agent_router)

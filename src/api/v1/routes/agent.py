from fastapi import APIRouter

from src.api.v1.dependencies import AgentServiceDep
from src.api.v1.schemas import AgentAskRequest, AgentAskResponse

router = APIRouter()


@router.post("/agent/ask", response_model=AgentAskResponse)
async def agent_ask(payload: AgentAskRequest, agent_service: AgentServiceDep):
    result = await agent_service.run(
        question=payload.question,
        session_id=payload.session_id,
        user_id = "2"
    )
    return AgentAskResponse(
        status=result.status,
        answer=result.answer,
        steps=result.steps,
        tools_used=result.tools_used,
    )

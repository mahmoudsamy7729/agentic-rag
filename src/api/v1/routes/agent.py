from fastapi import APIRouter, HTTPException, Query

from src.agents import DocumentNotFoundError
from src.api.v1.dependencies import (
    AgentAskPipelineDep,
)
from src.api.v1.schemas import AgentAskRequest, AgentAskResponse
from src.modules.users.dependencies import ActiveUserDep

router = APIRouter()


@router.post("/agent/ask", response_model=AgentAskResponse)
async def agent_ask(
    payload: AgentAskRequest,
    ask_pipeline: AgentAskPipelineDep,
    current_user: ActiveUserDep,
    use_cache: bool = Query(default=True),
):
    try:
        result = await ask_pipeline.ask(
            owner_user_id=current_user.id,
            question=payload.question,
            doc_id=payload.doc_id,
            session_id=payload.session_id,
            use_cache=use_cache,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return AgentAskResponse(
        status=result.status,
        cache_status=result.cache_status,  # type: ignore[arg-type]
        refined_query=result.refined_query,
        answer=result.answer,
        steps=result.steps,
        tools_used=result.tools_used,
        citations=result.citations,
    )

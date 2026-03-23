from fastapi import APIRouter

from src.api.v1.dependencies import AgentServiceDep
from src.api.v1.schemas import AgentAskRequest, AgentAskResponse

router = APIRouter()


@router.post("/agent/ask", response_model=AgentAskResponse)
async def agent_ask(payload: AgentAskRequest, agent_service: AgentServiceDep):
    result = await agent_service.run(
        question=payload.question,
        doc_id=payload.doc_id,
        session_id=payload.session_id,
        user_id=payload.user_id,
    )
    return AgentAskResponse(
        status=result.status,
        answer=result.answer,
        steps=result.steps,
        tools_used=result.tools_used,
        citations=[
            {
                "source": citation.source,
                "doc_id": citation.doc_id,
                "chunk_id": citation.chunk_id,
                "snippet": citation.snippet,
                "page_number": citation.page_number,
            }
            for citation in result.citations
        ],
    )

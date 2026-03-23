from fastapi import APIRouter, HTTPException

from src.api.v1.dependencies import AgentServiceDep
from src.api.v1.schemas import AgentAskRequest, AgentAskResponse
from src.modules.documents import DocumentsRepositoryDep
from src.modules.users.dependencies import ActiveUserDep

router = APIRouter()


@router.post("/agent/ask", response_model=AgentAskResponse)
async def agent_ask(
    payload: AgentAskRequest,
    agent_service: AgentServiceDep,
    repository: DocumentsRepositoryDep,
    current_user: ActiveUserDep,
):
    document = await repository.get_owned_document(
        owner_user_id=current_user.id,
        doc_id=payload.doc_id,
        include_deleted=False,
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    result = await agent_service.run(
        question=payload.question,
        doc_id=payload.doc_id,
        session_id=payload.session_id,
        user_id=str(current_user.id),
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

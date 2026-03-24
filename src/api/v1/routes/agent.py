from fastapi import APIRouter, HTTPException

from src.agents.cache_policy import is_cacheable_rag_answer
from src.api.v1.dependencies import (
    AgentServiceDep,
    EmbeddingProviderDep,
    LLMDep,
    SemanticCacheServiceDep,
)
from src.api.v1.schemas import AgentAskRequest, AgentAskResponse
from src.modules.documents import DocumentsRepositoryDep
from src.modules.users.dependencies import ActiveUserDep

router = APIRouter()


@router.post("/agent/ask", response_model=AgentAskResponse)
async def agent_ask(
    payload: AgentAskRequest,
    agent_service: AgentServiceDep,
    llm: LLMDep,
    embedding_provider: EmbeddingProviderDep,
    semantic_cache_service: SemanticCacheServiceDep,
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

    normalized_question = semantic_cache_service.normalize_question(payload.question)
    query_embedding: list[float] | None = None
    if semantic_cache_service.enabled and document.last_indexed_at is not None:
        try:
            query_embedding = await embedding_provider.embed_query(normalized_question)
            cache_hit = await semantic_cache_service.lookup(
                owner_user_id=current_user.id,
                doc_id=payload.doc_id,
                doc_version=document.last_indexed_at,
                model_name=llm.model_name,
                query_embedding=query_embedding,
            )
            if cache_hit is not None:
                return AgentAskResponse(
                    status="ok",
                    cache_status="hit",
                    answer=cache_hit.answer,
                    steps=0,
                    tools_used=[],
                    citations=cache_hit.citations,
                )
        except Exception:
            # Cache failures should not block generation.
            pass

    result = await agent_service.run(
        question=payload.question,
        doc_id=payload.doc_id,
        session_id=payload.session_id,
        user_id=str(current_user.id),
    )

    citations = [
        {
            "source": citation.source,
            "doc_id": citation.doc_id,
            "chunk_id": citation.chunk_id,
            "snippet": citation.snippet,
            "page_number": citation.page_number,
        }
        for citation in result.citations
    ]
    is_rag_backed = is_cacheable_rag_answer(
        tools_used=result.tools_used,
        citations=citations,
    )
    if (
        semantic_cache_service.enabled
        and document.last_indexed_at is not None
        and is_rag_backed
    ):
        try:
            if query_embedding is None:
                query_embedding = await embedding_provider.embed_query(normalized_question)
            await semantic_cache_service.store(
                owner_user_id=current_user.id,
                doc_id=payload.doc_id,
                doc_version=document.last_indexed_at,
                model_name=llm.model_name,
                question_normalized=normalized_question,
                question_embedding=query_embedding,
                answer=result.answer,
                citations=citations,
            )
        except Exception:
            # Cache write failures should not block generation.
            pass

    return AgentAskResponse(
        status=result.status,
        cache_status="miss",
        answer=result.answer,
        steps=result.steps,
        tools_used=result.tools_used,
        citations=citations,
    )

from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request

from src.agents import DocumentNotFoundError
from src.api.v1.dependencies import (
    AgentAskPipelineDep,
)
from src.api.v1.schemas import AgentAskRequest, AgentAskResponse
from src.modules.users.dependencies import ActiveUserDep
from src.shared.tracing import TraceContext, trace_event

router = APIRouter(tags=["agent"])


@router.post("/agent/ask", response_model=AgentAskResponse)
async def agent_ask(
    request: Request,
    payload: AgentAskRequest,
    ask_pipeline: AgentAskPipelineDep,
    current_user: ActiveUserDep,
    use_cache: bool = Query(default=True),
):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    trace_context = TraceContext(
        request_id=request_id,
        doc_id=payload.doc_id,
        owner_user_id=str(current_user.id),
        session_id=payload.session_id,
    )
    started_at = perf_counter()
    trace_event(
        "ask.request.started",
        trace_context=trace_context,
        question=payload.question,
        use_cache=use_cache,
    )
    try:
        result = await ask_pipeline.ask(
            owner_user_id=current_user.id,
            question=payload.question,
            doc_id=payload.doc_id,
            session_id=payload.session_id,
            use_cache=use_cache,
            request_id=request_id,
        )
    except DocumentNotFoundError as exc:
        trace_event(
            "ask.request.failed",
            trace_context=trace_context,
            question=payload.question,
            error=str(exc),
            status_code=404,
            elapsed_ms=round((perf_counter() - started_at) * 1000, 3),
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        trace_event(
            "ask.request.failed",
            trace_context=trace_context,
            question=payload.question,
            error=str(exc),
            status_code=500,
            elapsed_ms=round((perf_counter() - started_at) * 1000, 3),
        )
        raise

    trace_event(
        "ask.request.succeeded",
        trace_context=trace_context,
        question=payload.question,
        refined_query=result.refined_query,
        cache_status=result.cache_status,
        tools_used=result.tools_used,
        citation_count=len(result.citations),
        elapsed_ms=round((perf_counter() - started_at) * 1000, 3),
    )

    return AgentAskResponse(
        status=result.status,
        cache_status=result.cache_status,  # type: ignore[arg-type]
        refined_query=result.refined_query,
        answer=result.answer,
        steps=result.steps,
        tools_used=result.tools_used,
        citations=result.citations,
    )

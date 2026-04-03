from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from src.agents.cache_policy import is_cacheable_rag_answer, is_no_answer_fallback
from src.agents.service import AgentService
from src.modules.documents.repository import DocumentsRepository
from src.modules.semantic_cache.service import SemanticCacheService
from src.rag.embeddings import EmbeddingProvider
from src.shared.tracing import TraceContext, trace_event
from src.shared.interfaces.llm import LLM

from .query_refinement import QueryRefinementService


class DocumentNotFoundError(Exception):
    pass


@dataclass(slots=True)
class AgentAskPipelineResult:
    status: str
    cache_status: str
    refined_query: str
    answer: str
    steps: int
    tools_used: list[str]
    citations: list[dict]


class AgentAskPipeline:
    def __init__(
        self,
        *,
        agent_service: AgentService,
        llm: LLM,
        query_refinement_service: QueryRefinementService,
        embedding_provider: EmbeddingProvider,
        semantic_cache_service: SemanticCacheService,
        documents_repository: DocumentsRepository,
    ) -> None:
        self._agent_service = agent_service
        self._llm = llm
        self._query_refinement_service = query_refinement_service
        self._embedding_provider = embedding_provider
        self._semantic_cache_service = semantic_cache_service
        self._documents_repository = documents_repository

    async def ask(
        self,
        *,
        owner_user_id: UUID,
        question: str,
        doc_id: str,
        session_id: str | None = None,
        use_cache: bool = True,
        request_id: str | None = None,
    ) -> AgentAskPipelineResult:
        trace_context = TraceContext(
            request_id=request_id or "unknown",
            doc_id=doc_id,
            owner_user_id=str(owner_user_id),
            session_id=session_id,
        )
        document = await self._documents_repository.get_owned_document(
            owner_user_id=owner_user_id,
            doc_id=doc_id,
            include_deleted=False,
        )
        if document is None:
            raise DocumentNotFoundError("Document not found.")
        trace_event(
            "ask.document.checked",
            trace_context=trace_context,
            document_found=True,
            use_cache=use_cache,
        )

        refinement = await self._query_refinement_service.refine(
            question=question,
            doc_id=doc_id,
        )
        refined_query = refinement.refined_query
        trace_event(
            "ask.query.refined",
            trace_context=trace_context,
            question=question,
            refined_query=refined_query,
        )
        normalized_question = self._semantic_cache_service.normalize_question(refined_query)
        query_embedding: list[float] | None = None

        should_use_cache = (
            use_cache
            and self._semantic_cache_service.enabled
            and document.last_indexed_at is not None
        )
        if should_use_cache:
            trace_event(
                "ask.cache.lookup.started",
                trace_context=trace_context,
                question=question,
                refined_query=refined_query,
                cache_enabled=self._semantic_cache_service.enabled,
                use_cache=use_cache,
            )
            try:
                query_embedding = await self._embedding_provider.embed_query(normalized_question)
                cache_hit = await self._semantic_cache_service.lookup(
                    owner_user_id=owner_user_id,
                    doc_id=doc_id,
                    doc_version=document.last_indexed_at,
                    model_name=self._llm.model_name,
                    query_embedding=query_embedding,
                )
                if cache_hit is not None:
                    trace_event(
                        "ask.cache.hit",
                        trace_context=trace_context,
                        cache_status="hit",
                        refined_query=refined_query,
                        citation_count=len(cache_hit.citations),
                    )
                    return AgentAskPipelineResult(
                        status="ok",
                        cache_status="hit",
                        refined_query=refined_query,
                        answer=cache_hit.answer,
                        steps=0,
                        tools_used=[],
                        citations=cache_hit.citations,
                    )
                trace_event(
                    "ask.cache.miss",
                    trace_context=trace_context,
                    cache_status="miss",
                    refined_query=refined_query,
                )
            except Exception:
                trace_event(
                    "ask.cache.lookup.failed",
                    trace_context=trace_context,
                    cache_status="miss",
                    refined_query=refined_query,
                )
                # Cache failures should never block generation.
                pass

        trace_event(
            "ask.agent.run.started",
            trace_context=trace_context,
            refined_query=refined_query,
        )
        result = await self._agent_service.run(
            question=refined_query,
            doc_id=doc_id,
            session_id=session_id,
            user_id=str(owner_user_id),
            request_id=request_id,
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
        trace_event(
            "ask.agent.run.completed",
            trace_context=trace_context,
            refined_query=refined_query,
            answer_status=result.status,
            tool_count=len(result.tools_used),
            citation_count=len(citations),
            no_answer_fallback=is_no_answer_fallback(result.answer),
        )

        if should_use_cache:
            is_rag_backed = is_cacheable_rag_answer(
                tools_used=result.tools_used,
                citations=citations,
            )
            should_skip_cache = is_no_answer_fallback(result.answer)
            if is_rag_backed and not should_skip_cache:
                try:
                    if query_embedding is None:
                        query_embedding = await self._embedding_provider.embed_query(
                            normalized_question
                        )
                    await self._semantic_cache_service.store(
                        owner_user_id=owner_user_id,
                        doc_id=doc_id,
                        doc_version=document.last_indexed_at,
                        model_name=self._llm.model_name,
                        question_normalized=normalized_question,
                        question_embedding=query_embedding,
                        answer=result.answer,
                        citations=citations,
                    )
                    trace_event(
                        "ask.cache.store.succeeded",
                        trace_context=trace_context,
                        cache_status="miss",
                        refined_query=refined_query,
                        citation_count=len(citations),
                    )
                except Exception:
                    trace_event(
                        "ask.cache.store.failed",
                        trace_context=trace_context,
                        cache_status="miss",
                        refined_query=refined_query,
                    )
                    # Cache failures should never block generation.
                    pass
            else:
                trace_event(
                    "ask.cache.store.skipped",
                    trace_context=trace_context,
                    cache_status="miss",
                    refined_query=refined_query,
                    is_rag_backed=is_rag_backed,
                    no_answer_fallback=should_skip_cache,
                )

        return AgentAskPipelineResult(
            status=result.status,
            cache_status="miss",
            refined_query=refined_query,
            answer=result.answer,
            steps=result.steps,
            tools_used=result.tools_used,
            citations=citations,
        )


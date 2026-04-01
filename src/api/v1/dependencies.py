from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Awaitable, Callable
from uuid import UUID

from fastapi import Depends
from src.infrastructure.database import AsyncSessionFactory
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents import AgentAskPipeline, AgentService, QueryRefinementService
from src.infrastructure.database import get_db
from src.infrastructure.llm.huggingface_embeddings import HuggingFaceEmbeddingProvider
from src.infrastructure.llm.openai_embeddings import OpenAIEmbeddingProvider
from src.infrastructure.llm.openai_llm import OpenAILLM
from src.rag.embeddings import EmbeddingProvider
from src.rag.ingestion import (
    ChunkingStrategyRegistry,
    FixedWindowChunkingStrategy,
    PDFExtractor,
    PDFPlumberExtractor,
)
from src.rag.pipeline import RAGIngestionService, RAGRetrievalService
from src.rag.reranker import Reranker
from src.rag.vectorstore import VectorStore
from src.settings.config import settings
from src.shared.interfaces.llm import LLM
from src.tools import PingTool, RetrieverTool, ToolRegistry
from src.modules.documents.dependencies import DocumentsRepositoryDep

if TYPE_CHECKING:
    from src.modules.documents.models import Document
    from src.modules.evaluation.config import EvaluationRunConfig
    from src.modules.evaluation.repository import EvaluationRepository
    from src.modules.evaluation.service import EvaluationService
    from src.modules.semantic_cache.repository import SemanticCacheRepository
    from src.modules.semantic_cache.service import SemanticCacheService

DbSessionDep = Annotated[AsyncSession, Depends(get_db)]


@lru_cache
def get_llm() -> LLM:
    if not settings.openai_key:
        raise RuntimeError("Missing OPENAI_KEY in environment.")
    if not settings.model:
        raise RuntimeError("Missing MODEL in environment.")

    return OpenAILLM(
        api_key=settings.openai_key,
        model=settings.model,
        base_url=settings.ollama_base_url,
    )


LLMDep = Annotated[LLM, Depends(get_llm)]


@lru_cache
def get_judge_llm() -> LLM:
    if not settings.openai_key:
        raise RuntimeError("Missing OPENAI_KEY in environment.")
    if not settings.eval_judge_model:
        raise RuntimeError("Missing EVAL_JUDGE_MODEL in environment.")
    return OpenAILLM(
        api_key=settings.openai_key,
        model=settings.eval_judge_model,
        base_url=settings.ollama_base_url,
    )


JudgeLLMDep = Annotated[LLM, Depends(get_judge_llm)]


def get_query_refinement_service(llm: LLMDep) -> QueryRefinementService:
    return QueryRefinementService(
        llm=llm,
        enabled=settings.query_refinement_enabled,
        temperature=settings.query_refinement_temperature,
        max_tokens=settings.query_refinement_max_tokens,
    )


QueryRefinementServiceDep = Annotated[
    QueryRefinementService, Depends(get_query_refinement_service)
]


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    if not settings.embedding_model:
        raise RuntimeError("Missing EMBEDDING_MODEL in environment.")
    provider_name = settings.embedding_provider

    if provider_name == "openai":
        if not settings.openai_key:
            raise RuntimeError("Missing OPENAI_KEY in environment.")
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_key,
            model=settings.embedding_model,
            base_url=settings.embedding_base_url,
        )

    if provider_name == "huggingface":
        return HuggingFaceEmbeddingProvider(model_name=settings.embedding_model)

    raise RuntimeError(
        f"Unsupported EMBEDDING_PROVIDER '{provider_name}'. Use 'openai' or 'huggingface'."
    )


EmbeddingProviderDep = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]


def get_semantic_cache_repository(session: DbSessionDep) -> "SemanticCacheRepository":
    from src.modules.semantic_cache.repository import SemanticCacheRepository

    return SemanticCacheRepository(session)


SemanticCacheRepositoryDep = Annotated[
    "SemanticCacheRepository", Depends(get_semantic_cache_repository)
]


def get_semantic_cache_service(
    repository: SemanticCacheRepositoryDep,
) -> "SemanticCacheService":
    from src.modules.semantic_cache.service import SemanticCacheService

    return SemanticCacheService(
        repository=repository,
        enabled=settings.semantic_cache_enabled,
        similarity_threshold=settings.semantic_cache_similarity_threshold,
    )


SemanticCacheServiceDep = Annotated["SemanticCacheService", Depends(get_semantic_cache_service)]


@lru_cache
def get_vector_store() -> VectorStore:
    from src.infrastructure.vector_db.chroma_vectorstore import ChromaVectorStore

    return ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.resolved_rag_collection_name(),
    )


VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]


@lru_cache
def get_reranker() -> Reranker | None:
    if not settings.reranker_enabled:
        return None
    if not settings.reranker_api_key:
        raise RuntimeError("Missing RERANKER_API_KEY in environment.")
    if not settings.reranker_model:
        raise RuntimeError("Missing RERANKER_MODEL in environment.")

    from src.infrastructure.reranker import CohereReranker

    return CohereReranker(
        api_key=settings.reranker_api_key,
        model=settings.reranker_model,
    )


RerankerDep = Annotated[Reranker | None, Depends(get_reranker)]


@lru_cache
def get_pdf_extractor() -> PDFExtractor:
    return PDFPlumberExtractor(
        dedupe_threshold=settings.rag_pdf_dedupe_threshold,
    )


PDFExtractorDep = Annotated[PDFExtractor, Depends(get_pdf_extractor)]


@lru_cache
def get_chunking_registry() -> ChunkingStrategyRegistry:
    return ChunkingStrategyRegistry([FixedWindowChunkingStrategy()])


ChunkingRegistryDep = Annotated[ChunkingStrategyRegistry, Depends(get_chunking_registry)]


@lru_cache
def get_rag_ingestion_service() -> RAGIngestionService:
    registry = get_chunking_registry()
    registry.resolve(settings.default_chunking_strategy)
    return RAGIngestionService(
        embedding_provider=get_embedding_provider(),
        vector_store=get_vector_store(),
        chunking_registry=registry,
        default_chunking_strategy=settings.default_chunking_strategy,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        pdf_extractor=get_pdf_extractor(),
        pdf_max_pages=settings.rag_pdf_max_pages,
    )


RAGIngestionServiceDep = Annotated[RAGIngestionService, Depends(get_rag_ingestion_service)]


@lru_cache
def get_rag_retrieval_service() -> RAGRetrievalService:
    return RAGRetrievalService(
        embedding_provider=get_embedding_provider(),
        vector_store=get_vector_store(),
        default_top_k=settings.rag_top_k,
        prefetch_k=settings.rag_prefetch_k,
        reranker=get_reranker(),
    )


RAGRetrievalServiceDep = Annotated[RAGRetrievalService, Depends(get_rag_retrieval_service)]


@lru_cache
def get_retriever_tool() -> RetrieverTool:
    return RetrieverTool(
        retrieval_service=get_rag_retrieval_service(),
        default_top_k=settings.rag_top_k,
    )


@lru_cache
def get_tool_registry() -> ToolRegistry:
    return ToolRegistry([PingTool(), get_retriever_tool()])


ToolRegistryDep = Annotated[ToolRegistry, Depends(get_tool_registry)]


def get_agent_service(llm: LLMDep, registry: ToolRegistryDep) -> AgentService:
    return AgentService(
        llm=llm,
        registry=registry,
        max_steps=settings.agent_max_steps,
        temperature=settings.agent_temperature,
        max_tokens=settings.agent_max_tokens,
        timeout_s=settings.agent_timeout_s,
        system_prompt=settings.agent_system_prompt,
    )


AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]


def get_agent_ask_pipeline(
    agent_service: AgentServiceDep,
    llm: LLMDep,
    query_refinement_service: QueryRefinementServiceDep,
    embedding_provider: EmbeddingProviderDep,
    semantic_cache_service: SemanticCacheServiceDep,
    documents_repository: DocumentsRepositoryDep,
) -> AgentAskPipeline:
    return AgentAskPipeline(
        agent_service=agent_service,
        llm=llm,
        query_refinement_service=query_refinement_service,
        embedding_provider=embedding_provider,
        semantic_cache_service=semantic_cache_service,
        documents_repository=documents_repository,
    )


AgentAskPipelineDep = Annotated[AgentAskPipeline, Depends(get_agent_ask_pipeline)]


@lru_cache
def get_evaluation_agent_service() -> AgentService:
    return AgentService(
        llm=get_llm(),
        registry=get_tool_registry(),
        max_steps=settings.agent_max_steps,
        temperature=0.0,
        max_tokens=settings.agent_max_tokens,
        timeout_s=settings.agent_timeout_s,
        system_prompt=settings.agent_system_prompt,
    )


def get_evaluation_repository(session: DbSessionDep) -> "EvaluationRepository":
    from src.modules.evaluation.repository import EvaluationRepository

    return EvaluationRepository(session)


EvaluationRepositoryDep = Annotated[
    "EvaluationRepository", Depends(get_evaluation_repository)
]


def get_evaluation_judge_service(judge_llm: JudgeLLMDep):
    from src.modules.evaluation.judge import EvaluationJudgeService

    return EvaluationJudgeService(
        llm=judge_llm,
        max_tokens=settings.eval_judge_max_tokens,
        timeout_s=settings.eval_judge_timeout_s,
    )


EvaluationJudgeServiceDep = Annotated[
    "EvaluationJudgeService", Depends(get_evaluation_judge_service)
]


def build_evaluation_run_config(
    *,
    document: "Document | None" = None,
) -> "EvaluationRunConfig":
    from src.modules.evaluation.config import EvaluationRunConfig

    chunk_strategy = (
        document.chunking_strategy
        if document is not None and document.chunking_strategy
        else settings.default_chunking_strategy
    )
    chunk_size = (
        document.chunk_size
        if document is not None and document.chunk_size is not None
        else settings.rag_chunk_size
    )
    chunk_overlap = (
        document.chunk_overlap
        if document is not None and document.chunk_overlap is not None
        else settings.rag_chunk_overlap
    )

    return EvaluationRunConfig(
        rag_top_k=settings.rag_top_k,
        rag_prefetch_k=settings.rag_prefetch_k,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        reranker_enabled=settings.reranker_enabled,
        reranker_model=settings.reranker_model if settings.reranker_enabled else None,
        answer_model=get_llm().model_name,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def get_evaluation_run_config() -> "EvaluationRunConfig":
    return build_evaluation_run_config()


EvaluationRunConfigDep = Annotated["EvaluationRunConfig", Depends(get_evaluation_run_config)]


def get_evaluation_service(
    repository: EvaluationRepositoryDep,
    judge_service: EvaluationJudgeServiceDep,
    run_config: EvaluationRunConfigDep,
    ask_pipeline: AgentAskPipelineDep,
) -> "EvaluationService":
    from src.modules.evaluation.service import EvaluationService

    return EvaluationService(
        repository=repository,
        retrieval_service=get_rag_retrieval_service(),
        agent_service=get_evaluation_agent_service(),
        ask_pipeline=ask_pipeline,
        judge_service=judge_service,
        max_cases=settings.eval_max_cases,
        run_config=run_config,
    )


EvaluationServiceDep = Annotated["EvaluationService", Depends(get_evaluation_service)]


def _build_agent_ask_pipeline_for_session(
    *,
    session: AsyncSession,
    agent_service: AgentService,
) -> AgentAskPipeline:
    from src.modules.documents.repository import DocumentsRepository

    return AgentAskPipeline(
        agent_service=agent_service,
        llm=get_llm(),
        query_refinement_service=get_query_refinement_service(get_llm()),
        embedding_provider=get_embedding_provider(),
        semantic_cache_service=get_semantic_cache_service(get_semantic_cache_repository(session)),
        documents_repository=DocumentsRepository(session),
    )


async def run_evaluation_job(run_id: UUID) -> None:
    from src.modules.evaluation.judge import EvaluationJudgeService
    from src.modules.evaluation.repository import EvaluationRepository
    from src.modules.evaluation.service import EvaluationService

    async with AsyncSessionFactory() as session:
        repository = EvaluationRepository(session)
        service = EvaluationService(
            repository=repository,
            retrieval_service=get_rag_retrieval_service(),
            agent_service=get_evaluation_agent_service(),
            ask_pipeline=_build_agent_ask_pipeline_for_session(
                session=session,
                agent_service=get_evaluation_agent_service(),
            ),
            judge_service=EvaluationJudgeService(
                llm=get_judge_llm(),
                max_tokens=settings.eval_judge_max_tokens,
                timeout_s=settings.eval_judge_timeout_s,
            ),
            max_cases=settings.eval_max_cases,
            run_config=get_evaluation_run_config(),
        )
        await service.execute_run(run_id=run_id)


def get_evaluation_job_runner() -> Callable[[UUID], Awaitable[None]]:
    return run_evaluation_job


EvaluationJobRunnerDep = Annotated[
    Callable[[UUID], Awaitable[None]], Depends(get_evaluation_job_runner)
]


async def run_evaluation_rerun_failed_job(run_id: UUID) -> None:
    from src.modules.evaluation.judge import EvaluationJudgeService
    from src.modules.evaluation.repository import EvaluationRepository
    from src.modules.evaluation.service import EvaluationService

    async with AsyncSessionFactory() as session:
        repository = EvaluationRepository(session)
        service = EvaluationService(
            repository=repository,
            retrieval_service=get_rag_retrieval_service(),
            agent_service=get_evaluation_agent_service(),
            ask_pipeline=_build_agent_ask_pipeline_for_session(
                session=session,
                agent_service=get_evaluation_agent_service(),
            ),
            judge_service=EvaluationJudgeService(
                llm=get_judge_llm(),
                max_tokens=settings.eval_judge_max_tokens,
                timeout_s=settings.eval_judge_timeout_s,
            ),
            max_cases=settings.eval_max_cases,
            run_config=get_evaluation_run_config(),
        )
        await service.execute_rerun_failed(run_id=run_id)


def get_evaluation_rerun_failed_job_runner() -> Callable[[UUID], Awaitable[None]]:
    return run_evaluation_rerun_failed_job


EvaluationRerunFailedJobRunnerDep = Annotated[
    Callable[[UUID], Awaitable[None]], Depends(get_evaluation_rerun_failed_job_runner)
]

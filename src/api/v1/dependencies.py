from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.agents import AgentService
from src.infrastructure.llm.huggingface_embeddings import HuggingFaceEmbeddingProvider
from src.infrastructure.llm.openai_embeddings import OpenAIEmbeddingProvider
from src.infrastructure.llm.openai_llm import OpenAILLM
from src.rag.embeddings import EmbeddingProvider
from src.rag.pipeline import RAGIngestionService, RAGRetrievalService
from src.rag.vectorstore import VectorStore
from src.settings.config import settings
from src.shared.interfaces.llm import LLM
from src.tools import PingTool, RetrieverTool, ToolRegistry


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


@lru_cache
def get_vector_store() -> VectorStore:
    from src.infrastructure.vector_db.chroma_vectorstore import ChromaVectorStore

    return ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.resolved_rag_collection_name(),
    )


VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]


@lru_cache
def get_rag_ingestion_service() -> RAGIngestionService:
    return RAGIngestionService(
        embedding_provider=get_embedding_provider(),
        vector_store=get_vector_store(),
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )


RAGIngestionServiceDep = Annotated[RAGIngestionService, Depends(get_rag_ingestion_service)]


@lru_cache
def get_rag_retrieval_service() -> RAGRetrievalService:
    return RAGRetrievalService(
        embedding_provider=get_embedding_provider(),
        vector_store=get_vector_store(),
        default_top_k=settings.rag_top_k,
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

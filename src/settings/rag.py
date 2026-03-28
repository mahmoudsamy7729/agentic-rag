import re
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class RAGSettings(BaseSettings):
    rag_collection_name: str | None = Field(default=None)
    rag_top_k: int = Field(default=12, ge=1, le=100)
    rag_prefetch_k: int = Field(default=80, ge=1, le=200)
    rag_chunk_size: int = Field(default=800, ge=100, le=4000)
    rag_chunk_overlap: int = Field(default=120, ge=0, le=1000)
    rag_pdf_max_pages: int = Field(default=300, ge=1, le=5000)
    rag_pdf_max_mb: int = Field(default=25, ge=1, le=500)
    rag_pdf_dedupe_threshold: int = Field(default=96, ge=0, le=100)
    reranker_enabled: bool = Field(default=False)
    reranker_model: str = Field(default="rerank-v4.0-fast")
    reranker_api_key: str | None = Field(default=None)
    embedding_provider: Literal["openai", "huggingface"] = Field(default="huggingface")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_base_url: str | None = Field(default=None)
    chroma_persist_dir: str = Field(
        default_factory=lambda: str(Path(__file__).resolve().parents[2] / "data" / "chroma")
    )
    semantic_cache_enabled: bool = Field(default=True)
    semantic_cache_similarity_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    query_refinement_enabled: bool = Field(default=True)
    query_refinement_max_tokens: int = Field(default=64, ge=1, le=512)
    query_refinement_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    def resolved_rag_collection_name(self) -> str:
        if self.rag_collection_name is not None:
            return self.rag_collection_name
        if not self.embedding_model:
            raise RuntimeError("Missing EMBEDDING_MODEL in environment.")

        # Separate persisted collections by embedding config to avoid cross-model
        # dimension collisions when deployments switch providers or models.
        model_slug = re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", self.embedding_model.lower()))
        model_slug = model_slug.strip("_")
        if not model_slug:
            raise RuntimeError("Could not derive RAG collection name from EMBEDDING_MODEL.")

        return f"agentic_rag_docs__{self.embedding_provider}__{model_slug}"

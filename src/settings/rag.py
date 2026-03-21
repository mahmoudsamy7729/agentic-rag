from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class RAGSettings(BaseSettings):
    rag_collection_name: str = Field(default="agentic_rag_docs")
    rag_top_k: int = Field(default=4, ge=1, le=20)
    rag_chunk_size: int = Field(default=800, ge=100, le=4000)
    rag_chunk_overlap: int = Field(default=120, ge=0, le=1000)
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_base_url: str | None = Field(default="None")
    chroma_persist_dir: str = Field(
        default_factory=lambda: str(Path(__file__).resolve().parents[2] / "data" / "chroma")
    )

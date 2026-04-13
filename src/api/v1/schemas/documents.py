from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentItem(BaseModel):
    doc_id: str = Field(description="Document identifier.")
    owner_user_id: UUID = Field(description="Owner user id.")
    source: str | None = Field(default=None, description="Document source label.")
    chunking_strategy: str | None = Field(default=None, description="Chunking strategy used for indexing.")
    chunk_size: int | None = Field(default=None, description="Chunk size used for indexing.")
    chunk_overlap: int | None = Field(default=None, description="Chunk overlap used for indexing.")
    created_at: datetime = Field(description="Document creation timestamp.")
    updated_at: datetime = Field(description="Document update timestamp.")
    deleted_at: datetime | None = Field(
        default=None,
        description="Soft delete timestamp; null for active documents.",
    )


class DocumentChunkItem(BaseModel):
    chunk_id: str = Field(description="Indexed chunk identifier.")
    source: str = Field(description="Chunk source label.")
    text: str = Field(description="Stored chunk text.")
    page_number: int | None = Field(default=None, description="PDF page number when available.")


class DocumentChunkSummary(BaseModel):
    doc_id: str = Field(description="Document identifier.")
    source: str | None = Field(default=None, description="Document source label.")
    chunking_strategy: str | None = Field(default=None, description="Chunking strategy used for indexing.")
    chunk_size: int | None = Field(default=None, description="Chunk size used for indexing.")
    chunk_overlap: int | None = Field(default=None, description="Chunk overlap used for indexing.")


class DocumentChunkListResponse(BaseModel):
    status: str = Field(description="Operation status.")
    document: DocumentChunkSummary = Field(description="Document metadata and chunking config.")
    total: int = Field(description="Total number of matched chunks.")
    limit: int = Field(description="Applied page size.")
    offset: int = Field(description="Applied offset.")
    page_number: int | None = Field(default=None, description="Applied page filter.")
    q: str | None = Field(default=None, description="Applied text search filter.")
    items: list[DocumentChunkItem] = Field(default_factory=list, description="Paginated document chunks.")


class DocumentListResponse(BaseModel):
    status: str = Field(description="Operation status.")
    items: list[DocumentItem] = Field(default_factory=list, description="Owned documents list.")
    limit: int = Field(description="Applied page size.")
    offset: int = Field(description="Applied offset.")


class DocumentDeleteResponse(BaseModel):
    status: str = Field(description="Operation status.")
    doc_id: str = Field(description="Deleted document id.")
    deleted: bool = Field(description="Whether this call performed a new soft delete.")


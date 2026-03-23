from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentItem(BaseModel):
    doc_id: str = Field(description="Document identifier.")
    owner_user_id: UUID = Field(description="Owner user id.")
    source: str | None = Field(default=None, description="Document source label.")
    created_at: datetime = Field(description="Document creation timestamp.")
    updated_at: datetime = Field(description="Document update timestamp.")
    deleted_at: datetime | None = Field(
        default=None,
        description="Soft delete timestamp; null for active documents.",
    )


class DocumentListResponse(BaseModel):
    status: str = Field(description="Operation status.")
    items: list[DocumentItem] = Field(default_factory=list, description="Owned documents list.")
    limit: int = Field(description="Applied page size.")
    offset: int = Field(description="Applied offset.")


class DocumentDeleteResponse(BaseModel):
    status: str = Field(description="Operation status.")
    doc_id: str = Field(description="Deleted document id.")
    deleted: bool = Field(description="Whether this call performed a new soft delete.")


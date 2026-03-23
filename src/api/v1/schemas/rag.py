from pydantic import BaseModel, Field


class RAGIngestTextRequest(BaseModel):
    text: str = Field(min_length=1, description="Raw text to index.")
    source: str | None = Field(
        default=None,
        description="Optional source label (file name, URL, or domain key).",
    )
    doc_id: str | None = Field(
        default=None,
        description="Optional document id; generated automatically if omitted.",
    )


class RAGIngestTextResponse(BaseModel):
    status: str = Field(description="Ingestion status.")
    doc_id: str = Field(description="Document id used for indexing.")
    chunks_ingested: int = Field(description="Number of indexed chunks.")


class RAGIngestPDFResponse(BaseModel):
    status: str = Field(description="Ingestion status.")
    doc_id: str = Field(description="Document id used for indexing.")
    chunks_ingested: int = Field(description="Number of indexed chunks.")
    pages_total: int = Field(description="Total number of PDF pages.")
    pages_ingested: int = Field(description="Number of pages successfully ingested.")
    skipped_pages: list[int] = Field(
        default_factory=list,
        description="1-based page numbers skipped during extraction.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Extraction warnings encountered during partial ingestion.",
    )

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.v1.dependencies import RAGIngestionServiceDep
from src.api.v1.schemas import (
    RAGIngestPDFResponse,
    RAGIngestTextRequest,
    RAGIngestTextResponse,
)
from src.settings.config import settings

router = APIRouter()


@router.post("/rag/ingest/text", response_model=RAGIngestTextResponse)
async def ingest_text(payload: RAGIngestTextRequest, ingestion_service: RAGIngestionServiceDep):
    result = await ingestion_service.ingest_text(
        text=payload.text,
        source=payload.source,
        doc_id=payload.doc_id,
    )
    return RAGIngestTextResponse(
        status="ok",
        doc_id=result.doc_id,
        chunks_ingested=result.chunks_ingested,
    )


@router.post("/rag/ingest/pdf", response_model=RAGIngestPDFResponse)
async def ingest_pdf(
    file: Annotated[UploadFile, File(...)],
    ingestion_service: RAGIngestionServiceDep,
    source: Annotated[str | None, Form()] = None,
    doc_id: Annotated[str | None, Form()] = None,
):
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=422, detail="Only PDF files are supported.")

    payload = await file.read()
    max_bytes = settings.rag_pdf_max_mb * 1024 * 1024
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds max size of {settings.rag_pdf_max_mb} MB.",
        )

    resolved_source = source or file.filename or "uploaded-pdf"
    try:
        result = await ingestion_service.ingest_pdf(
            pdf_bytes=payload,
            source=resolved_source,
            doc_id=doc_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return RAGIngestPDFResponse(
        status="ok",
        doc_id=result.doc_id,
        chunks_ingested=result.chunks_ingested,
        pages_total=result.pages_total,
        pages_ingested=result.pages_ingested,
        skipped_pages=result.skipped_pages,
        warnings=result.warnings,
    )


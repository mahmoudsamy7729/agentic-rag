from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlalchemy.exc import IntegrityError

from src.api.v1.dependencies import RAGIngestionServiceDep, VectorStoreDep
from src.api.v1.schemas import (
    RAGIngestPDFResponse,
    RAGIngestTextRequest,
    RAGIngestTextResponse,
)
from src.modules.documents import DocumentsRepositoryDep
from src.modules.users.dependencies import ActiveUserDep
from src.settings.config import settings

router = APIRouter(tags=["rag"])


@router.post("/rag/ingest/text", response_model=RAGIngestTextResponse)
async def ingest_text(
    payload: RAGIngestTextRequest,
    ingestion_service: RAGIngestionServiceDep,
    repository: DocumentsRepositoryDep,
    vector_store: VectorStoreDep,
    current_user: ActiveUserDep,
):
    resolved_doc_id = payload.doc_id or str(uuid4())
    if await repository.doc_id_exists(doc_id=resolved_doc_id, include_deleted=True):
        raise HTTPException(status_code=409, detail="Document id already exists.")

    try:
        await repository.create_document(
            owner_user_id=current_user.id,
            doc_id=resolved_doc_id,
            source=payload.source or "inline-text",
        )
    except IntegrityError as exc:
        await repository.rollback()
        raise HTTPException(status_code=409, detail="Document id already exists.") from exc

    try:
        result = await ingestion_service.ingest_text(
            text=payload.text,
            source=payload.source,
            doc_id=resolved_doc_id,
            chunking_strategy=payload.chunking_strategy,
        )
        await repository.mark_document_indexed(
            owner_user_id=current_user.id,
            doc_id=resolved_doc_id,
            chunking_strategy=result.chunking_strategy,
            chunk_size=result.chunk_size,
            chunk_overlap=result.chunk_overlap,
        )
        await repository.commit()
    except ValueError as exc:
        await repository.rollback()
        await vector_store.delete_by_doc_id(doc_id=resolved_doc_id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        await repository.rollback()
        await vector_store.delete_by_doc_id(doc_id=resolved_doc_id)
        raise HTTPException(status_code=500, detail="Ingestion failed.") from exc

    return RAGIngestTextResponse(
        status="ok",
        doc_id=resolved_doc_id,
        chunks_ingested=result.chunks_ingested,
    )


@router.post("/rag/ingest/pdf", response_model=RAGIngestPDFResponse)
async def ingest_pdf(
    file: Annotated[UploadFile, File(...)],
    ingestion_service: RAGIngestionServiceDep,
    repository: DocumentsRepositoryDep,
    vector_store: VectorStoreDep,
    current_user: ActiveUserDep,
    source: Annotated[str | None, Form()] = None,
    doc_id: Annotated[str | None, Form()] = None,
    chunking_strategy: Annotated[str | None, Form()] = None,
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
    resolved_doc_id = doc_id or str(uuid4())
    if await repository.doc_id_exists(doc_id=resolved_doc_id, include_deleted=True):
        raise HTTPException(status_code=409, detail="Document id already exists.")
    try:
        await repository.create_document(
            owner_user_id=current_user.id,
            doc_id=resolved_doc_id,
            source=resolved_source,
        )
    except IntegrityError as exc:
        await repository.rollback()
        raise HTTPException(status_code=409, detail="Document id already exists.") from exc

    try:
        result = await ingestion_service.ingest_pdf(
            pdf_bytes=payload,
            source=resolved_source,
            doc_id=resolved_doc_id,
            chunking_strategy=chunking_strategy,
        )
        await repository.mark_document_indexed(
            owner_user_id=current_user.id,
            doc_id=resolved_doc_id,
            chunking_strategy=result.chunking_strategy,
            chunk_size=result.chunk_size,
            chunk_overlap=result.chunk_overlap,
        )
        await repository.commit()
    except ValueError as exc:
        await repository.rollback()
        await vector_store.delete_by_doc_id(doc_id=resolved_doc_id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        await repository.rollback()
        await vector_store.delete_by_doc_id(doc_id=resolved_doc_id)
        raise HTTPException(status_code=500, detail="Ingestion failed.") from exc

    return RAGIngestPDFResponse(
        status="ok",
        doc_id=resolved_doc_id,
        chunks_ingested=result.chunks_ingested,
        pages_total=result.pages_total,
        pages_ingested=result.pages_ingested,
        skipped_pages=result.skipped_pages,
        warnings=result.warnings,
    )


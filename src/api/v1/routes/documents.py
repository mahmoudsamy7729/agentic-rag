from fastapi import APIRouter, HTTPException, Query

from src.api.v1.dependencies import VectorStoreDep
from src.api.v1.schemas import (
    DocumentChunkItem,
    DocumentChunkListResponse,
    DocumentChunkSummary,
    DocumentDeleteResponse,
    DocumentItem,
    DocumentListResponse,
)
from src.modules.documents import DocumentsRepositoryDep
from src.modules.users.dependencies import ActiveUserDep
from src.rag.models import RAGChunk

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    current_user: ActiveUserDep,
    repository: DocumentsRepositoryDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    items = await repository.list_owned_documents(
        owner_user_id=current_user.id,
        limit=limit,
        offset=offset,
    )
    return DocumentListResponse(
        status="ok",
        items=[
            DocumentItem(
                doc_id=item.id,
                owner_user_id=item.owner_user_id,
                source=item.source,
                created_at=item.created_at,
                updated_at=item.updated_at,
                deleted_at=item.deleted_at,
            )
            for item in items
        ],
        limit=limit,
        offset=offset,
    )


@router.get("/documents/{doc_id}", response_model=DocumentItem)
async def get_document(
    doc_id: str,
    current_user: ActiveUserDep,
    repository: DocumentsRepositoryDep,
):
    item = await repository.get_owned_document(
        owner_user_id=current_user.id,
        doc_id=doc_id,
        include_deleted=False,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentItem(
        doc_id=item.id,
        owner_user_id=item.owner_user_id,
        source=item.source,
        created_at=item.created_at,
        updated_at=item.updated_at,
        deleted_at=item.deleted_at,
    )


@router.get("/documents/{doc_id}/chunks", response_model=DocumentChunkListResponse)
async def list_document_chunks(
    doc_id: str,
    current_user: ActiveUserDep,
    repository: DocumentsRepositoryDep,
    vector_store: VectorStoreDep,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    page_number: int | None = Query(default=None, ge=1),
    q: str | None = Query(default=None),
):
    item = await repository.get_owned_document(
        owner_user_id=current_user.id,
        doc_id=doc_id,
        include_deleted=False,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    chunks = await vector_store.list_chunks(doc_id=doc_id)
    filtered_chunks = _filter_chunks(chunks=chunks, page_number=page_number, q=q)
    page = filtered_chunks[offset : offset + limit]

    return DocumentChunkListResponse(
        status="ok",
        document=DocumentChunkSummary(
            doc_id=item.id,
            source=item.source,
            chunking_strategy=item.chunking_strategy or _first_non_null(chunks, "chunking_strategy"),
            chunk_size=item.chunk_size or _first_non_null(chunks, "chunk_size"),
            chunk_overlap=item.chunk_overlap or _first_non_null(chunks, "chunk_overlap"),
        ),
        total=len(filtered_chunks),
        limit=limit,
        offset=offset,
        page_number=page_number,
        q=q.strip() if q and q.strip() else None,
        items=[
            DocumentChunkItem(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                text=chunk.text,
                page_number=chunk.page_number,
            )
            for chunk in page
        ],
    )


@router.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    doc_id: str,
    current_user: ActiveUserDep,
    repository: DocumentsRepositoryDep,
    vector_store: VectorStoreDep,
):
    existing = await repository.get_owned_document(
        owner_user_id=current_user.id,
        doc_id=doc_id,
        include_deleted=True,
    )
    if existing is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    was_already_deleted = existing.deleted_at is not None

    item = await repository.soft_delete_owned_document(
        owner_user_id=current_user.id,
        doc_id=doc_id,
    )
    await repository.commit()
    await vector_store.delete_by_doc_id(doc_id=doc_id)

    return DocumentDeleteResponse(
        status="ok",
        doc_id=doc_id,
        deleted=not was_already_deleted and item is not None,
    )


def _filter_chunks(
    *,
    chunks: list[RAGChunk],
    page_number: int | None,
    q: str | None,
) -> list[RAGChunk]:
    filtered = chunks
    if page_number is not None:
        filtered = [chunk for chunk in filtered if chunk.page_number == page_number]
    if q and q.strip():
        needle = q.strip().lower()
        filtered = [chunk for chunk in filtered if needle in chunk.text.lower()]
    return filtered


def _first_non_null(chunks: list[RAGChunk], field_name: str):
    for chunk in chunks:
        value = getattr(chunk, field_name, None)
        if value is not None:
            return value
    return None

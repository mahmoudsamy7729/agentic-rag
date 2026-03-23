from fastapi import APIRouter, HTTPException, Query

from src.api.v1.dependencies import VectorStoreDep
from src.api.v1.schemas import DocumentDeleteResponse, DocumentItem, DocumentListResponse
from src.modules.documents import DocumentsRepositoryDep
from src.modules.users.dependencies import ActiveUserDep

router = APIRouter()


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

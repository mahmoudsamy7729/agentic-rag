from fastapi import APIRouter

from src.api.v1.dependencies import RAGIngestionServiceDep
from src.api.v1.schemas import RAGIngestTextRequest, RAGIngestTextResponse

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

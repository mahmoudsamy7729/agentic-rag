from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from src.rag.embeddings.interface import EmbeddingProvider
from src.rag.ingestion.chunker import ChunkingStrategyRegistry
from src.rag.ingestion.pdf_extractor import PDFExtractor
from src.rag.models import RetrievedChunk
from src.rag.reranker import Reranker
from src.rag.vectorstore.interface import VectorStore


@dataclass(slots=True)
class IngestionResult:
    doc_id: str
    chunks_ingested: int
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int


@dataclass(slots=True)
class PDFIngestionResult:
    doc_id: str
    chunks_ingested: int
    pages_total: int
    pages_ingested: int
    skipped_pages: list[int]
    warnings: list[str]
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int


class RAGIngestionService:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chunking_registry: ChunkingStrategyRegistry,
        default_chunking_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        pdf_extractor: PDFExtractor | None = None,
        pdf_max_pages: int = 300,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._chunking_registry = chunking_registry
        self._default_chunking_strategy = default_chunking_strategy
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._pdf_extractor = pdf_extractor
        self._pdf_max_pages = pdf_max_pages

    async def ingest_text(
        self,
        *,
        text: str,
        source: str | None = None,
        doc_id: str | None = None,
        chunking_strategy: str | None = None,
    ) -> IngestionResult:
        resolved_doc_id = doc_id or str(uuid4())
        resolved_source = source or "inline-text"
        resolved_strategy_name = chunking_strategy or self._default_chunking_strategy
        strategy = self._chunking_registry.resolve(resolved_strategy_name)

        chunks = strategy.chunk(
            text=text,
            doc_id=resolved_doc_id,
            source=resolved_source,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        embeddings = await self._embedding_provider.embed_documents(
            [chunk.text for chunk in chunks]
        )
        await self._vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)

        return IngestionResult(
            doc_id=resolved_doc_id,
            chunks_ingested=len(chunks),
            chunking_strategy=strategy.name,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    async def ingest_pdf(
        self,
        *,
        pdf_bytes: bytes,
        source: str | None = None,
        doc_id: str | None = None,
        chunking_strategy: str | None = None,
    ) -> PDFIngestionResult:
        if self._pdf_extractor is None:
            raise RuntimeError("PDF extractor is not configured.")

        resolved_doc_id = doc_id or str(uuid4())
        resolved_source = source or "uploaded-pdf"
        resolved_strategy_name = chunking_strategy or self._default_chunking_strategy
        strategy = self._chunking_registry.resolve(resolved_strategy_name)
        extraction = await self._pdf_extractor.extract(
            pdf_bytes=pdf_bytes,
            max_pages=self._pdf_max_pages,
        )

        if extraction.pages_ingested == 0:
            raise ValueError("No extractable pages were found in this PDF.")

        chunks = []
        global_index = 0
        for segment in extraction.segments:
            segment_chunks = strategy.chunk(
                text=segment.text,
                doc_id=resolved_doc_id,
                source=resolved_source,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                page_number=segment.page_number,
            )
            for chunk in segment_chunks:
                chunk.chunk_id = f"chunk-{global_index}"
                global_index += 1
                chunks.append(chunk)

        if not chunks:
            raise ValueError("No chunks were generated from extracted PDF content.")

        embeddings = await self._embedding_provider.embed_documents(
            [chunk.text for chunk in chunks]
        )
        await self._vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)

        return PDFIngestionResult(
            doc_id=resolved_doc_id,
            chunks_ingested=len(chunks),
            pages_total=extraction.pages_total,
            pages_ingested=extraction.pages_ingested,
            skipped_pages=extraction.skipped_pages,
            warnings=extraction.warnings,
            chunking_strategy=strategy.name,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )


class RAGRetrievalService:
    RERANK_MAX_ATTEMPTS = 2
    RERANK_EXHAUSTED_ERROR = "Reranker failed after one retry."

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        default_top_k: int,
        prefetch_k: int,
        reranker: Reranker | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._default_top_k = default_top_k
        self._prefetch_k = prefetch_k
        self._reranker = reranker

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int | None = None,
        doc_id: str | None = None,
    ) -> list[RetrievedChunk]:
        final_k = top_k or self._default_top_k
        if final_k < 1:
            raise ValueError("top_k must be >= 1")

        candidate_k = max(self._prefetch_k, final_k)
        query_embedding = await self._embedding_provider.embed_query(query)
        candidates = await self._vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=candidate_k,
            doc_id=doc_id,
        )
        if not candidates:
            return []
        if self._reranker is None:
            return candidates[:final_k]

        for attempt in range(1, self.RERANK_MAX_ATTEMPTS + 1):
            try:
                reranked = await self._reranker.rerank(
                    query=query,
                    chunks=candidates,
                    top_n=final_k,
                )
                return reranked[:final_k]
            except Exception as exc:
                if attempt >= self.RERANK_MAX_ATTEMPTS:
                    raise RuntimeError(self.RERANK_EXHAUSTED_ERROR) from exc

        raise RuntimeError(self.RERANK_EXHAUSTED_ERROR)

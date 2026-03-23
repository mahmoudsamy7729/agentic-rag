import asyncio

import pytest

from src.rag.ingestion.pdf_extractor import PDFExtractionResult, PDFSegment
from src.rag.pipeline.services import RAGIngestionService


class FakeEmbeddingProvider:
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1), 0.0] for index, _ in enumerate(texts)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.last_chunks = []
        self.last_embeddings = []

    async def upsert_chunks(self, *, chunks, embeddings):
        self.last_chunks = chunks
        self.last_embeddings = embeddings


class FakePDFExtractor:
    def __init__(self, result: PDFExtractionResult) -> None:
        self._result = result
        self.last_max_pages: int | None = None

    async def extract(self, *, pdf_bytes: bytes, max_pages: int) -> PDFExtractionResult:
        self.last_max_pages = max_pages
        return self._result


def test_ingest_pdf_maps_page_number_and_returns_warnings():
    extraction = PDFExtractionResult(
        pages_total=3,
        pages_ingested=2,
        skipped_pages=[2],
        warnings=["Page 2: extraction failed"],
        segments=[
            PDFSegment(page_number=1, segment_type="text", text="Alpha section", y0=1.0, x0=1.0),
            PDFSegment(page_number=3, segment_type="table", text="Table:\n| a |\n|---|\n| 1 |", y0=2.0, x0=1.0),
        ],
    )
    pdf_extractor = FakePDFExtractor(extraction)
    vector_store = FakeVectorStore()

    service = RAGIngestionService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        chunk_size=100,
        chunk_overlap=0,
        pdf_extractor=pdf_extractor,
        pdf_max_pages=300,
    )

    result = asyncio.run(
        service.ingest_pdf(
            pdf_bytes=b"%PDF-sample",
            source="policy.pdf",
            doc_id="doc-123",
        )
    )

    assert result.doc_id == "doc-123"
    assert result.pages_total == 3
    assert result.pages_ingested == 2
    assert result.skipped_pages == [2]
    assert result.warnings == ["Page 2: extraction failed"]
    assert result.chunks_ingested == 2
    assert pdf_extractor.last_max_pages == 300

    assert len(vector_store.last_chunks) == 2
    assert [chunk.page_number for chunk in vector_store.last_chunks] == [1, 3]
    assert [chunk.chunk_id for chunk in vector_store.last_chunks] == ["chunk-0", "chunk-1"]


def test_ingest_pdf_raises_when_no_pages_extracted():
    extraction = PDFExtractionResult(
        pages_total=2,
        pages_ingested=0,
        skipped_pages=[1, 2],
        warnings=["Page 1: failed", "Page 2: failed"],
        segments=[],
    )
    service = RAGIngestionService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
        chunk_size=100,
        chunk_overlap=0,
        pdf_extractor=FakePDFExtractor(extraction),
        pdf_max_pages=300,
    )

    with pytest.raises(ValueError, match="No extractable pages"):
        asyncio.run(service.ingest_pdf(pdf_bytes=b"%PDF-sample"))

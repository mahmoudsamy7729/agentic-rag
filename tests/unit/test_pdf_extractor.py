import asyncio

from src.rag.ingestion.pdf_extractor import PDFPlumberExtractor, PDFSegment


class FakeTable:
    def __init__(self, bbox, rows):
        self.bbox = bbox
        self._rows = rows

    def extract(self):
        return self._rows


class FakePage:
    def __init__(self, *, words=None, tables=None, should_fail=False):
        self._words = words or []
        self._tables = tables or []
        self._should_fail = should_fail

    def find_tables(self):
        return self._tables

    def extract_words(self):
        if self._should_fail:
            raise RuntimeError("boom")
        return self._words


class FakePDF:
    def __init__(self, pages):
        self.pages = pages


class FakePDFContext:
    def __init__(self, pages):
        self._pdf = FakePDF(pages)

    def __enter__(self):
        return self._pdf

    def __exit__(self, exc_type, exc, tb):
        return None


def test_pdf_extractor_preserves_order_and_skips_failed_pages(monkeypatch):
    pages = [
        FakePage(
            words=[
                {"text": "Intro", "x0": 10, "x1": 20, "top": 10, "bottom": 15},
                {"text": "inside", "x0": 15, "x1": 30, "top": 50, "bottom": 56},
                {"text": "Outro", "x0": 10, "x1": 22, "top": 90, "bottom": 95},
            ],
            tables=[
                FakeTable(
                    bbox=(0, 40, 200, 80),
                    rows=[["item", "value"], ["refund_window", "7_days"]],
                )
            ],
        ),
        FakePage(should_fail=True),
    ]

    fake_pdfplumber = type(
        "FakePdfPlumber",
        (),
        {"open": staticmethod(lambda _: FakePDFContext(pages))},
    )
    monkeypatch.setattr("src.rag.ingestion.pdf_extractor._pdfplumber", fake_pdfplumber)

    extractor = PDFPlumberExtractor(dedupe_threshold=96)
    result = asyncio.run(extractor.extract(pdf_bytes=b"%PDF-sample", max_pages=10))

    assert result.pages_total == 2
    assert result.pages_ingested == 1
    assert result.skipped_pages == [2]
    assert len(result.warnings) == 1

    texts = [segment.text for segment in result.segments]
    joined = "\n".join(texts)

    intro_idx = next(index for index, text in enumerate(texts) if "Intro" in text)
    table_idx = next(index for index, text in enumerate(texts) if text.startswith("Table:"))
    outro_idx = next(index for index, text in enumerate(texts) if "Outro" in text)

    assert intro_idx < table_idx < outro_idx
    assert "inside" not in joined


def test_pdf_extractor_dedupes_adjacent_similar_segments():
    extractor = PDFPlumberExtractor(dedupe_threshold=96)

    segments = [
        PDFSegment(page_number=1, segment_type="text", text="Alpha", y0=1.0, x0=1.0),
        PDFSegment(page_number=1, segment_type="text", text="Alpha", y0=2.0, x0=1.0),
        PDFSegment(page_number=1, segment_type="text", text="Beta", y0=3.0, x0=1.0),
    ]

    deduped = extractor._dedupe_adjacent_segments(segments)

    assert len(deduped) == 2
    assert [segment.text for segment in deduped] == ["Alpha", "Beta"]

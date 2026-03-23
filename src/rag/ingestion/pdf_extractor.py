from __future__ import annotations

import asyncio
import difflib
import io
from dataclasses import dataclass
from typing import Any, Protocol

try:
    import pandas as pd
except ImportError:  # pragma: no cover - dependency presence is environment-specific
    pd = None

try:
    import pdfplumber as _pdfplumber
except ImportError:  # pragma: no cover - dependency presence is environment-specific
    _pdfplumber = None

try:
    from rapidfuzz import fuzz as _rapidfuzz
except ImportError:  # pragma: no cover - dependency presence is environment-specific
    _rapidfuzz = None


@dataclass(slots=True)
class PDFSegment:
    page_number: int
    segment_type: str
    text: str
    y0: float
    x0: float


@dataclass(slots=True)
class PDFExtractionResult:
    pages_total: int
    pages_ingested: int
    skipped_pages: list[int]
    warnings: list[str]
    segments: list[PDFSegment]


class PDFExtractor(Protocol):
    async def extract(
        self,
        *,
        pdf_bytes: bytes,
        max_pages: int,
    ) -> PDFExtractionResult: ...


class PDFPlumberExtractor:
    def __init__(self, *, dedupe_threshold: int = 96) -> None:
        self._dedupe_threshold = dedupe_threshold

    async def extract(
        self,
        *,
        pdf_bytes: bytes,
        max_pages: int,
    ) -> PDFExtractionResult:
        return await asyncio.to_thread(
            self._extract_sync,
            pdf_bytes=pdf_bytes,
            max_pages=max_pages,
        )

    def _extract_sync(
        self,
        *,
        pdf_bytes: bytes,
        max_pages: int,
    ) -> PDFExtractionResult:
        if _pdfplumber is None:
            raise RuntimeError("pdfplumber is not installed. Install it to use PDF ingestion.")
        if not pdf_bytes:
            raise ValueError("Uploaded PDF is empty.")

        with _pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = list(pdf.pages)
            pages_total = len(pages)
            if pages_total == 0:
                raise ValueError("PDF has no pages.")
            if pages_total > max_pages:
                raise ValueError(
                    f"PDF has {pages_total} pages; maximum allowed is {max_pages}."
                )

            all_segments: list[PDFSegment] = []
            skipped_pages: list[int] = []
            warnings: list[str] = []
            pages_ingested = 0

            for page_number, page in enumerate(pages, start=1):
                try:
                    page_segments = self._extract_page_segments(
                        page=page,
                        page_number=page_number,
                    )
                except Exception as exc:
                    skipped_pages.append(page_number)
                    warnings.append(f"Page {page_number}: extraction failed ({exc})")
                    continue

                if not page_segments:
                    skipped_pages.append(page_number)
                    warnings.append(f"Page {page_number}: no extractable text or tables.")
                    continue

                pages_ingested += 1
                all_segments.extend(page_segments)

            return PDFExtractionResult(
                pages_total=pages_total,
                pages_ingested=pages_ingested,
                skipped_pages=skipped_pages,
                warnings=warnings,
                segments=all_segments,
            )

    def _extract_page_segments(self, *, page: Any, page_number: int) -> list[PDFSegment]:
        table_segments: list[PDFSegment] = []
        table_bboxes: list[tuple[float, float, float, float]] = []

        tables = page.find_tables() or []
        for table in tables:
            bbox_raw = getattr(table, "bbox", None)
            if not bbox_raw:
                continue
            x0, y0, x1, y1 = self._normalize_bbox(bbox_raw)
            table_text = self._table_to_text(table.extract() or [])
            if not table_text:
                continue
            table_bboxes.append((x0, y0, x1, y1))
            table_segments.append(
                PDFSegment(
                    page_number=page_number,
                    segment_type="table",
                    text=table_text,
                    y0=y0,
                    x0=x0,
                )
            )

        words = page.extract_words() or []
        filtered_words = [
            word
            for word in words
            if not self._word_in_any_table(word=word, table_bboxes=table_bboxes)
        ]
        text_segments = self._words_to_text_segments(
            words=filtered_words,
            page_number=page_number,
        )

        merged = sorted(
            [*text_segments, *table_segments],
            key=lambda segment: (segment.y0, segment.x0),
        )
        return self._dedupe_adjacent_segments(merged)

    @staticmethod
    def _normalize_bbox(bbox: Any) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = bbox
        return float(x0), float(y0), float(x1), float(y1)

    @staticmethod
    def _table_to_text(rows: list[list[Any]]) -> str:
        normalized_rows: list[list[str]] = []
        for row in rows:
            if row is None:
                continue
            values = [str(cell).strip() if cell is not None else "" for cell in row]
            if any(values):
                normalized_rows.append(values)
        if not normalized_rows:
            return ""

        max_cols = max(len(row) for row in normalized_rows)
        padded = [row + [""] * (max_cols - len(row)) for row in normalized_rows]
        if pd is not None:
            frame = pd.DataFrame(padded, columns=[f"col_{idx + 1}" for idx in range(max_cols)])
            try:
                markdown = frame.to_markdown(index=False)
                return f"Table:\n{markdown}".strip()
            except Exception:
                # pandas.to_markdown may require optional `tabulate`; fallback to manual markdown.
                pass

        header = " | ".join(f"col_{idx + 1}" for idx in range(max_cols))
        separator = " | ".join("---" for _ in range(max_cols))
        row_lines = "\n".join(f"| {' | '.join(row)} |" for row in padded)
        return f"Table:\n| {header} |\n| {separator} |\n{row_lines}".strip()

    @staticmethod
    def _word_in_any_table(
        *,
        word: dict[str, Any],
        table_bboxes: list[tuple[float, float, float, float]],
    ) -> bool:
        if not table_bboxes:
            return False
        x0 = float(word.get("x0", 0.0))
        x1 = float(word.get("x1", x0))
        y0 = float(word.get("top", 0.0))
        y1 = float(word.get("bottom", y0))
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        for tx0, ty0, tx1, ty1 in table_bboxes:
            if tx0 <= cx <= tx1 and ty0 <= cy <= ty1:
                return True
        return False

    @staticmethod
    def _words_to_text_segments(
        *,
        words: list[dict[str, Any]],
        page_number: int,
    ) -> list[PDFSegment]:
        if not words:
            return []

        ordered_words = sorted(
            words,
            key=lambda word: (float(word.get("top", 0.0)), float(word.get("x0", 0.0))),
        )

        lines: list[dict[str, Any]] = []
        current_words: list[dict[str, Any]] = []
        current_top: float | None = None
        line_tolerance = 2.5

        for word in ordered_words:
            top = float(word.get("top", 0.0))
            if current_top is None or abs(top - current_top) <= line_tolerance:
                current_words.append(word)
                current_top = top if current_top is None else min(current_top, top)
                continue

            line = PDFPlumberExtractor._finalize_line(current_words=current_words)
            if line:
                lines.append(line)
            current_words = [word]
            current_top = top

        line = PDFPlumberExtractor._finalize_line(current_words=current_words)
        if line:
            lines.append(line)
        if not lines:
            return []

        segments: list[PDFSegment] = []
        block_lines: list[dict[str, Any]] = []
        max_line_gap = 8.0

        for line in lines:
            if not block_lines:
                block_lines.append(line)
                continue

            previous = block_lines[-1]
            if line["y0"] - previous["y1"] <= max_line_gap:
                block_lines.append(line)
                continue

            segments.append(
                PDFPlumberExtractor._lines_to_segment(
                    lines=block_lines,
                    page_number=page_number,
                )
            )
            block_lines = [line]

        if block_lines:
            segments.append(
                PDFPlumberExtractor._lines_to_segment(
                    lines=block_lines,
                    page_number=page_number,
                )
            )

        return [segment for segment in segments if segment.text.strip()]

    @staticmethod
    def _finalize_line(*, current_words: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not current_words:
            return None
        ordered = sorted(current_words, key=lambda word: float(word.get("x0", 0.0)))
        text = " ".join(str(word.get("text", "")).strip() for word in ordered).strip()
        if not text:
            return None
        return {
            "text": text,
            "x0": min(float(word.get("x0", 0.0)) for word in ordered),
            "y0": min(float(word.get("top", 0.0)) for word in ordered),
            "y1": max(float(word.get("bottom", 0.0)) for word in ordered),
        }

    @staticmethod
    def _lines_to_segment(*, lines: list[dict[str, Any]], page_number: int) -> PDFSegment:
        text = "\n".join(line["text"] for line in lines).strip()
        return PDFSegment(
            page_number=page_number,
            segment_type="text",
            text=text,
            y0=float(lines[0]["y0"]),
            x0=float(min(line["x0"] for line in lines)),
        )

    def _dedupe_adjacent_segments(self, segments: list[PDFSegment]) -> list[PDFSegment]:
        if not segments:
            return []

        deduped: list[PDFSegment] = [segments[0]]
        for segment in segments[1:]:
            previous = deduped[-1]
            if previous.page_number == segment.page_number:
                score = self._similarity_ratio(previous.text, segment.text)
                if score >= self._dedupe_threshold:
                    continue
            deduped.append(segment)
        return deduped

    @staticmethod
    def _similarity_ratio(left: str, right: str) -> float:
        if _rapidfuzz is not None:
            return float(_rapidfuzz.ratio(left, right))
        return difflib.SequenceMatcher(None, left, right).ratio() * 100.0

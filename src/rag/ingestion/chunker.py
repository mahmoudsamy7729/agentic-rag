from __future__ import annotations

from abc import ABC, abstractmethod
import re

from src.rag.models import RAGChunk


class ChunkingStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Stable strategy identifier."""

    @abstractmethod
    def chunk(
        self,
        *,
        text: str,
        doc_id: str,
        source: str,
        chunk_size: int,
        chunk_overlap: int,
        page_number: int | None = None,
    ) -> list[RAGChunk]:
        """Chunk text into RAG chunks."""


class FixedWindowChunkingStrategy(ChunkingStrategy):
    @property
    def name(self) -> str:
        return "fixed_window"

    def chunk(
        self,
        *,
        text: str,
        doc_id: str,
        source: str,
        chunk_size: int,
        chunk_overlap: int,
        page_number: int | None = None,
    ) -> list[RAGChunk]:
        normalized = text.strip()
        if not normalized:
            return []
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        chunks: list[RAGChunk] = []
        start = 0
        step = chunk_size - chunk_overlap
        index = 0

        while start < len(normalized):
            end = start + chunk_size
            chunk_body = normalized[start:end].strip()
            if chunk_body:
                chunks.append(
                    RAGChunk(
                        doc_id=doc_id,
                        chunk_id=f"chunk-{index}",
                        source=source,
                        text=chunk_body,
                        page_number=page_number,
                        chunking_strategy=self.name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )
                index += 1
            start += step

        return chunks


class RecursiveSemanticChunkingStrategy(ChunkingStrategy):
    _markdown_heading_re = re.compile(r"^\s{0,3}#{1,6}\s+\S")
    _list_item_re = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
    _sentence_boundary_re = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

    @property
    def name(self) -> str:
        return "recursive_semantic"

    def chunk(
        self,
        *,
        text: str,
        doc_id: str,
        source: str,
        chunk_size: int,
        chunk_overlap: int,
        page_number: int | None = None,
    ) -> list[RAGChunk]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        structural_units = self._split_structural_units(normalized)
        leaf_units: list[str] = []
        for unit in structural_units:
            leaf_units.extend(self._split_unit(unit, chunk_size))

        packed_units = self._pack_units(leaf_units, chunk_size)
        return [
            RAGChunk(
                doc_id=doc_id,
                chunk_id=f"chunk-{index}",
                source=source,
                text=chunk_text,
                page_number=page_number,
                chunking_strategy=self.name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for index, chunk_text in enumerate(packed_units)
            if chunk_text
        ]

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()

    def _split_structural_units(self, text: str) -> list[str]:
        lines = text.split("\n")
        units: list[str] = []
        buffer: list[str] = []
        index = 0

        def flush_buffer() -> None:
            if buffer:
                joined = "\n".join(buffer).strip()
                if joined:
                    units.append(joined)
                buffer.clear()

        while index < len(lines):
            line = lines[index]
            stripped = line.strip()

            if not stripped:
                flush_buffer()
                index += 1
                continue

            next_nonempty = self._next_nonempty_line(lines, index + 1)
            if self._is_heading_like(stripped) or self._is_title_like(stripped, next_nonempty):
                flush_buffer()
                units.append(stripped)
                index += 1
                continue

            if self._is_list_item(stripped):
                flush_buffer()
                list_lines = [stripped]
                index += 1
                while index < len(lines):
                    next_line = lines[index].strip()
                    if not next_line:
                        break
                    if self._is_list_item(next_line):
                        list_lines.append(next_line)
                        index += 1
                        continue
                    if lines[index].startswith((" ", "\t")):
                        list_lines.append(next_line)
                        index += 1
                        continue
                    break
                units.append("\n".join(list_lines).strip())
                continue

            buffer.append(stripped)
            index += 1

        flush_buffer()
        return units

    @staticmethod
    def _next_nonempty_line(lines: list[str], start: int) -> str | None:
        for index in range(start, len(lines)):
            stripped = lines[index].strip()
            if stripped:
                return stripped
        return None

    def _is_heading_like(self, line: str) -> bool:
        return bool(self._markdown_heading_re.match(line))

    def _is_title_like(self, line: str, next_line: str | None) -> bool:
        if next_line is None:
            return False
        if len(line) > 80:
            return False
        if len(line.split()) > 12:
            return False
        if line.endswith((".", "!", "?", ";", ":")):
            return False
        if self._is_list_item(line):
            return False
        if line.isupper():
            return True
        return bool(re.match(r"^[A-Z][\w\s/&()'-]+$", line))

    def _is_list_item(self, line: str) -> bool:
        return bool(self._list_item_re.match(line))

    def _split_unit(self, text: str, chunk_size: int) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        if len(normalized) <= chunk_size:
            return [normalized]

        sentences = self._split_sentences(normalized)
        if len(sentences) > 1:
            sentence_units: list[str] = []
            for sentence in sentences:
                sentence_units.extend(self._split_unit(sentence, chunk_size))
            return sentence_units

        return self._hard_split(normalized, chunk_size)

    def _split_sentences(self, text: str) -> list[str]:
        if "\n" in text and self._is_list_item(text.split("\n", 1)[0].strip()):
            items = [item.strip() for item in text.split("\n") if item.strip()]
            return items if len(items) > 1 else [text]

        sentences = [piece.strip() for piece in self._sentence_boundary_re.split(text) if piece.strip()]
        return sentences if sentences else [text]

    def _pack_units(self, units: list[str], chunk_size: int) -> list[str]:
        packed: list[str] = []
        current = ""

        for unit in units:
            candidate = unit if not current else f"{current}\n\n{unit}"
            if current and len(candidate) > chunk_size:
                packed.append(current)
                current = unit
                continue
            if len(unit) > chunk_size:
                if current:
                    packed.append(current)
                    current = ""
                packed.extend(self._hard_split(unit, chunk_size))
                continue
            current = candidate

        if current:
            packed.append(current)
        return packed

    @staticmethod
    def _hard_split(text: str, chunk_size: int) -> list[str]:
        pieces: list[str] = []
        start = 0
        normalized = text.strip()
        while start < len(normalized):
            end = start + chunk_size
            piece = normalized[start:end].strip()
            if piece:
                pieces.append(piece)
            start = end
        return pieces


class ChunkingStrategyRegistry:
    def __init__(self, strategies: list[ChunkingStrategy]) -> None:
        self._strategies = {strategy.name: strategy for strategy in strategies}
        if not self._strategies:
            raise ValueError("At least one chunking strategy must be registered.")

    def resolve(self, name: str) -> ChunkingStrategy:
        strategy = self._strategies.get(name)
        if strategy is None:
            available = ", ".join(sorted(self._strategies))
            raise ValueError(
                f"Unsupported chunking_strategy '{name}'. Available: {available}."
            )
        return strategy

    def names(self) -> list[str]:
        return sorted(self._strategies)


def chunk_text(
    *,
    text: str,
    doc_id: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
    page_number: int | None = None,
) -> list[RAGChunk]:
    return FixedWindowChunkingStrategy().chunk(
        text=text,
        doc_id=doc_id,
        source=source,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        page_number=page_number,
    )

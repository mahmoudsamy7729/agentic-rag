import pytest

from src.rag.ingestion.chunker import (
    ChunkingStrategyRegistry,
    FixedWindowChunkingStrategy,
    chunk_text,
)


def test_chunker_uses_overlap_and_stable_ids():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(
        text=text,
        doc_id="doc-1",
        source="unit-test",
        chunk_size=10,
        chunk_overlap=3,
    )

    assert [chunk.chunk_id for chunk in chunks] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
    assert chunks[0].text == "abcdefghij"
    assert chunks[1].text.startswith("hij")
    assert all(chunk.doc_id == "doc-1" for chunk in chunks)
    assert all(chunk.chunking_strategy == "fixed_window" for chunk in chunks)
    assert all(chunk.chunk_size == 10 for chunk in chunks)
    assert all(chunk.chunk_overlap == 3 for chunk in chunks)


def test_chunking_registry_rejects_unknown_strategy():
    registry = ChunkingStrategyRegistry([FixedWindowChunkingStrategy()])

    with pytest.raises(ValueError, match="Unsupported chunking_strategy"):
        registry.resolve("recursive_semantic")

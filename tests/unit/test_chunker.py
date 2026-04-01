import pytest

from src.rag.ingestion.chunker import (
    ChunkingStrategyRegistry,
    FixedWindowChunkingStrategy,
    RecursiveSemanticChunkingStrategy,
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


def test_recursive_semantic_returns_structure_aware_chunks():
    strategy = RecursiveSemanticChunkingStrategy()
    text = """# Privacy Policy

Introduction

We collect account information and usage data to operate the service.

- We collect names
- We collect emails
- We collect device identifiers
"""

    chunks = strategy.chunk(
        text=text,
        doc_id="doc-1",
        source="unit-test",
        chunk_size=120,
        chunk_overlap=40,
    )

    assert chunks
    assert all(chunk.chunking_strategy == "recursive_semantic" for chunk in chunks)
    assert any("Privacy Policy" in chunk.text for chunk in chunks)
    assert any("We collect names" in chunk.text for chunk in chunks)


def test_recursive_semantic_ignores_overlap_in_output_boundaries():
    strategy = RecursiveSemanticChunkingStrategy()
    text = (
        "Alpha section.\n\n"
        "Beta section with more explanation. Gamma section continues the same topic. "
        "Delta section adds more supporting details."
    )

    chunks_a = strategy.chunk(
        text=text,
        doc_id="doc-1",
        source="unit-test",
        chunk_size=60,
        chunk_overlap=0,
    )
    chunks_b = strategy.chunk(
        text=text,
        doc_id="doc-1",
        source="unit-test",
        chunk_size=60,
        chunk_overlap=30,
    )

    assert [chunk.text for chunk in chunks_a] == [chunk.text for chunk in chunks_b]
    assert all(chunk.chunk_overlap == 0 for chunk in chunks_a)
    assert all(chunk.chunk_overlap == 30 for chunk in chunks_b)


def test_recursive_semantic_preserves_page_number_and_max_size():
    strategy = RecursiveSemanticChunkingStrategy()
    text = (
        "Overview\n\n"
        "This is a long paragraph. It has several sentences. "
        "Each sentence should help the recursive splitter create bounded chunks. "
        "The fallback splitter should only be used when needed."
    )

    chunks = strategy.chunk(
        text=text,
        doc_id="doc-1",
        source="unit-test",
        chunk_size=70,
        chunk_overlap=10,
        page_number=7,
    )

    assert chunks
    assert all(chunk.page_number == 7 for chunk in chunks)
    assert all(len(chunk.text) <= 70 for chunk in chunks)

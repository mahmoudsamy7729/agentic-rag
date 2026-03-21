from src.rag.ingestion.chunker import chunk_text


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

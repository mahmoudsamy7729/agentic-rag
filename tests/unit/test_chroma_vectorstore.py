import asyncio
import shutil
import tempfile

import pytest

pytest.importorskip("chromadb")

from src.infrastructure.vector_db.chroma_vectorstore import ChromaVectorStore
from src.rag.models import RAGChunk


def test_chroma_vectorstore_upsert_and_search():
    persist_dir = tempfile.mkdtemp(prefix="chroma-store-", dir=".")
    try:
        async def _run():
            store = ChromaVectorStore(
                persist_dir=persist_dir,
                collection_name="unit-test-collection",
            )

            chunks = [
                RAGChunk(
                    doc_id="doc-1",
                    chunk_id="chunk-0",
                    source="a.txt",
                    text="alpha",
                    page_number=2,
                ),
                RAGChunk(doc_id="doc-1", chunk_id="chunk-1", source="a.txt", text="beta"),
            ]
            embeddings = [
                [1.0, 0.0],
                [0.0, 1.0],
            ]

            await store.upsert_chunks(chunks=chunks, embeddings=embeddings)
            return await store.similarity_search(query_embedding=[0.9, 0.1], top_k=1)

        results = asyncio.run(_run())

        assert len(results) == 1
        assert results[0].chunk_id == "chunk-0"
        assert results[0].doc_id == "doc-1"
        assert results[0].source == "a.txt"
        assert results[0].page_number == 2
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)


def test_chroma_vectorstore_filters_by_doc_id():
    persist_dir = tempfile.mkdtemp(prefix="chroma-store-", dir=".")
    try:
        async def _run():
            store = ChromaVectorStore(
                persist_dir=persist_dir,
                collection_name="unit-test-doc-filter",
            )

            chunks = [
                RAGChunk(doc_id="doc-a", chunk_id="chunk-0", source="a.txt", text="alpha"),
                RAGChunk(doc_id="doc-b", chunk_id="chunk-0", source="b.txt", text="alpha"),
            ]
            embeddings = [
                [1.0, 0.0],
                [1.0, 0.0],
            ]

            await store.upsert_chunks(chunks=chunks, embeddings=embeddings)
            return await store.similarity_search(
                query_embedding=[1.0, 0.0],
                top_k=5,
                doc_id="doc-b",
            )

        results = asyncio.run(_run())

        assert len(results) == 1
        assert results[0].doc_id == "doc-b"
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)


def test_chroma_vectorstore_allows_distinct_collections_per_embedding_mode():
    persist_dir = tempfile.mkdtemp(prefix="chroma-store-", dir=".")
    try:
        async def _run():
            openai_store = ChromaVectorStore(
                persist_dir=persist_dir,
                collection_name="agentic_rag_docs__openai__text_embedding_3_small",
            )
            huggingface_store = ChromaVectorStore(
                persist_dir=persist_dir,
                collection_name="agentic_rag_docs__huggingface__sentence_transformers_all_minilm_l6_v2",
            )

            await openai_store.upsert_chunks(
                chunks=[RAGChunk(doc_id="doc-openai", chunk_id="chunk-0", source="openai.txt", text="alpha")],
                embeddings=[[1.0, 0.0]],
            )
            await huggingface_store.upsert_chunks(
                chunks=[RAGChunk(doc_id="doc-hf", chunk_id="chunk-0", source="hf.txt", text="beta")],
                embeddings=[[1.0, 0.0, 0.0]],
            )

            openai_results = await openai_store.similarity_search(
                query_embedding=[0.9, 0.1],
                top_k=1,
            )
            huggingface_results = await huggingface_store.similarity_search(
                query_embedding=[0.9, 0.1, 0.0],
                top_k=1,
            )

            return openai_results, huggingface_results

        openai_results, huggingface_results = asyncio.run(_run())

        assert len(openai_results) == 1
        assert openai_results[0].doc_id == "doc-openai"
        assert len(huggingface_results) == 1
        assert huggingface_results[0].doc_id == "doc-hf"
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)


def test_chroma_vectorstore_delete_by_doc_id():
    persist_dir = tempfile.mkdtemp(prefix="chroma-store-", dir=".")
    try:
        async def _run():
            store = ChromaVectorStore(
                persist_dir=persist_dir,
                collection_name="unit-test-delete-doc",
            )
            chunks = [
                RAGChunk(doc_id="doc-a", chunk_id="chunk-0", source="a.txt", text="alpha"),
                RAGChunk(doc_id="doc-b", chunk_id="chunk-0", source="b.txt", text="beta"),
            ]
            embeddings = [[1.0, 0.0], [0.0, 1.0]]
            await store.upsert_chunks(chunks=chunks, embeddings=embeddings)
            await store.delete_by_doc_id(doc_id="doc-a")
            return await store.similarity_search(query_embedding=[1.0, 0.0], top_k=10)

        results = asyncio.run(_run())
        assert all(item.doc_id != "doc-a" for item in results)
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)


def test_chroma_vectorstore_list_chunks_returns_stable_chunk_order():
    persist_dir = tempfile.mkdtemp(prefix="chroma-store-", dir=".")
    try:
        async def _run():
            store = ChromaVectorStore(
                persist_dir=persist_dir,
                collection_name="unit-test-list-chunks",
            )
            chunks = [
                RAGChunk(doc_id="doc-a", chunk_id="chunk-10", source="a.txt", text="ten", page_number=2),
                RAGChunk(doc_id="doc-a", chunk_id="chunk-2", source="a.txt", text="two", page_number=1),
                RAGChunk(doc_id="doc-a", chunk_id="chunk-1", source="a.txt", text="one", page_number=1),
            ]
            embeddings = [[1.0, 0.0], [0.5, 0.5], [0.2, 0.8]]
            await store.upsert_chunks(chunks=chunks, embeddings=embeddings)
            return await store.list_chunks(doc_id="doc-a")

        results = asyncio.run(_run())
        assert [item.chunk_id for item in results] == ["chunk-1", "chunk-2", "chunk-10"]
        assert results[0].text == "one"
        assert results[1].page_number == 1
        assert results[2].page_number == 2
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)

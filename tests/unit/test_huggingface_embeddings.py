import asyncio

import pytest

from src.infrastructure.llm.huggingface_embeddings import HuggingFaceEmbeddingProvider


class Fake_HuggingFaceEmbeddings:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(idx), 0.5] for idx, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [0.25, 0.75]


def test_huggingface_embed_documents_returns_vectors(monkeypatch):
    monkeypatch.setattr(
        "src.infrastructure.llm.huggingface_embeddings._HuggingFaceEmbeddings",
        Fake_HuggingFaceEmbeddings,
    )
    provider = HuggingFaceEmbeddingProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectors = asyncio.run(provider.embed_documents(["a", "b"]))

    assert len(vectors) == 2
    assert vectors[0] == [0.0, 0.5]
    assert vectors[1] == [1.0, 0.5]


def test_huggingface_embed_query_returns_vector(monkeypatch):
    monkeypatch.setattr(
        "src.infrastructure.llm.huggingface_embeddings._HuggingFaceEmbeddings",
        Fake_HuggingFaceEmbeddings,
    )
    provider = HuggingFaceEmbeddingProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector = asyncio.run(provider.embed_query("framework"))

    assert vector == [0.25, 0.75]


def test_huggingface_malformed_query_vector_raises(monkeypatch):
    class BadEmbeddings(Fake_HuggingFaceEmbeddings):
        def embed_query(self, text: str):  # type: ignore[override]
            return {"bad": "shape"}

    monkeypatch.setattr(
        "src.infrastructure.llm.huggingface_embeddings._HuggingFaceEmbeddings",
        BadEmbeddings,
    )
    provider = HuggingFaceEmbeddingProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")

    with pytest.raises(ValueError, match="non-list vector for query"):
        asyncio.run(provider.embed_query("framework"))

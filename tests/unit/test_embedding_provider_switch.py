import shutil
import sys
import tempfile
from types import SimpleNamespace

import pytest

from src.api.v1 import dependencies as deps
from src.infrastructure.llm.huggingface_embeddings import HuggingFaceEmbeddingProvider
from src.infrastructure.llm.openai_embeddings import OpenAIEmbeddingProvider
from src.settings.config import settings


@pytest.fixture(autouse=True)
def _restore_settings_and_cache():
    fields = [
        "rag_collection_name",
        "chroma_persist_dir",
        "embedding_provider",
        "openai_key",
        "embedding_model",
        "embedding_base_url",
    ]
    snapshot = {field: getattr(settings, field) for field in fields}
    deps.get_embedding_provider.cache_clear()
    deps.get_vector_store.cache_clear()
    try:
        yield
    finally:
        for field, value in snapshot.items():
            setattr(settings, field, value)
        deps.get_embedding_provider.cache_clear()
        deps.get_vector_store.cache_clear()


def test_get_embedding_provider_returns_openai_provider():
    settings.embedding_provider = "openai"
    settings.openai_key = "openai-key"
    settings.embedding_model = "text-embedding-3-small"
    settings.embedding_base_url = None

    provider = deps.get_embedding_provider()

    assert isinstance(provider, OpenAIEmbeddingProvider)


def test_get_embedding_provider_returns_huggingface_provider(monkeypatch):
    class FakeHF:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def embed_documents(self, texts):
            return [[0.1] for _ in texts]

        def embed_query(self, text):
            return [0.1]

    monkeypatch.setattr(
        "src.infrastructure.llm.huggingface_embeddings._HuggingFaceEmbeddings",
        FakeHF,
    )

    settings.embedding_provider = "huggingface"
    settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    provider = deps.get_embedding_provider()

    assert isinstance(provider, HuggingFaceEmbeddingProvider)


def test_get_embedding_provider_openai_missing_key_raises():
    settings.embedding_provider = "openai"
    settings.openai_key = None
    settings.embedding_model = "text-embedding-3-small"

    with pytest.raises(RuntimeError, match="OPENAI_KEY"):
        deps.get_embedding_provider()


def test_resolved_rag_collection_name_for_openai():
    settings.rag_collection_name = None
    settings.embedding_provider = "openai"
    settings.embedding_model = "text-embedding-3-small"

    collection_name = settings.resolved_rag_collection_name()

    assert collection_name == "agentic_rag_docs__openai__text_embedding_3_small"


def test_resolved_rag_collection_name_for_huggingface():
    settings.rag_collection_name = None
    settings.embedding_provider = "huggingface"
    settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    collection_name = settings.resolved_rag_collection_name()

    assert (
        collection_name
        == "agentic_rag_docs__huggingface__sentence_transformers_all_minilm_l6_v2"
    )


def test_resolved_rag_collection_name_uses_explicit_override():
    settings.rag_collection_name = "custom_collection"
    settings.embedding_provider = "huggingface"
    settings.embedding_model = "Sentence-Transformers/All-MiniLM-L6-v2"

    collection_name = settings.resolved_rag_collection_name()

    assert collection_name == "custom_collection"


def test_resolved_rag_collection_name_sanitizes_model_name():
    settings.rag_collection_name = None
    settings.embedding_provider = "huggingface"
    settings.embedding_model = "Sentence-Transformers/All-MiniLM-L6-v2"

    collection_name = settings.resolved_rag_collection_name()

    assert (
        collection_name
        == "agentic_rag_docs__huggingface__sentence_transformers_all_minilm_l6_v2"
    )


def test_get_vector_store_uses_resolved_collection_name_and_cache_clear(monkeypatch):
    created: list[tuple[str, str]] = []

    class FakeStore:
        def __init__(self, *, persist_dir: str, collection_name: str) -> None:
            self.persist_dir = persist_dir
            self.collection_name = collection_name
            created.append((persist_dir, collection_name))

    monkeypatch.setitem(
        sys.modules,
        "src.infrastructure.vector_db.chroma_vectorstore",
        SimpleNamespace(ChromaVectorStore=FakeStore),
    )

    workspace_tmp = tempfile.mkdtemp(prefix="vector-store-", dir=".")
    try:
        settings.chroma_persist_dir = workspace_tmp
        settings.rag_collection_name = None
        settings.embedding_provider = "openai"
        settings.embedding_model = "text-embedding-3-small"

        first_store = deps.get_vector_store()

        settings.embedding_provider = "huggingface"
        settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        deps.get_vector_store.cache_clear()
        second_store = deps.get_vector_store()

        assert first_store.collection_name == "agentic_rag_docs__openai__text_embedding_3_small"
        assert (
            second_store.collection_name
            == "agentic_rag_docs__huggingface__sentence_transformers_all_minilm_l6_v2"
        )
        assert created == [
            (workspace_tmp, "agentic_rag_docs__openai__text_embedding_3_small"),
            (
                workspace_tmp,
                "agentic_rag_docs__huggingface__sentence_transformers_all_minilm_l6_v2",
            ),
        ]
    finally:
        shutil.rmtree(workspace_tmp, ignore_errors=True)

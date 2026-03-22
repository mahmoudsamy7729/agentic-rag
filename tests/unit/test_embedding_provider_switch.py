import pytest

from src.api.v1 import dependencies as deps
from src.infrastructure.llm.huggingface_embeddings import HuggingFaceEmbeddingProvider
from src.infrastructure.llm.openai_embeddings import OpenAIEmbeddingProvider
from src.settings.config import settings


@pytest.fixture(autouse=True)
def _restore_settings_and_cache():
    fields = [
        "embedding_provider",
        "openai_key",
        "embedding_model",
        "embedding_base_url",
    ]
    snapshot = {field: getattr(settings, field) for field in fields}
    deps.get_embedding_provider.cache_clear()
    try:
        yield
    finally:
        for field, value in snapshot.items():
            setattr(settings, field, value)
        deps.get_embedding_provider.cache_clear()


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

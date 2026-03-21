import ast

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("chromadb")

from main import app
from src.agents import AgentService
from src.api.v1 import dependencies as deps
from src.rag.models import RetrievedChunk
from src.rag.pipeline import IngestionResult
from src.shared.interfaces.llm import (
    ChatMessage,
    GenerationConfig,
    LLM,
    LLMResponse,
    MessageRole,
    ToolCall,
)
from src.tools import PingTool, RetrieverTool, ToolRegistry


class InMemoryRAGStore:
    def __init__(self) -> None:
        self.items: list[dict] = []


class FakeIngestionService:
    def __init__(self, store: InMemoryRAGStore) -> None:
        self._store = store

    async def ingest_text(self, *, text: str, source: str | None = None, doc_id: str | None = None):
        resolved_doc_id = doc_id or f"doc-{len(self._store.items) + 1}"
        self._store.items.append(
            {
                "doc_id": resolved_doc_id,
                "chunk_id": "chunk-0",
                "source": source or "inline-text",
                "text": text,
            }
        )
        return IngestionResult(doc_id=resolved_doc_id, chunks_ingested=1)


class FakeRetrievalService:
    def __init__(self, store: InMemoryRAGStore) -> None:
        self._store = store

    async def retrieve(self, *, query: str, top_k: int | None = None):
        lowered_query = query.lower()
        hits = [
            item
            for item in self._store.items
            if any(token in item["text"].lower() for token in lowered_query.split())
        ]
        limit = top_k or 4
        return [
            RetrievedChunk(
                doc_id=item["doc_id"],
                chunk_id=item["chunk_id"],
                source=item["source"],
                text=item["text"],
                score=0.9,
            )
            for item in hits[:limit]
        ]


class FakeLLM(LLM):
    @property
    def model_name(self) -> str:
        return "fake-llm"

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        config: GenerationConfig | None = None,
        tools=None,
    ) -> LLMResponse:
        has_tool_result = any(
            msg.role == MessageRole.TOOL and msg.name == "retrieve_context"
            for msg in messages
        )

        if not has_tool_result:
            user_question = next(
                (msg.content for msg in reversed(messages) if msg.role == MessageRole.USER),
                "",
            )
            return LLMResponse(
                content="",
                model=self.model_name,
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="retrieve_context",
                        arguments={"query": user_question, "top_k": 4},
                    )
                ],
            )

        tool_message = next(
            (
                msg
                for msg in reversed(messages)
                if msg.role == MessageRole.TOOL and msg.name == "retrieve_context"
            ),
            None,
        )

        results: list[dict] = []
        if tool_message:
            try:
                payload = ast.literal_eval(tool_message.content)
                if isinstance(payload, dict):
                    results = payload.get("results", []) or []
            except (ValueError, SyntaxError):
                results = []

        if results:
            answer = f"Based on retrieved context: {results[0].get('text', '')}"
        else:
            answer = "No relevant context found."

        return LLMResponse(content=answer, model=self.model_name)

    async def stream(self, messages, *, config=None, tools=None):
        if False:
            yield ""


def _build_client():
    store = InMemoryRAGStore()
    ingestion_service = FakeIngestionService(store)
    retrieval_service = FakeRetrievalService(store)
    registry = ToolRegistry(
        [
            PingTool(),
            RetrieverTool(retrieval_service=retrieval_service, default_top_k=4),
        ]
    )
    llm = FakeLLM()

    app.dependency_overrides[deps.get_rag_ingestion_service] = lambda: ingestion_service
    app.dependency_overrides[deps.get_tool_registry] = lambda: registry
    app.dependency_overrides[deps.get_llm] = lambda: llm
    app.dependency_overrides[deps.get_agent_service] = lambda: AgentService(
        llm=llm,
        registry=registry,
        max_steps=4,
    )

    return TestClient(app)


def test_end_to_end_rag_flow_with_citations():
    client = _build_client()
    try:
        llm_health = client.get("/llm/health")
        tools_health = client.get("/tools/health")

        assert llm_health.status_code == 200
        assert llm_health.json()["llm_ok"] is True
        assert tools_health.status_code == 200
        assert tools_health.json()["tools_ok"] is True

        ask_before_ingest = client.post(
            "/agent/ask",
            json={"question": "What is the capital of France?", "session_id": "s-1"},
        )
        before_body = ask_before_ingest.json()
        assert ask_before_ingest.status_code == 200
        assert before_body["status"] == "ok"
        assert before_body["citations"] == []

        ingest = client.post(
            "/rag/ingest/text",
            json={
                "text": "Paris is the capital of France.",
                "source": "wiki-france",
                "doc_id": "doc-fr",
            },
        )
        ingest_body = ingest.json()
        assert ingest.status_code == 200
        assert ingest_body["status"] == "ok"
        assert ingest_body["chunks_ingested"] == 1

        ask_after_ingest = client.post(
            "/agent/ask",
            json={"question": "What is the capital of France?", "session_id": "s-1"},
        )
        after_body = ask_after_ingest.json()
        assert ask_after_ingest.status_code == 200
        assert after_body["status"] == "ok"
        assert len(after_body["citations"]) >= 1
        assert after_body["citations"][0]["source"] == "wiki-france"
    finally:
        app.dependency_overrides.clear()


def test_ingest_empty_text_returns_422():
    client = _build_client()
    try:
        response = client.post(
            "/rag/ingest/text",
            json={"text": "", "source": "invalid"},
        )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()

import json
import logging
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi_users")

from main import app
from src.agents.ask_pipeline import AgentAskPipelineResult
from src.api.v1 import dependencies as deps
from src.modules.users.dependencies import active_user
from src.shared.tracing import TRACE_LOGGER_NAME


@dataclass
class _User:
    id: UUID
    is_active: bool = True


class _SuccessfulPipeline:
    def __init__(self) -> None:
        self.request_id: str | None = None

    async def ask(
        self,
        *,
        owner_user_id,
        question: str,
        doc_id: str,
        session_id: str | None = None,
        use_cache: bool = True,
        request_id: str | None = None,
    ) -> AgentAskPipelineResult:
        self.request_id = request_id
        return AgentAskPipelineResult(
            status="ok",
            cache_status="miss",
            refined_query="refined q",
            answer="answer",
            steps=1,
            tools_used=["retrieve_context"],
            citations=[],
        )


class _FailingPipeline:
    async def ask(
        self,
        *,
        owner_user_id,
        question: str,
        doc_id: str,
        session_id: str | None = None,
        use_cache: bool = True,
        request_id: str | None = None,
    ) -> AgentAskPipelineResult:
        raise RuntimeError("boom")


def _trace_events(caplog) -> list[dict]:
    return [
        json.loads(record.message)
        for record in caplog.records
        if record.name == TRACE_LOGGER_NAME
    ]


def test_agent_route_reuses_request_id_and_logs_success(caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER_NAME)
    pipeline = _SuccessfulPipeline()
    user = _User(id=uuid4())
    client = TestClient(app)
    app.dependency_overrides[deps.get_agent_ask_pipeline] = lambda: pipeline
    app.dependency_overrides[active_user] = lambda: user
    try:
        response = client.post(
            "/agent/ask",
            headers={"X-Request-ID": "req-route-1"},
            json={
                "question": "What is the capital?",
                "doc_id": "doc-1",
                "session_id": "s1",
            },
        )
        assert response.status_code == 200
        assert pipeline.request_id == "req-route-1"
        events = _trace_events(caplog)
        assert [event["event"] for event in events] == [
            "ask.request.started",
            "ask.request.succeeded",
        ]
        assert all(event["request_id"] == "req-route-1" for event in events)
    finally:
        app.dependency_overrides.clear()


def test_agent_route_logs_failure(caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER_NAME)
    user = _User(id=uuid4())
    client = TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides[deps.get_agent_ask_pipeline] = lambda: _FailingPipeline()
    app.dependency_overrides[active_user] = lambda: user
    try:
        response = client.post(
            "/agent/ask",
            headers={"X-Request-ID": "req-route-2"},
            json={
                "question": "What is the capital?",
                "doc_id": "doc-1",
                "session_id": "s1",
            },
        )
        assert response.status_code == 500
        events = _trace_events(caplog)
        assert [event["event"] for event in events] == [
            "ask.request.started",
            "ask.request.failed",
        ]
        assert events[-1]["request_id"] == "req-route-2"
        assert events[-1]["error"] == "boom"
        assert events[-1]["status_code"] == 500
    finally:
        app.dependency_overrides.clear()

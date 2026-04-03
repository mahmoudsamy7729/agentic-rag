from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from src.agents.ask_pipeline import AgentAskPipeline
from src.agents.service import AgentCitation, AgentResult
from src.shared.tracing import TRACE_LOGGER_NAME


@dataclass
class _Doc:
    id: str
    owner_user_id: object
    last_indexed_at: datetime | None


class _DocsRepo:
    async def get_owned_document(self, *, owner_user_id, doc_id: str, include_deleted: bool = False):
        return _Doc(id=doc_id, owner_user_id=owner_user_id, last_indexed_at=datetime.now(timezone.utc))


class _Agent:
    async def run(
        self,
        *,
        question: str,
        doc_id: str,
        session_id: str | None = None,
        user_id: str | None = None,
        request_id: str | None = None,
    ):
        return AgentResult(
            answer="answer",
            steps=1,
            tools_used=["retrieve_context"],
            status="ok",
            citations=[
                AgentCitation(
                    source="inline",
                    doc_id=doc_id,
                    chunk_id="chunk-1",
                    snippet="snippet",
                    page_number=None,
                )
            ],
        )


class _LLM:
    @property
    def model_name(self) -> str:
        return "model-x"


class _Refiner:
    class _Out:
        refined_query = "refined q"

    async def refine(self, *, question: str, doc_id: str):
        return self._Out()


class _Embed:
    def __init__(self) -> None:
        self.calls = 0

    async def embed_query(self, text: str):
        self.calls += 1
        return [0.1, 0.2]


class _Cache:
    def __init__(self) -> None:
        self.enabled = True
        self.lookup_calls = 0
        self.store_calls = 0

    @staticmethod
    def normalize_question(question: str) -> str:
        return question

    async def lookup(self, **kwargs):
        self.lookup_calls += 1
        return None

    async def store(self, **kwargs):
        self.store_calls += 1


def _trace_events(caplog) -> list[dict]:
    return [
        json.loads(record.message)
        for record in caplog.records
        if record.name == TRACE_LOGGER_NAME
    ]


def test_ask_pipeline_use_cache_false_skips_cache_lookup_and_store():
    embed = _Embed()
    cache = _Cache()
    pipeline = AgentAskPipeline(
        agent_service=_Agent(),
        llm=_LLM(),
        query_refinement_service=_Refiner(),
        embedding_provider=embed,
        semantic_cache_service=cache,
        documents_repository=_DocsRepo(),
    )

    result = asyncio.run(
        pipeline.ask(
            owner_user_id=uuid4(),
            question="q",
            doc_id="doc-1",
            session_id="s1",
            use_cache=False,
        )
    )

    assert result.cache_status == "miss"
    assert cache.lookup_calls == 0
    assert cache.store_calls == 0
    assert embed.calls == 0


def test_ask_pipeline_emits_traces_without_chunk_text(caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER_NAME)
    embed = _Embed()
    cache = _Cache()
    pipeline = AgentAskPipeline(
        agent_service=_Agent(),
        llm=_LLM(),
        query_refinement_service=_Refiner(),
        embedding_provider=embed,
        semantic_cache_service=cache,
        documents_repository=_DocsRepo(),
    )

    result = asyncio.run(
        pipeline.ask(
            owner_user_id=uuid4(),
            question="q",
            doc_id="doc-1",
            session_id="s1",
            use_cache=True,
            request_id="req-pipeline",
        )
    )

    assert result.cache_status == "miss"
    events = _trace_events(caplog)
    assert [event["event"] for event in events] == [
        "ask.document.checked",
        "ask.query.refined",
        "ask.cache.lookup.started",
        "ask.cache.miss",
        "ask.agent.run.started",
        "ask.agent.run.completed",
        "ask.cache.store.succeeded",
    ]
    assert all(event["request_id"] == "req-pipeline" for event in events)
    assert events[1]["question"] == "q"
    assert events[1]["refined_query"] == "refined q"
    joined_logs = "\n".join(record.message for record in caplog.records)
    assert "\"snippet\"" not in joined_logs
    assert "\"answer\"" not in joined_logs

import asyncio
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

pytest.importorskip("fastapi_users")

from src.infrastructure.database import Base
from src.modules.documents.models import Document
from src.modules.evaluation.judge import ContextRelevanceJudgeResult
from src.modules.evaluation.repository import EvaluationRepository
from src.modules.evaluation.service import (
    RetrievalEvaluationRunConfig,
    RetrievalEvaluationService,
)
from src.modules.users.models import User
from src.rag.models import RetrievedChunk


class FakeRetriever:
    async def retrieve(self, *, question: str, file_id: str, k: int):
        return [
            RetrievedChunk(
                doc_id=file_id,
                chunk_id="chunk-1",
                source="policy",
                text="Users can request a refund within 30 days.",
                score=0.9,
            ),
            RetrievedChunk(
                doc_id=file_id,
                chunk_id="chunk-2",
                source="policy",
                text="Refunds apply only when the subscription was not used.",
                score=0.8,
            ),
        ][:k]


class FakeJudge:
    async def judge(self, *, question: str, retrieved_chunks: list[dict[str, str]]):
        return ContextRelevanceJudgeResult(
            score=5,
            explanation="The retrieved chunks are sufficient.",
        )


class FlakyRetriever:
    def __init__(self) -> None:
        self._attempts: dict[str, int] = {}

    async def retrieve(self, *, question: str, file_id: str, k: int):
        self._attempts[question] = self._attempts.get(question, 0) + 1
        if "support" in question.lower() and self._attempts[question] == 1:
            raise RuntimeError("temporary retriever failure")
        return [
            RetrievedChunk(
                doc_id=file_id,
                chunk_id="chunk-1",
                source="policy",
                text="Users can request a refund within 30 days.",
                score=0.9,
            ),
            RetrievedChunk(
                doc_id=file_id,
                chunk_id="chunk-2",
                source="policy",
                text="Refunds apply only when the subscription was not used.",
                score=0.8,
            ),
        ][:k]


def test_retrieval_evaluation_service_persists_completed_run_and_case():
    temp_db = Path(tempfile.mkdtemp(prefix="eval-service-", dir=".")) / "test.db"
    database_url = f"sqlite+aiosqlite:///{temp_db.as_posix()}"
    owner_id = uuid4()

    async def _run():
        engine = create_async_engine(database_url)
        session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            session.add(
                User(
                    id=owner_id,
                    email="owner@example.com",
                    hashed_password="hashed",
                    is_active=True,
                    is_superuser=False,
                    is_verified=False,
                )
            )
            session.add(
                Document(
                    id="doc-1",
                    owner_user_id=owner_id,
                    source="policy",
                )
            )
            await session.commit()

        service = RetrievalEvaluationService(
            session_factory=session_factory,
            dataset_storage_dir=str(Path(tempfile.mkdtemp(prefix="eval-data-", dir="."))),
            retriever_factory=lambda: FakeRetriever(),
            judge_factory=lambda: FakeJudge(),
        )
        config = RetrievalEvaluationRunConfig(
            k=2,
            strip_punctuation=True,
            min_keyword_hits=2,
            min_keyword_ratio=0.4,
            store_retrieved_chunk_texts=True,
            judge_enabled=True,
            rag_top_k=12,
            rag_prefetch_k=80,
            embedding_provider="huggingface",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            reranker_enabled=False,
            reranker_model=None,
            judge_model="fake-judge",
        )
        raw_dataset = (
            b'{"question":"What is the refund policy?","answer":"Users can request a refund within 30 days if the subscription was not used.","must_include_keywords":["refund","30","days","subscription","used"],"must_include_phrases":["refund within 30 days","subscription was not used"],"difficulty":"easy","category":"billing"}\n'
        )
        created = await service.create_run_from_upload(
            owner_user_id=owner_id,
            file_id="doc-1",
            dataset_name="dataset.jsonl",
            dataset_bytes=raw_dataset,
            config=config,
        )

        await service.process_run(run_id=created.run_id)

        async with session_factory() as session:
            repo = EvaluationRepository(session)
            run = await repo.get_owned_run(owner_user_id=owner_id, run_id=created.run_id)
            assert run is not None
            assert run.status == "completed"
            assert run.processed_cases == 1
            assert run.hit_at_k_avg == 1.0
            assert run.keyword_coverage_avg == 1.0
            assert run.context_relevance_score_avg == 5.0
            assert run.grouped_summary["category"]["billing"]["count"] == 1

            cases = await repo.list_cases_for_run(run_id=created.run_id)
            assert len(cases) == 1
            assert cases[0].status == "completed"
            assert cases[0].retrieved_chunk_ids == ["chunk-1", "chunk-2"]
            assert len(cases[0].retrieved_chunk_texts) == 2
            assert cases[0].context_relevance_score == 5

        await engine.dispose()

    asyncio.run(_run())


def test_retrieval_evaluation_service_reruns_failed_cases_in_place():
    temp_db = Path(tempfile.mkdtemp(prefix="eval-rerun-service-", dir=".")) / "test.db"
    database_url = f"sqlite+aiosqlite:///{temp_db.as_posix()}"
    owner_id = uuid4()

    async def _run():
        engine = create_async_engine(database_url)
        session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            session.add(
                User(
                    id=owner_id,
                    email="owner@example.com",
                    hashed_password="hashed",
                    is_active=True,
                    is_superuser=False,
                    is_verified=False,
                )
            )
            session.add(Document(id="doc-1", owner_user_id=owner_id, source="policy"))
            await session.commit()

        retriever = FlakyRetriever()
        service = RetrievalEvaluationService(
            session_factory=session_factory,
            dataset_storage_dir=str(Path(tempfile.mkdtemp(prefix="eval-rerun-data-", dir="."))),
            retriever_factory=lambda: retriever,
            judge_factory=lambda: FakeJudge(),
        )
        config = RetrievalEvaluationRunConfig(
            k=2,
            strip_punctuation=True,
            min_keyword_hits=2,
            min_keyword_ratio=0.4,
            store_retrieved_chunk_texts=True,
            judge_enabled=True,
            rag_top_k=12,
            rag_prefetch_k=80,
            embedding_provider="huggingface",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            reranker_enabled=False,
            reranker_model=None,
            judge_model="fake-judge",
        )
        raw_dataset = (
            b'{"question":"What is the refund policy?","answer":"Users can request a refund within 30 days if the subscription was not used.","must_include_keywords":["refund","30","days","subscription","used"],"must_include_phrases":["refund within 30 days","subscription was not used"],"difficulty":"easy","category":"billing"}\n'
            b'{"question":"What are the support hours?","answer":"Support hours are Monday to Friday.","must_include_keywords":["support","monday","friday"],"must_include_phrases":["Monday to Friday"],"difficulty":"easy","category":"support"}\n'
        )
        created = await service.create_run_from_upload(
            owner_user_id=owner_id,
            file_id="doc-1",
            dataset_name="dataset.jsonl",
            dataset_bytes=raw_dataset,
            config=config,
        )

        await service.process_run(run_id=created.run_id)

        async with session_factory() as session:
            repo = EvaluationRepository(session)
            run = await repo.get_owned_run(owner_user_id=owner_id, run_id=created.run_id)
            assert run is not None
            assert run.status == "completed"
            assert run.processed_cases == 2
            cases = await repo.list_cases_for_run(run_id=created.run_id)
            completed_case = next(case for case in cases if case.question.startswith("What is the refund"))
            failed_case = next(case for case in cases if case.question.startswith("What are the support"))
            assert completed_case.status == "completed"
            assert failed_case.status == "failed"
            completed_chunk_ids = list(completed_case.retrieved_chunk_ids)

        rerun = await service.rerun_failed_cases(owner_user_id=owner_id, run_id=created.run_id)
        assert rerun.rerun_case_count == 1

        async with session_factory() as session:
            repo = EvaluationRepository(session)
            run = await repo.get_owned_run(owner_user_id=owner_id, run_id=created.run_id)
            assert run is not None
            assert run.status == "queued"
            assert run.processed_cases == 1
            cases = await repo.list_cases_for_run(run_id=created.run_id)
            completed_case = next(case for case in cases if case.question.startswith("What is the refund"))
            reset_case = next(case for case in cases if case.question.startswith("What are the support"))
            assert completed_case.status == "completed"
            assert completed_case.retrieved_chunk_ids == completed_chunk_ids
            assert reset_case.status == "queued"
            assert reset_case.retrieved_chunk_ids == []
            assert reset_case.error_message is None

        await service.process_selected_cases(run_id=created.run_id, case_ids=rerun.case_ids)

        async with session_factory() as session:
            repo = EvaluationRepository(session)
            run = await repo.get_owned_run(owner_user_id=owner_id, run_id=created.run_id)
            assert run is not None
            assert run.status == "completed"
            assert run.processed_cases == 2
            assert run.grouped_summary["category"]["billing"]["count"] == 1
            assert run.grouped_summary["category"]["support"]["count"] == 1

            cases = await repo.list_cases_for_run(run_id=created.run_id)
            assert all(case.status == "completed" for case in cases)
            rerun_case = next(case for case in cases if case.question.startswith("What are the support"))
            assert rerun_case.error_message is None
            assert rerun_case.retrieved_chunk_ids == ["chunk-1", "chunk-2"]

        await engine.dispose()

    asyncio.run(_run())

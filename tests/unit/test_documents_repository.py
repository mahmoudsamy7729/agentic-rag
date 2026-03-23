import asyncio
import tempfile
from pathlib import Path
from uuid import uuid4

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.infrastructure.database import Base
from src.modules.documents.repository import DocumentsRepository
from src.modules.users.models import User


def test_documents_repository_ownership_and_soft_delete():
    temp_db = Path(tempfile.mkdtemp(prefix="docs-repo-", dir=".")) / "test.db"
    database_url = f"sqlite+aiosqlite:///{temp_db.as_posix()}"

    async def _run():
        engine = create_async_engine(database_url)
        session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        owner_id = uuid4()
        other_user_id = uuid4()

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
                User(
                    id=other_user_id,
                    email="other@example.com",
                    hashed_password="hashed",
                    is_active=True,
                    is_superuser=False,
                    is_verified=False,
                )
            )
            await session.commit()

        async with session_factory() as session:
            repo = DocumentsRepository(session)
            await repo.create_document(
                owner_user_id=owner_id,
                doc_id="doc-1",
                source="inline-text",
            )
            await repo.commit()

            assert await repo.doc_id_exists(doc_id="doc-1", include_deleted=True) is True
            owned = await repo.get_owned_document(
                owner_user_id=owner_id,
                doc_id="doc-1",
                include_deleted=False,
            )
            assert owned is not None
            assert owned.owner_user_id == owner_id

            wrong_owner = await repo.get_owned_document(
                owner_user_id=other_user_id,
                doc_id="doc-1",
                include_deleted=False,
            )
            assert wrong_owner is None

            listed = await repo.list_owned_documents(owner_user_id=owner_id, limit=10, offset=0)
            assert len(listed) == 1
            assert listed[0].id == "doc-1"

            deleted = await repo.soft_delete_owned_document(owner_user_id=owner_id, doc_id="doc-1")
            assert deleted is not None
            await repo.commit()

            active_after_delete = await repo.get_owned_document(
                owner_user_id=owner_id,
                doc_id="doc-1",
                include_deleted=False,
            )
            assert active_after_delete is None

            with_deleted = await repo.get_owned_document(
                owner_user_id=owner_id,
                doc_id="doc-1",
                include_deleted=True,
            )
            assert with_deleted is not None
            assert with_deleted.deleted_at is not None

            listed_after_delete = await repo.list_owned_documents(
                owner_user_id=owner_id,
                limit=10,
                offset=0,
            )
            assert listed_after_delete == []

        await engine.dispose()

    asyncio.run(_run())

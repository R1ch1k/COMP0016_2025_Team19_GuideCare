import os
import sys
import asyncio
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.db import models

# On Windows the default event loop policy is Proactor which can cause issues
# with asyncpg/greenlet interplay. Use the selector policy for compatibility.
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

@pytest.fixture(scope="session")
def event_loop():
    # Use a fresh event loop for the test session and set it as current.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    # Ensure async generators are shut down and loop is closed.
    try:
        loop.run_until_complete(loop.shutdown_asyncgens())
    finally:
        loop.close()

@pytest.fixture(scope="session")
async def engine():
    """Create an async engine for tests.

    Behavior:
      - If TEST_DATABASE_URL is set, use that database.
      - Otherwise, attempt to start a temporary Postgres container using Testcontainers.
    """
    container = None
    db_url = TEST_DATABASE_URL

    # Default testing behavior: use an in-memory SQLite DB unless TEST_DATABASE_URL
    # is provided or TESTCONTAINERS env var is set to force starting a Postgres
    # container. This avoids Docker-related async issues on Windows by default.
    use_testcontainers = os.environ.get("USE_TESTCONTAINERS")

    if not db_url and not use_testcontainers:
        # Use in-memory sqlite for fast, local unit tests
        db_url = "sqlite+aiosqlite:///:memory:"
        # Use StaticPool to make the in-memory DB accessible across connections
        engine = create_async_engine(
            db_url,
            future=True,
            connect_args={"check_same_thread": False},
            poolclass=__import__("sqlalchemy.pool").pool.StaticPool,
        )

        # Normalize Postgres-only types (JSONB, UUID) to generic types so our
        # models work on SQLite too.
        from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB, UUID as PG_UUID
        from sqlalchemy import JSON as GEN_JSON, String as GEN_STRING

        def normalize_metadata_for_sqlite(metadata):
            for table in metadata.tables.values():
                for col in list(table.columns):
                    t = col.type
                    if isinstance(t, PG_JSONB):
                        col.type = GEN_JSON()
                    elif isinstance(t, PG_UUID):
                        col.type = GEN_STRING(36)
                        # Replace the column default with a callable that returns a
                        # string UUID so SQLite can bind the value.
                        try:
                            from sqlalchemy.sql.schema import ColumnDefault
                            import uuid
                            col.default = ColumnDefault(lambda: str(uuid.uuid4()))
                        except Exception:
                            pass

        # Ensure UUID Python objects are coerced to strings before flushing when using SQLite
        import uuid
        from sqlalchemy import event
        from sqlalchemy.orm import Session as OrmSession

        def _coerce_uuid_before_flush(session, flush_context, instances):
            for obj in list(session.new) + list(session.dirty):
                for attr in ("id", "patient_id", "conversation_id"):
                    val = getattr(obj, attr, None)
                    if isinstance(val, uuid.UUID):
                        setattr(obj, attr, str(val))

        # Install the listener on the ORM Session class so the sync session
        # used internally by AsyncSession will convert UUID objects to strings.
        event.listen(OrmSession, "before_flush", _coerce_uuid_before_flush)

        # Coerce UUID bind parameters in SQL queries for SQLite compatibility.
        # SQLAlchemy's UUID type converts to .hex (no hyphens), but we store
        # hyphenated strings — so we re-format 32-char hex strings to 36-char.
        import re
        _hex_uuid_re = re.compile(r'^[0-9a-f]{32}$')

        def _fix_param(v):
            if isinstance(v, uuid.UUID):
                return str(v)
            if isinstance(v, str) and _hex_uuid_re.match(v):
                return str(uuid.UUID(v))
            return v

        @event.listens_for(engine.sync_engine, "before_cursor_execute", retval=True)
        def _coerce_uuid_params(conn, cursor, statement, parameters, context, executemany):
            if parameters:
                if isinstance(parameters, dict):
                    parameters = {k: _fix_param(v) for k, v in parameters.items()}
                elif isinstance(parameters, (list, tuple)):
                    parameters = tuple(_fix_param(v) for v in parameters)
            return statement, parameters
        # create tables once per test session using normalized metadata
        async with engine.begin() as conn:
            await conn.run_sync(lambda sync_conn: (normalize_metadata_for_sqlite(models.Base.metadata), models.Base.metadata.create_all(sync_conn)))

    else:
        # Use TEST_DATABASE_URL or Testcontainers if requested
        if not db_url and use_testcontainers:
            try:
                from testcontainers.postgres import PostgresContainer
            except Exception as e:
                raise RuntimeError(
                    "USE_TESTCONTAINERS set but testcontainers is not available. "
                    "Install testcontainers and ensure Docker is running."
                ) from e

            # Start a temporary Postgres container (postgres:15)
            container = PostgresContainer("postgres:15")
            container.start()
            connection_url = container.get_connection_url()
            if connection_url.startswith("postgresql+"):
                db_url = "postgresql+asyncpg://" + connection_url.split("://", 1)[1]
            else:
                db_url = connection_url.replace("postgresql://", "postgresql+asyncpg://")

        engine = create_async_engine(db_url, future=True)

        # create tables once per test session
        async with engine.begin() as conn:
            await conn.run_sync(models.Base.metadata.create_all)

    yield engine

    # Attempt cleanup: drop tables if possible, dispose engine, and stop container
    try:
        async with engine.begin() as conn:
            await conn.run_sync(models.Base.metadata.drop_all)
    except Exception:
        pass
    finally:
        await engine.dispose()

    if container:
        try:
            container.stop()
        except Exception:
            pass


@pytest.fixture
async def db_session(engine):
    _AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with _AsyncSessionLocal() as session:
        yield session
        # rollback any lingering transaction state between tests
        await session.rollback()


@pytest.fixture
async def async_client(engine):
    """httpx AsyncClient wired to the FastAPI app with DB override."""
    from httpx import AsyncClient, ASGITransport
    from main import app
    from app.db.session import get_async_session

    _AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _override_session():
        async with _AsyncSessionLocal() as session:
            yield session

    app.dependency_overrides[get_async_session] = _override_session

    # Suppress the startup event (which tries to connect to real DB)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()

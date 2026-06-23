"""
Shared test fixtures.

Uses an in-memory SQLite database so tests are hermetic and fast.
"""
import os
import pytest
import pytest_asyncio

# Point at in-memory SQLite before any app code imports
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("API_KEY", "test-key-123")
os.environ.setdefault("WS_TOKEN", "test-ws-token")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("ANTHROPIC_API_KEY", "")  # no real calls in tests


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture(scope="function")
async def db_session():
    """Fresh in-memory DB session per test."""
    from database import engine, Base
    import models.orm  # noqa: F401 — register all models

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        from database import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            yield session
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

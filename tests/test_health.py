"""
Health and readiness endpoint tests.
These are the first things a load balancer checks — they must always pass.
"""
import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_health_returns_200():
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_health_requires_no_auth():
    """Health endpoint must be accessible without an API key."""
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_readiness_path_exists():
    """Readiness endpoint must exist (may return 503 in test env, but must not 404)."""
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/readiness")
    assert resp.status_code in (200, 503)  # 503 is acceptable if DB not ready

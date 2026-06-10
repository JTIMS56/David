"""
API key authentication tests.
Every /api/* endpoint must reject requests without a valid key.
"""
import os
import pytest
from httpx import ASGITransport, AsyncClient

_VALID_KEY  = os.environ.get("API_KEY", "test-key-123")
_WRONG_KEY  = "definitely-wrong-key"


async def _client():
    from main import app
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_api_requires_key_when_configured():
    """Without key → 401."""
    async with await _client() as client:
        resp = await client.get("/api/status")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_api_rejects_wrong_key():
    """Wrong key → 401."""
    async with await _client() as client:
        resp = await client.get("/api/status", headers={"X-API-Key": _WRONG_KEY})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_api_accepts_valid_header_key():
    """Correct X-API-Key → not 401."""
    async with await _client() as client:
        resp = await client.get("/api/status", headers={"X-API-Key": _VALID_KEY})
    assert resp.status_code != 401


@pytest.mark.asyncio
async def test_api_accepts_bearer_token():
    """Correct Bearer token → not 401."""
    async with await _client() as client:
        resp = await client.get(
            "/api/status",
            headers={"Authorization": f"Bearer {_VALID_KEY}"}
        )
    assert resp.status_code != 401


@pytest.mark.asyncio
async def test_health_bypasses_auth():
    """/health must never require authentication."""
    async with await _client() as client:
        resp = await client.get("/health")
    assert resp.status_code == 200

"""
API key authentication for all /api/* routes.

Usage:
  Set API_KEY in .env (or environment).
  Every request to a protected route must include:
    Header:  X-API-Key: <your-key>
    OR
    Header:  Authorization: Bearer <your-key>

  If API_KEY is empty (default), the dependency is a no-op
  so development works without configuration. In production
  API_KEY must be set — startup will log a warning if it isn't.
"""
from __future__ import annotations

import logging
import secrets

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from config import settings

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_scheme   = HTTPBearer(auto_error=False)


def _constant_time_eq(a: str, b: str) -> bool:
    """Timing-safe string comparison."""
    return secrets.compare_digest(a.encode(), b.encode())


async def require_api_key(
    header_key: str | None = Security(_api_key_header),
    bearer:     HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> None:
    """
    FastAPI dependency — attach to any route or router that needs protection.

    Accepts key via:
      X-API-Key: <key>           (preferred)
      Authorization: Bearer <key>  (alternative, for OpenAPI clients)
    """
    configured_key = settings.api_key
    if not configured_key:
        # Dev mode — warn once, let through
        return

    # Extract presented key from either header
    presented = header_key or (bearer.credentials if bearer else None)

    if not presented or not _constant_time_eq(presented, configured_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

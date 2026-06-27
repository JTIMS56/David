from __future__ import annotations

import ssl
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from config import settings


def _normalize_db_url(url: str) -> tuple[str, dict]:
    """
    Prepare a DATABASE_URL for SQLAlchemy asyncpg.
    - Rewrite postgresql:// -> postgresql+asyncpg://
    - Strip sslmode query param (asyncpg uses an ssl= connect_arg instead)
    """
    connect_args: dict = {}

    # SQLite (local/dev) needs no normalization — and urlunparse would mangle
    # its triple-slash (sqlite:///./x.db -> sqlite:/./x.db), so pass it through.
    if url.startswith("sqlite"):
        return url, connect_args

    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            url = "postgresql+asyncpg://" + url[len(prefix):]
            break

    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    sslmode = params.pop("sslmode", [None])[0]

    if sslmode and sslmode != "disable":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        connect_args["ssl"] = ctx

    new_query = urlencode({k: v[0] for k, v in params.items()})
    url = urlunparse(parsed._replace(query=new_query))

    return url, connect_args


_db_url, _connect_args = _normalize_db_url(settings.database_url)
engine = create_async_engine(_db_url, echo=False, connect_args=_connect_args)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def _migrate_schema(engine) -> None:
    """Add columns introduced after initial schema creation."""
    async with engine.begin() as conn:
        db_url = str(engine.url)
        if "sqlite" in db_url:
            result = await conn.execute(text("PRAGMA table_info(positions)"))
            cols = [row[1] for row in result.fetchall()]
            if "oanda_trade_id" not in cols:
                await conn.execute(
                    text("ALTER TABLE positions ADD COLUMN oanda_trade_id VARCHAR(20)")
                )
        else:
            # PostgreSQL: IF NOT EXISTS avoids errors on repeated startups
            await conn.execute(
                text("ALTER TABLE positions ADD COLUMN IF NOT EXISTS oanda_trade_id VARCHAR(20)")
            )
            # forecast_logs: BS comparison columns added after initial schema
            for _col, _typ in [
                ("bs_expected_price",     "FLOAT"),
                ("bs_expected_direction", "VARCHAR(4)"),
                ("bs_direction_correct",  "BOOLEAN"),
            ]:
                await conn.execute(
                    text(f"ALTER TABLE forecast_logs ADD COLUMN IF NOT EXISTS {_col} {_typ}")
                )


async def init_db() -> None:
    from models.orm import Position, Trade, PriceTick, AgentDecision, PortfolioSnapshot, ForecastLog  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _migrate_schema(engine)

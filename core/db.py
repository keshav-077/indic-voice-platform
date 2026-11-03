"""Database setup using SQLAlchemy async. Optional — the API runs fine without a database."""

from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.config import settings

_engine = None
_session_factory = None


async def init_db() -> None:
    global _engine, _session_factory
    if not settings.VAI_DB_URL:
        return
    _engine = create_async_engine(
        settings.VAI_DB_URL,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        echo=settings.DEBUG,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


def is_db_available() -> bool:
    return _session_factory is not None


@asynccontextmanager
async def get_db_session():
    if _session_factory is None:
        raise RuntimeError("Database not configured. Set VAI_DB_URL or run without DB features.")
    async with _session_factory() as session:
        yield session

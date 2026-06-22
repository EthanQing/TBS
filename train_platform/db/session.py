from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import logging

from sqlalchemy import create_engine
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session, sessionmaker

from train_platform.core.config import settings


logger = logging.getLogger("train_platform.db")


engine = create_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
    pool_recycle=settings.db_pool_recycle,
    pool_pre_ping=settings.db_pool_pre_ping,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def pool_status() -> str:
    try:
        return engine.pool.status()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def log_pool_configuration() -> None:
    logger.info(
        "Database pool configured: size=%s max_overflow=%s timeout=%s recycle=%s pre_ping=%s status=%s",
        settings.db_pool_size,
        settings.db_max_overflow,
        settings.db_pool_timeout,
        settings.db_pool_recycle,
        settings.db_pool_pre_ping,
        pool_status(),
    )


def _log_timeout(exc: SQLAlchemyTimeoutError) -> None:
    logger.error("Database connection pool timeout: %s; pool_status=%s", exc, pool_status())


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except SQLAlchemyTimeoutError as exc:
        _log_timeout(exc)
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyTimeoutError as exc:
        _log_timeout(exc)
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

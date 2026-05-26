from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from train_platform.api.v3 import router as api_router
from train_platform.core.config import settings
from train_platform.core.license import assert_valid_license
from train_platform.db.init_db import init_db
from train_platform.db.session import SessionLocal
from train_platform.services.v3.dataset_upload_service import DatasetUploadService
from train_platform.utils.exceptions import ConflictError, NotFoundError, ValidationError


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_platform")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    assert_valid_license()
    try:
        init_db()
    except Exception as e:
        logger.error("Database init failed (did you run 'alembic -c alembic.ini upgrade head'?)")
        raise
    try:
        with SessionLocal() as db:
            cleaned = DatasetUploadService().cleanup_expired_sessions(db)
            if cleaned:
                logger.info("Cleaned %s expired dataset upload sessions on startup.", cleaned)
    except Exception as e:
        logger.warning("Failed to clean expired dataset upload sessions on startup: %s", e)
    yield


def create_app() -> FastAPI:
    # Ensure static directories exist before mounting (Starlette requires this at startup).
    settings.ensure_dirs()

    app = FastAPI(
        title="Train Platform Backend (v3)",
        version="3.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v3")

    app.mount("/static/datasets", StaticFiles(directory=str(settings.datasets_dir)), name="datasets")
    app.mount("/static/thumbnails", StaticFiles(directory=str(settings.thumbnails_dir)), name="thumbnails")
    app.mount("/static/training", StaticFiles(directory=str(settings.training_dir)), name="training")
    app.mount("/static/temp", StaticFiles(directory=str(settings.temp_dir)), name="temp")
    app.mount("/static/pretrain", StaticFiles(directory=str(settings.pretrain_models_dir)), name="pretrain")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.exception_handler(ValidationError)
    async def validation_error_handler(_request, exc: ValidationError):
        return JSONResponse(status_code=400, content={"message": "Validation error", "detail": str(exc)})

    @app.exception_handler(NotFoundError)
    async def not_found_error_handler(_request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"message": "Not found", "detail": str(exc)})

    @app.exception_handler(ConflictError)
    async def conflict_error_handler(_request, exc: ConflictError):
        return JSONResponse(status_code=409, content={"message": "Conflict", "detail": str(exc)})

    @app.exception_handler(500)
    async def internal_error_handler(_request, exc):  # pragma: no cover
        logger.error("Internal server error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error", "detail": "An unexpected error occurred"},
        )

    return app


app = create_app()

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()


def _default_home_dir() -> Path:
    # <repo>/backend_v2/train_platform/core/config.py -> parents[2] == <repo>/backend_v2
    return Path(__file__).resolve().parents[2]


def _csv_env(name: str, default: str = "") -> Tuple[str, ...]:
    raw = os.getenv(name, default)
    if raw is None:
        return tuple()
    items = [x.strip() for x in str(raw).split(",")]
    return tuple(x for x in items if x)


def _datetime_env(name: str) -> datetime | None:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return None
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO8601 datetime for {name}: {raw}") from exc
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class Settings:
    database_url_override: str | None = os.getenv("DATABASE_URL")

    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "password")
    mysql_host: str = os.getenv("MYSQL_HOST", "localhost")
    mysql_port: str = os.getenv("MYSQL_PORT", "3306")
    mysql_database: str = os.getenv("MYSQL_DATABASE", "train_backend_v2")

    home_dir: Path = Path(os.getenv("TRAIN_PLATFORM_HOME") or _default_home_dir()).resolve()

    datasets_dir: Path = Path(os.getenv("BASE_DATASETS_DIR") or (home_dir / "datasets")).resolve()
    training_dir: Path = Path(os.getenv("BASE_TRAINING_DIR") or (home_dir / "training_runs")).resolve()
    temp_dir: Path = Path(os.getenv("BASE_TEMP_DIR") or (home_dir / "temp")).resolve()
    pretrain_models_dir: Path = Path(os.getenv("BASE_PRETRAIN_MODELS_DIR") or (home_dir / "pretrain_models")).resolve()
    paddle_det_dir: Path = Path(os.getenv("PADDLE_DET_DIR") or (home_dir / "PaddleDetection")).resolve()
    disable_append_upload: bool = os.getenv("DISABLE_APPEND_UPLOAD", "1") in ("1", "true", "True")
    internal_api_token: str = os.getenv("INTERNAL_API_TOKEN", "")
    inference_max_download_bytes: int = int(os.getenv("INFERENCE_MAX_DOWNLOAD_BYTES", str(20 * 1024 * 1024)))
    inference_download_timeout_sec: float = float(os.getenv("INFERENCE_DOWNLOAD_TIMEOUT_SEC", "20"))
    inference_allowed_schemes: Tuple[str, ...] = _csv_env("INFERENCE_ALLOWED_SCHEMES", "http,https")
    inference_allowed_hosts: Tuple[str, ...] = _csv_env("INFERENCE_ALLOWED_HOSTS", "")
    worker_bind_host: str = os.getenv("WORKER_BIND_HOST", "").strip()
    thumbnail_max_workers: int = int(os.getenv("THUMBNAIL_MAX_WORKERS", "4"))
    thumbnail_first_page_prewarm: int = int(os.getenv("THUMBNAIL_FIRST_PAGE_PREWARM", "32"))
    thumbnail_size: int = int(os.getenv("THUMBNAIL_SIZE", "200"))
    view_index_max_workers: int = int(os.getenv("VIEW_INDEX_MAX_WORKERS", "8"))
    upload_chunk_size_mb: int = max(1, int(os.getenv("UPLOAD_CHUNK_SIZE_MB", "64") or "64"))
    upload_session_ttl_hours: int = max(1, int(os.getenv("UPLOAD_SESSION_TTL_HOURS", "24") or "24"))
    upload_part_max_retries: int = max(0, int(os.getenv("UPLOAD_PART_MAX_RETRIES", "5") or "5"))
    upload_max_parallel_parts: int = max(1, int(os.getenv("UPLOAD_MAX_PARALLEL_PARTS", "3") or "3"))
    software_not_before_at: datetime | None = _datetime_env("SOFTWARE_NOT_BEFORE_AT")
    software_not_after_at: datetime | None = _datetime_env("SOFTWARE_NOT_AFTER_AT")
    software_expires_at: datetime | None = _datetime_env("SOFTWARE_EXPIRES_AT")
    software_clock_rollback_tolerance_seconds: int = max(
        0,
        int(os.getenv("SOFTWARE_CLOCK_ROLLBACK_TOLERANCE_SECONDS", "300") or "300"),
    )
    software_guard_persist_interval_seconds: int = max(
        1,
        int(os.getenv("SOFTWARE_GUARD_PERSIST_INTERVAL_SECONDS", "60") or "60"),
    )

    @property
    def thumbnails_dir(self) -> Path:
        # Keep thumbnails under the datasets root so they can be managed together.
        return (self.datasets_dir / ".thumbnails").resolve()

    @property
    def usage_limit_state_path(self) -> Path:
        return (self.home_dir / ".runtime" / ".node_state.dat").resolve()

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return str(self.database_url_override)
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    def ensure_dirs(self) -> None:
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.pretrain_models_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()

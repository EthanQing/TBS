from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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


def _path_list_env(name: str) -> Tuple[str, ...]:
    raw = os.getenv(name, "")
    if raw is None:
        return tuple()
    # Use comma / semicolon separators. Avoid splitting on ":" because Windows
    # drive names use it.
    normalized = str(raw).replace("\n", ";")
    for sep in (",", ";"):
        normalized = normalized.replace(sep, ";")
    return tuple(x.strip() for x in normalized.split(";") if x.strip())


def _ensure_mysql_charset(url: str) -> str:
    value = str(url or "")
    if not value.lower().startswith("mysql"):
        return value
    try:
        parts = urlsplit(value)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        if "charset" not in {key.lower(): key for key in query}:
            query["charset"] = "utf8mb4"
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
    except Exception:
        joiner = "&" if "?" in value else "?"
        return f"{value}{joiner}charset=utf8mb4"


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
    upload_sessions_dir: Path = Path(os.getenv("BASE_UPLOAD_SESSIONS_DIR") or (datasets_dir / ".uploads")).resolve()
    dataset_staging_dir: Path = Path(os.getenv("BASE_DATASET_STAGING_DIR") or (datasets_dir / ".staging")).resolve()
    imports_dir: Path = Path(os.getenv("BASE_IMPORTS_DIR") or (home_dir / "imports")).resolve()
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
    illegal_dataset_id_start: int = max(1, int(os.getenv("ILLEGAL_DATASET_ID_START", "1000000") or "1000000"))
    standard_dataset_id_start: int = max(1, int(os.getenv("STANDARD_DATASET_ID_START", "2000000") or "2000000"))

    @property
    def dataset_import_roots(self) -> Tuple[Path, ...]:
        roots: list[Path] = [self.imports_dir.resolve()]
        for raw in _path_list_env("DATASET_IMPORT_ROOTS"):
            try:
                root = Path(raw).expanduser().resolve(strict=False)
            except Exception:
                continue
            if root not in roots:
                roots.append(root)
        return tuple(roots)

    @property
    def thumbnails_dir(self) -> Path:
        # Keep thumbnails under the datasets root so they can be managed together.
        return (self.datasets_dir / ".thumbnails").resolve()

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return _ensure_mysql_charset(str(self.database_url_override))
        return _ensure_mysql_charset(
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    def ensure_dirs(self) -> None:
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.upload_sessions_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_staging_dir.mkdir(parents=True, exist_ok=True)
        self.imports_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.pretrain_models_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()

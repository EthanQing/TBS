from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _default_home_dir() -> Path:
    # <repo>/backend_v2/train_platform/core/config.py -> parents[2] == <repo>/backend_v2
    return Path(__file__).resolve().parents[2]


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

    @property
    def thumbnails_dir(self) -> Path:
        # Keep thumbnails under the datasets root so they can be managed together.
        return (self.datasets_dir / ".thumbnails").resolve()

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return str(self.database_url_override)
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    def ensure_dirs(self) -> None:
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.pretrain_models_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()

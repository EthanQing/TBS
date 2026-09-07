from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy.orm import Session

from train_platform.core.config import settings
from train_platform.utils.exceptions import ValidationError

from .service import TrainingRunService


def tail_text_file(path: Path, *, lines: int) -> str:
    """Read the last ``lines`` from a worker log without loading the whole file."""

    try:
        if not path or not path.exists() or not path.is_file():
            return ""
    except Exception:
        return ""

    chunk_size = 4096
    data = b""
    try:
        with open(path, "rb") as file:
            file.seek(0, os.SEEK_END)
            pos = file.tell()
            while pos > 0 and data.count(b"\n") <= int(lines):
                read_size = min(chunk_size, pos)
                pos -= read_size
                file.seek(pos, os.SEEK_SET)
                data = file.read(read_size) + data
                if pos == 0:
                    break
    except Exception:
        return ""

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        text = str(data)
    parts = text.splitlines()
    return "\n".join(parts[-int(lines) :]) if parts else ""


def tail_logs(db: Session, run_id: str, *, which: str = "stdout", lines: int = 200) -> str:
    TrainingRunService().get_run(db, run_id)

    which = (which or "").strip().lower()
    if which not in ("stdout", "stderr"):
        raise ValidationError("which must be 'stdout' or 'stderr'")

    lines = int(lines)
    if lines < 1 or lines > 20000:
        raise ValidationError("lines must be between 1 and 20000")

    log_name = "train.stdout.log" if which == "stdout" else "train.stderr.log"
    path = settings.training_dir / str(run_id) / "logs" / log_name
    return tail_text_file(path, lines=lines)


__all__ = ["tail_logs", "tail_text_file"]

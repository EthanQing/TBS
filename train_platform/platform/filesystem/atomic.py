from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any] | list[Any],
    *,
    ensure_ascii: bool = False,
    indent: int = 2,
    sort_keys: bool = True,
) -> Path:
    content = json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys)
    return atomic_write_text(Path(path), content)

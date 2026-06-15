from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from train_platform.core.config import settings


PADDLE_DET_REQUIRED_CONFIG = Path("configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml")


def _candidate_roots(raw: Path | None = None) -> Iterable[Path]:
    bases: list[Path] = []
    if raw is not None:
        bases.append(raw)
    bases.append(settings.paddle_det_dir)
    bases.append(settings.home_dir / "PaddleDetection")

    seen: set[str] = set()
    for base in bases:
        for candidate in (base, base / "PaddleDetection", base.parent if base.name.lower() == "ppdet" else base):
            try:
                resolved = candidate.expanduser().resolve(strict=False)
            except Exception:
                resolved = candidate
            key = str(resolved).lower()
            if key in seen:
                continue
            seen.add(key)
            yield resolved


def is_paddledet_repo(path: Path) -> bool:
    root = path.parent if path.name.lower() == "ppdet" else path
    return (
        (root / "ppdet" / "__init__.py").is_file()
        and (root / "configs").is_dir()
        and (root / PADDLE_DET_REQUIRED_CONFIG).is_file()
    )


def resolve_paddledet_repo(raw: Path | str | None = None) -> Path | None:
    raw_path = Path(str(raw)) if raw is not None and str(raw).strip() else None
    for candidate in _candidate_roots(raw_path):
        root = candidate.parent if candidate.name.lower() == "ppdet" else candidate
        if is_paddledet_repo(root):
            return root
    return None


def paddledet_missing_message() -> str:
    return (
        "PaddleDetection source checkout is required. Clone PaddleDetection release/2.6 "
        f"to {settings.paddle_det_dir}, or set PADDLE_DET_DIR to a complete checkout containing "
        f"`ppdet/` and `{PADDLE_DET_REQUIRED_CONFIG.as_posix()}`."
    )


def ensure_paddledet_repo_on_syspath() -> Path:
    repo = resolve_paddledet_repo()
    if repo is None:
        raise RuntimeError(paddledet_missing_message())
    repo_s = str(repo)
    if repo_s not in sys.path:
        sys.path.insert(0, repo_s)
    return repo


def resolve_paddledet_config_path(config_path: object) -> Path | None:
    txt = str(config_path or "").strip().replace("\\", "/")
    if not txt:
        return None

    direct = Path(txt)
    if direct.is_absolute():
        return direct.resolve(strict=False) if direct.is_file() else None

    repo = resolve_paddledet_repo()
    candidates: list[Path] = []
    if repo is not None:
        candidates.append((repo / txt).resolve(strict=False))
    candidates.append((settings.home_dir / txt).resolve(strict=False))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None

from __future__ import annotations

from pathlib import Path
from typing import Optional

from train_platform.core.config import settings
from train_platform.utils.exceptions import ValidationError


def _resolve_under_base(
    *,
    raw_path: Optional[str],
    base_dir: Path,
    marker: str,
    label: str,
) -> Path:
    base = base_dir.resolve()
    if not raw_path:
        return base

    p = str(raw_path).strip().replace("\\", "/")
    if marker in p:
        p = p.split(marker, 1)[1]

    p = p.strip("/\\")
    if not p:
        return base

    rel = Path(p)
    if rel.is_absolute():
        raise ValidationError(f"{label} must be a relative path under {base}")
    if ".." in rel.parts:
        raise ValidationError(f"{label} cannot contain parent traversal")

    resolved = (base / rel).resolve(strict=False)
    try:
        resolved.relative_to(base)
    except Exception as e:
        raise ValidationError(f"{label} resolves outside allowed base directory") from e
    return resolved


def resolve_training_path(raw_path: Optional[str]) -> Path:
    return _resolve_under_base(
        raw_path=raw_path,
        base_dir=settings.training_dir,
        marker="/static/training/",
        label="training path",
    )


def resolve_temp_path(raw_path: Optional[str]) -> Path:
    return _resolve_under_base(
        raw_path=raw_path,
        base_dir=settings.temp_dir,
        marker="/static/temp/",
        label="temp path",
    )


def resolve_pretrain_path(raw_path: Optional[str]) -> Path:
    base_dir = settings.pretrain_models_dir.resolve()
    if not raw_path:
        return base_dir

    p = str(raw_path).strip().replace("\\", "/")
    marker = "/static/pretrain/"
    if marker in p:
        p = p.split(marker, 1)[1]

    p = p.strip("/\\")
    if not p:
        return base_dir

    abs_candidate = Path(p)
    if abs_candidate.is_absolute() and abs_candidate.exists():
        return abs_candidate.resolve()

    # Prevent path traversal for relative paths.
    rel = Path(p)
    if ".." in rel.parts:
        return base_dir

    return (base_dir / p).resolve(strict=False)


PADDLE_DET_REQUIRED_CONFIG = Path("configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml")


def _paddledet_candidate_roots(raw: Path | None = None) -> Iterable[Path]:
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
    for candidate in _paddledet_candidate_roots(raw_path):
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


__all__ = [
    "PADDLE_DET_REQUIRED_CONFIG",
    "is_paddledet_repo",
    "paddledet_missing_message",
    "resolve_paddledet_config_path",
    "resolve_paddledet_repo",
    "resolve_pretrain_path",
    "resolve_temp_path",
    "resolve_training_path",
]




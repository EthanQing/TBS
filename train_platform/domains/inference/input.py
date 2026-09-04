from __future__ import annotations

import ipaddress
import shutil
import uuid
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlparse

import requests

from train_platform.core.config import settings
from train_platform.utils.exceptions import NotFoundError, ValidationError
from train_platform.platform.filesystem.locations import resolve_temp_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
ALL_UPLOAD_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES


def save_uploaded_file(
    filename: str,
    stream: BinaryIO,
    *,
    subdir: str = "inference_uploads",
    validate_suffix: bool = True,
) -> str:
    """Persist an uploaded inference file and return its temp token."""
    raw_filename = str(filename or "")
    suffix = Path(raw_filename).suffix or ".jpg"
    suffix_for_validation = suffix.lower()
    if validate_suffix and suffix_for_validation not in ALL_UPLOAD_SUFFIXES:
        allowed = ", ".join(sorted(ALL_UPLOAD_SUFFIXES))
        raise ValidationError(f"Unsupported format: {suffix_for_validation}. Allowed: {allowed}")
    if validate_suffix:
        suffix = suffix_for_validation

    settings.ensure_dirs()
    directory = str(subdir or "").strip().replace("\\", "/").strip("/")
    if not directory or any(part in {".", ".."} for part in Path(directory).parts):
        raise ValidationError("Unsafe upload directory")
    out_dir = (settings.temp_dir / directory).resolve(strict=False)
    try:
        out_dir.relative_to(settings.temp_dir.resolve())
    except ValueError as exc:
        raise ValidationError("Unsafe upload directory") from exc
    out_dir.mkdir(parents=True, exist_ok=True)

    token = f"{directory}/{uuid.uuid4().hex}{suffix}"
    out_path = (settings.temp_dir / token).resolve(strict=False)
    try:
        out_path.relative_to(settings.temp_dir.resolve())
    except ValueError as exc:
        raise ValidationError("Unsafe upload path") from exc
    with out_path.open("wb") as output:
        shutil.copyfileobj(stream, output)
    return token


def materialize_input(
    *,
    input_path: str | None,
    image_url: str | None,
) -> tuple[Path, str, dict[str, Any]]:
    """Resolve an existing temp token or download one remote image safely."""
    if not input_path and not image_url:
        raise ValidationError("Either input_path or image_url is required")
    if input_path and image_url:
        raise ValidationError("Provide only one of input_path and image_url")

    meta: dict[str, Any] = {}
    if image_url:
        url = str(image_url).strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValidationError("image_url must be http(s)")
        local_path, token = _download_to_temp(url)
        meta["image_url"] = url
        return local_path, token, meta

    assert input_path is not None
    raw = str(input_path).strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        local_path, token = _download_to_temp(raw)
        meta["image_url"] = raw
        return local_path, token, meta

    path = resolve_temp_path(raw)
    if path.exists() and path.is_file():
        token = path.relative_to(settings.temp_dir.resolve()).as_posix()
        return path, token, meta
    raise NotFoundError(f"Input file not found: {raw}")


def resolve_temp_token(raw_token: str) -> str:
    """Validate a temp token and return its portable relative form."""
    raw = str(raw_token or "").strip()
    path = resolve_temp_path(raw)
    if not path.exists() or not path.is_file():
        raise NotFoundError(f"Input file not found: {raw}")
    return path.relative_to(settings.temp_dir.resolve()).as_posix()


def _download_to_temp(url: str) -> tuple[Path, str]:
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    out_dir = settings.temp_dir / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(url.split("?", 1)[0]).suffix or ".jpg"
    out_path = out_dir / f"{uuid.uuid4().hex}{suffix}"
    try:
        _validate_remote_url(url)
        max_bytes = max(1, int(settings.inference_max_download_bytes))
        timeout = max(1.0, float(settings.inference_download_timeout_sec))

        written = 0
        with requests.get(url, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            with out_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValidationError(f"image_url exceeds max allowed size ({max_bytes} bytes)")
                    output.write(chunk)
    except Exception as exc:
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise ValidationError(f"Failed to download image_url: {exc}") from exc

    token = out_path.relative_to(settings.temp_dir.resolve()).as_posix()
    return out_path, token


def _validate_remote_url(url: str) -> None:
    parsed = urlparse(url)
    scheme = str(parsed.scheme or "").strip().lower()
    if not scheme:
        raise ValidationError("image_url scheme is required")

    allowed_schemes = {str(item).strip().lower() for item in settings.inference_allowed_schemes if str(item).strip()}
    if not allowed_schemes:
        allowed_schemes = {"http", "https"}
    if scheme not in allowed_schemes:
        raise ValidationError(f"image_url scheme not allowed: {scheme}")

    host = str(parsed.hostname or "").strip().lower()
    if not host:
        raise ValidationError("image_url host is required")

    allowed_hosts = {str(item).strip().lower() for item in settings.inference_allowed_hosts if str(item).strip()}
    if allowed_hosts and host not in allowed_hosts:
        raise ValidationError(f"image_url host not allowed: {host}")

    if not allowed_hosts:
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValidationError("image_url host resolves to a disallowed private address")


__all__ = [
    "ALL_UPLOAD_SUFFIXES",
    "IMAGE_SUFFIXES",
    "VIDEO_SUFFIXES",
    "materialize_input",
    "resolve_temp_token",
    "save_uploaded_file",
]

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


PUBLIC_KEY_B64 = "qk/C58JiQDFp8UfxCp1TX+ABNZkD4yq+NsZ2LjNHuHE="
DEFAULT_LICENSE_PATH = "/app/license/license.dat"


class LicenseError(RuntimeError):
    pass


@dataclass(frozen=True)
class LicenseInfo:
    customer: str
    deployment: str
    expires_at: datetime


def license_required() -> bool:
    value = os.getenv("TRAIN_PLATFORM_LICENSE_REQUIRED", "0")
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def assert_valid_license() -> LicenseInfo | None:
    if not license_required():
        return None

    license_path = Path(os.getenv("TRAIN_PLATFORM_LICENSE_PATH", DEFAULT_LICENSE_PATH))
    if not license_path.exists() or not license_path.is_file():
        raise LicenseError(
            f"License file not found: {license_path}. "
            "Set TRAIN_PLATFORM_LICENSE_PATH or mount license.dat into /app/license."
        )

    try:
        raw = json.loads(license_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise LicenseError(f"Invalid license file format: {e}") from e

    payload = raw.get("payload")
    signature_b64 = raw.get("signature")
    if not isinstance(payload, dict) or not isinstance(signature_b64, str):
        raise LicenseError("Invalid license file: expected payload and signature.")

    payload_bytes = _canonical_payload_bytes(payload)
    signature = _decode_b64(signature_b64, "signature")
    public_key = Ed25519PublicKey.from_public_bytes(_decode_b64(PUBLIC_KEY_B64, "public key"))

    try:
        public_key.verify(signature, payload_bytes)
    except InvalidSignature as e:
        raise LicenseError("Invalid license signature.") from e

    expires_at = _parse_datetime(str(payload.get("expires_at", "")), "expires_at")
    now = datetime.now(timezone.utc)
    if expires_at <= now:
        raise LicenseError(f"License expired at {expires_at.isoformat()}.")

    customer = str(payload.get("customer", "")).strip()
    deployment = str(payload.get("deployment", "")).strip()
    if not customer:
        raise LicenseError("Invalid license payload: customer is required.")
    if not deployment:
        raise LicenseError("Invalid license payload: deployment is required.")

    return LicenseInfo(customer=customer, deployment=deployment, expires_at=expires_at)


def _canonical_payload_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _decode_b64(value: str, label: str) -> bytes:
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as e:
        raise LicenseError(f"Invalid base64 value for {label}.") from e


def _parse_datetime(value: str, label: str) -> datetime:
    raw = value.strip()
    if not raw:
        raise LicenseError(f"Invalid license payload: {label} is required.")
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as e:
        raise LicenseError(f"Invalid datetime for {label}: {value}") from e
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

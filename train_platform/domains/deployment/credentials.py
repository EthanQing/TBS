from __future__ import annotations

import hashlib
import hmac
import secrets


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(str(raw_key).encode("utf-8")).hexdigest()


def verify_api_key(raw_key: str, key_hash: str) -> bool:
    return hmac.compare_digest(hash_api_key(raw_key), str(key_hash or ""))


def generate_api_key() -> tuple[str, str, str]:
    key = f"dpk_{secrets.token_urlsafe(32)}"
    key_hash = hash_api_key(key)
    hint = f"{key[:6]}...{key[-4:]}" if len(key) >= 12 else key
    return key, key_hash, hint


__all__ = ["generate_api_key", "hash_api_key", "verify_api_key"]

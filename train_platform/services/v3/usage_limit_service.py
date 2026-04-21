from __future__ import annotations

import calendar
import hashlib
import hmac
import json
import logging
import os
import socket
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from train_platform.core.config import settings


logger = logging.getLogger(__name__)


class UsageLimitService:
    _state_lock = Lock()
    _cached_state_payload: dict[str, Any] | None = None
    # Local-only hardcoded trial window: 6 calendar months from first successful startup.
    _hardcoded_trial_months = 6
    _exempt_exact_paths = frozenset(
        {
            "/health",
            "/openapi.json",
            "/favicon.ico",
        }
    )
    _exempt_prefixes = (
        "/docs",
        "/redoc",
    )
    _state_sig_salt = "tp.runtime.guard.v2"

    @classmethod
    def _utcnow(cls) -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _normalize_path(path: str | None) -> str:
        raw = str(path or "").strip()
        if not raw:
            return "/"
        normalized = raw.split("?", 1)[0].strip()
        if not normalized.startswith("/"):
            normalized = "/" + normalized
        normalized = normalized.rstrip("/")
        return normalized or "/"

    @staticmethod
    def _ensure_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _parse_datetime(raw: str | None) -> datetime | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @classmethod
    def _machine_fingerprint(cls) -> str:
        parts = [
            str(settings.home_dir),
            socket.gethostname(),
            f"{uuid.getnode():012x}",
            os.name,
        ]
        raw = "|".join(parts).encode("utf-8", errors="ignore")
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def _state_secret(cls) -> bytes:
        raw = (
            f"{cls._state_sig_salt}|{cls._machine_fingerprint()}|"
            f"{str(getattr(settings, 'internal_api_token', '') or '').strip()}"
        ).encode("utf-8")
        return hashlib.sha256(raw).digest()

    @classmethod
    def _serialize_payload(cls, payload: dict[str, Any]) -> bytes:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    @classmethod
    def _sign_payload(cls, payload: dict[str, Any]) -> str:
        return hmac.new(cls._state_secret(), cls._serialize_payload(payload), hashlib.sha256).hexdigest()

    @classmethod
    def _load_state(cls) -> dict[str, Any] | None:
        if cls._cached_state_payload is not None:
            return dict(cls._cached_state_payload)

        path = settings.usage_limit_state_path
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read runtime-guard state file %s: %s", path, exc)
            return {"_invalid": True}

        if not isinstance(envelope, dict):
            return {"_invalid": True}

        payload = envelope.get("payload")
        signature = str(envelope.get("sig") or "").strip()
        if not isinstance(payload, dict) or not signature:
            return {"_invalid": True}

        expected = cls._sign_payload(payload)
        if not hmac.compare_digest(signature, expected):
            logger.warning("Runtime-guard state signature verification failed: %s", path)
            return {"_invalid": True}

        cls._cached_state_payload = dict(payload)
        return dict(payload)

    @classmethod
    def _save_state(cls, payload: dict[str, Any]) -> None:
        path = settings.usage_limit_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        envelope = {
            "payload": payload,
            "sig": cls._sign_payload(payload),
        }
        tmp_path.write_text(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        tmp_path.replace(path)
        cls._cached_state_payload = dict(payload)

    @classmethod
    def _configured_window_bounds(cls) -> tuple[datetime | None, datetime | None]:
        window_start = cls._ensure_utc(getattr(settings, "software_not_before_at", None))
        window_end = cls._ensure_utc(getattr(settings, "software_not_after_at", None))
        legacy_end = cls._ensure_utc(getattr(settings, "software_expires_at", None))
        if window_end is None:
            window_end = legacy_end
        return window_start, window_end

    @staticmethod
    def _add_months(value: datetime, months: int) -> datetime:
        if months <= 0:
            return value

        total_month = value.month - 1 + months
        year = value.year + total_month // 12
        month = total_month % 12 + 1
        day = min(value.day, calendar.monthrange(year, month)[1])
        return value.replace(year=year, month=month, day=day)

    @classmethod
    def _hardcoded_window_bounds(cls, *, first_seen_at: datetime | None) -> tuple[datetime | None, datetime | None]:
        months = max(0, int(cls._hardcoded_trial_months or 0))
        if first_seen_at is None or months <= 0:
            return None, None

        window_start = cls._ensure_utc(first_seen_at)
        if window_start is None:
            return None, None
        return window_start, cls._add_months(window_start, months)

    @classmethod
    def _load_or_init_state(cls, *, now: datetime) -> tuple[dict[str, Any] | None, bool]:
        with cls._state_lock:
            state = cls._load_state()
            if isinstance(state, dict) and state.get("_invalid"):
                return None, True
            if state is not None:
                return state, False

            payload = {
                "schema": 2,
                "first_seen_at": now.isoformat(),
                "last_seen_at": now.isoformat(),
            }
            cls._save_state(payload)
            logger.info("Initialized runtime-guard state at %s", now.isoformat())
            return payload, False

    @classmethod
    def _touch_last_seen(cls, *, state: dict[str, Any], now: datetime) -> dict[str, Any]:
        updated = dict(state)
        updated["last_seen_at"] = now.isoformat()
        cls._save_state(updated)
        return updated

    @staticmethod
    def _render_timestamp(value: datetime | None) -> str:
        if value is None:
            return "N/A"
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @classmethod
    def is_exempt_path(cls, path: str | None) -> bool:
        normalized = cls._normalize_path(path)
        if normalized in cls._exempt_exact_paths:
            return True
        return any(
            normalized == prefix or normalized.startswith(prefix + "/")
            for prefix in cls._exempt_prefixes
        )

    @classmethod
    def get_status(cls) -> dict[str, Any]:
        now = cls._utcnow()
        configured_window_start, configured_window_end = cls._configured_window_bounds()
        hardcoded_window_enabled = max(0, int(cls._hardcoded_trial_months or 0)) > 0
        window_start = configured_window_start
        window_end = configured_window_end
        enabled = hardcoded_window_enabled or window_start is not None or window_end is not None
        rollback_tolerance = max(
            0,
            int(getattr(settings, "software_clock_rollback_tolerance_seconds", 0) or 0),
        )
        persist_interval_seconds = max(
            1,
            int(getattr(settings, "software_guard_persist_interval_seconds", 60) or 60),
        )
        state = None
        invalid_state = False
        first_seen_at = None
        last_seen_at = None
        blocked = False
        reason = "disabled"
        message = "Runtime guard disabled."

        if enabled:
            state, invalid_state = cls._load_or_init_state(now=now)
            if state:
                first_seen_at = cls._parse_datetime(state.get("first_seen_at"))
                last_seen_at = cls._parse_datetime(state.get("last_seen_at"))
                if window_start is None and window_end is None and hardcoded_window_enabled:
                    window_start, window_end = cls._hardcoded_window_bounds(first_seen_at=first_seen_at)

            if window_start and window_end and window_start > window_end:
                blocked = True
                reason = "invalid_window"
                message = "Configured time window is invalid."
            elif invalid_state:
                blocked = True
                reason = "tampered_state"
                message = "Runtime guard state verification failed."
            elif last_seen_at and now.timestamp() + rollback_tolerance < last_seen_at.timestamp():
                blocked = True
                reason = "clock_rollback"
                message = (
                    "Clock rollback detected: "
                    f"current={cls._render_timestamp(now)}, last_seen={cls._render_timestamp(last_seen_at)}."
                )
            elif window_start and now < window_start:
                blocked = True
                reason = "before_start"
                message = (
                    "Current time is earlier than the allowed window: "
                    f"{cls._render_timestamp(now)} < {cls._render_timestamp(window_start)}."
                )
            elif window_end and now > window_end:
                blocked = True
                reason = "after_end"
                message = (
                    "Current time is later than the allowed window: "
                    f"{cls._render_timestamp(now)} > {cls._render_timestamp(window_end)}."
                )
            else:
                reason = "ok"
                message = "Runtime guard passed."
                if state is not None:
                    should_persist = last_seen_at is None or (now - last_seen_at).total_seconds() >= persist_interval_seconds
                    if should_persist:
                        state = cls._touch_last_seen(state=state, now=now)
                        last_seen_at = now

        remaining_seconds = None
        if window_end is not None:
            remaining_seconds = max(0, int((window_end - now).total_seconds()))

        return {
            "enabled": enabled,
            "blocked": blocked,
            "expired": reason == "after_end",
            "reason": reason,
            "message": message,
            "server_time": now,
            "window_start": window_start,
            "window_end": window_end,
            "first_seen_at": first_seen_at,
            "last_seen_at": last_seen_at,
            "remaining_seconds": remaining_seconds,
        }

    @classmethod
    def get_denial_payload(cls, status: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "message": "Forbidden",
            "code": "access_denied",
        }

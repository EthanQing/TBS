from __future__ import annotations

import os
from typing import Any

from train_platform.utils.exceptions import ValidationError


STATUS_ACTIVE = "active"
STATUS_RESOLVED = "resolved"
SOURCE_TRAINING_RUN = "training_run"

RULE_TYPE_TRAINING_FAILED = "training_run_failed"
RULE_TYPE_TRAINING_STALE = "training_run_stale"

ALLOWED_RULE_TYPES = {
    RULE_TYPE_TRAINING_FAILED,
    RULE_TYPE_TRAINING_STALE,
}
ALLOWED_SEVERITIES = {"critical", "high", "medium", "low"}


def _default_stale_after_seconds() -> int:
    try:
        value = int(os.getenv("WORKER_STALE_AFTER_SECONDS", "120"))
    except (TypeError, ValueError):
        value = 120
    return max(1, min(value, 86400))


RULE_CATALOG: dict[str, dict[str, Any]] = {
    RULE_TYPE_TRAINING_FAILED: {
        "name": "训练任务失败",
        "description": "当训练任务状态变为 failed 时触发。",
        "default_severity": "high",
        "default_enabled": True,
        "default_cooldown_seconds": 300,
        "config_schema": {},
    },
    RULE_TYPE_TRAINING_STALE: {
        "name": "训练任务心跳超时",
        "description": "训练任务处于 running 且心跳超过阈值未更新时触发。",
        "default_severity": "high",
        "default_enabled": True,
        "default_cooldown_seconds": 300,
        "config_schema": {
            "stale_after_seconds": {
                "type": "integer",
                "minimum": 1,
                "default": _default_stale_after_seconds(),
                "description": "覆盖系统默认心跳超时秒数。",
            }
        },
    },
}


def list_rule_types() -> list[dict[str, Any]]:
    return [{"rule_type": rule_type, **meta} for rule_type, meta in RULE_CATALOG.items()]


def validate_rule_type(raw: Any) -> str:
    rule_type = str(raw or "").strip()
    if rule_type not in ALLOWED_RULE_TYPES:
        raise ValidationError(f"Unsupported rule_type: {rule_type}")
    return rule_type


def validate_severity(raw: Any) -> str:
    severity = str(raw or "").strip().lower()
    if severity not in ALLOWED_SEVERITIES:
        raise ValidationError(f"Invalid severity: {raw}")
    return severity


def validate_cooldown(raw: Any) -> int:
    try:
        cooldown = int(raw)
    except (TypeError, ValueError):
        raise ValidationError("cooldown_seconds must be an integer") from None
    if cooldown < 0 or cooldown > 86400:
        raise ValidationError("cooldown_seconds must be between 0 and 86400")
    return cooldown


def normalize_config(rule_type: str, raw: Any) -> dict[str, Any]:
    if raw is None:
        config: dict[str, Any] = {}
    elif isinstance(raw, dict):
        config = dict(raw)
    else:
        raise ValidationError("config must be a JSON object")

    if rule_type == RULE_TYPE_TRAINING_STALE and "stale_after_seconds" in config:
        try:
            stale_after = int(config.get("stale_after_seconds"))
        except (TypeError, ValueError):
            raise ValidationError("config.stale_after_seconds must be an integer") from None
        if stale_after < 1 or stale_after > 86400:
            raise ValidationError("config.stale_after_seconds must be between 1 and 86400")
        config["stale_after_seconds"] = stale_after
    return config


def resolve_stale_after_seconds(config: Any) -> int:
    normalized = normalize_config(RULE_TYPE_TRAINING_STALE, config)
    raw = normalized.get("stale_after_seconds")
    return _default_stale_after_seconds() if raw is None else int(raw)

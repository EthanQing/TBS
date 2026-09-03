from __future__ import annotations

from fastapi import APIRouter, Query

from train_platform.domains.training.frameworks import get_plugin, list_plugins
from train_platform.schemas.v3.frameworks import (
    FrameworkConfigSchemaOut,
    FrameworkConfigValidateOut,
    FrameworkConfigValidateRequest,
    FrameworkPluginOut,
)
from train_platform.utils.exceptions import NotFoundError, ValidationError


router = APIRouter(prefix="/frameworks", tags=["frameworks"])


@router.get("", response_model=list[FrameworkPluginOut])
def list_frameworks(
    implemented: bool | None = Query(
        None,
        description="Filter by plugin implementation status. Omit to return all plugins.",
    ),
):
    items = []
    for row in list_plugins():
        if implemented is not None and bool(row.implemented) != bool(implemented):
            continue
        items.append(
            {
                "plugin_id": str(row.plugin_id),
                "name": str(row.name),
                "display_name": str(row.display_name),
                "implemented": bool(row.implemented),
                "config_schema": dict(row.config_schema or {}),
            }
        )
    return items


@router.get("/{plugin_id}/config-schema", response_model=FrameworkConfigSchemaOut)
def get_framework_config_schema(plugin_id: str):
    normalized_plugin_id = str(plugin_id or "").strip().lower()
    if not normalized_plugin_id:
        raise ValidationError("plugin_id is required")
    try:
        plugin = get_plugin(normalized_plugin_id)
    except Exception as exc:
        raise NotFoundError(f"Framework plugin not found: {normalized_plugin_id}") from exc
    try:
        schema = dict(plugin.get_config_schema() or {})
    except Exception:
        schema = {}
    return {"plugin_id": normalized_plugin_id, "config_schema": schema}


@router.post("/{plugin_id}/validate-config", response_model=FrameworkConfigValidateOut)
def validate_framework_config(plugin_id: str, payload: FrameworkConfigValidateRequest):
    if payload.config is not None and not isinstance(payload.config, dict):
        raise ValidationError("config must be an object")
    try:
        plugin = get_plugin(plugin_id)
    except Exception as exc:
        raise NotFoundError(f"Framework plugin not found: {plugin_id}") from exc
    try:
        normalized = plugin.normalize_config(payload.config or {})
    except Exception as exc:
        raise ValidationError(f"Invalid framework config: {exc}") from exc
    if not isinstance(normalized, dict):
        raise ValidationError("Framework plugin normalize_config() must return an object")
    canonical_plugin_id = str(getattr(plugin, "plugin_id", plugin_id) or plugin_id).strip().lower()
    return {"plugin_id": canonical_plugin_id, "normalized_config": dict(normalized)}

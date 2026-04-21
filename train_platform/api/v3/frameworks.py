from __future__ import annotations

from fastapi import APIRouter, Query

from train_platform.schemas.v3.frameworks import (
    FrameworkConfigSchemaOut,
    FrameworkConfigValidateOut,
    FrameworkConfigValidateRequest,
    FrameworkPluginOut,
)
from train_platform.services.v3.framework_service import FrameworkService


router = APIRouter(prefix="/frameworks", tags=["frameworks"])


@router.get("", response_model=list[FrameworkPluginOut])
def list_frameworks(
    implemented: bool | None = Query(
        None,
        description="Filter by plugin implementation status. Omit to return all plugins.",
    ),
):
    return FrameworkService().list_frameworks(implemented=implemented)


@router.get("/{plugin_id}/config-schema", response_model=FrameworkConfigSchemaOut)
def get_framework_config_schema(plugin_id: str):
    return FrameworkService().get_config_schema(plugin_id)


@router.post("/{plugin_id}/validate-config", response_model=FrameworkConfigValidateOut)
def validate_framework_config(plugin_id: str, payload: FrameworkConfigValidateRequest):
    return FrameworkService().normalize_config(plugin_id, payload.config)

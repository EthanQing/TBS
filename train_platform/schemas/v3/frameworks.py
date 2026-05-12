from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class FrameworkPluginOut(BaseModel):
    plugin_id: str
    name: str
    display_name: str
    implemented: bool
    config_schema: Dict[str, Any] = Field(default_factory=dict)


class FrameworkConfigSchemaOut(BaseModel):
    plugin_id: str
    config_schema: Dict[str, Any] = Field(default_factory=dict)


class FrameworkConfigValidateRequest(BaseModel):
    config: Dict[str, Any] = Field(default_factory=dict)


class FrameworkConfigValidateOut(BaseModel):
    plugin_id: str
    normalized_config: Dict[str, Any] = Field(default_factory=dict)

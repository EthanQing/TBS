from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contract import TrainerPlugin
from .custom_source import CustomSourceTrainer
from .paddle_det.plugin import PaddleDetTrainer
from .ultralytics_yolo import UltralyticsYOLOTrainer


@dataclass(frozen=True)
class FrameworkPluginInfo:
    plugin_id: str
    name: str
    display_name: str
    implemented: bool
    config_schema: dict[str, Any]


_PLUGIN_MAP: dict[str, TrainerPlugin] = {}


def register_plugin(plugin: TrainerPlugin) -> None:
    plugin_id = str(getattr(plugin, "plugin_id", "") or "").strip().lower()
    if not plugin_id:
        raise ValueError("plugin_id is required for trainer plugin")
    _PLUGIN_MAP[plugin_id] = plugin


def _bootstrap_plugins() -> None:
    if _PLUGIN_MAP:
        return
    register_plugin(UltralyticsYOLOTrainer())
    register_plugin(PaddleDetTrainer())
    register_plugin(CustomSourceTrainer())


def list_plugins() -> list[FrameworkPluginInfo]:
    _bootstrap_plugins()
    out: list[FrameworkPluginInfo] = []
    for plugin_id, plugin in sorted(_PLUGIN_MAP.items(), key=lambda item: item[0]):
        try:
            schema = dict(plugin.get_config_schema() or {})
        except Exception:
            schema = {}
        out.append(
            FrameworkPluginInfo(
                plugin_id=plugin_id,
                name=str(getattr(plugin, "name", plugin_id) or plugin_id),
                display_name=str(getattr(plugin, "display_name", getattr(plugin, "name", plugin_id)) or plugin_id),
                implemented=bool(getattr(plugin, "implemented", True)),
                config_schema=schema,
            )
        )
    return out


def get_trainer(*, model_family: str, engine: str | None = None) -> TrainerPlugin:
    _bootstrap_plugins()
    engine_key = str(engine or "").strip().lower()
    if engine_key:
        plugin = _PLUGIN_MAP.get(engine_key)
        if plugin is not None:
            return plugin

    family = (model_family or "").strip()
    for plugin in _PLUGIN_MAP.values():
        try:
            if plugin.can_handle(family):
                return plugin
        except Exception:
            continue
    raise ValueError(f"No trainer registered for model_family='{family}', engine='{engine_key}'")


def get_plugin(plugin_id: str) -> TrainerPlugin:
    _bootstrap_plugins()
    normalized = str(plugin_id or "").strip().lower()
    plugin = _PLUGIN_MAP.get(normalized)
    if plugin is None:
        raise ValueError(f"Plugin not found: {plugin_id}")
    return plugin


__all__ = ["FrameworkPluginInfo", "get_plugin", "get_trainer", "list_plugins", "register_plugin"]

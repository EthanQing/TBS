from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from train_platform.training.plugins.base import TrainerPlugin
from train_platform.training.plugins.mmdet import MMDetTrainer
from train_platform.training.plugins.paddle_det import PaddleDetTrainer
from train_platform.training.plugins.ultralytics_yolo import UltralyticsYOLOTrainer


@dataclass(frozen=True)
class FrameworkPluginInfo:
    plugin_id: str
    name: str
    display_name: str
    implemented: bool
    config_schema: Dict[str, Any]


_PLUGIN_MAP: Dict[str, TrainerPlugin] = {}


def register_plugin(plugin: TrainerPlugin) -> None:
    pid = str(getattr(plugin, "plugin_id", "") or "").strip().lower()
    if not pid:
        raise ValueError("plugin_id is required for trainer plugin")
    _PLUGIN_MAP[pid] = plugin


def _bootstrap_plugins() -> None:
    if _PLUGIN_MAP:
        return
    register_plugin(UltralyticsYOLOTrainer())
    register_plugin(PaddleDetTrainer())
    register_plugin(MMDetTrainer())


def list_plugins() -> List[FrameworkPluginInfo]:
    _bootstrap_plugins()
    out: List[FrameworkPluginInfo] = []
    for pid, p in sorted(_PLUGIN_MAP.items(), key=lambda x: x[0]):
        schema = {}
        try:
            schema = dict(p.get_config_schema() or {})
        except Exception:
            schema = {}
        out.append(
            FrameworkPluginInfo(
                plugin_id=pid,
                name=str(getattr(p, "name", pid) or pid),
                display_name=str(getattr(p, "display_name", getattr(p, "name", pid)) or pid),
                implemented=bool(getattr(p, "implemented", True)),
                config_schema=schema,
            )
        )
    return out


def get_trainer(*, model_family: str, engine: str | None = None) -> TrainerPlugin:
    _bootstrap_plugins()
    engine_key = str(engine or "").strip().lower()
    if engine_key:
        p = _PLUGIN_MAP.get(engine_key)
        if p is not None:
            return p

    mf = (model_family or "").strip()
    for p in _PLUGIN_MAP.values():
        try:
            if p.can_handle(mf):
                return p
        except Exception:
            continue
    raise ValueError(f"No trainer registered for model_family='{mf}', engine='{engine_key}'")


def get_plugin(plugin_id: str) -> TrainerPlugin:
    _bootstrap_plugins()
    pid = str(plugin_id or "").strip().lower()
    p = _PLUGIN_MAP.get(pid)
    if p is None:
        raise ValueError(f"Plugin not found: {plugin_id}")
    return p

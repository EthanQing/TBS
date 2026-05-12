from __future__ import annotations

from typing import Any, Dict, List

from train_platform.training.registry import FrameworkPluginInfo, get_plugin, list_plugins
from train_platform.utils.exceptions import NotFoundError, ValidationError


def _to_plugin_info_dict(item: FrameworkPluginInfo) -> Dict[str, Any]:
    return {
        "plugin_id": str(item.plugin_id),
        "name": str(item.name),
        "display_name": str(item.display_name),
        "implemented": bool(item.implemented),
        "config_schema": dict(item.config_schema or {}),
    }


class FrameworkService:
    def list_frameworks(self, *, implemented: bool | None = None) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for row in list_plugins():
            if implemented is not None and bool(row.implemented) != bool(implemented):
                continue
            items.append(_to_plugin_info_dict(row))
        return items

    def get_framework(self, plugin_id: str) -> Dict[str, Any]:
        pid = str(plugin_id or "").strip().lower()
        if not pid:
            raise ValidationError("plugin_id is required")

        plugin = self._get_plugin_or_404(pid)
        schema = {}
        try:
            schema = dict(plugin.get_config_schema() or {})
        except Exception:
            schema = {}

        return {
            "plugin_id": pid,
            "name": str(getattr(plugin, "name", pid) or pid),
            "display_name": str(getattr(plugin, "display_name", getattr(plugin, "name", pid)) or pid),
            "implemented": bool(getattr(plugin, "implemented", True)),
            "config_schema": schema,
        }

    def get_config_schema(self, plugin_id: str) -> Dict[str, Any]:
        info = self.get_framework(plugin_id)
        return {"plugin_id": info["plugin_id"], "config_schema": dict(info.get("config_schema") or {})}

    def normalize_config(self, plugin_id: str, raw: Dict[str, Any] | None) -> Dict[str, Any]:
        if raw is not None and not isinstance(raw, dict):
            raise ValidationError("config must be an object")
        plugin = self._get_plugin_or_404(plugin_id)
        try:
            normalized = plugin.normalize_config(raw or {})
        except Exception as e:
            raise ValidationError(f"Invalid framework config: {e}") from e
        if not isinstance(normalized, dict):
            raise ValidationError("Framework plugin normalize_config() must return an object")
        return {"plugin_id": str(plugin_id).strip().lower(), "normalized_config": dict(normalized)}

    @staticmethod
    def _get_plugin_or_404(plugin_id: str):
        try:
            return get_plugin(plugin_id)
        except Exception as e:
            raise NotFoundError(f"Framework plugin not found: {plugin_id}") from e

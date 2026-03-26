# 框架插件系统 API（后端）

> 基础前缀：`/api/v2/frameworks`

本接口用于让前端动态发现训练框架插件、读取配置表单 schema、提交前做配置归一化校验。

---

## 1) 获取插件列表

`GET /api/v2/frameworks`

### Query 参数
- `implemented`（可选，`true|false`）：按是否已实现过滤。

### 响应示例
```json
[
  {
    "plugin_id": "ultralytics-yolo",
    "name": "ultralytics-yolo",
    "display_name": "Ultralytics YOLO",
    "implemented": true,
    "config_schema": {
      "type": "object",
      "properties": {
        "use_pretrained": { "type": "boolean", "default": true }
      }
    }
  }
]
```

---

## 2) 获取插件配置 schema

`GET /api/v2/frameworks/{plugin_id}/config-schema`

### 响应示例
```json
{
  "plugin_id": "paddle-det",
  "config_schema": {
    "type": "object",
    "properties": {
      "config_path": { "type": "string" },
      "resume_training": { "type": "boolean", "default": false }
    }
  }
}
```

---

## 3) 归一化/校验插件配置

`POST /api/v2/frameworks/{plugin_id}/validate-config`

### 请求体
```json
{
  "config": {
    "resume_training": true,
    "eval_interval": 2
  }
}
```

### 响应示例
```json
{
  "plugin_id": "paddle-det",
  "normalized_config": {
    "resume_training": true,
    "eval_interval": 2
  }
}
```

---

## 4) 与训练创建接口的配合

在创建训练任务 `POST /api/v2/training-runs` 时，将插件配置放入：

`parameters.additional_params.framework_config`

示例：
```json
{
  "project_id": 1,
  "architecture_id": 12,
  "parameters": {
    "epochs": 100,
    "batch_size": -1,
    "device": "0",
    "additional_params": {
      "framework_config": {
        "resume_training": false,
        "save_period": 1
      }
    }
  }
}
```

训练 worker 会按 `architecture.engine` 选择插件，并将 `framework_config` 传入插件执行。

### 训练参数补充说明

- `batch_size = -1`：仅 `ultralytics-yolo`（YOLO / RT-DETR）支持，表示启用 Ultralytics 自动 batch。
- `device = "0,1"`：仅 `ultralytics-yolo` 支持，表示使用多卡训练。
- Ultralytics 多卡训练时，`batch_size` 必须为正整数，且能被 GPU 数量整除；`batch_size=-1` 不支持多卡。

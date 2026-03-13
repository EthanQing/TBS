# 离线告警 API（v2）

路由前缀：`/alarms`  
若后端挂在 `/api/v2`，实际路径为 `/api/v2/alarms`。

> 本期仅支持离线展示闭环（规则配置、触发、活跃、历史、确认），不包含邮件/webhook发送。

## 一、规则类型（动态表单）

### 1) 获取规则类型
GET `/alarms/rule-types`

响应示例：
```json
[
  {
    "rule_type": "training_run_failed",
    "name": "训练任务失败",
    "description": "当训练任务状态变为 failed 时触发。",
    "default_severity": "high",
    "default_enabled": true,
    "default_cooldown_seconds": 300,
    "config_schema": {}
  },
  {
    "rule_type": "training_run_stale",
    "name": "训练任务心跳超时",
    "description": "训练任务处于 running 且心跳超过阈值未更新时触发。",
    "default_severity": "high",
    "default_enabled": true,
    "default_cooldown_seconds": 300,
    "config_schema": {
      "stale_after_seconds": {
        "type": "integer",
        "minimum": 1,
        "default": 120,
        "description": "覆盖系统默认心跳超时秒数。"
      }
    }
  }
]
```

## 二、规则配置（6.1 / 6.2）

字段说明：
- `rule_type`：`training_run_failed` | `training_run_stale`
- `severity`：`critical` | `high` | `medium` | `low`
- `cooldown_seconds`：同一告警重复命中的冷却秒数（0~86400）
- `config`：
  - `training_run_failed`：可为空对象
  - `training_run_stale`：可含 `stale_after_seconds`

### 1) 规则列表
GET `/alarms/rules?page=1&page_size=50&enabled=true`

### 2) 创建规则
POST `/alarms/rules`
```json
{
  "rule_type": "training_run_failed",
  "name": "训练失败告警",
  "description": "训练失败立即告警",
  "severity": "high",
  "enabled": true,
  "cooldown_seconds": 300,
  "config": {}
}
```

### 3) 更新规则
PATCH `/alarms/rules/{rule_id}`
```json
{
  "enabled": false,
  "cooldown_seconds": 600
}
```

### 4) 删除规则
DELETE `/alarms/rules/{rule_id}`

## 三、触发与补偿评估（6.3 / 6.4）

### 手动评估
POST `/alarms/evaluate`

请求体（可选）：
```json
{
  "run_ids": ["run-id-1", "run-id-2"]
}
```

响应：
```json
{
  "evaluated_runs": 2,
  "triggered_new": 1,
  "touched_active": 0,
  "resolved": 1,
  "active_total": 3,
  "timestamp": "2026-03-13T10:00:00+00:00"
}
```

说明：
- 后端在训练状态关键变更时会自动触发评估（事件驱动）。
- 该接口用于定时补偿扫描，防止漏触发。

## 四、活跃告警 / 历史告警（6.5 / 6.6）

告警字段：
- `status`: `active` | `resolved`
- `source_type`: 目前固定 `training_run`
- `source_id`: 对应训练任务 `run_id`
- `trigger_count`: 命中次数
- `first_triggered_at` / `last_triggered_at` / `resolved_at`
- `acked_at` / `acked_by`

### 1) 活跃告警列表
GET `/alarms/active?page=1&page_size=50&severity=high&rule_type=training_run_failed&source_id=...`

### 2) 确认活跃告警（ACK）
POST `/alarms/active/{alert_id}/ack`
```json
{
  "acked_by": "alice"
}
```

说明：ACK 仅标记确认，不会将 `active` 直接改为 `resolved`。

### 3) 历史告警列表
GET `/alarms/history?page=1&page_size=50&severity=high&rule_type=training_run_failed&source_id=...`

### 4) 告警摘要（角标/概览）
GET `/alarms/summary`
```json
{
  "active_total": 5,
  "by_severity": {
    "critical": 1,
    "high": 3,
    "medium": 1,
    "low": 0
  }
}
```

## 五、前端联调建议

1. 页面初始化：
   - 调 `GET /alarms/rule-types` 构建规则表单；
   - 调 `GET /alarms/rules` 获取当前配置。
2. 活跃告警页：
   - 轮询 `GET /alarms/summary`（例如 10~30 秒）更新角标；
   - 轮询或手动刷新 `GET /alarms/active`。
3. 历史告警页：
   - 使用 `GET /alarms/history` 分页与筛选。
4. 运维补偿：
   - 可由前端按钮或离线任务定时调用 `POST /alarms/evaluate`。


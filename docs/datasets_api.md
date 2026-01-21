# 数据集管理 API（v2）

路由前缀：`/datasets`

说明：如果你的应用将 v2 路由挂到 `/api/v2`，则实际路径是 `/api/v2/datasets`。

## 通用分页返回

```json
{
  "items": [ ... ],
  "meta": { "page": 1, "page_size": 50, "total": 123 }
}
```

## 枚举值

- `dataset_type`: `detection` | `segmentation` | `classification`
- `dataset_version_status`: `created` | `finalized` | `failed`
- `split`: `train` | `val` | `test`

## 1) 数据集列表

GET `/datasets`

Query
- `page` int, default 1, min 1
- `page_size` int, default 50, max 500

Response：`Page[DatasetOut]`

`DatasetOut`
```json
{
  "dataset_id": 1,
  "name": "my-ds",
  "dataset_type": "detection",
  "storage_path": "token_or_relative_path",
  "description": "desc",
  "active_version_id": 3,
  "created_at": "2025-01-01T12:00:00",
  "updated_at": "2025-01-02T12:00:00"
}
```

前端示例（axios）
```ts
axios.get('/api/v2/datasets', { params: { page: 1, page_size: 50 } })
```

## 2) 创建数据集

POST `/datasets`

Body：JSON `DatasetCreate`
- `name` string (1–255)
- `dataset_type` enum
- `storage_path` string | null (1–500, optional; defaults to `name`)
- `description` string | null

Response：`DatasetOut`（201）

前端示例
```ts
axios.post('/api/v2/datasets', {
  name: 'my-ds',
  dataset_type: 'detection',
  description: 'demo'
})
```

## 3) 获取数据集

GET `/datasets/{dataset_id}`

Response：`DatasetOut`

## 4) 获取数据集详情

GET `/datasets/{dataset_id}/detail`

Query
- `versions_limit` int, default 20, max 200
- `events_limit` int, default 20, max 200

Response：`DatasetDetailOut`
```json
{
  "dataset": { ...DatasetOut },
  "statistics": { ...DatasetStatisticsOut } | null,
  "active_version": { ...DatasetVersionOut } | null,
  "versions": [ ...DatasetVersionOut ],
  "events": [ ...DatasetEventOut ]
}
```

## 5) 更新数据集

PATCH `/datasets/{dataset_id}`

Body：JSON `DatasetUpdate`
- `name` string | null
- `description` string | null
- `active_version_id` int | null

Response：`DatasetOut`

## 6) 删除数据集

DELETE `/datasets/{dataset_id}`

Query
- `delete_files` bool, default false
- `force` bool, default false（删除数据集及其关联项目/训练/模型版本）

Response：`DeleteResponse`
```json
{ "ok": true, "message": "Dataset deleted" }
```

## 7) 上传数据集压缩包（更新已有数据集）

POST `/datasets/{dataset_id}/upload`

Content-Type：`multipart/form-data`

Note: dataset must already exist (create it via POST `/datasets`).

FormData
- `file` 上传文件（必填）
- `message` string | null
- `created_by` string | null
- `create_version` bool, default true
- `activate` bool, default true

Response：`DatasetOut`（201）

前端示例（fetch）
```ts
const fd = new FormData();
fd.append('file', file);
fd.append('create_version', 'true');
fd.append('activate', 'true');

fetch('/api/v2/datasets/1/upload', { method: 'POST', body: fd });
```

## 8) 导入/上传（兼容别名，上传到已有数据集）

POST `/datasets/import`

POST `/datasets/upload`

Content-Type：`multipart/form-data`

FormData
- `dataset_id` int（必填）
- `file` 上传文件（必填）
- `message` string | null
- `created_by` string | null

Response：`DatasetOut`（201）

Note: these endpoints no longer create datasets. Use POST `/datasets` then POST `/datasets/{id}/upload`.

## 9) 上传图片/标注文件

POST `/datasets/{dataset_id}/uploads/images`

Content-Type：`multipart/form-data`

FormData
- `files` or `images` 图片文件数组（必填，可多次 append）
- `relative_dir` string, default `images`
- `labels` or `annotations` 标注文件数组（检测数据集必填；需与图片一一对应）
- `labels_relative_dir` string | null
- `message` string | null
- `created_by` string | null
- `create_snapshot` bool, default false

Note: backend always validates image-label pairing for detection datasets and always creates/activates a new dataset version.

Response：`DatasetImageUploadOut`
```json
{
  "dataset_id": 1,
  "event_id": 10,
  "version_id": 3,
  "version": 3,
  "active_version_id": 3,
  "relative_dir": "images",
  "saved_count": 100,
  "saved_files": ["images/a.jpg", "..."],
  "truncated": false,
  "total_bytes": 123456,
  "labels_relative_dir": "labels",
  "saved_label_count": 100,
  "saved_label_files": ["labels/a.txt", "..."],
  "max_class_id": 5,
  "nc_before": 3,
  "nc_after": 6,
  "added_class_ids": [4,5],
  "class_names_updated": true,
  "created_at": "2025-01-01T12:00:00"
}
```

前端示例（axios）
```ts
const fd = new FormData();
files.forEach(f => fd.append('images', f));
labels.forEach(l => fd.append('annotations', l));
fd.append('relative_dir', 'images');
axios.post('/api/v2/datasets/1/uploads/images', fd);
```

## 10) 事件列表

GET `/datasets/{dataset_id}/events`

Query
- `page` int, default 1
- `page_size` int, default 50
- `event_type` string | null

Response：`Page[DatasetEventOut]`

`DatasetEventOut`
```json
{
  "event_id": 1,
  "dataset_id": 1,
  "version_id": 3,
  "event_type": "upload",
  "message": "xxx",
  "data": {},
  "created_by": "user",
  "created_at": "2025-01-01T12:00:00"
}
```

## 11) 版本列表

GET `/datasets/{dataset_id}/versions`

Query
- `page` int, default 1
- `page_size` int, default 50

Response：`Page[DatasetVersionOut]`

`DatasetVersionOut`（示例）
```json
{
  "version_id": 3,
  "dataset_id": 1,
  "version": 3,
  "parent_version_id": 2,
  "status": "finalized",
  "message": "v3",
  "manifest_path": "path/to/manifest.json",
  "snapshot_path": "path/to/snapshot",
  "file_count": 123,
  "size_bytes": 123456,
  "meta": {},
  "created_by": "user",
  "created_at": "2025-01-01T12:00:00"
}
```

## 12) 创建版本

POST `/datasets/{dataset_id}/versions`

Body：JSON `DatasetVersionCreate`
- `message` string | null（版本说明/变更说明）
- `created_by` string | null
- `create_snapshot` bool, default false（是否创建快照拷贝）

Response：`DatasetVersionOut`（201）

## 13) 激活版本

POST `/datasets/{dataset_id}/versions/{version_id}/activate`

Response：`DatasetOut`

## 14) 数据集统计

GET `/datasets/{dataset_id}/statistics`

Query
- `version_id` int | null（不传则取 active version）

Response：`DatasetStatisticsOut`
```json
{
  "dataset_id": 1,
  "version_id": 3,
  "version": 3,
  "total_files": 1000,
  "total_size_bytes": 123456789,
  "total_size_mb": 117.7,
  "total_images": 1000,
  "annotations_count": 1000
}
```

## 15) 版本差异

GET `/datasets/{dataset_id}/versions/{version_id}/diff`

Query
- `base_version_id` int | null（不传可能默认对比父版本）
- `limit` int, default 200

Response：`DatasetVersionDiffOut`
```json
{
  "dataset_id": 1,
  "base_version_id": 2,
  "base_version": 2,
  "version_id": 3,
  "version": 3,
  "summary": { "added": 10, "removed": 2, "modified": 3 },
  "added": ["images/a.jpg"],
  "removed": ["images/b.jpg"],
  "modified": ["images/c.jpg"],
  "truncated": { "added": false, "removed": false, "modified": false }
}
```

## 16) 文件列表

GET `/datasets/{dataset_id}/files`

Query
- `page` int, default 1
- `page_size` int, default 50
- `version_id` int | null
- `kind` string, default `image`（可选：`image` / `label` / `all`）
- `prefix` string | null
- `q` string | null
- `include_missing` bool, default false

Response：`Page[DatasetFileOut]`

`DatasetFileOut`
```json
{
  "path": "images/a.jpg",
  "size_bytes": 12345,
  "mtime": 1730000000.0,
  "url": "/static/datasets/...",
  "exists": true
}
```

## 17) 划分训练/验证集

POST `/datasets/{dataset_id}/split`

Body：JSON `DatasetSplitRequest`
- `version_id` int | null
- `train_ratio` float (0,1), default 0.8
- `val_ratio` float (0,1) | null
- `seed` int | null
- `shuffle` bool, default true
- `overwrite` bool, default true

Response：`DatasetSplitSummary`
```json
{
  "dataset_id": 1,
  "version_id": 3,
  "version": 3,
  "total_images": 1000,
  "train_count": 800,
  "val_count": 200,
  "train_ratio": 0.8,
  "val_ratio": 0.2,
  "seed": 42,
  "shuffle": true
}
```

## 18) 获取划分结果

GET `/datasets/{dataset_id}/split`

Query
- `page` int, default 1
- `page_size` int, default 50
- `version_id` int | null
- `split` string | null（`train` / `val` / `unassigned` / `none` / `null` / `unsplit`）

Response：`DatasetSplitResultOut`
```json
{
  "summary": { ...DatasetSplitSummary },
  "items": [
    {
      "image_id": 1,
      "dataset_id": 1,
      "dataset_version_id": 3,
      "path": "images/a.jpg",
      "split": "train",
      "created_at": "2025-01-01T12:00:00",
      "updated_at": "2025-01-01T12:00:00"
    }
  ],
  "meta": { "page": 1, "page_size": 50, "total": 1000 }
}
```

## 错误说明（常见）

- 参数校验失败通常返回 422（FastAPI 默认）
- `dataset_type` 非法会触发 `ValidationError("Invalid dataset_type")`

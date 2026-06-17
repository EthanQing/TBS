from __future__ import annotations

import json
from pathlib import Path

from train_platform.schemas.v3.dataset_imports import DatasetImportRootOut
from train_platform.services.v3.dataset_import_service import DatasetImportService


def test_list_entries_counts_large_child_directory_without_fast_scan_truncation(tmp_path: Path, monkeypatch) -> None:
    imports_root = tmp_path / "imports"
    dataset_root = imports_root / "People Detection.illegal_labelme"
    dataset_root.mkdir(parents=True)
    for idx in range(901):
        (dataset_root / f"{idx:04d}.jpg").write_bytes(b"image")
        (dataset_root / f"{idx:04d}.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")

    monkeypatch.setattr(
        DatasetImportService,
        "roots",
        lambda _self: [
            DatasetImportRootOut(
                root_id="default",
                path=str(imports_root),
                label="default",
                exists=True,
                readable=True,
            )
        ],
    )

    result = DatasetImportService().list_entries(root_id="default", path="")

    entry = next(item for item in result.entries if item.name == "People Detection.illegal_labelme")
    assert entry.image_count == 901
    assert entry.json_count == 901
    assert entry.label_count == 0

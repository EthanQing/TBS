from __future__ import annotations

import csv
import json
import shutil
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping


LAYOUT_VERSION = 1
ULTRALYTICS_ENGINE = "ultralytics-yolo"
ULTRALYTICS_OUTPUT_DIR = "output"


@dataclass(frozen=True)
class TrainingExecutionPaths:
    run_root: Path
    runtime_dir: Path
    logs_dir: Path
    output_dir: Path

    @property
    def weights_dir(self) -> Path:
        return self.output_dir / "weights"

    @property
    def data_runtime_yaml(self) -> Path:
        return self.runtime_dir / "data.runtime.yaml"

    @property
    def layout_manifest(self) -> Path:
        return self.runtime_dir / "layout.json"

    @property
    def stdout_log(self) -> Path:
        return self.logs_dir / "train.stdout.log"

    @property
    def stderr_log(self) -> Path:
        return self.logs_dir / "train.stderr.log"

    def output_file(self, relative_path: str) -> Path:
        candidate = (self.output_dir / relative_path).resolve(strict=False)
        output_root = self.output_dir.resolve(strict=False)
        if candidate != output_root and output_root not in candidate.parents:
            raise ValueError("output path escapes the framework output directory")
        return candidate


def _base_paths(run_root: Path, output_dir: Path) -> TrainingExecutionPaths:
    root = Path(run_root).resolve(strict=False)
    return TrainingExecutionPaths(
        run_root=root,
        runtime_dir=root / "runtime",
        logs_dir=root / "logs",
        output_dir=output_dir,
    )


def new_ultralytics_paths(run_root: Path) -> TrainingExecutionPaths:
    root = Path(run_root).resolve(strict=False)
    output = (root / ULTRALYTICS_OUTPUT_DIR).resolve(strict=False)
    if root not in output.parents:
        raise ValueError("Ultralytics output directory escapes the run root")
    return _base_paths(root, output)


def _layout_output_dir(run_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("training layout output_dir must be a non-empty relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    relative = Path(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part == ".." for part in posix.parts)
        or any(part == ".." for part in windows.parts)
    ):
        raise ValueError("training layout output_dir must be confined to the run root")
    root = run_root.resolve(strict=False)
    output = (root / relative).resolve(strict=False)
    if output != root and root not in output.parents:
        raise ValueError("training layout output_dir escapes the run root")
    return output


def read_ultralytics_paths(run_root: Path) -> TrainingExecutionPaths:
    """Resolve recorded output layout, or the legacy run-root layout when unrecorded."""

    root = Path(run_root).resolve(strict=False)
    manifest = root / "runtime" / "layout.json"
    if not manifest.exists():
        return _base_paths(root, root)
    try:
        layout = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid training layout manifest: {manifest}") from exc
    if not isinstance(layout, dict):
        raise ValueError(f"invalid training layout manifest: {manifest}")
    if layout.get("version") != LAYOUT_VERSION:
        raise ValueError(f"unsupported training layout version: {layout.get('version')!r}")
    if layout.get("engine") != ULTRALYTICS_ENGINE:
        raise ValueError(f"unexpected training layout engine: {layout.get('engine')!r}")
    return _base_paths(root, _layout_output_dir(root, layout.get("output_dir")))


def prepare_ultralytics_execution(run_root: Path) -> TrainingExecutionPaths:
    paths = new_ultralytics_paths(run_root)
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = paths.layout_manifest
    temporary = paths.runtime_dir / "layout.json.tmp"
    temporary.write_text(
        json.dumps(
            {"version": LAYOUT_VERSION, "engine": ULTRALYTICS_ENGINE, "output_dir": ULTRALYTICS_OUTPUT_DIR},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest)
    return paths


def _epoch_column(header: list[str]) -> int | None:
    return next((index for index, name in enumerate(header) if name.strip() == "epoch"), None)


def _positive_epoch(value: object) -> int | None:
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite() or number != number.to_integral_value() or number < 1:
        return None
    return int(number)


def _checkpoint_history(train_results: Mapping[str, object] | None, cutoff: int) -> tuple[list[str], list[list[str]]] | None:
    if not isinstance(train_results, Mapping) or not train_results:
        return None
    header = [str(name) for name in train_results.keys()]
    epoch_index = _epoch_column(header)
    columns = list(train_results.values())
    if epoch_index is None or not columns or any(not isinstance(column, (list, tuple)) for column in columns):
        return None
    lengths = {len(column) for column in columns}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        return None
    rows: list[list[str]] = []
    for values in zip(*columns):
        epoch = _positive_epoch(values[epoch_index])
        if epoch is not None and epoch <= cutoff:
            rows.append(["" if value is None else str(value) for value in values])
    return (header, rows) if rows else None


def _csv_history(path: Path, cutoff: int) -> tuple[list[str], list[list[str]]] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            records = list(csv.reader(file))
    except (OSError, UnicodeError, csv.Error):
        return None
    if not records or not records[0]:
        return None
    header = records[0]
    epoch_index = _epoch_column(header)
    if epoch_index is None:
        return None
    rows = []
    for row in records[1:]:
        if len(row) != len(header):
            continue
        epoch = _positive_epoch(row[epoch_index])
        if epoch is not None and epoch <= cutoff:
            rows.append(row)
    return (header, rows) if rows else None


def _raw_csv(path: Path) -> tuple[list[str], list[list[str]]] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            records = list(csv.reader(file))
    except (OSError, UnicodeError, csv.Error):
        return None
    return (records[0], records[1:]) if records and records[0] else None


def _normalized_cell(value: str) -> tuple[str, object]:
    stripped = value.strip()
    try:
        number = Decimal(stripped)
        if number.is_finite():
            return "number", number.normalize()
    except InvalidOperation:
        pass
    return "text", stripped


def _rows_equivalent(left: list[str], right: list[str]) -> bool:
    return len(left) == len(right) and all(_normalized_cell(a) == _normalized_cell(b) for a, b in zip(left, right))


def _rows_by_epoch(header: list[str], rows: list[list[str]], cutoff: int) -> dict[int, list[str]]:
    epoch_index = _epoch_column(header)
    if epoch_index is None:
        return {}
    result: dict[int, list[str]] = {}
    for row in rows:
        if len(row) != len(header):
            continue
        epoch = _positive_epoch(row[epoch_index])
        if epoch is not None and epoch <= cutoff and epoch not in result:
            result[epoch] = row
    return result


def _write_csv_atomic(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as file:
            temporary_name = file.name
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _csv_matches(path: Path, header: list[str], rows: list[list[str]]) -> bool:
    raw = _raw_csv(path)
    return bool(
        raw is not None
        and raw[0] == header
        and len(raw[1]) == len(rows)
        and all(_rows_equivalent(existing, expected) for existing, expected in zip(raw[1], rows))
    )


def _copy_file_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as file:
            temporary_name = file.name
        shutil.copy2(source, temporary_name)
        Path(temporary_name).replace(target)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def prepare_ultralytics_resume_output(
    source_output_dir: Path,
    target_output_dir: Path,
    *,
    checkpoint_epoch: int,
    train_results: Mapping[str, object] | None,
) -> None:
    """Carry resume history into a different Ultralytics output directory."""

    if isinstance(checkpoint_epoch, bool) or not isinstance(checkpoint_epoch, int) or checkpoint_epoch < 0:
        raise ValueError("resume checkpoint epoch must be a non-negative integer")
    source = Path(source_output_dir).resolve(strict=False)
    target = Path(target_output_dir).resolve(strict=False)
    if source == target:
        return

    target.mkdir(parents=True, exist_ok=True)
    source_best = source / "weights" / "best.pt"
    target_best = target / "weights" / "best.pt"
    if source_best.is_file() and not target_best.is_file():
        _copy_file_atomic(source_best, target_best)

    cutoff = checkpoint_epoch + 1
    selected = _checkpoint_history(train_results, cutoff) or _csv_history(source / "results.csv", cutoff)
    if selected is None:
        raw_target = _raw_csv(target / "results.csv")
        if raw_target is not None and _epoch_column(raw_target[0]) is not None:
            valid_rows = list(_rows_by_epoch(raw_target[0], raw_target[1], cutoff).values())
            if not _csv_matches(target / "results.csv", raw_target[0], valid_rows):
                _write_csv_atomic(target / "results.csv", raw_target[0], valid_rows)
        return
    header, selected_rows = selected
    selected_by_epoch = _rows_by_epoch(header, selected_rows, cutoff)
    if not selected_by_epoch:
        return

    target_csv = target / "results.csv"
    target_history = _csv_history(target_csv, cutoff)
    final_by_epoch = dict(selected_by_epoch)
    if target_history is not None and target_history[0] == header:
        target_by_epoch = _rows_by_epoch(*target_history, cutoff)
        overlap = set(target_by_epoch) & set(selected_by_epoch)
        if overlap and all(_rows_equivalent(target_by_epoch[epoch], selected_by_epoch[epoch]) for epoch in overlap):
            final_by_epoch = dict(target_by_epoch)
            final_by_epoch.update(selected_by_epoch)

    final_rows = [final_by_epoch[epoch] for epoch in sorted(final_by_epoch)]
    if _csv_matches(target_csv, header, final_rows):
        return
    _write_csv_atomic(target_csv, header, final_rows)


def resolve_ultralytics_checkpoint(run_root: Path, kind: str = "last") -> Path | None:
    if kind not in {"best", "last"}:
        raise ValueError("checkpoint kind must be 'best' or 'last'")
    candidate = read_ultralytics_paths(run_root).weights_dir / f"{kind}.pt"
    return candidate if candidate.is_file() else None


def resolve_ultralytics_resume_checkpoint(run_root: Path) -> Path | None:
    paths = read_ultralytics_paths(run_root)
    candidates = [paths.weights_dir / "last.pt"]
    legacy = paths.run_root / "weights" / "last.pt"
    if legacy != candidates[0]:
        candidates.append(legacy)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


__all__ = [
    "TrainingExecutionPaths",
    "new_ultralytics_paths",
    "prepare_ultralytics_execution",
    "prepare_ultralytics_resume_output",
    "read_ultralytics_paths",
    "resolve_ultralytics_checkpoint",
    "resolve_ultralytics_resume_checkpoint",
]

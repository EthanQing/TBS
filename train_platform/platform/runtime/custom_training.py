from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from train_platform.platform.filesystem import atomic_write_json, atomic_write_text


@dataclass(frozen=True)
class CustomTrainingArtifactEvent:
    """Transport payload for one artifact event emitted by a custom trainer."""

    role: str
    path: str
    format: str | None = None
    meta: Mapping[str, Any] | None = None


MetricsCallback = Callable[[int, Mapping[str, float]], None]
LogCallback = Callable[[str], None]
ArtifactCallback = Callable[[CustomTrainingArtifactEvent], None]


class CustomTrainingRuntimeError(RuntimeError):
    """The custom training child process could not complete normally."""


class CustomTrainingProtocolError(CustomTrainingRuntimeError):
    """The child emitted an invalid JSONL control event."""


class CustomTrainingCancelled(CustomTrainingRuntimeError):
    """The parent observed cancellation and signalled the child process."""


def _validate_event(raw_line: str) -> tuple[str, Any]:
    try:
        event = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise CustomTrainingProtocolError(f"Child emitted invalid JSON: {exc.msg}") from exc
    if not isinstance(event, dict):
        raise CustomTrainingProtocolError("Child control event must be a JSON object")

    event_type = event.get("type")
    if event_type == "log":
        message = event.get("message")
        if not isinstance(message, str):
            raise CustomTrainingProtocolError("Log event message must be a string")
        return "log", message

    if event_type == "metrics":
        epoch = event.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise CustomTrainingProtocolError("Metrics event epoch must be a non-negative integer")
        raw_metrics = event.get("metrics")
        if not isinstance(raw_metrics, dict):
            raise CustomTrainingProtocolError("Metrics event metrics must be a JSON object")
        metrics: dict[str, float] = {}
        for key, value in raw_metrics.items():
            if not isinstance(key, str) or not key.strip():
                raise CustomTrainingProtocolError("Metric names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CustomTrainingProtocolError(f"Metric '{key}' must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise CustomTrainingProtocolError(f"Metric '{key}' must be finite")
            metrics[key] = number
        return "metrics", (epoch, metrics)

    if event_type == "artifact":
        role = event.get("role")
        path = event.get("path")
        format_value = event.get("format")
        meta = event.get("meta")
        if not isinstance(role, str):
            raise CustomTrainingProtocolError("Artifact event role must be a string")
        if not isinstance(path, str):
            raise CustomTrainingProtocolError("Artifact event path must be a string")
        if format_value is not None and not isinstance(format_value, str):
            raise CustomTrainingProtocolError("Artifact event format must be a string or null")
        if meta is not None and not isinstance(meta, dict):
            raise CustomTrainingProtocolError("Artifact event meta must be a JSON object or null")
        return "artifact", CustomTrainingArtifactEvent(
            role=role,
            path=path,
            format=format_value,
            meta=meta,
        )

    raise CustomTrainingProtocolError(f"Unknown child control event type: {event_type!r}")


def _terminate_child_process_tree(
    process: subprocess.Popen,
    *,
    wait_timeout: float = 2.0,
) -> None:
    """Best-effort termination for the custom child and its descendants."""

    timeout = max(0.1, float(wait_timeout))
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        try:
            process.terminate()
        except OSError:
            pass

    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        try:
            process.kill()
        except OSError:
            pass

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


def _consume_event_line(
    raw_line: str,
    *,
    on_metrics: MetricsCallback,
    on_log: LogCallback | None,
    on_artifact: ArtifactCallback | None,
) -> None:
    if not raw_line.strip():
        return
    parsed_kind, payload = _validate_event(raw_line)
    if parsed_kind == "metrics":
        epoch, metrics = payload
        on_metrics(epoch, metrics)
    elif parsed_kind == "artifact":
        if on_artifact is not None:
            on_artifact(payload)
    elif on_log is not None:
        on_log(payload)
    else:
        print(payload, flush=True)


def _drain_event_file(
    event_file: Any,
    pending: str,
    *,
    on_metrics: MetricsCallback,
    on_log: LogCallback | None,
    on_artifact: ArtifactCallback | None,
) -> str:
    pending += event_file.read()
    complete_lines = pending.splitlines(keepends=True)
    if complete_lines and not complete_lines[-1].endswith(("\n", "\r")):
        pending = complete_lines.pop()
    else:
        pending = ""

    for raw_line in complete_lines:
        _consume_event_line(raw_line, on_metrics=on_metrics, on_log=on_log, on_artifact=on_artifact)
    return pending


def run_custom_training(
    context: Mapping[str, Any],
    *,
    context_path: Path,
    cancel_marker_path: Path,
    cancel_requested: Callable[[], bool],
    on_metrics: MetricsCallback,
    on_log: LogCallback | None = None,
    on_artifact: ArtifactCallback | None = None,
    cancel_grace_seconds: float = 3.0,
) -> int:
    """Run one materialized custom trainer and consume its JSONL events.

    The caller owns lifecycle persistence.  This function only owns the child
    process, its context/control files, and the protocol between both processes.
    """

    context_path = Path(context_path)
    cancel_marker_path = Path(cancel_marker_path)
    try:
        event_path = Path(context["event_path"])
    except (KeyError, TypeError) as exc:
        raise CustomTrainingRuntimeError("Custom training context is missing 'event_path'") from exc

    cancel_marker_path.unlink(missing_ok=True)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    event_path.write_text("", encoding="utf-8")
    atomic_write_json(context_path, context)

    command = [
        sys.executable,
        "-m",
        "train_platform.workers.training.custom_entry",
        "--context",
        str(context_path),
    ]
    try:
        process = subprocess.Popen(
            command,
            start_new_session=(os.name != "nt"),
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),  # type: ignore[attr-defined]
        )
    except OSError as exc:
        raise CustomTrainingRuntimeError(f"Failed to start custom training subprocess: {exc}") from exc

    cancel_started_at: float | None = None
    cancel_marker_written = False
    failure: BaseException | None = None
    grace = max(0.0, float(cancel_grace_seconds))
    pending_events = ""

    try:
        with event_path.open("r", encoding="utf-8") as event_file:
            while process.poll() is None:
                try:
                    pending_events = _drain_event_file(
                        event_file,
                        pending_events,
                        on_metrics=on_metrics,
                        on_log=on_log,
                        on_artifact=on_artifact,
                    )
                except BaseException as exc:
                    failure = exc
                    break

                if not cancel_marker_written:
                    try:
                        requested = bool(cancel_requested())
                    except BaseException as exc:
                        failure = exc
                        break
                    if requested:
                        atomic_write_text(cancel_marker_path, "cancel\n")
                        cancel_marker_written = True
                        cancel_started_at = time.monotonic()

                if cancel_marker_written and cancel_started_at is not None:
                    if time.monotonic() - cancel_started_at >= grace:
                        _terminate_child_process_tree(process, wait_timeout=max(0.1, grace))
                        break
                time.sleep(0.1)

            if failure is None:
                try:
                    pending_events = _drain_event_file(
                        event_file,
                        pending_events,
                        on_metrics=on_metrics,
                        on_log=on_log,
                        on_artifact=on_artifact,
                    )
                except BaseException as exc:
                    failure = exc
    finally:
        if failure is not None:
            _terminate_child_process_tree(process)

    if cancel_marker_written:
        _terminate_child_process_tree(process)
        raise CustomTrainingCancelled("Custom training cancellation requested")

    if failure is not None:
        if isinstance(failure, CustomTrainingRuntimeError):
            raise failure
        raise CustomTrainingRuntimeError("Custom training subprocess event handling failed") from failure

    exit_code = int(process.returncode or 0)
    if exit_code != 0:
        _terminate_child_process_tree(process)
    return exit_code


__all__ = [
    "CustomTrainingCancelled",
    "CustomTrainingArtifactEvent",
    "CustomTrainingProtocolError",
    "CustomTrainingRuntimeError",
    "run_custom_training",
]

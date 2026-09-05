from __future__ import annotations

import json
import math
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from train_platform.platform.filesystem import atomic_write_json, atomic_write_text


MetricsCallback = Callable[[int, Mapping[str, float]], None]
LogCallback = Callable[[str], None]


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

    raise CustomTrainingProtocolError(f"Unknown child control event type: {event_type!r}")


def _read_stdout(stream: Any, events: queue.Queue[tuple[str, str | None]]) -> None:
    try:
        for line in stream:
            events.put(("stdout", line))
    finally:
        events.put(("stdout_eof", None))


def _forward_stderr(stream: Any) -> None:
    for line in stream:
        print(line.rstrip("\r\n"), file=sys.stderr, flush=True)


def _stop_process(process: subprocess.Popen[str], *, wait_timeout: float = 2.0) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except OSError:
        pass
    try:
        process.wait(timeout=max(0.1, float(wait_timeout)))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=max(0.1, float(wait_timeout)))
    except subprocess.TimeoutExpired:
        pass


def run_custom_training(
    context: Mapping[str, Any],
    *,
    context_path: Path,
    cancel_marker_path: Path,
    cancel_requested: Callable[[], bool],
    on_metrics: MetricsCallback,
    on_log: LogCallback | None = None,
    cancel_grace_seconds: float = 3.0,
) -> int:
    """Run one materialized custom trainer and consume its JSONL events.

    The caller owns lifecycle persistence.  This function only owns the child
    process, its context/control files, and the protocol between both processes.
    """

    context_path = Path(context_path)
    cancel_marker_path = Path(cancel_marker_path)
    cancel_marker_path.unlink(missing_ok=True)
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
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise CustomTrainingRuntimeError(f"Failed to start custom training subprocess: {exc}") from exc

    assert process.stdout is not None
    assert process.stderr is not None
    if process.stdin is not None:
        try:
            process.stdin.close()
        except OSError:
            pass

    events: queue.Queue[tuple[str, str | None]] = queue.Queue()
    stdout_thread = threading.Thread(target=_read_stdout, args=(process.stdout, events), daemon=True)
    stderr_thread = threading.Thread(target=_forward_stderr, args=(process.stderr,), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    cancel_started_at: float | None = None
    cancel_marker_written = False
    stdout_eof = False
    failure: BaseException | None = None
    grace = max(0.0, float(cancel_grace_seconds))

    try:
        while not stdout_eof:
            try:
                event_kind, raw_line = events.get(timeout=0.1)
            except queue.Empty:
                event_kind, raw_line = "", None

            if event_kind == "stdout":
                if raw_line and raw_line.strip():
                    try:
                        parsed_kind, payload = _validate_event(raw_line)
                        if parsed_kind == "metrics":
                            epoch, metrics = payload
                            on_metrics(epoch, metrics)
                        elif on_log is not None:
                            on_log(payload)
                        else:
                            print(payload, flush=True)
                    except BaseException as exc:
                        failure = exc
                        break
            elif event_kind == "stdout_eof":
                stdout_eof = True

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

            if cancel_marker_written and process.poll() is None and cancel_started_at is not None:
                if time.monotonic() - cancel_started_at >= grace:
                    _stop_process(process, wait_timeout=max(0.1, grace))

    finally:
        if failure is not None or process.poll() is None:
            _stop_process(process)
        stdout_thread.join(timeout=2.0)
        stderr_thread.join(timeout=2.0)

    if cancel_marker_written:
        raise CustomTrainingCancelled("Custom training cancellation requested")

    if failure is not None:
        if isinstance(failure, CustomTrainingRuntimeError):
            raise failure
        raise CustomTrainingRuntimeError("Custom training subprocess event handling failed") from failure

    return int(process.returncode or 0)


__all__ = [
    "CustomTrainingCancelled",
    "CustomTrainingProtocolError",
    "CustomTrainingRuntimeError",
    "run_custom_training",
]

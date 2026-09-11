from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

import psutil


logger = logging.getLogger(__name__)


class UltralyticsDDPCancelled(RuntimeError):
    pass


class UltralyticsDDPError(RuntimeError):
    pass


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def process_identity(pid: int, **extra: Any) -> dict[str, Any]:
    process = psutil.Process(int(pid))
    identity: dict[str, Any] = {"pid": int(pid), "create_time": float(process.create_time()), **extra}
    if os.name != "nt":
        try:
            identity["pgid"] = int(os.getpgid(int(pid)))
        except OSError:
            pass
    return identity


def register_process(processes_dir: Path, name: str, **identity: Any) -> None:
    _atomic_json(Path(processes_dir) / f"{name}.json", identity)


def _load_identities(processes_dir: Path) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    if not processes_dir.is_dir():
        return identities
    for path in processes_dir.glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                identities.append(value)
        except (OSError, ValueError):
            continue
    return identities


def _matching_process(identity: Mapping[str, Any]) -> psutil.Process | None:
    try:
        process = psutil.Process(int(identity["pid"]))
        if float(process.create_time()) != float(identity["create_time"]):
            return None
        return process
    except (KeyError, TypeError, ValueError, psutil.Error):
        return None


def _identity_matches_scope(
    identity: Mapping[str, Any],
    *,
    run_id: str,
    attempt_id: str,
    owner: Mapping[str, Any],
) -> bool:
    if identity.get("run_id") != run_id or identity.get("attempt_id") != attempt_id:
        return False
    identity_owner = identity.get("execution_owner")
    if not isinstance(identity_owner, Mapping):
        return False
    return not owner or all(
        identity_owner.get(key) == value
        for key, value in owner.items()
        if value is not None
    )


def _is_live(process: psutil.Process) -> bool:
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def terminate_registered_processes(
    run_root: Path,
    *,
    run_id: str,
    owner: Mapping[str, Any],
    attempt_id: str | None = None,
    grace_seconds: float = 5.0,
) -> list[psutil.Process]:
    ddp_root = Path(run_root) / "runtime" / "ddp"
    if attempt_id is not None:
        attempts = [ddp_root / attempt_id]
    else:
        attempts = list(ddp_root.glob("*")) if ddp_root.is_dir() else []
    processes: dict[tuple[int, float], psutil.Process] = {}
    for attempt in attempts:
        context_path = attempt / "context.json"
        try:
            context = json.loads(context_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(context, dict) or context.get("run_id") != str(run_id):
            continue
        if context.get("attempt_id") != attempt.name:
            continue
        registered_owner = context.get("execution_owner")
        if not isinstance(registered_owner, Mapping):
            continue
        if owner and any(registered_owner.get(key) != value for key, value in owner.items() if value is not None):
            continue
        for identity in _load_identities(attempt / "processes"):
            if not _identity_matches_scope(
                identity,
                run_id=str(run_id),
                attempt_id=str(context["attempt_id"]),
                owner=owner,
            ):
                continue
            process = _matching_process(identity)
            if process is not None and process.pid != os.getpid():
                processes[(process.pid, process.create_time())] = process
                try:
                    for child in process.children(recursive=True):
                        processes[(child.pid, child.create_time())] = child
                except psutil.Error:
                    pass
    for process in sorted(processes.values(), key=lambda item: item.pid, reverse=True):
        try:
            process.terminate()
        except psutil.Error:
            pass
    candidates = [process for process in processes.values() if _is_live(process)]
    _, alive = psutil.wait_procs(candidates, timeout=max(0.0, grace_seconds))
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            pass
    if alive:
        _, alive = psutil.wait_procs(alive, timeout=2.0)
    return [process for process in alive if _is_live(process)]


class MetricsJSONLReader:
    def __init__(self, path: Path, *, run_id: str, attempt_id: str) -> None:
        self.path = Path(path)
        self.run_id = str(run_id)
        self.attempt_id = str(attempt_id)
        self.offset = 0
        self.pending = b""

    def read(self, emit: Callable[[int, dict[str, float]], None], *, final: bool = False) -> None:
        try:
            with self.path.open("rb") as file:
                file.seek(self.offset)
                chunk = file.read()
                self.offset = file.tell()
        except FileNotFoundError:
            return
        data = self.pending + chunk
        lines = data.split(b"\n")
        self.pending = b"" if data.endswith(b"\n") else lines.pop()
        for line in lines:
            if not line.strip():
                continue
            try:
                event = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue
            if not isinstance(event, dict):
                continue
            epoch = event.get("epoch")
            if (
                event.get("type") != "epoch_metrics"
                or event.get("run_id") != self.run_id
                or event.get("attempt_id") != self.attempt_id
                or not isinstance(event.get("metrics"), dict)
                or isinstance(epoch, bool)
                or not isinstance(epoch, int)
                or epoch < 0
            ):
                continue
            metrics: dict[str, float] = {}
            for key, value in event["metrics"].items():
                try:
                    metrics[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            emit(epoch, metrics)
        if final:
            self.pending = b""


def run_ultralytics_ddp(
    context: Mapping[str, Any],
    *,
    cancel_requested: Callable[[], bool],
    upsert_epoch_metrics: Callable[[int, dict[str, float]], None],
    poll_seconds: float = 0.25,
) -> None:
    if not isinstance(context, Mapping):
        raise TypeError("distributed context must be a mapping")
    context_owner = context.get("execution_owner", {})
    if not isinstance(context_owner, Mapping):
        raise ValueError("distributed context execution_owner must be a mapping")
    run_root = Path(str(context["run_root"])).resolve(strict=False)
    attempt_id = uuid.uuid4().hex
    attempt_dir = run_root / "runtime" / "ddp" / attempt_id
    processes_dir = attempt_dir / "processes"
    metrics_path = attempt_dir / "metrics.jsonl"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    processes_dir.mkdir()
    metrics_path.touch()
    payload = dict(context)
    payload.update(
        attempt_id=attempt_id,
        metrics_path=str(metrics_path.resolve()),
        processes_dir=str(processes_dir.resolve()),
        supervisor=process_identity(os.getpid(), role="supervisor"),
    )
    context_path = attempt_dir / "context.json"
    _atomic_json(context_path, payload)
    command = [
        sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
        f"--nproc-per-node={int(payload['world_size'])}", "--rdzv-backend=c10d",
        "--rdzv-endpoint=localhost:0", f"--rdzv-id={attempt_id}", "--max-restarts=0",
        "--module", "train_platform.workers.training.ultralytics_ddp_entry",
        "--context", str(context_path.resolve()),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(payload["cuda_visible_devices"])
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    popen_kwargs: dict[str, Any] = {"env": env}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    reader = MetricsJSONLReader(metrics_path, run_id=str(payload["run_id"]), attempt_id=attempt_id)
    owner = dict(context_owner)
    cancelled = False
    interrupted = False
    return_code: int | None = None
    primary_error: BaseException | None = None
    cleanup_survivors: list[psutil.Process] = []
    previous_handlers: dict[int, Any] = {}
    known_processes: dict[tuple[int, float], psutil.Process] = {}

    def stop_requested(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signum] = signal.signal(signum, stop_requested)
        except (ValueError, OSError):
            pass
    launcher: subprocess.Popen[Any] | None = None

    def observe_descendants() -> None:
        roots: dict[tuple[int, float], psutil.Process] = {}
        if launcher is not None:
            try:
                launcher_process = psutil.Process(launcher.pid)
                roots[(launcher_process.pid, launcher_process.create_time())] = launcher_process
                known_processes[(launcher_process.pid, launcher_process.create_time())] = launcher_process
            except psutil.Error:
                pass
        for identity in _load_identities(processes_dir):
            if not _identity_matches_scope(
                identity,
                run_id=str(payload["run_id"]),
                attempt_id=attempt_id,
                owner=owner,
            ):
                continue
            process = _matching_process(identity)
            if process is not None:
                roots[(process.pid, process.create_time())] = process
        for root in roots.values():
            try:
                descendants = root.children(recursive=True)
            except psutil.Error:
                continue
            for child in descendants:
                try:
                    known_processes[(child.pid, child.create_time())] = child
                    identity = process_identity(
                        child.pid,
                        role="descendant",
                        run_id=payload["run_id"],
                        attempt_id=attempt_id,
                        execution_owner=owner,
                    )
                    register_process(processes_dir, f"observed-{child.pid}", **identity)
                except (OSError, psutil.Error):
                    continue

    def stop_known_processes() -> list[psutil.Process]:
        candidates = [process for process in known_processes.values() if _is_live(process)]
        for process in sorted(candidates, key=lambda item: item.pid, reverse=True):
            try:
                process.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(candidates, timeout=2.0)
        for process in alive:
            try:
                process.kill()
            except psutil.Error:
                pass
        if alive:
            _, alive = psutil.wait_procs(alive, timeout=1.0)
        return [process for process in alive if _is_live(process)]

    def stop_launcher() -> None:
        if launcher is None or launcher.poll() is not None:
            return
        try:
            observe_descendants()
        except (OSError, psutil.Error):
            pass
        try:
            launcher.terminate()
        except OSError:
            pass
        deadline = time.monotonic() + 4.0
        while launcher.poll() is None and time.monotonic() < deadline:
            try:
                observe_descendants()
            except (OSError, psutil.Error):
                pass
            time.sleep(0.1)
        if launcher.poll() is None:
            try:
                launcher.kill()
            except OSError:
                pass

    try:
        logger.info(
            "Starting Ultralytics torchrun run_id=%s attempt_id=%s world_size=%s CUDA_VISIBLE_DEVICES=%s",
            payload["run_id"], attempt_id, payload["world_size"], payload["cuda_visible_devices"],
        )
        launcher = subprocess.Popen(command, **popen_kwargs)
        launcher_process = psutil.Process(launcher.pid)
        known_processes[(launcher_process.pid, launcher_process.create_time())] = launcher_process
        register_process(
            processes_dir,
            "launcher",
            **process_identity(
                launcher.pid,
                role="launcher",
                run_id=payload["run_id"],
                attempt_id=attempt_id,
                execution_owner=owner,
            ),
        )
        while launcher.poll() is None:
            reader.read(upsert_epoch_metrics)
            observe_descendants()
            platform_cancelled = bool(cancel_requested())
            if interrupted or platform_cancelled:
                cancelled = platform_cancelled
                logger.info(
                    "Stopping Ultralytics torchrun run_id=%s attempt_id=%s cancel_requested=%s signal_received=%s",
                    payload["run_id"], attempt_id, cancelled, interrupted,
                )
                stop_launcher()
                break
            time.sleep(max(0.05, poll_seconds))
        if launcher.poll() is None:
            stop_launcher()
        try:
            return_code = int(launcher.wait(timeout=2.0))
        except subprocess.TimeoutExpired:
            launcher.kill()
            return_code = int(launcher.wait(timeout=1.0))
    except BaseException as exc:
        primary_error = exc
    finally:
        try:
            try:
                stop_launcher()
            except BaseException as exc:
                if primary_error is None:
                    primary_error = exc
            try:
                cleanup_survivors = terminate_registered_processes(
                    run_root,
                    run_id=str(payload["run_id"]),
                    owner=owner,
                    attempt_id=attempt_id,
                    grace_seconds=2.0,
                )
            except BaseException as exc:
                if primary_error is None:
                    primary_error = exc
            known_survivors = stop_known_processes()
            known_survivor_pids = {process.pid for process in cleanup_survivors}
            cleanup_survivors.extend(
                process for process in known_survivors if process.pid not in known_survivor_pids
            )
            if launcher is not None:
                try:
                    if launcher.poll() is None:
                        launcher.kill()
                    launcher.wait(timeout=1.0)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    if primary_error is None:
                        primary_error = exc
            try:
                reader.read(upsert_epoch_metrics, final=True)
            except BaseException as exc:
                if primary_error is None:
                    primary_error = exc
        finally:
            for signum, handler in previous_handlers.items():
                try:
                    signal.signal(signum, handler)
                except (ValueError, OSError):
                    pass
    if primary_error is not None:
        raise primary_error
    if cleanup_survivors:
        survivor_pids = ",".join(str(process.pid) for process in cleanup_survivors)
        raise UltralyticsDDPError(f"distributed processes did not exit: pids={survivor_pids} attempt_id={attempt_id}")
    if cancelled:
        raise UltralyticsDDPCancelled(f"distributed training cancelled: attempt_id={attempt_id}")
    if interrupted:
        raise UltralyticsDDPError(f"distributed supervisor received a termination signal: attempt_id={attempt_id}")
    if return_code != 0:
        raise UltralyticsDDPError(f"torchrun exited with code {return_code}: attempt_id={attempt_id}")


__all__ = ["MetricsJSONLReader", "UltralyticsDDPCancelled", "UltralyticsDDPError", "process_identity", "register_process", "run_ultralytics_ddp", "terminate_registered_processes"]

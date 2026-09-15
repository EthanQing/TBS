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

from train_platform.platform.runtime import process_scope
from train_platform.platform.runtime.execution_identity import process_identity, process_state as _process_state


logger = logging.getLogger(__name__)


class UltralyticsDDPCancelled(RuntimeError):
    pass


class UltralyticsDDPError(RuntimeError):
    pass


class UltralyticsDDPCleanupIncomplete(UltralyticsDDPError):
    def __init__(
        self,
        message: str,
        *,
        run_id: str,
        attempt_id: str,
        execution_owner: Mapping[str, Any],
        survivors: list[dict[str, Any]],
        original_error: BaseException | None = None,
        cleanup_errors: list[BaseException] | None = None,
    ) -> None:
        super().__init__(message)
        self.run_id = str(run_id)
        self.attempt_id = str(attempt_id)
        self.execution_owner = dict(execution_owner)
        self.survivors = survivors
        self.original_error = original_error
        self.cleanup_errors = list(cleanup_errors or [])


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def _load_identities_strict(processes_dir: Path) -> list[dict[str, Any]]:
    if not processes_dir.is_dir():
        raise ValueError(f"distributed process registration directory is missing: {processes_dir}")
    identities: list[dict[str, Any]] = []
    for path in processes_dir.glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid distributed process registration: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"invalid distributed process registration: {path}")
        identities.append(value)
    return identities


def _matching_process(identity: Mapping[str, Any]) -> psutil.Process | None:
    if process_scope.compare_process_scope(
        process_scope.identity_process_scope(identity), process_scope.get_process_scope()
    ) != "same":
        return None
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


def _identity_from_process(process: psutil.Process, **extra: Any) -> dict[str, Any]:
    return {
        "pid": int(process.pid), "create_time": float(process.create_time()),
        "process_scope": process_scope.get_process_scope(), **extra,
    }


def _candidate_identity_key(identity: Mapping[str, Any]) -> tuple[int, float, str]:
    pid = int(identity["pid"])
    create_time = float(identity["create_time"])
    extracted = process_scope.identity_process_scope(identity)
    if extracted is not None:
        scope_value: object = extracted
    else:
        owner = identity.get("execution_owner")
        scope_value = {
            "direct": identity.get("process_scope"),
            "owner": owner.get("process_scope") if isinstance(owner, Mapping) else None,
        }
    return pid, create_time, json.dumps(scope_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def terminate_registered_processes(
    run_root: Path,
    *,
    run_id: str,
    owner: Mapping[str, Any],
    attempt_id: str | None = None,
    grace_seconds: float = 5.0,
) -> list[psutil.Process]:
    owner_scope_status = process_scope.compare_process_scope(
        owner.get("process_scope"), process_scope.get_process_scope()
    )
    if owner_scope_status != "same":
        raise UltralyticsDDPCleanupIncomplete(
            f"execution owner process scope is {owner_scope_status}",
            run_id=str(run_id), attempt_id=str(attempt_id or "unknown"),
            execution_owner=owner, survivors=[],
        )
    ddp_root = Path(run_root) / "runtime" / "ddp"
    if attempt_id is not None:
        attempts = [ddp_root / attempt_id]
    else:
        attempts = list(ddp_root.glob("*")) if ddp_root.is_dir() else []
    processes: dict[tuple[int, float], psutil.Process] = {}
    scoped_identities: dict[tuple[int, float], dict[str, Any]] = {}
    selected_attempts: list[Path] = []
    for attempt in attempts:
        context_path = attempt / "context.json"
        try:
            context = json.loads(context_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise UltralyticsDDPCleanupIncomplete(
                f"distributed context is unreadable: attempt_id={attempt.name}",
                run_id=str(run_id), attempt_id=attempt.name,
                execution_owner=owner, survivors=[], cleanup_errors=[exc],
            ) from exc
        if not isinstance(context, dict) or context.get("run_id") != str(run_id):
            if attempt_id is not None:
                error = ValueError(f"distributed context does not match run: {context_path}")
                raise UltralyticsDDPCleanupIncomplete(
                    f"distributed context is invalid: attempt_id={attempt.name}",
                    run_id=str(run_id), attempt_id=attempt.name,
                    execution_owner=owner, survivors=[], cleanup_errors=[error],
                ) from error
            continue
        if context.get("attempt_id") != attempt.name:
            error = ValueError(f"distributed context attempt does not match directory: {context_path}")
            raise UltralyticsDDPCleanupIncomplete(
                f"distributed context is invalid: attempt_id={attempt.name}",
                run_id=str(run_id), attempt_id=attempt.name,
                execution_owner=owner, survivors=[], cleanup_errors=[error],
            ) from error
        registered_owner = context.get("execution_owner")
        if not isinstance(registered_owner, Mapping):
            error = ValueError(f"distributed context owner is invalid: {context_path}")
            raise UltralyticsDDPCleanupIncomplete(
                f"distributed context is invalid: attempt_id={attempt.name}",
                run_id=str(run_id), attempt_id=attempt.name,
                execution_owner=owner, survivors=[], cleanup_errors=[error],
            ) from error
        if owner and any(registered_owner.get(key) != value for key, value in owner.items() if value is not None):
            basic_keys = ("guard_pid", "guard_create_time", "worker_id", "allocation_id")
            if any(registered_owner.get(key) != owner.get(key) for key in basic_keys):
                continue
        context_scope_status = process_scope.compare_process_scope(
            registered_owner.get("process_scope"), process_scope.get_process_scope()
        )
        if context_scope_status != "same":
            raise UltralyticsDDPCleanupIncomplete(
                f"distributed context process scope is {context_scope_status}: attempt_id={attempt.name}",
                run_id=str(run_id), attempt_id=attempt.name,
                execution_owner=owner, survivors=[],
            )
        selected_attempts.append(attempt)
        try:
            identities = _load_identities_strict(attempt / "processes")
        except (OSError, ValueError) as exc:
            raise UltralyticsDDPCleanupIncomplete(
                f"distributed process registrations are unreadable: attempt_id={context['attempt_id']}",
                run_id=str(run_id), attempt_id=str(context["attempt_id"]),
                execution_owner=owner, survivors=[], cleanup_errors=[exc],
            ) from exc
        pending_path = attempt / "cleanup-pending.json"
        try:
            pending = json.loads(pending_path.read_text(encoding="utf-8"))
            if not isinstance(pending, dict) or not isinstance(pending.get("survivors"), list):
                raise ValueError(f"invalid distributed cleanup marker: {pending_path}")
            if pending.get("run_id") != str(run_id) or pending.get("attempt_id") != str(context["attempt_id"]):
                raise ValueError(f"distributed cleanup marker scope mismatch: {pending_path}")
            pending_owner = pending.get("execution_owner")
            if not isinstance(pending_owner, Mapping):
                raise ValueError(f"distributed cleanup marker owner is invalid: {pending_path}")
            if any(pending_owner.get(key) != owner.get(key) for key in ("guard_pid", "guard_create_time", "worker_id")):
                raise ValueError(f"distributed cleanup marker owner mismatch: {pending_path}")
            pending_scope_status = process_scope.compare_process_scope(
                pending_owner.get("process_scope"), process_scope.get_process_scope()
            )
            if pending_scope_status != "same":
                raise UltralyticsDDPCleanupIncomplete(
                    f"distributed cleanup marker process scope is {pending_scope_status}: attempt_id={context['attempt_id']}",
                    run_id=str(run_id), attempt_id=str(context["attempt_id"]),
                    execution_owner=owner, survivors=[],
                )
            if any(not isinstance(item, dict) for item in pending["survivors"]):
                raise ValueError(f"invalid distributed cleanup survivor: {pending_path}")
            identities.extend(pending["survivors"])
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            raise UltralyticsDDPCleanupIncomplete(
                f"distributed cleanup marker is unreadable: attempt_id={context['attempt_id']}",
                run_id=str(run_id),
                attempt_id=str(context["attempt_id"]),
                execution_owner=owner,
                survivors=[],
            )
        for identity in identities:
            if not all(key in identity for key in ("pid", "create_time", "run_id", "attempt_id", "execution_owner")):
                error = ValueError("distributed process identity is incomplete")
                raise UltralyticsDDPCleanupIncomplete(
                    f"distributed process identity is invalid: attempt_id={context['attempt_id']}",
                    run_id=str(run_id), attempt_id=str(context["attempt_id"]),
                    execution_owner=owner, survivors=[], cleanup_errors=[error],
                ) from error
            identity_scope_status = process_scope.compare_process_scope(
                process_scope.identity_process_scope(identity), process_scope.get_process_scope()
            )
            if identity_scope_status != "same":
                raise UltralyticsDDPCleanupIncomplete(
                    f"distributed process scope is {identity_scope_status}: attempt_id={context['attempt_id']}",
                    run_id=str(run_id), attempt_id=str(context["attempt_id"]),
                    execution_owner=owner, survivors=[dict(identity, state="unknown")],
                )
            if not _identity_matches_scope(
                identity,
                run_id=str(run_id),
                attempt_id=str(context["attempt_id"]),
                owner=owner,
            ):
                continue
            try:
                identity_key = (int(identity["pid"]), float(identity["create_time"]))
                scoped_identities[identity_key] = dict(identity)
            except (KeyError, TypeError, ValueError) as exc:
                raise UltralyticsDDPCleanupIncomplete(
                    f"distributed process identity is invalid: attempt_id={context['attempt_id']}",
                    run_id=str(run_id), attempt_id=str(context["attempt_id"]),
                    execution_owner=owner, survivors=[], cleanup_errors=[exc],
                ) from exc
            process = _matching_process(identity)
            if process is not None and process.pid != os.getpid():
                processes[(process.pid, process.create_time())] = process
                try:
                    for child in process.children(recursive=True):
                        processes[(child.pid, child.create_time())] = child
                        child_identity = process_identity(
                            child.pid,
                            role="descendant",
                            run_id=str(run_id),
                            attempt_id=str(context["attempt_id"]),
                            execution_owner=dict(owner),
                        )
                        scoped_identities[(child.pid, child.create_time())] = child_identity
                        try:
                            register_process(attempt / "processes", f"helper-observed-{child.pid}", **child_identity)
                        except OSError:
                            # Keep the in-memory identity so this cleanup pass still
                            # terminates and verifies the child before returning.
                            pass
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
    final_alive: list[psutil.Process] = []
    survivors: list[dict[str, Any]] = []
    for identity in scoped_identities.values():
        state, process = _process_state(identity)
        if state != "dead":
            survivors.append(dict(identity, state=state))
        if state == "live" and process is not None:
            final_alive.append(process)
    for attempt in selected_attempts:
        remaining = [item for item in survivors if item["attempt_id"] == attempt.name]
        pending_path = attempt / "cleanup-pending.json"
        if remaining:
            try:
                _atomic_json(pending_path, {
                    "run_id": str(run_id), "attempt_id": attempt.name,
                    "execution_owner": dict(owner), "survivors": remaining,
                })
            except OSError as exc:
                raise UltralyticsDDPCleanupIncomplete(
                    f"distributed cleanup identities could not be persisted: attempt_id={attempt.name}",
                    run_id=str(run_id), attempt_id=attempt.name,
                    execution_owner=owner, survivors=survivors, cleanup_errors=[exc],
                ) from exc
        else:
            pending_path.unlink(missing_ok=True)
    if any(item["state"] == "unknown" for item in survivors):
        selected_attempt = attempt_id or "multiple"
        raise UltralyticsDDPCleanupIncomplete(
            f"distributed process state could not be confirmed: attempt_id={selected_attempt}",
            run_id=str(run_id), attempt_id=selected_attempt,
            execution_owner=owner, survivors=survivors,
        )
    return final_alive



class MetricsJSONLReader:
    def __init__(self, path: Path, *, run_id: str, attempt_id: str, allocation_id: str | None = None) -> None:
        self.path = Path(path)
        self.run_id = str(run_id)
        self.attempt_id = str(attempt_id)
        self.allocation_id = allocation_id
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
                or event.get("allocation_id") != self.allocation_id
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
    if context.get("allocation_id") and context_owner.get("allocation_id") != context["allocation_id"]:
        raise ValueError("distributed context allocation identity mismatch")
    run_root = Path(str(context["run_root"])).resolve(strict=False)
    attempt_id = uuid.uuid4().hex
    owner_scope_status = process_scope.compare_process_scope(
        context_owner.get("process_scope"), process_scope.get_process_scope()
    )
    if owner_scope_status != "same":
        raise UltralyticsDDPCleanupIncomplete(
            f"distributed supervisor process scope is {owner_scope_status}",
            run_id=str(context["run_id"]), attempt_id=attempt_id,
            execution_owner=context_owner, survivors=[],
        )
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
    reader = MetricsJSONLReader(metrics_path, run_id=str(payload["run_id"]), attempt_id=attempt_id,
                               allocation_id=payload.get("allocation_id"))
    owner = dict(context_owner)
    cancelled = False
    interrupted = False
    return_code: int | None = None
    primary_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
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
        _, alive = psutil.wait_procs(candidates, timeout=0.5)
        for process in alive:
            try:
                process.kill()
            except psutil.Error:
                pass
        if alive:
            _, alive = psutil.wait_procs(alive, timeout=0.5)
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
        deadline = time.monotonic() + 2.0
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
        if payload.get("allocation_id"):
            from train_platform.platform.runtime.execution_processes import register_execution_process

            register_execution_process(
                run_root, run_id=payload["run_id"], allocation_id=payload["allocation_id"],
                execution_owner=owner, pid=launcher.pid, role="torchrun",
                assigned_gpu_uuids=list(payload["assigned_gpu_uuids"]),
            )
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
            if primary_error is None and not cancelled and not interrupted and return_code not in (None, 0):
                primary_error = UltralyticsDDPError(
                    f"torchrun exited with code {return_code}: attempt_id={attempt_id}"
                )
            candidate_identities: dict[tuple[int, float, str], dict[str, Any]] = {}
            for (pid, create_time) in known_processes:
                identity = {
                    "pid": pid, "create_time": create_time,
                    "process_scope": owner["process_scope"],
                    "run_id": str(payload["run_id"]), "attempt_id": attempt_id,
                    "execution_owner": owner,
                }
                candidate_identities[_candidate_identity_key(identity)] = identity
            cleanup_unconfirmed = False
            try:
                registered_identities = _load_identities_strict(processes_dir)
            except BaseException as exc:
                registered_identities = []
                cleanup_errors.append(exc)
            for identity in registered_identities:
                if not _identity_matches_scope(
                    identity,
                    run_id=str(payload["run_id"]),
                    attempt_id=attempt_id,
                    owner=owner,
                ):
                    continue
                try:
                    key = _candidate_identity_key(identity)
                except (KeyError, TypeError, ValueError):
                    cleanup_unconfirmed = True
                    continue
                candidate_identities[key] = dict(identity)
            try:
                stop_launcher()
            except BaseException as exc:
                cleanup_errors.append(exc)
            round_survivors: list[psutil.Process] = []
            cleanup_pass_unconfirmed = False
            try:
                round_survivors.extend(terminate_registered_processes(
                    run_root,
                    run_id=str(payload["run_id"]),
                    owner=owner,
                    attempt_id=attempt_id,
                    grace_seconds=0.5,
                ))
            except UltralyticsDDPCleanupIncomplete as exc:
                cleanup_errors.append(exc)
                cleanup_pass_unconfirmed = not bool(exc.survivors)
                for identity in exc.survivors:
                    try:
                        key = _candidate_identity_key(identity)
                    except (KeyError, TypeError, ValueError):
                        cleanup_pass_unconfirmed = True
                        continue
                    candidate_identities[key] = dict(identity)
            except BaseException as exc:
                cleanup_errors.append(exc)
                cleanup_pass_unconfirmed = True
            try:
                round_survivors.extend(stop_known_processes())
            except BaseException as exc:
                cleanup_errors.append(exc)
            for process in round_survivors:
                try:
                    identity = _identity_from_process(
                        process,
                        process_scope=owner["process_scope"],
                        run_id=str(payload["run_id"]), attempt_id=attempt_id,
                        execution_owner=owner,
                    )
                    candidate_identities[_candidate_identity_key(identity)] = identity
                except psutil.Error:
                    continue
            if launcher is not None:
                try:
                    if launcher.poll() is None:
                        launcher.kill()
                    launcher.wait(timeout=1.0)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    cleanup_errors.append(exc)
            try:
                reader.read(upsert_epoch_metrics, final=True)
            except BaseException as exc:
                if primary_error is None:
                    primary_error = exc
            for (pid, create_time) in known_processes:
                identity = {
                    "pid": pid, "create_time": create_time,
                    "process_scope": owner["process_scope"],
                    "run_id": str(payload["run_id"]), "attempt_id": attempt_id,
                    "execution_owner": owner,
                }
                candidate_identities.setdefault(_candidate_identity_key(identity), identity)
            try:
                final_registered = _load_identities_strict(processes_dir)
            except BaseException as exc:
                final_registered = []
                cleanup_errors.append(exc)
                cleanup_unconfirmed = True
            else:
                cleanup_unconfirmed = cleanup_pass_unconfirmed
            for identity in final_registered:
                if not _identity_matches_scope(
                    identity,
                    run_id=str(payload["run_id"]),
                    attempt_id=attempt_id,
                    owner=owner,
                ):
                    continue
                try:
                    key = _candidate_identity_key(identity)
                except (KeyError, TypeError, ValueError):
                    cleanup_unconfirmed = True
                    continue
                candidate_identities[key] = dict(identity)
            final_survivors: list[dict[str, Any]] = []
            for identity in candidate_identities.values():
                state, _ = _process_state(identity)
                if state != "dead":
                    final_survivors.append(dict(identity, state=state))
            pending_path = attempt_dir / "cleanup-pending.json"
            if final_survivors:
                try:
                    _atomic_json(
                        pending_path,
                        {
                            "run_id": str(payload["run_id"]),
                            "attempt_id": attempt_id,
                            "execution_owner": owner,
                            "survivors": final_survivors,
                        },
                    )
                except BaseException as exc:
                    cleanup_errors.append(exc)
                    cleanup_unconfirmed = True
            else:
                try:
                    pending_path.unlink(missing_ok=True)
                except OSError as exc:
                    cleanup_errors.append(exc)
        finally:
            for signum, handler in previous_handlers.items():
                try:
                    signal.signal(signum, handler)
                except (ValueError, OSError):
                    pass
    if final_survivors or cleanup_unconfirmed:
        survivor_pids = ",".join(str(item.get("pid")) for item in final_survivors)
        error = UltralyticsDDPCleanupIncomplete(
            f"distributed process cleanup incomplete: pids={survivor_pids} attempt_id={attempt_id}",
            run_id=str(payload["run_id"]),
            attempt_id=attempt_id,
            execution_owner=owner,
            survivors=final_survivors,
            original_error=primary_error,
            cleanup_errors=cleanup_errors,
        )
        if primary_error is not None:
            raise error from primary_error
        if cleanup_errors:
            raise error from cleanup_errors[0]
        raise error
    if primary_error is not None:
        raise primary_error
    if cancelled:
        raise UltralyticsDDPCancelled(f"distributed training cancelled: attempt_id={attempt_id}")
    if interrupted:
        raise UltralyticsDDPError(f"distributed supervisor received a termination signal: attempt_id={attempt_id}")
    if return_code != 0:
        raise UltralyticsDDPError(f"torchrun exited with code {return_code}: attempt_id={attempt_id}")


__all__ = ["MetricsJSONLReader", "UltralyticsDDPCancelled", "UltralyticsDDPCleanupIncomplete", "UltralyticsDDPError", "process_identity", "register_process", "run_ultralytics_ddp", "terminate_registered_processes"]

from __future__ import annotations

import logging
import os
import socket
import threading
import uuid
import psutil
from datetime import datetime, timezone
from dataclasses import replace

from train_platform.core.config import settings
from train_platform.db.session import session_scope
from train_platform.domains.training.resources.inventory import (
    mark_worker_stopped,
    register_worker_instance,
    save_inventory,
    save_cuda_bindings,
    update_worker_heartbeat,
)
from train_platform.platform.runtime import process_scope
from train_platform.platform.runtime.gpu_probe import GpuProbeResult, probe_gpus
from train_platform.platform.runtime.cuda_devices import probe_cuda_devices

logger = logging.getLogger(__name__)


def enrich_process_attribution(result: GpuProbeResult, node_id: str | None) -> GpuProbeResult:
    """Read identities outside the inventory write transaction; never guess driver PIDs."""
    host_proc_root = getattr(settings, "gpu_host_proc_root", None)
    if not host_proc_root or not node_id or result.status != "success":
        return result
    from train_platform.models.v3.gpu_allocation import GpuAllocation
    from train_platform.platform.runtime.gpu_processes import attribute_driver_processes, load_execution_registrations

    with session_scope() as db:
        allocations = db.query(GpuAllocation).filter(
            GpuAllocation.node_id == node_id, GpuAllocation.state != "released",
        ).all()
        owners = {item.allocation_id: dict(item.execution_owner) for item in allocations if item.execution_owner}
        run_ids = {item.run_id for item in allocations if item.execution_owner}
    registrations = []
    for run_id in run_ids:
        registrations.extend(load_execution_registrations(settings.training_dir / run_id / "runtime/executions"))
    registrations = [item for item in registrations
                     if isinstance(item.get("create_time"), (int, float))
                     and item["create_time"] <= result.sampled_at.timestamp()]
    devices = []
    for device in result.devices:
        snapshot = dict(device.process_snapshot or {})
        if not device.gpu_uuid or not snapshot.get("complete"):
            devices.append(device)
            continue
        process_memory = [row.get("memory_used_mib") for row in snapshot.get("processes", [])]
        known_memory = sum(value for value in process_memory if isinstance(value, int) and value >= 0)
        if device.memory_used_mib is None or known_memory > device.memory_used_mib:
            snapshot.update(attribution_complete=False, usage_by_allocation={},
                            attribution_error="Process memory contradicts the whole-card sample")
            devices.append(replace(device, process_snapshot=snapshot))
            continue
        attribution = attribute_driver_processes(
            snapshot.get("processes", []), registrations,
            host_proc_root=host_proc_root, gpu_uuid=device.gpu_uuid, active_owners=owners,
        )
        snapshot.update(usage_by_allocation=attribution.usage_by_allocation,
                        attribution_complete=attribution.complete,
                        has_duplicates=attribution.has_duplicates,
                        attribution_error=attribution.error)
        devices.append(replace(device, process_snapshot=snapshot))
    return replace(result, devices=devices)


class GpuResourceReporter:
    def __init__(self, *, worker_id: str, allowed_engines: set[str] | None):
        self.instance_id = str(uuid.uuid4())
        self.worker_id = worker_id
        self.allowed_engines = sorted(allowed_engines or set())
        self.started_at = datetime.now(timezone.utc)
        self.node_id = getattr(settings, "gpu_node_id", None)
        self.hostname = socket.gethostname()
        self.process_scope = process_scope.get_process_scope()
        self.launcher_identity = {"pid": os.getpid(), "create_time": psutil.Process().create_time(),
                                  "process_scope": self.process_scope}
        self.nvidia_visible_devices = os.getenv("NVIDIA_VISIBLE_DEVICES")
        self.cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._registered = False

    def start(self) -> None:
        if self._thread is not None or not getattr(settings, "gpu_inventory_enabled", True):
            return
        self._thread = threading.Thread(target=self._run, name="gpu-resource-reporter", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        interval = getattr(settings, "gpu_inventory_interval_seconds", 5)
        try:
            while not self._stop.is_set():
                if not self._registered:
                    try:
                        with session_scope() as db:
                            register_worker_instance(
                                db,
                                instance_id=self.instance_id,
                                worker_id=self.worker_id,
                                node_id=self.node_id,
                                hostname=self.hostname,
                                process_scope=self.process_scope,
                                allowed_engines=self.allowed_engines,
                                nvidia_visible_devices=self.nvidia_visible_devices,
                                cuda_visible_devices=self.cuda_visible_devices,
                                started_at=self.started_at,
                                max_training_slots=getattr(settings, "worker_max_concurrent_trainings", 2),
                                accepting_tasks=getattr(settings, "gpu_scheduler_enabled", False),
                                launcher_identity=self.launcher_identity,
                                scheduling_policy={
                                    "shared_execution_enabled": getattr(settings, "gpu_shared_execution_enabled", False),
                                    "max_shared_tasks_per_device": getattr(settings, "gpu_max_shared_tasks_per_device", 2),
                                    "memory_safety_mib": getattr(settings, "gpu_memory_safety_mib", 4096),
                                } if getattr(settings, "gpu_scheduler_enabled", False) else None,
                            )
                        self._registered = True
                    except Exception:
                        logger.exception("GPU worker registration failed for instance %s", self.instance_id)
                if self._registered:
                    try:
                        with session_scope() as db:
                            update_worker_heartbeat(db, self.instance_id)
                    except Exception:
                        logger.exception("GPU worker heartbeat failed for instance %s", self.instance_id)
                    try:
                        result = probe_gpus()
                        try:
                            result = enrich_process_attribution(result, self.node_id)
                        except Exception:
                            logger.exception("GPU process attribution unavailable; retaining conservative samples")
                    except Exception as exc:
                        result = GpuProbeResult(
                            status="failed",
                            source=None,
                            sampled_at=datetime.now(timezone.utc),
                            error=str(exc),
                            complete=False,
                        )
                    try:
                        with session_scope() as db:
                            save_inventory(db, self.instance_id, result)
                    except Exception as exc:
                        logger.exception("GPU resource report failed for worker instance %s", self.instance_id)
                        report_error = GpuProbeResult(
                            status="failed",
                            source=result.source,
                            sampled_at=result.sampled_at,
                            error=f"report_error: {exc}",
                            complete=False,
                        )
                        try:
                            with session_scope() as db:
                                save_inventory(db, self.instance_id, report_error)
                        except Exception:
                            logger.exception("Failed to persist GPU report error for instance %s", self.instance_id)
                    cuda_result = probe_cuda_devices()
                    try:
                        with session_scope() as db:
                            save_cuda_bindings(db, self.instance_id, cuda_result)
                    except Exception:
                        logger.exception("CUDA binding report failed for worker instance %s", self.instance_id)
                self._stop.wait(interval)
        finally:
            if self._registered:
                try:
                    with session_scope() as db:
                        mark_worker_stopped(db, self.instance_id)
                except Exception:
                    logger.exception("Failed to mark GPU worker instance stopped: %s", self.instance_id)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        timeout = max(1, getattr(settings, "gpu_inventory_interval_seconds", 5) + 7)
        self._thread.join(timeout=timeout)
        if not self._thread.is_alive():
            self._thread = None

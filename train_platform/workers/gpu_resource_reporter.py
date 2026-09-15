from __future__ import annotations

import logging
import os
import socket
import threading
import uuid
from datetime import datetime, timezone

from train_platform.core.config import settings
from train_platform.db.session import session_scope
from train_platform.domains.training.resources.inventory import (
    mark_worker_stopped,
    register_worker_instance,
    save_inventory,
    update_worker_heartbeat,
)
from train_platform.platform.runtime import process_scope
from train_platform.platform.runtime.gpu_probe import GpuProbeResult, probe_gpus

logger = logging.getLogger(__name__)


class GpuResourceReporter:
    def __init__(self, *, worker_id: str, allowed_engines: set[str] | None):
        self.instance_id = str(uuid.uuid4())
        self.worker_id = worker_id
        self.allowed_engines = sorted(allowed_engines or set())
        self.started_at = datetime.now(timezone.utc)
        self.node_id = getattr(settings, "gpu_node_id", None)
        self.hostname = socket.gethostname()
        self.process_scope = process_scope.get_process_scope()
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

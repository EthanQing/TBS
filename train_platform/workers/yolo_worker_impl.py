from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import urlopen

from train_platform.core.config import settings
from train_platform.core.license import assert_valid_license
from train_platform.workers.model_conversion_queue import ModelConversionQueueWorker
from train_platform.workers.worker import DbQueueWorker


def _inference_worker_enabled() -> bool:
    raw = str(os.getenv("YOLO_WORKER_START_INFERENCE", "1") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _inference_worker_endpoint() -> tuple[str, int]:
    raw_url = str(os.getenv("INFERENCE_WORKER_URL") or "").strip()
    parsed = urlparse(raw_url) if raw_url else None
    host = str(os.getenv("INFERENCE_WORKER_HOST") or "").strip()
    port_raw = str(os.getenv("INFERENCE_WORKER_PORT") or "").strip()

    if parsed and parsed.hostname:
        host = host or parsed.hostname
        if not port_raw and parsed.port:
            port_raw = str(parsed.port)

    bind_host = str(settings.worker_bind_host or "").strip() or os.getenv("WORKER_BIND_HOST", "").strip()
    host = bind_host or host or "127.0.0.1"
    port = int(port_raw or "18002")
    return host, port


def _is_port_listening(host: str, port: int) -> bool:
    import socket

    probe_host = "127.0.0.1" if host in {"", "0.0.0.0", "::"} else host
    try:
        with socket.create_connection((probe_host, int(port)), timeout=1.0):
            return True
    except Exception:
        return False


def _sidecar_has_validation_endpoint(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in {"", "0.0.0.0", "::"} else host
    try:
        with urlopen(f"http://{probe_host}:{int(port)}/openapi.json", timeout=2.0) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        return "/internal/model-evaluations/yolo-val" in text
    except Exception:
        return False


def _start_inference_worker_if_needed() -> Optional[subprocess.Popen]:
    if not _inference_worker_enabled():
        print("[worker] inference sidecar disabled by YOLO_WORKER_START_INFERENCE=0", flush=True)
        return None

    host, port = _inference_worker_endpoint()
    if _is_port_listening(host, port):
        if _sidecar_has_validation_endpoint(host, port):
            print(f"[worker] inference sidecar already listening on {host}:{port}", flush=True)
            return None
        print(
            f"[worker] port {host}:{port} is in use but does not expose the current inference sidecar API",
            file=sys.stderr,
            flush=True,
        )
        return None

    log_dir = settings.temp_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "inference_worker.stdout.log"
    stderr_path = log_dir / "inference_worker.stderr.log"
    stdout_f = stdout_path.open("a", encoding="utf-8")
    stderr_f = stderr_path.open("a", encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("INFERENCE_WORKER_HOST", host)
    env.setdefault("INFERENCE_WORKER_PORT", str(port))
    if host in {"127.0.0.1", "localhost"}:
        env.setdefault("WORKER_BIND_HOST", host)

    args = [sys.executable, "-m", "train_platform.workers.inference_worker"]
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0  # type: ignore[attr-defined]
    proc = subprocess.Popen(
        args,
        cwd=str(Path.cwd()),
        env=env,
        stdout=stdout_f,
        stderr=stderr_f,
        creationflags=creationflags,
        start_new_session=(os.name != "nt"),
    )
    print(f"[worker] inference sidecar starting pid={proc.pid} on {host}:{port}", flush=True)
    return proc


def main() -> None:
    assert_valid_license()
    # Dedicated entrypoint for Ultralytics YOLO training jobs and YOLO-side utility jobs.
    training_worker = DbQueueWorker(
        worker_id=os.getenv("WORKER_ID") or "worker-yolo",
        allowed_engines={"ultralytics-yolo", "custom-source"},
    )
    conversion_worker = ModelConversionQueueWorker(worker_id=training_worker.worker_id)
    engines_text = ",".join(sorted(training_worker.allowed_engines)) if training_worker.allowed_engines else "*"
    print(f"[worker] starting worker_id={training_worker.worker_id} engines={engines_text}", flush=True)
    settings.ensure_dirs()
    inference_proc = _start_inference_worker_if_needed()

    while True:
        try:
            if inference_proc is not None and inference_proc.poll() is not None:
                print(
                    f"[worker] inference sidecar exited rc={inference_proc.returncode}; restarting",
                    file=sys.stderr,
                    flush=True,
                )
                inference_proc = _start_inference_worker_if_needed()
            training_worker.tick()
            if getattr(training_worker, "_running", None) is None:
                conversion_worker.tick()
        except Exception as e:
            print(f"[worker] tick error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        time.sleep(training_worker.poll_interval)


if __name__ == "__main__":
    main()
